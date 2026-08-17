from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openpatch_ptw.attacks import (
    batch_roll_donor,
    cross_image_patch_transfer,
    local_plain_replacement,
    residual_transfer,
)
from openpatch_ptw.checkpoint import load_checkpoint, save_checkpoint
from openpatch_ptw.config import dump_config, load_config
from openpatch_ptw.data import build_dataset_from_config, collate_openpatch
from openpatch_ptw.degradations import random_training_degradation
from openpatch_ptw.genptw_bridge import freeze_batch_norm_stats, set_module_requires_grad
from openpatch_ptw.losses import local_code_loss, mask_loss, status_loss
from openpatch_ptw.metrics import bit_accuracy, mask_scores
from openpatch_ptw.models import (
    OpenPatchModels,
    build_openpatch_models,
    encode_decode_pair,
    extract_openpatch,
    sample_bits,
)
from openpatch_ptw.quality import QualityObjective
from openpatch_ptw.runtime import (
    append_jsonl,
    count_trainable_parameters,
    environment_report,
    make_generator,
    seed_everything,
    seed_worker,
    utc_timestamp,
    write_json,
)
from openpatch_ptw.visualize import save_forensic_grid


def parse_args():
    parser = argparse.ArgumentParser("Train OpenPatch-PTW")
    parser.add_argument("--config", default="configs/openpatch_ptw.yaml")
    parser.add_argument("--variant", default=None, help="Optional YAML merged over the base config")
    parser.add_argument("--override", action="append", default=[], help="Config override, e.g. train.batch_size=1")
    parser.add_argument("--stage", choices=["smoke", "warmup", "finetune"], default="warmup")
    parser.add_argument("--max-steps", type=int, default=None, help="Optimizer steps, not micro-batches")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--init-checkpoint", default=None, help="Load model weights only, reset optimizer/step")
    parser.add_argument("--resume", default=None, help="Resume model, optimizer, scheduler and step")
    parser.add_argument("--no-strict-assets", action="store_true")
    return parser.parse_args()


def choose_sample_kind(mix: dict) -> str:
    value = random.random()
    if value < float(mix["valid"]):
        return "valid"
    if value < float(mix["valid"]) + float(mix["unwatermarked"]):
        return "unwatermarked"
    return "forged"


def set_trainability(models: OpenPatchModels, cfg: dict, stage: str) -> list[torch.nn.Parameter]:
    for module in models.modules():
        set_module_requires_grad(module, False)

    flags = cfg["experiment"]
    spatial = models.vae.decoder.watermark_2
    if bool(flags.get("use_position_code", True)):
        set_module_requires_grad(spatial.position_gate, True)
        set_module_requires_grad(spatial.position_residual, True)
        spatial.alpha_logit.requires_grad_(True)
    if stage == "finetune" and bool(cfg["train"].get("finetune_base_sf", False)):
        set_module_requires_grad(spatial.global_proj, True)
        set_module_requires_grad(spatial.base_fuse, True)

    if bool(flags.get("use_consistency", True)):
        set_module_requires_grad(models.code_head, True)
        set_module_requires_grad(models.localizer.convnext.consistency_stem, True)
    if bool(flags.get("use_status", True)):
        set_module_requires_grad(models.status_head, True)

    # Preserve the official localization backbone early; adapt decoder and high stages later.
    set_module_requires_grad(models.localizer.maskdecoder, True)
    if stage == "finetune":
        set_module_requires_grad(models.localizer.convnext.stages[2], True)
        set_module_requires_grad(models.localizer.convnext.stages[3], True)

    trainable = []
    seen = set()
    for module in models.modules():
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                trainable.append(parameter)
                seen.add(id(parameter))
    if not trainable:
        raise RuntimeError("No trainable parameters; check experiment flags")
    return trainable


def set_modes(models: OpenPatchModels, cfg: dict, stage: str) -> None:
    models.vae.eval()
    models.message_decoder.eval()
    models.code_head.train(bool(cfg["experiment"].get("use_consistency", True)))
    models.status_head.train(bool(cfg["experiment"].get("use_status", True)))
    models.localizer.train()
    models.vae.decoder.watermark_2.train()
    # The official base SF and frozen upstream modules must not update BN statistics.
    freeze_batch_norm_stats(models.vae)
    if stage != "finetune":
        models.localizer.convnext.stages[0].eval()
        models.localizer.convnext.stages[1].eval()
        models.localizer.convnext.stages[2].eval()
        models.localizer.convnext.stages[3].eval()


def build_scheduler(optimizer, max_steps: int, warmup_steps: int):
    def scale(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def quick_validate(models: OpenPatchModels, loader, cfg: dict, device, max_batches: int = 8) -> dict:
    models.eval()
    bit_values, f1_values, status_values = [], [], []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            image = batch["pixel_values"].to(device)
            mask = batch["masks"].to(device).clamp(0, 1)
            bits = sample_bits(image.shape[0], int(cfg["model"]["bit_dim"]), device, image.dtype)
            plain, watermarked, _, _ = encode_decode_pair(models.vae, image, bits)
            edited = local_plain_replacement(watermarked, batch_roll_donor(plain), mask).image
            output = extract_openpatch(models, edited, mask)
            bit_values.append(bit_accuracy(output["decoded_bits"], bits))
            f1_values.append(mask_scores(output["pred_mask"], mask)["f1"])

            if bool(cfg["experiment"].get("use_status", True)):
                plain_output = extract_openpatch(models, plain, torch.zeros_like(mask))
                valid_output = extract_openpatch(models, watermarked, torch.zeros_like(mask))
                forged = cross_image_patch_transfer(watermarked, batch_roll_donor(watermarked), mask).image
                forged_output = extract_openpatch(models, forged, mask)
                logits = torch.cat(
                    [plain_output["status_logits"], valid_output["status_logits"], forged_output["status_logits"]]
                )
                labels = torch.cat(
                    [
                        torch.zeros(image.shape[0], device=device, dtype=torch.long),
                        torch.ones(image.shape[0], device=device, dtype=torch.long),
                        torch.full((image.shape[0],), 2, device=device, dtype=torch.long),
                    ]
                )
                status_values.append(float((logits.argmax(1) == labels).float().mean().item()))
    metrics = {
        "bit_acc": float(np.mean(bit_values)) if bit_values else float("nan"),
        "mask_f1": float(np.mean(f1_values)) if f1_values else float("nan"),
        "status_acc": float(np.mean(status_values)) if status_values else 0.0,
    }
    metrics["selection_score"] = (
        0.35 * metrics["bit_acc"] + 0.45 * metrics["mask_f1"] + 0.20 * metrics["status_acc"]
    )
    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config, args.variant, args.override)
    seed = int(cfg["train"]["seed"])
    seed_everything(seed, deterministic=bool(cfg["train"].get("deterministic", False)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.stage != "smoke":
        raise RuntimeError("Warmup/finetune require CUDA. Use --stage smoke for CPU validation only.")

    run_name = args.run_name or cfg["project"].get("run_name") or f"{args.stage}_{utc_timestamp()}"
    run_root = Path(cfg["project"]["output_dir"]) / run_name
    stage_dir = run_root / args.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_root / "resolved_config.yaml")
    write_json(run_root / "environment.json", environment_report(cfg["runtime"]["repo_root"]))

    train_dataset = build_dataset_from_config(cfg, "train", deterministic=False)
    dev_dataset = build_dataset_from_config(cfg, "dev", deterministic=True)
    generator = make_generator(seed)
    loader_kwargs = dict(
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=collate_openpatch,
        worker_init_fn=seed_worker,
        persistent_workers=int(cfg["data"]["num_workers"]) > 0,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_kwargs,
    )
    dev_loader = DataLoader(dev_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    models = build_openpatch_models(cfg, device, strict_assets=not args.no_strict_assets)
    models.vae.decoder.watermark_2.set_position_enabled(bool(cfg["experiment"].get("use_position_code", True)))
    models.localizer.set_consistency_enabled(bool(cfg["experiment"].get("use_consistency", True)))
    write_json(run_root / "upstream_load_report.json", models.load_report)

    trainable = set_trainability(models, cfg, args.stage)
    set_modes(models, cfg, args.stage)
    total_parameters, trainable_parameters = count_trainable_parameters(models.modules())
    write_json(
        run_root / "parameter_report.json",
        {"total": total_parameters, "trainable": trainable_parameters},
    )

    stage_defaults = cfg["train"]["steps"]
    max_steps = int(args.max_steps or stage_defaults[args.stage])
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["train"]["learning_rate"]),
        betas=tuple(cfg["train"].get("betas", [0.9, 0.999])),
        eps=float(cfg["train"].get("epsilon", 1e-8)),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = build_scheduler(optimizer, max_steps, int(cfg["train"].get("lr_warmup_steps", 0)))

    precision = str(cfg["train"].get("mixed_precision", "bf16"))
    use_bf16 = device.type == "cuda" and precision == "bf16" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and precision == "fp16"
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = make_scaler(use_fp16)

    quality_objective = QualityObjective(
        use_lpips=bool(cfg["quality"].get("use_lpips", True)),
        use_jnd=bool(cfg["quality"].get("use_jnd", True)),
        lpips_net=str(cfg["quality"].get("lpips_net", "vgg")),
    ).to(device)

    global_step = 0
    best_metric = -float("inf")
    if args.init_checkpoint:
        payload = load_checkpoint(args.init_checkpoint, models, strict=False)
        print(f"[init] loaded model weights from {args.init_checkpoint}, source step={payload.get('step')}")
    if args.resume:
        payload = load_checkpoint(args.resume, models, optimizer, scheduler, scaler, strict=True)
        global_step = int(payload.get("step", 0))
        best_metric = float(payload.get("best_metric") or -float("inf"))
        print(f"[resume] step={global_step} from {args.resume}")

    accumulation = int(cfg["train"]["gradient_accumulation_steps"])
    save_every = int(cfg["train"]["save_every"])
    eval_every = int(cfg["train"]["eval_every"])
    visual_every = int(cfg["train"].get("visual_every", 500))
    optimizer.zero_grad(set_to_none=True)
    micro_step = global_step * accumulation
    progress = tqdm(total=max_steps, initial=global_step, desc=f"OpenPatch-{args.stage}")
    last_visual = None

    while global_step < max_steps:
        for batch in train_loader:
            if global_step >= max_steps:
                break
            image = batch["pixel_values"].to(device, non_blocking=True)
            mask = batch["masks"].to(device, non_blocking=True).clamp(0, 1)
            batch_size = image.shape[0]

            amp_enabled = device.type == "cuda" and (use_bf16 or use_fp16)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                with torch.no_grad():
                    latents = models.vae.encode(image).latent_dist.sample() * 0.18215
                    plain = models.vae.decode_plain(latents / 0.18215, return_dict=False)[0].clamp(-1, 1)
                bits = sample_bits(batch_size, int(cfg["model"]["bit_dim"]), device, image.dtype)
                wm_values = models.vae.decode_wm(latents / 0.18215, bits, return_dict=False)
                watermarked = wm_values[0].clamp(-1, 1)

                kind = choose_sample_kind(cfg["sample_mix"])
                localization_supervised = True
                code_supervised = False
                watermark_supervised = False

                if kind == "valid":
                    if random.random() < float(cfg["sample_mix"].get("valid_local_edit_prob", 0.5)):
                        result = local_plain_replacement(watermarked, batch_roll_donor(plain.detach()), mask)
                        attacked, target_mask = result.image, result.mask
                    else:
                        attacked, target_mask = watermarked, torch.zeros_like(mask)
                    status_target = torch.ones(batch_size, device=device, dtype=torch.long)
                    watermark_supervised = True
                    code_supervised = bool(cfg["experiment"].get("use_position_code", True))
                elif kind == "unwatermarked":
                    attacked, target_mask = plain.detach(), torch.zeros_like(mask)
                    status_target = torch.zeros(batch_size, device=device, dtype=torch.long)
                    localization_supervised = False
                else:
                    if not bool(cfg["experiment"].get("use_forgery_training", True)):
                        attacked, target_mask = watermarked, torch.zeros_like(mask)
                        status_target = torch.ones(batch_size, device=device, dtype=torch.long)
                        watermark_supervised = True
                        code_supervised = bool(cfg["experiment"].get("use_position_code", True))
                        kind = "valid_fallback"
                    else:
                        attack_name = random.choice(list(cfg["attacks"]["train_forgery_types"]))
                        if attack_name == "residual_transfer":
                            result = residual_transfer(
                                plain.detach(),
                                batch_roll_donor(plain.detach()),
                                batch_roll_donor(watermarked.detach()),
                                tuple(cfg["attacks"]["residual_beta"]),
                            )
                            localization_supervised = False
                        elif attack_name == "cross_image_patch":
                            result = cross_image_patch_transfer(
                                watermarked,
                                batch_roll_donor(watermarked.detach()),
                                mask,
                            )
                        else:
                            raise ValueError(f"Unknown training forgery: {attack_name}")
                        attacked, target_mask = result.image, result.mask
                        status_target = torch.full((batch_size,), 2, device=device, dtype=torch.long)
                        kind = f"forged:{attack_name}"

                attacked = random_training_degradation(attacked, cfg["degradation"])
                output = extract_openpatch(models, attacked, target_mask, step=global_step)

                losses = {}
                total_loss = attacked.new_zeros(())
                if bool(cfg["experiment"].get("use_status", True)):
                    losses["status"] = status_loss(output["status_logits"], status_target)
                    total_loss = total_loss + float(cfg["loss"]["status"]) * losses["status"]
                if watermark_supervised:
                    losses["wm"] = F.binary_cross_entropy(output["decoded_bits"], bits)
                    total_loss = total_loss + float(cfg["loss"]["wm"]) * losses["wm"]
                if code_supervised and bool(cfg["experiment"].get("use_consistency", True)):
                    expected_gt = models.vae.decoder.watermark_2.code_generator(
                        bits, output["predicted_code"].shape[-2:]
                    )
                    losses["code"] = local_code_loss(output["predicted_code"], expected_gt, target_mask)
                    total_loss = total_loss + float(cfg["loss"]["code"]) * losses["code"]
                if localization_supervised:
                    losses["mask"] = mask_loss(
                        output["pred_mask_logits"],
                        target_mask,
                        dice_weight=float(cfg["loss"]["dice"]),
                        edge_weight=float(cfg["loss"]["edge"]),
                        max_pos_weight=float(cfg["loss"].get("max_pos_weight", 20.0)),
                    )
                    total_loss = total_loss + float(cfg["loss"]["mask"]) * losses["mask"]

                quality = quality_objective(
                    watermarked,
                    plain,
                    rec_weight=float(cfg["loss"]["rec"]),
                    lpips_weight=float(cfg["loss"]["lpips"]),
                    jnd_weight=float(cfg["loss"]["jnd"]),
                )
                losses["quality"] = quality.total
                losses["rec"] = quality.reconstruction
                losses["lpips"] = quality.lpips
                losses["jnd"] = quality.jnd
                total_loss = total_loss + quality.total
                scaled_loss = total_loss / accumulation

            if use_fp16:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            micro_step += 1

            if micro_step % accumulation != 0:
                continue

            if use_fp16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, float(cfg["train"]["max_grad_norm"]))
            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            progress.update(1)

            with torch.no_grad():
                log = {
                    "step": global_step,
                    "micro_step": micro_step,
                    "kind": kind,
                    "loss": float(total_loss.detach().float().item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "alpha": float(torch.sigmoid(models.vae.decoder.watermark_2.alpha_logit).item()),
                    "mask_area": float(target_mask.mean().item()),
                }
                for name, value in losses.items():
                    log[f"loss_{name}"] = float(value.detach().float().item())
                if watermark_supervised:
                    log["bit_acc"] = bit_accuracy(output["decoded_bits"], bits)
                if localization_supervised:
                    log.update({f"mask_{key}": value for key, value in mask_scores(output["pred_mask"], target_mask).items()})
                if bool(cfg["experiment"].get("use_status", True)):
                    log["status_acc"] = float((output["status_logits"].argmax(1) == status_target).float().mean().item())
                append_jsonl(stage_dir / "train_log.jsonl", log)
                progress.set_postfix({key: value for key, value in log.items() if key in {"loss", "bit_acc", "mask_f1", "status_acc"}})
                last_visual = (plain, watermarked, attacked, target_mask, output)

            if global_step % visual_every == 0 and last_visual is not None:
                p, w, a, m, o = last_visual
                save_forensic_grid(
                    stage_dir / "visuals" / f"step_{global_step}.png",
                    p,
                    w,
                    a,
                    m,
                    o["pred_mask"],
                    o["consistency"],
                    max_items=int(cfg["eval"].get("num_visuals", 4)),
                )

            if global_step % eval_every == 0 or global_step == max_steps:
                validation = quick_validate(
                    models,
                    dev_loader,
                    cfg,
                    device,
                    max_batches=int(cfg["train"].get("dev_batches", 8)),
                )
                validation["step"] = global_step
                append_jsonl(stage_dir / "validation_log.jsonl", validation)
                if validation["selection_score"] > best_metric:
                    best_metric = validation["selection_score"]
                    save_checkpoint(
                        stage_dir / "best.pth",
                        global_step,
                        models,
                        optimizer,
                        scheduler,
                        scaler,
                        cfg,
                        best_metric,
                    )
                set_modes(models, cfg, args.stage)

            if global_step % save_every == 0 or global_step == max_steps:
                save_checkpoint(
                    stage_dir / f"step_{global_step}.pth",
                    global_step,
                    models,
                    optimizer,
                    scheduler,
                    scaler,
                    cfg,
                    best_metric,
                )

    progress.close()
    write_json(
        stage_dir / "training_complete.json",
        {"run_name": run_name, "stage": args.stage, "steps": global_step, "best_metric": best_metric},
    )
    print(f"Training finished: {stage_dir}")


if __name__ == "__main__":
    main()
