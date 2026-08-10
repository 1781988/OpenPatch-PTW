from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from tqdm import tqdm

from openpatch_ptw.attacks import (
    batch_roll_donor,
    cross_image_patch_transfer,
    residual_transfer,
)
from openpatch_ptw.data import OpenPatchCocoDataset, collate_openpatch
from openpatch_ptw.genptw_bridge import (
    add_genptw_to_path,
    build_upstream_message_decoder,
    inject_openpatch_adapter,
    load_message_decoder_weights,
    warmstart_adapter_from_genptw,
)
from openpatch_ptw.heads import LocalCodeHead, OpenSetStatusHead, consistency_map
from openpatch_ptw.localizer import OpenPatchLocalizer, load_genptw_localizer_weights
from openpatch_ptw.losses import local_code_loss, mask_loss, status_loss
from openpatch_ptw.metrics import bit_accuracy, mask_scores


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/openpatch_ptw.yaml")
    p.add_argument("--stage", choices=["smoke", "warmup", "finetune"], default="warmup")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_sample_kind(mix: dict) -> str:
    r = random.random()
    if r < mix["valid"]:
        return "valid"
    if r < mix["valid"] + mix["unwatermarked"]:
        return "unwatermarked"
    return "forged"


def build_models(cfg, device):
    add_genptw_to_path(cfg["upstream"]["root"])
    model_cfg = cfg["model"]
    resolution = cfg["data"]["resolution"]

    vae = AutoencoderKL.from_pretrained(cfg["upstream"]["vae"]).to(device)
    vae = inject_openpatch_adapter(
        vae,
        bit_dim=model_cfg["bit_dim"],
        image_size=resolution,
        code_dim=model_cfg["code_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        fourier_bands=model_cfg["fourier_bands"],
    )

    ckpt_dir = Path(cfg["upstream"]["checkpoint_dir"])
    adapter_ckpt = ckpt_dir / "diffusion_pytorch_model.safetensors"
    if adapter_ckpt.exists():
        info = warmstart_adapter_from_genptw(vae, str(adapter_ckpt))
        print(f"[warmstart] adapter loaded tensors: {info['loaded']}")

    msg_decoder = build_upstream_message_decoder(model_cfg["bit_dim"], resolution).to(device)
    msg_ckpt = ckpt_dir / "msg_decoder.pth"
    if msg_ckpt.exists():
        print("[warmstart] message decoder:", load_message_decoder_weights(msg_decoder, str(msg_ckpt)))

    code_head = LocalCodeHead(code_dim=model_cfg["code_dim"]).to(device)
    status_head = OpenSetStatusHead(hidden_dim=model_cfg["status_hidden_dim"]).to(device)
    localizer = OpenPatchLocalizer(
        image_size=resolution,
        conv_pretrain=bool(model_cfg.get("localizer_pretrained", False)),
        conv_ckpt=cfg["upstream"].get("convnext"),
    ).to(device)
    loc_ckpt = ckpt_dir / "localizer.pth"
    if loc_ckpt.exists():
        info = load_genptw_localizer_weights(localizer, str(loc_ckpt))
        print(f"[warmstart] localizer loaded tensors: {info['loaded']}")

    return vae, msg_decoder, code_head, status_head, localizer


def configure_trainable(stage, vae, msg_decoder, code_head, status_head, localizer):
    for p in vae.parameters():
        p.requires_grad = False
    for p in msg_decoder.parameters():
        p.requires_grad = False
    for p in code_head.parameters():
        p.requires_grad = True
    for p in status_head.parameters():
        p.requires_grad = True
    for p in localizer.parameters():
        p.requires_grad = True

    # Only the new spatial injection is updated in the VAE. CAF1/CAF2 stay frozen.
    for p in vae.decoder.watermark_2.parameters():
        p.requires_grad = True

    if stage in ("smoke", "warmup"):
        # Freeze early ConvNeXt stages to preserve upstream localization features.
        for p in localizer.convnext.stages[0].parameters():
            p.requires_grad = False
        for p in localizer.convnext.stages[1].parameters():
            p.requires_grad = False


def save_checkpoint(path, step, vae, code_head, status_head, localizer, optimizer, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "position_sf": vae.decoder.watermark_2.state_dict(),
            "code_head": code_head.state_dict(),
            "status_head": status_head.state_dict(),
            "localizer": localizer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    seed_everything(int(cfg["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warning] CUDA not found; smoke test is possible but full training is not recommended.")

    bins = cfg["mask"]["bins"]
    shape_probs = {k: v for k, v in cfg["mask"]["shapes"].items() if k != "coco"}
    dataset = OpenPatchCocoDataset(
        cfg["data"]["train_img_dir"],
        cfg["data"]["train_ann_file"],
        resolution=cfg["data"]["resolution"],
        bins=bins,
        shape_probs=shape_probs,
        coco_prob=float(cfg["mask"]["shapes"].get("coco", 0.35)),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_openpatch,
        drop_last=True,
    )

    vae, msg_decoder, code_head, status_head, localizer = build_models(cfg, device)
    configure_trainable(args.stage, vae, msg_decoder, code_head, status_head, localizer)

    trainable = [
        p
        for module in (vae, code_head, status_head, localizer)
        for p in module.parameters()
        if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        vae.decoder.watermark_2.load_state_dict(state["position_sf"])
        code_head.load_state_dict(state["code_head"])
        status_head.load_state_dict(state["status_head"])
        localizer.load_state_dict(state["localizer"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])

    default_steps = {
        "smoke": 200,
        "warmup": int(cfg["train"]["warmup_new_modules_steps"]),
        "finetune": int(cfg["train"]["finetune_steps"]),
    }
    max_steps = args.max_steps or default_steps[args.stage]
    accum = int(cfg["train"]["gradient_accumulation_steps"])
    out_dir = Path(cfg["project"]["output_dir"]) / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = cfg["train"].get("mixed_precision") == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    vae.train()
    code_head.train()
    status_head.train()
    localizer.train()
    msg_decoder.eval()

    step = start_step
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=max_steps, initial=step, desc=f"OpenPatch-{args.stage}")

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            image = batch["pixel_values"].to(device, non_blocking=True)
            mask = batch["masks"].to(device, non_blocking=True).clamp(0, 1)
            bsz = image.shape[0]

            amp_enabled = device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                with torch.no_grad():
                    latents = vae.encode(image).latent_dist.sample() * 0.18215
                    plain = vae.decode_plain(latents / 0.18215, return_dict=False)[0].clamp(-1, 1)

                bits = torch.bernoulli(torch.full((bsz, cfg["model"]["bit_dim"]), 0.5, device=device, dtype=image.dtype))
                wm_values = vae.decode_wm(latents / 0.18215, bits, return_dict=False)
                watermarked = wm_values[0].clamp(-1, 1)

                kind = choose_sample_kind(cfg["sample_mix"])
                loc_supervise = True
                code_supervise = False
                wm_supervise = False

                if kind == "valid":
                    # Half of valid samples stay clean; half receive a local content replacement.
                    if random.random() < 0.5:
                        donor_plain = batch_roll_donor(plain.detach())
                        m3 = mask.expand(-1, 3, -1, -1)
                        attacked = watermarked * (1.0 - m3) + donor_plain * m3
                        target_mask = mask
                    else:
                        attacked = watermarked
                        target_mask = torch.zeros_like(mask)
                    status_target = torch.ones(bsz, device=device, dtype=torch.long)
                    wm_supervise = True
                    code_supervise = True
                elif kind == "unwatermarked":
                    attacked = plain.detach()
                    target_mask = torch.zeros_like(mask)
                    status_target = torch.zeros(bsz, device=device, dtype=torch.long)
                    loc_supervise = False
                else:
                    train_types = cfg["attacks"]["train_forgery_types"]
                    forgery_kind = random.choice(train_types)
                    if forgery_kind == "residual_transfer":
                        result = residual_transfer(
                            target_plain=plain.detach(),
                            donor_plain=batch_roll_donor(plain.detach()),
                            donor_watermarked=batch_roll_donor(watermarked.detach()),
                            beta_range=tuple(cfg["attacks"]["residual_beta"]),
                        )
                        loc_supervise = False
                    else:
                        result = cross_image_patch_transfer(
                            watermarked,
                            batch_roll_donor(watermarked.detach()),
                            mask,
                        )
                    attacked, target_mask = result.image, result.mask
                    status_target = torch.full((bsz,), 2, device=device, dtype=torch.long)

                decoded_bits, wm_feature = msg_decoder(attacked, target_mask, step)
                pred_code = code_head(wm_feature)
                expected_pred = vae.decoder.watermark_2.code_generator(decoded_bits.detach(), pred_code.shape[-2:])
                residual = consistency_map(pred_code, expected_pred, detach_expected=True)
                loc_out = localizer(attacked, wm_feature, residual)
                status_logits = status_head(wm_feature, decoded_bits, residual)

                loss = status_loss(status_logits, status_target) * float(cfg["loss"]["status"])

                if wm_supervise:
                    loss = loss + F.binary_cross_entropy(decoded_bits, bits) * float(cfg["loss"]["wm"])
                if code_supervise:
                    expected_gt = vae.decoder.watermark_2.code_generator(bits, pred_code.shape[-2:])
                    loss = loss + local_code_loss(pred_code, expected_gt, target_mask) * float(cfg["loss"]["code"])
                if loc_supervise:
                    loss = loss + mask_loss(
                        loc_out["pred_mask_logits"],
                        target_mask,
                        dice_weight=float(cfg["loss"]["dice"]),
                    ) * float(cfg["loss"]["mask_bce"])

                # Quality anchor for the newly initialized position-bound SF.
                loss = loss + F.mse_loss(watermarked, plain.detach()) * float(cfg["loss"]["rec"])
                loss = loss / accum

            loss.backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg["train"]["max_grad_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 20 == 0:
                with torch.no_grad():
                    logs = {
                        "step": step,
                        "kind": kind,
                        "loss": float(loss.item() * accum),
                        "status_acc": float((status_logits.argmax(1) == status_target).float().mean().item()),
                    }
                    if wm_supervise:
                        logs["bit_acc"] = bit_accuracy(decoded_bits, bits)
                    if loc_supervise:
                        logs.update({f"mask_{k}": v for k, v in mask_scores(loc_out["pred_mask"], target_mask).items()})
                    progress.set_postfix(logs)
                    with open(out_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(logs, ensure_ascii=False) + "\n")

            step += 1
            progress.update(1)
            if step % int(cfg["train"]["save_every"]) == 0 or step == max_steps:
                save_checkpoint(out_dir / f"step_{step}.pth", step, vae, code_head, status_head, localizer, optimizer, cfg)

    progress.close()
    print(f"Training finished. Checkpoints: {out_dir}")


if __name__ == "__main__":
    main()
