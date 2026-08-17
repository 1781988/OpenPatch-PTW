from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from openpatch_ptw.attacks import (
    batch_roll_donor,
    copy_move,
    cross_image_patch_transfer,
    local_plain_replacement,
    residual_transfer,
)
from openpatch_ptw.checkpoint import find_checkpoint, load_checkpoint
from openpatch_ptw.config import dump_config, load_config
from openpatch_ptw.data import build_dataset_from_config, collate_openpatch
from openpatch_ptw.degradations import apply_degradation
from openpatch_ptw.masks import generate_multiscale_mask, mask_area
from openpatch_ptw.metrics import (
    bit_accuracy_per_sample,
    equal_error_rate,
    far_at_tpr,
    mask_scores_per_sample,
    status_metrics,
)
from openpatch_ptw.models import (
    BaselineModels,
    OpenPatchModels,
    build_genptw_baseline,
    build_openpatch_models,
    encode_decode_pair,
    extract_baseline,
    extract_openpatch,
    sample_bits,
)
from openpatch_ptw.quality import image_quality_per_sample
from openpatch_ptw.results import ResultWriter
from openpatch_ptw.runtime import environment_report, make_generator, seed_everything, seed_worker, write_json
from openpatch_ptw.visualize import save_forensic_grid


CLASS_NAMES = {0: "unwatermarked", 1: "valid", 2: "forged"}


def _attack_label(name: str, kwargs: dict) -> str:
    if not kwargs:
        return name
    suffix = "_".join(f"{key}{value}" for key, value in sorted(kwargs.items()))
    return f"{name}_{suffix}"


def parse_args():
    parser = argparse.ArgumentParser("Evaluate OpenPatch-PTW and the official GenPTW baseline")
    parser.add_argument("--config", default="configs/openpatch_ptw.yaml")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--model", choices=["openpatch", "genptw"], default="openpatch")
    parser.add_argument(
        "--suite",
        choices=["standard", "small_tamper", "open_set", "forgery", "real_edits", "all"],
        required=True,
    )
    parser.add_argument("--checkpoint", default=None, help="OpenPatch checkpoint; ignored for GenPTW")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-visuals", type=int, default=None)
    parser.add_argument("--no-lpips", action="store_true")
    return parser.parse_args()


def _make_loader(cfg: dict):
    dataset = build_dataset_from_config(cfg, "test", deterministic=True)
    return DataLoader(
        dataset,
        batch_size=int(cfg["eval"].get("batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_openpatch,
        worker_init_fn=seed_worker,
        generator=make_generator(int(cfg["train"]["seed"])),
        persistent_workers=int(cfg["data"]["num_workers"]) > 0,
    )


def _load_models(args, cfg, device):
    if args.model == "openpatch":
        models = build_openpatch_models(cfg, device, strict_assets=True)
        checkpoint = find_checkpoint(cfg["project"]["output_dir"], args.checkpoint)
        payload = load_checkpoint(checkpoint, models, strict=True)
        models.eval()
        return models, checkpoint, int(payload.get("step", 0))
    models = build_genptw_baseline(cfg, device).eval()
    return models, None, 0


@torch.no_grad()
def _pair(models, model_name: str, image: torch.Tensor, bits: torch.Tensor):
    if model_name == "openpatch":
        return encode_decode_pair(models.vae, image, bits)[:2]
    latent = models.vae.encode(image).latent_dist.sample() * 0.18215
    plain = models.vae.decode_plain(latent / 0.18215, return_dict=False)[0].clamp(-1, 1)
    watermarked = models.vae.decode_wm(latent / 0.18215, bits, return_dict=False)[0].clamp(-1, 1)
    return plain, watermarked


@torch.no_grad()
def _extract(models, model_name: str, image: torch.Tensor, mask: torch.Tensor):
    if model_name == "openpatch":
        return extract_openpatch(models, image, mask)
    return extract_baseline(models, image, mask)


def _status_fields(output: dict, sample: int) -> dict:
    if "status_logits" not in output:
        return {"valid_score": float(output["valid_score"][sample].item())}
    probability = torch.softmax(output["status_logits"], dim=1)[sample]
    prediction = int(probability.argmax().item())
    return {
        "status_pred": prediction,
        "status_name": CLASS_NAMES[prediction],
        "prob_unwatermarked": float(probability[0].item()),
        "prob_valid": float(probability[1].item()),
        "prob_forged": float(probability[2].item()),
        "valid_score": float(probability[1].item()),
    }


def _visualize_once(output_dir: Path, key: str, visual_state: tuple | None, num_visuals: int):
    if visual_state is None:
        return
    plain, watermarked, attacked, mask, output = visual_state
    consistency = output.get("consistency", torch.zeros_like(mask))
    save_forensic_grid(
        output_dir / "visuals" / f"{key}.png",
        plain,
        watermarked,
        attacked,
        mask,
        output["pred_mask"],
        consistency,
        max_items=num_visuals,
    )


def suite_standard(loader, cfg, models, model_name, device, output_dir, max_samples, num_visuals, lpips_model):
    writer = ResultWriter(output_dir, "standard")
    attacks = cfg["eval"]["common_attacks"]
    seen = 0
    visual_state = None
    for batch in tqdm(loader, desc=f"{model_name}:standard"):
        image = batch["pixel_values"].to(device)
        image_ids = batch["image_ids"].tolist()
        batch_size = image.shape[0]
        bits = sample_bits(batch_size, int(cfg["model"]["bit_dim"]), device, image.dtype)
        plain, watermarked = _pair(models, model_name, image, bits)
        qualities = image_quality_per_sample(watermarked, plain, lpips_model)
        zero_mask = torch.zeros((batch_size, 1, image.shape[-2], image.shape[-1]), device=device, dtype=image.dtype)

        for spec in attacks:
            name = spec["name"]
            attack_kwargs = spec.get("kwargs", {})
            label = _attack_label(name, attack_kwargs)
            attacked = apply_degradation(watermarked, name, **attack_kwargs)
            output = _extract(models, model_name, attacked, zero_mask)
            bit_acc = bit_accuracy_per_sample(output["decoded_bits"], bits).cpu().tolist()
            false_positive = output["pred_mask"].mean(dim=(1, 2, 3)).cpu().tolist()
            for i in range(batch_size):
                writer.add(
                    {
                        "image_id": image_ids[i],
                        "attack": label,
                        "mask_area": 0.0,
                        "bit_acc": float(bit_acc[i]),
                        "pred_positive_rate": float(false_positive[i]),
                        **qualities[i],
                        **_status_fields(output, i),
                    }
                )

        mask = batch["masks"].to(device).clamp(0, 1)
        edited = local_plain_replacement(watermarked, batch_roll_donor(plain), mask).image
        output = _extract(models, model_name, edited, mask)
        bit_acc = bit_accuracy_per_sample(output["decoded_bits"], bits).cpu().tolist()
        scores = mask_scores_per_sample(output["pred_mask"], mask)
        areas = mask_area(mask).cpu().tolist()
        for i in range(batch_size):
            writer.add(
                {
                    "image_id": image_ids[i],
                    "attack": "local_splice",
                    "mask_area": float(areas[i]),
                    "bit_acc": float(bit_acc[i]),
                    **scores[i],
                    **qualities[i],
                    **_status_fields(output, i),
                }
            )
        if visual_state is None:
            visual_state = (plain, watermarked, edited, mask, output)
        seen += batch_size
        if seen >= max_samples:
            break

    _visualize_once(output_dir, "standard", visual_state, num_visuals)
    return writer.finalize({"model": model_name, "suite": "standard", "max_samples": max_samples})


def suite_small_tamper(loader, cfg, models, model_name, device, output_dir, max_samples, num_visuals, _lpips_model):
    writer = ResultWriter(output_dir, "small_tamper")
    visual_by_bin = {}
    for low, high in cfg["eval"]["tamper_area_bins"]:
        seen = 0
        label = f"{int(low*100)}-{int(high*100)}%"
        for batch in tqdm(loader, desc=f"{model_name}:small:{label}"):
            image = batch["pixel_values"].to(device)
            image_ids = batch["image_ids"].tolist()
            masks = torch.stack(
                [
                    generate_multiscale_mask(
                        int(cfg["data"]["resolution"]),
                        [[float(low), float(high), 1.0]],
                        cfg["mask"]["shapes"],
                        seed=int(cfg["train"]["seed"]) + int(image_id) + int(low * 10000),
                    )
                    for image_id in image_ids
                ]
            ).to(device)
            bits = sample_bits(image.shape[0], int(cfg["model"]["bit_dim"]), device, image.dtype)
            plain, watermarked = _pair(models, model_name, image, bits)
            edited = local_plain_replacement(watermarked, batch_roll_donor(plain), masks).image
            output = _extract(models, model_name, edited, masks)
            bit_acc = bit_accuracy_per_sample(output["decoded_bits"], bits).cpu().tolist()
            scores = mask_scores_per_sample(output["pred_mask"], masks)
            areas = mask_area(masks).cpu().tolist()
            for i in range(image.shape[0]):
                writer.add(
                    {
                        "image_id": image_ids[i],
                        "attack": "small_tamper",
                        "area_bin": label,
                        "mask_area": float(areas[i]),
                        "bit_acc": float(bit_acc[i]),
                        **scores[i],
                        **_status_fields(output, i),
                    }
                )
            visual_by_bin.setdefault(label, (plain, watermarked, edited, masks, output))
            seen += image.shape[0]
            if seen >= max_samples:
                break
    for label, state in visual_by_bin.items():
        _visualize_once(output_dir, f"small_{label.replace('%','')}", state, num_visuals)
    return writer.finalize(
        {"model": model_name, "suite": "small_tamper", "max_samples_per_bin": max_samples},
        group_keys=("area_bin",),
    )


def suite_open_set(loader, cfg, models, model_name, device, output_dir, max_samples, num_visuals, _lpips_model):
    writer = ResultWriter(output_dir, "open_set")
    logits, labels, baseline_scores, baseline_labels = [], [], [], []
    seen = 0
    visual_state = None
    for batch in tqdm(loader, desc=f"{model_name}:open_set"):
        image = batch["pixel_values"].to(device)
        mask = batch["masks"].to(device).clamp(0, 1)
        image_ids = batch["image_ids"].tolist()
        bits = sample_bits(image.shape[0], int(cfg["model"]["bit_dim"]), device, image.dtype)
        plain, watermarked = _pair(models, model_name, image, bits)
        forged = cross_image_patch_transfer(watermarked, batch_roll_donor(watermarked), mask).image
        samples = [
            ("unwatermarked", 0, plain, torch.zeros_like(mask)),
            ("valid", 1, watermarked, torch.zeros_like(mask)),
            ("forged", 2, forged, mask),
        ]
        for attack, label, sample_image, sample_mask in samples:
            output = _extract(models, model_name, sample_image, sample_mask)
            if model_name == "openpatch":
                logits.append(output["status_logits"].cpu())
                labels.append(torch.full((image.shape[0],), label, dtype=torch.long))
            else:
                baseline_scores.extend(output["valid_score"].cpu().tolist())
                baseline_labels.extend([1 if label == 1 else 0] * image.shape[0])
            for i in range(image.shape[0]):
                writer.add(
                    {
                        "image_id": image_ids[i],
                        "attack": attack,
                        "target_class": label,
                        "target_name": CLASS_NAMES[label],
                        "mask_area": float(sample_mask[i].mean().item()),
                        **_status_fields(output, i),
                    }
                )
        if visual_state is None:
            visual_state = (plain, watermarked, forged, mask, _extract(models, model_name, forged, mask))
        seen += image.shape[0]
        if seen >= max_samples:
            break

    if model_name == "openpatch":
        logits_tensor = torch.cat(logits)
        labels_tensor = torch.cat(labels)
        aggregate = status_metrics(logits_tensor, labels_tensor)
        valid_score = torch.softmax(logits_tensor, dim=1)[:, 1].numpy()
        valid_label = (labels_tensor.numpy() == 1).astype(np.int32)
        aggregate["far_at_95_tpr"] = far_at_tpr(valid_score, valid_label, 0.95)
    else:
        score = np.asarray(baseline_scores, dtype=np.float64)
        label = np.asarray(baseline_labels, dtype=np.int32)
        aggregate = {
            "valid_auroc": float("nan") if np.unique(label).size < 2 else float(roc_auc_score(label, score)),
            "valid_eer": equal_error_rate(score, label),
            "far_at_95_tpr": far_at_tpr(score, label, 0.95),
            "note": "GenPTW has no open-set classifier; valid score is mean bit confidence.",
        }
    _visualize_once(output_dir, "open_set_forged", visual_state, num_visuals)
    payload = {"model": model_name, "suite": "open_set", "aggregate": aggregate}
    return writer.finalize(payload, group_keys=("target_name",))


def _calibrate_baseline_threshold(loader, cfg, models, device, max_samples):
    scores = []
    seen = 0
    for batch in loader:
        image = batch["pixel_values"].to(device)
        bits = sample_bits(image.shape[0], int(cfg["model"]["bit_dim"]), device, image.dtype)
        _, watermarked = _pair(models, "genptw", image, bits)
        zero = torch.zeros((image.shape[0], 1, image.shape[-2], image.shape[-1]), device=device, dtype=image.dtype)
        output = _extract(models, "genptw", watermarked, zero)
        scores.extend(output["valid_score"].cpu().tolist())
        seen += image.shape[0]
        if seen >= max_samples:
            break
    return float(np.quantile(np.asarray(scores), 0.05))


def suite_forgery(loader, cfg, models, model_name, device, output_dir, max_samples, num_visuals, _lpips_model):
    writer = ResultWriter(output_dir, "forgery")
    threshold = None
    if model_name == "genptw":
        threshold = _calibrate_baseline_threshold(loader, cfg, models, device, min(max_samples, 500))
    seen = 0
    visual_states = {}
    for batch in tqdm(loader, desc=f"{model_name}:forgery"):
        image = batch["pixel_values"].to(device)
        mask = batch["masks"].to(device).clamp(0, 1)
        image_ids = batch["image_ids"].tolist()
        bits = sample_bits(image.shape[0], int(cfg["model"]["bit_dim"]), device, image.dtype)
        plain, watermarked = _pair(models, model_name, image, bits)
        donor_plain = batch_roll_donor(plain)
        donor_wm = batch_roll_donor(watermarked)
        donor_bits = batch_roll_donor(bits)

        attack_results = {
            "cross_image_patch": cross_image_patch_transfer(watermarked, donor_wm, mask),
            "copy_move": copy_move(watermarked, mask, rng=random.Random(int(cfg["train"]["seed"]) + seen)),
        }
        for beta in cfg["eval"]["residual_beta_grid"]:
            beta_tensor = torch.full((image.shape[0], 1, 1, 1), float(beta), device=device, dtype=image.dtype)
            attack_results[f"residual_transfer_b{beta}"] = residual_transfer(
                plain, donor_plain, donor_wm, beta=(beta_tensor)
            )

        for attack_name, result in attack_results.items():
            output = _extract(models, model_name, result.image, result.mask)
            scores = mask_scores_per_sample(output["pred_mask"], result.mask) if float(result.mask.mean()) > 0 else [None] * image.shape[0]
            reference_bits = bits if attack_name == "copy_move" else donor_bits
            donor_match = ((output["decoded_bits"] > 0.5) == (reference_bits > 0.5)).float().mean(dim=1)
            for i in range(image.shape[0]):
                status = _status_fields(output, i)
                accepted = (
                    status.get("status_pred") == 1
                    if model_name == "openpatch"
                    else status["valid_score"] >= float(threshold)
                )
                row = {
                    "image_id": image_ids[i],
                    "attack": attack_name,
                    "mask_area": float(result.mask[i].mean().item()),
                    "accepted_as_valid": int(accepted),
                    "donor_bit_match": float(donor_match[i].item()),
                    "attribution_success": int(accepted and donor_match[i].item() >= 0.95),
                    **status,
                }
                if scores[i] is not None:
                    row.update(scores[i])
                writer.add(row)
            visual_states.setdefault(attack_name, (plain, watermarked, result.image, result.mask, output))
        seen += image.shape[0]
        if seen >= max_samples:
            break

    for attack_name, state in list(visual_states.items())[:4]:
        _visualize_once(output_dir, attack_name.replace(".", "_"), state, num_visuals)
    payload = {
        "model": model_name,
        "suite": "forgery",
        "baseline_valid_threshold": threshold,
    }
    return writer.finalize(payload, group_keys=("attack",))


def _lama_edit(watermarked, mask, cfg):
    os.environ["LAMA_MODEL"] = cfg["upstream"]["lama"]
    from lama import SimpleLama

    model = SimpleLama(device=watermarked.device)
    unit = (watermarked / 2.0 + 0.5).clamp(0, 1)
    output = model(unit, mask).view_as(watermarked)
    return (output.clamp(0, 1) - 0.5) * 2.0


def _sd_inpaint(watermarked, mask, cfg, device):
    from diffusers import StableDiffusionInpaintPipeline

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        cfg["upstream"]["sd_inpaint"], torch_dtype=dtype, local_files_only=True
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    outputs = []
    for image_tensor, mask_tensor in zip(watermarked, mask):
        image_pil = TF.to_pil_image((image_tensor / 2.0 + 0.5).clamp(0, 1).cpu())
        mask_pil = TF.to_pil_image(mask_tensor.clamp(0, 1).cpu())
        generated = pipeline(
            prompt=str(cfg["eval"].get("inpaint_prompt", "")),
            image=image_pil,
            mask_image=mask_pil,
            height=int(cfg["data"]["resolution"]),
            width=int(cfg["data"]["resolution"]),
            strength=float(cfg["eval"].get("inpaint_strength", 1.0)),
            num_inference_steps=int(cfg["eval"].get("inpaint_steps", 25)),
        ).images[0]
        outputs.append(TF.to_tensor(generated) * 2.0 - 1.0)
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.stack(outputs).to(device=device, dtype=watermarked.dtype)


def suite_real_edits(loader, cfg, models, model_name, device, output_dir, max_samples, num_visuals, _lpips_model):
    writer = ResultWriter(output_dir, "real_edits")
    methods = list(cfg["eval"].get("real_edit_methods", ["lama", "sd_inpaint"]))
    seen = 0
    visuals = {}
    lama_model = None
    sd_pipeline = None
    for batch in tqdm(loader, desc=f"{model_name}:real_edits"):
        image = batch["pixel_values"].to(device)
        mask = batch["masks"].to(device).clamp(0, 1)
        image_ids = batch["image_ids"].tolist()
        bits = sample_bits(image.shape[0], int(cfg["model"]["bit_dim"]), device, image.dtype)
        plain, watermarked = _pair(models, model_name, image, bits)
        for method in methods:
            if method == "lama":
                if lama_model is None:
                    os.environ["LAMA_MODEL"] = cfg["upstream"]["lama"]
                    from lama import SimpleLama

                    lama_model = SimpleLama(device=device)
                unit = (watermarked / 2.0 + 0.5).clamp(0, 1)
                edited = (lama_model(unit, mask).view_as(watermarked).clamp(0, 1) - 0.5) * 2.0
            elif method == "sd_inpaint":
                if sd_pipeline is None:
                    from diffusers import StableDiffusionInpaintPipeline

                    dtype = torch.float16 if device.type == "cuda" else torch.float32
                    sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                        cfg["upstream"]["sd_inpaint"], torch_dtype=dtype, local_files_only=True
                    ).to(device)
                    sd_pipeline.set_progress_bar_config(disable=True)
                outputs = []
                for sample, sample_mask in zip(watermarked, mask):
                    result = sd_pipeline(
                        prompt=str(cfg["eval"].get("inpaint_prompt", "")),
                        image=TF.to_pil_image((sample / 2.0 + 0.5).clamp(0, 1).cpu()),
                        mask_image=TF.to_pil_image(sample_mask.clamp(0, 1).cpu()),
                        height=int(cfg["data"]["resolution"]),
                        width=int(cfg["data"]["resolution"]),
                        strength=float(cfg["eval"].get("inpaint_strength", 1.0)),
                        num_inference_steps=int(cfg["eval"].get("inpaint_steps", 25)),
                    ).images[0]
                    outputs.append(TF.to_tensor(result) * 2.0 - 1.0)
                edited = torch.stack(outputs).to(device=device, dtype=watermarked.dtype)
            else:
                raise ValueError(f"Unknown real edit method: {method}")

            output = _extract(models, model_name, edited, mask)
            bit_acc = bit_accuracy_per_sample(output["decoded_bits"], bits).cpu().tolist()
            scores = mask_scores_per_sample(output["pred_mask"], mask)
            for i in range(image.shape[0]):
                writer.add(
                    {
                        "image_id": image_ids[i],
                        "attack": method,
                        "mask_area": float(mask[i].mean().item()),
                        "bit_acc": float(bit_acc[i]),
                        **scores[i],
                        **_status_fields(output, i),
                    }
                )
            visuals.setdefault(method, (plain, watermarked, edited, mask, output))
        seen += image.shape[0]
        if seen >= max_samples:
            break
    for method, state in visuals.items():
        _visualize_once(output_dir, method, state, num_visuals)
    return writer.finalize({"model": model_name, "suite": "real_edits"}, group_keys=("attack",))


def main():
    args = parse_args()
    cfg = load_config(args.config, args.variant, args.override)
    seed_everything(int(cfg["train"]["seed"]), deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, checkpoint, checkpoint_step = _load_models(args, cfg, device)
    loader = _make_loader(cfg)

    run_name = args.run_name or cfg["project"].get("run_name") or "default"
    max_samples = int(args.max_samples or cfg["eval"]["max_samples"])
    num_visuals = int(args.num_visuals or cfg["eval"].get("num_visuals", 4))
    output_root = Path(cfg["project"]["output_dir"]) / run_name / "eval" / args.model
    output_root.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, output_root / "resolved_config.yaml")
    write_json(output_root / "environment.json", environment_report(cfg["runtime"]["repo_root"]))

    lpips_model = None
    if not args.no_lpips and args.suite in {"standard", "all"}:
        import lpips

        lpips_model = lpips.LPIPS(net=str(cfg["quality"].get("lpips_net", "vgg"))).to(device).eval()

    suites = {
        "standard": suite_standard,
        "small_tamper": suite_small_tamper,
        "open_set": suite_open_set,
        "forgery": suite_forgery,
        "real_edits": suite_real_edits,
    }
    selected = list(suites) if args.suite == "all" else [args.suite]
    summary = {
        "model": args.model,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_step": checkpoint_step,
        "suites": {},
    }
    for suite_name in selected:
        suite_max = max_samples
        if suite_name == "real_edits":
            suite_max = min(max_samples, int(cfg["eval"].get("real_edit_max_samples", 200)))
        suite_output = output_root / suite_name
        result = suites[suite_name](
            loader,
            cfg,
            models,
            args.model,
            device,
            suite_output,
            suite_max,
            num_visuals,
            lpips_model,
        )
        summary["suites"][suite_name] = result
    write_json(output_root / "evaluation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
