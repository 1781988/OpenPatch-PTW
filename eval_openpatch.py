from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from openpatch_ptw.attacks import batch_roll_donor, copy_move, cross_image_patch_transfer, residual_transfer
from openpatch_ptw.data import OpenPatchCocoDataset, collate_openpatch
from openpatch_ptw.heads import consistency_map
from openpatch_ptw.masks import generate_multiscale_mask
from openpatch_ptw.metrics import bit_accuracy, far_at_tpr, forgery_acceptance_rate, mask_scores, status_metrics
from train_openpatch import build_models, load_cfg, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/openpatch_ptw.yaml")
    p.add_argument("--suite", choices=["standard", "small_tamper", "open_set", "forgery"], required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--max_samples", type=int, default=1000)
    return p.parse_args()


def find_latest_checkpoint(cfg, explicit=None):
    if explicit:
        return Path(explicit)
    root = Path(cfg["project"]["output_dir"])
    candidates = list(root.glob("**/step_*.pth"))
    if not candidates:
        raise FileNotFoundError("No OpenPatch checkpoint found. Pass --checkpoint explicitly.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_openpatch_state(path, vae, code_head, status_head, localizer):
    state = torch.load(path, map_location="cpu")
    vae.decoder.watermark_2.load_state_dict(state["position_sf"])
    code_head.load_state_dict(state["code_head"])
    status_head.load_state_dict(state["status_head"])
    localizer.load_state_dict(state["localizer"])
    return int(state.get("step", 0))


@torch.no_grad()
def extract_all(image, target_mask, vae, msg_decoder, code_head, status_head, localizer):
    bits_pred, wm_feature = msg_decoder(image, target_mask, 0)
    pred_code = code_head(wm_feature)
    expected = vae.decoder.watermark_2.code_generator(bits_pred, pred_code.shape[-2:])
    residual = consistency_map(pred_code, expected, detach_expected=True)
    loc = localizer(image, wm_feature, residual)
    status = status_head(wm_feature, bits_pred, residual)
    return bits_pred, loc, status, residual


def aggregate_mask(items):
    if not items:
        return {}
    return {k: float(np.nanmean([x[k] for x in items])) for k in items[0]}


@torch.no_grad()
def make_plain_wm(image, cfg, vae):
    b = image.shape[0]
    lat = vae.encode(image).latent_dist.sample() * 0.18215
    plain = vae.decode_plain(lat / 0.18215, return_dict=False)[0].clamp(-1, 1)
    bits = torch.bernoulli(torch.full((b, cfg["model"]["bit_dim"]), 0.5, device=image.device, dtype=image.dtype))
    wm = vae.decode_wm(lat / 0.18215, bits, return_dict=False)[0].clamp(-1, 1)
    return plain, wm, bits


def suite_standard(loader, cfg, models, device, max_samples):
    vae, msg_decoder, code_head, status_head, localizer = models
    bit_accs, edits = [], []
    seen = 0
    for batch in tqdm(loader, desc="standard"):
        image = batch["pixel_values"].to(device)
        mask = batch["masks"].to(device).clamp(0, 1)
        plain, wm, bits = make_plain_wm(image, cfg, vae)
        pred, _, _, _ = extract_all(wm, torch.zeros_like(mask), *models)
        bit_accs.append(bit_accuracy(pred, bits))

        donor = batch_roll_donor(plain)
        m3 = mask.expand(-1, 3, -1, -1)
        edited = wm * (1 - m3) + donor * m3
        pred_e, loc_e, _, _ = extract_all(edited, mask, *models)
        bit_accs.append(bit_accuracy(pred_e, bits))
        edits.append(mask_scores(loc_e["pred_mask"], mask))
        seen += image.shape[0]
        if seen >= max_samples:
            break
    return {"bit_acc": float(np.mean(bit_accs)), "local_splice": aggregate_mask(edits)}


def suite_small_tamper(loader, cfg, models, device, max_samples):
    vae, *_ = models
    results = {}
    for lo, hi in cfg["eval"]["tamper_area_bins"]:
        scores, seen = [], 0
        for batch in tqdm(loader, desc=f"small_{lo:.2f}_{hi:.2f}"):
            image = batch["pixel_values"].to(device)
            b = image.shape[0]
            masks = torch.stack([
                generate_multiscale_mask(cfg["data"]["resolution"], [[lo, hi, 1.0]]) for _ in range(b)
            ]).to(device)
            plain, wm, _ = make_plain_wm(image, cfg, vae)
            donor = batch_roll_donor(plain)
            m3 = masks.expand(-1, 3, -1, -1)
            edited = wm * (1 - m3) + donor * m3
            _, loc, _, _ = extract_all(edited, masks, *models)
            scores.append(mask_scores(loc["pred_mask"], masks))
            seen += b
            if seen >= max_samples:
                break
        results[f"{int(lo*100)}-{int(hi*100)}%"] = aggregate_mask(scores)
    return results


def suite_open_set(loader, cfg, models, device, max_samples):
    vae, *_ = models
    logits, labels, seen = [], [], 0
    for batch in tqdm(loader, desc="open_set"):
        image = batch["pixel_values"].to(device)
        mask = batch["masks"].to(device).clamp(0, 1)
        b = image.shape[0]
        plain, wm, _ = make_plain_wm(image, cfg, vae)

        _, _, l0, _ = extract_all(plain, torch.zeros_like(mask), *models)
        _, _, l1, _ = extract_all(wm, torch.zeros_like(mask), *models)
        forged = cross_image_patch_transfer(wm, batch_roll_donor(wm), mask).image
        _, _, l2, _ = extract_all(forged, mask, *models)

        logits.extend([l0.cpu(), l1.cpu(), l2.cpu()])
        labels.extend([
            torch.zeros(b, dtype=torch.long),
            torch.ones(b, dtype=torch.long),
            torch.full((b,), 2, dtype=torch.long),
        ])
        seen += b
        if seen >= max_samples:
            break

    logits = torch.cat(logits)
    labels = torch.cat(labels)
    out = status_metrics(logits, labels)
    valid_score = torch.softmax(logits, dim=1)[:, 1].numpy()
    valid_label = (labels.numpy() == 1).astype(np.int32)
    out["far_at_95_tpr"] = far_at_tpr(valid_score, valid_label, 0.95)
    return out


def suite_forgery(loader, cfg, models, device, max_samples):
    vae, *_ = models
    names = ["residual_transfer", "cross_image_patch", "copy_move"]
    raw = {n: {"far": [], "asr": [], "mask": []} for n in names}
    seen = 0
    for batch in tqdm(loader, desc="forgery"):
        image = batch["pixel_values"].to(device)
        mask = batch["masks"].to(device).clamp(0, 1)
        plain, wm, bits = make_plain_wm(image, cfg, vae)
        donor_bits = batch_roll_donor(bits)
        attacks = {
            "residual_transfer": residual_transfer(
                plain, batch_roll_donor(plain), batch_roll_donor(wm), tuple(cfg["attacks"]["residual_beta"])
            ),
            "cross_image_patch": cross_image_patch_transfer(wm, batch_roll_donor(wm), mask),
            "copy_move": copy_move(wm, mask),
        }
        for name, result in attacks.items():
            pred_bits, loc, status, _ = extract_all(result.image, result.mask, *models)
            raw[name]["far"].append(forgery_acceptance_rate(status))
            if name != "copy_move":
                donor_match = ((pred_bits > 0.5) == (donor_bits > 0.5)).float().mean(dim=1)
                accepted = status.argmax(dim=1) == 1
                raw[name]["asr"].append(float(((donor_match > 0.95) & accepted).float().mean().item()))
            if float(result.mask.mean()) > 0:
                raw[name]["mask"].append(mask_scores(loc["pred_mask"], result.mask))
        seen += image.shape[0]
        if seen >= max_samples:
            break

    return {
        name: {
            "forgery_acceptance_rate": float(np.mean(v["far"])),
            "attribution_asr": float(np.mean(v["asr"])) if v["asr"] else None,
            "localization": aggregate_mask(v["mask"]),
        }
        for name, v in raw.items()
    }


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    seed_everything(int(cfg["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OpenPatchCocoDataset(
        cfg["data"]["val_img_dir"], cfg["data"]["val_ann_file"],
        resolution=cfg["data"]["resolution"], bins=cfg["mask"]["bins"],
        shape_probs={k: v for k, v in cfg["mask"]["shapes"].items() if k != "coco"},
        coco_prob=float(cfg["mask"]["shapes"].get("coco", 0.35)),
    )
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"], collate_fn=collate_openpatch)

    models = build_models(cfg, device)
    vae, _, code_head, status_head, localizer = models
    ckpt = find_latest_checkpoint(cfg, args.checkpoint)
    step = load_openpatch_state(ckpt, vae, code_head, status_head, localizer)
    for m in models:
        m.eval()

    suites = {"standard": suite_standard, "small_tamper": suite_small_tamper,
              "open_set": suite_open_set, "forgery": suite_forgery}
    result = suites[args.suite](loader, cfg, models, device, args.max_samples)
    payload = {"checkpoint": str(ckpt), "step": step, "suite": args.suite, "result": result}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    out_dir = Path(cfg["project"]["output_dir"]) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.suite}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
