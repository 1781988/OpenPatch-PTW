from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from openpatch_ptw.config import load_config

GENPTW_CHECKPOINT_URL = "https://drive.google.com/file/d/1nC85Jc0B6K5ycqRHN0NFWVQP2jSLHsoT/view?usp=drive_link"
CONVNEXT_URL = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"
LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"


def parse_args():
    parser = argparse.ArgumentParser("Download OpenPatch/GenPTW external model assets")
    parser.add_argument("--config", default="configs/openpatch_ptw.yaml")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--components",
        nargs="+",
        default=["vae", "convnext", "checkpoint"],
        choices=["vae", "convnext", "checkpoint", "lama", "sd_inpaint", "all"],
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download_file(url: str, output: Path, force: bool = False) -> Path:
    if output.exists() and not force:
        print(f"[skip] {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, stream=True, timeout=60, headers=headers) as response:
        response.raise_for_status()
        # Some servers ignore Range and return 200. Appending that response would
        # silently corrupt the asset, so restart from byte zero unless status 206.
        resumed = bool(existing and response.status_code == 206)
        if existing and not resumed:
            existing = 0
        mode = "ab" if resumed else "wb"
        remaining = int(response.headers.get("content-length", 0))
        total = existing + remaining if remaining else None
        with partial.open(mode) as handle, tqdm(
            total=total, initial=existing, unit="B", unit_scale=True, desc=output.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
                    bar.update(len(chunk))
    partial.replace(output)
    return output


def download_hf(repo_id: str, target: Path, token: str | None, allow_patterns=None, subfolder=None, force=False):
    if target.exists() and any(target.iterdir()) and not force:
        print(f"[skip] {target}")
        return
    from huggingface_hub import snapshot_download

    target.parent.mkdir(parents=True, exist_ok=True)
    if subfolder:
        with tempfile.TemporaryDirectory(prefix="openpatch_hf_") as temp:
            root = Path(
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=allow_patterns,
                    local_dir=temp,
                    token=token,
                )
            )
            source = root / subfolder
            if not source.exists():
                raise FileNotFoundError(f"Subfolder {subfolder} not found in downloaded {repo_id}")
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
    else:
        if target.exists() and force:
            shutil.rmtree(target)
        snapshot_download(repo_id=repo_id, local_dir=target, token=token)


def _extract_archive(path: Path, destination: Path) -> bool:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(destination)
        return True
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            archive.extractall(destination)
        return True
    return False


def download_genptw_checkpoint(target: Path, force: bool = False):
    expected = ["msg_decoder.pth", "localizer.pth", "diffusion_pytorch_model.safetensors"]
    if all((target / name).exists() for name in expected) and not force:
        print(f"[skip] official checkpoint: {target}")
        return
    import gdown

    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="genptw_ckpt_") as temp:
        downloaded = gdown.download(url=GENPTW_CHECKPOINT_URL, output=temp + os.sep, quiet=False, fuzzy=True)
        if not downloaded:
            raise RuntimeError("gdown failed to download the official GenPTW checkpoint")
        downloaded_path = Path(downloaded)
        extracted = Path(temp) / "extracted"
        extracted.mkdir()
        if _extract_archive(downloaded_path, extracted):
            search_root = extracted
        else:
            search_root = downloaded_path.parent
        for name in expected:
            matches = list(search_root.rglob(name))
            if matches:
                shutil.copy2(matches[0], target / name)
        missing = [name for name in expected if not (target / name).exists()]
        if missing:
            raise RuntimeError(
                "Downloaded Google Drive object did not contain the expected checkpoint files. "
                f"Missing: {missing}. Download manually from the GenPTW README and place them in {target}."
            )


def main():
    args = parse_args()
    cfg = load_config(args.config, args.variant, args.override)
    components = set(args.components)
    if "all" in components:
        components = {"vae", "convnext", "checkpoint", "lama", "sd_inpaint"}

    if "vae" in components:
        download_hf(
            "Manojb/stable-diffusion-2-base",
            Path(cfg["upstream"]["vae"]),
            args.hf_token,
            allow_patterns=["vae/**"],
            subfolder="vae",
            force=args.force,
        )
    if "convnext" in components:
        download_file(CONVNEXT_URL, Path(cfg["upstream"]["convnext"]), args.force)
    if "checkpoint" in components:
        download_genptw_checkpoint(Path(cfg["upstream"]["checkpoint_dir"]), args.force)
    if "lama" in components:
        download_file(LAMA_URL, Path(cfg["upstream"]["lama"]), args.force)
    if "sd_inpaint" in components:
        download_hf(
            "sd2-community/stable-diffusion-2-inpainting",
            Path(cfg["upstream"]["sd_inpaint"]),
            args.hf_token,
            force=args.force,
        )
    print("Asset preparation completed.")


if __name__ == "__main__":
    main()
