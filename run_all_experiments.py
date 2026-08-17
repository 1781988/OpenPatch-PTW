from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from openpatch_ptw.config import load_config
from openpatch_ptw.runtime import write_json


def parse_args():
    parser = argparse.ArgumentParser("Run the reproducible OpenPatch-PTW ICASSP experiment pipeline")
    parser.add_argument("--config", default="configs/openpatch_ptw.yaml")
    parser.add_argument("--run-name", default="icassp_main")
    parser.add_argument("--mode", choices=["quick", "core", "paper"], default="core")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    parser.add_argument("--resume-from", choices=["smoke", "warmup", "finetune", "eval"], default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()




def override_args(overrides: list[str]) -> list[str]:
    result: list[str] = []
    for item in overrides:
        result.extend(["--override", item])
    return result


def run_command(command: list[str], log_file: Path, state: dict, key: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(command), flush=True)
    state["steps"][key] = {"command": command, "status": "running"}
    write_json(log_file.parent / "pipeline_state.json", state)
    with log_file.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    state["steps"][key]["return_code"] = return_code
    state["steps"][key]["status"] = "done" if return_code == 0 else "failed"
    write_json(log_file.parent / "pipeline_state.json", state)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def should_run(resume_from: str | None, stage: str) -> bool:
    order = ["smoke", "warmup", "finetune", "eval"]
    if resume_from is None:
        return True
    return order.index(stage) >= order.index(resume_from)


def train_variant(
    python: str,
    config: str,
    variant: str | None,
    run_name: str,
    output_root: Path,
    state: dict,
    mode: str,
    resume_from: str | None = None,
    overrides: list[str] | None = None,
):
    common = ["--config", config, "--run-name", run_name, *override_args(overrides or [])]
    if variant:
        common += ["--variant", variant]

    smoke_steps = "20" if mode == "quick" else "200"
    warmup_steps = "200" if mode == "quick" else None
    finetune_steps = "500" if mode == "quick" else None

    smoke_best = output_root / run_name / "smoke" / "best.pth"
    warmup_best = output_root / run_name / "warmup" / "best.pth"
    finetune_best = output_root / run_name / "finetune" / "best.pth"

    if should_run(resume_from, "smoke"):
        command = [python, "train_openpatch.py", *common, "--stage", "smoke", "--max-steps", smoke_steps]
        run_command(command, output_root / run_name / "pipeline_logs" / "train_smoke.log", state, f"{run_name}:smoke")
    if should_run(resume_from, "warmup"):
        command = [python, "train_openpatch.py", *common, "--stage", "warmup", "--init-checkpoint", str(smoke_best)]
        if warmup_steps:
            command += ["--max-steps", warmup_steps]
        run_command(command, output_root / run_name / "pipeline_logs" / "train_warmup.log", state, f"{run_name}:warmup")
    if should_run(resume_from, "finetune"):
        command = [python, "train_openpatch.py", *common, "--stage", "finetune", "--init-checkpoint", str(warmup_best)]
        if finetune_steps:
            command += ["--max-steps", finetune_steps]
        run_command(command, output_root / run_name / "pipeline_logs" / "train_finetune.log", state, f"{run_name}:finetune")
    return finetune_best


def evaluate_model(
    python: str,
    config: str,
    variant: str | None,
    run_name: str,
    model: str,
    checkpoint: Path | None,
    suites: list[str],
    output_root: Path,
    state: dict,
    quick: bool,
    overrides: list[str] | None = None,
):
    for suite in suites:
        command = [
            python,
            "eval_openpatch.py",
            "--config",
            config,
            "--run-name",
            run_name,
            "--model",
            model,
            "--suite",
            suite,
            *override_args(overrides or []),
        ]
        if variant:
            command += ["--variant", variant]
        if checkpoint:
            command += ["--checkpoint", str(checkpoint)]
        if quick:
            command += ["--max-samples", "16", "--num-visuals", "2", "--no-lpips"]
        run_command(
            command,
            output_root / run_name / "pipeline_logs" / f"eval_{model}_{suite}.log",
            state,
            f"{run_name}:{model}:{suite}",
        )


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    output_root = Path(cfg["project"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    state = {
        "run_name": args.run_name,
        "mode": args.mode,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "steps": {},
    }

    if not args.skip_data:
        run_command(
            [python, "prepare_data.py", "--config", args.config, *override_args(args.override)],
            output_root / args.run_name / "pipeline_logs" / "prepare_data.log",
            state,
            "prepare_data",
        )
    run_command(
        [python, "doctor.py", "--config", args.config, "--strict", "--forward", *override_args(args.override)],
        output_root / args.run_name / "pipeline_logs" / "doctor.log",
        state,
        "doctor",
    )

    final_checkpoint = output_root / args.run_name / "finetune" / "best.pth"
    if not args.skip_train:
        final_checkpoint = train_variant(
            python,
            args.config,
            None,
            args.run_name,
            output_root,
            state,
            args.mode,
            args.resume_from,
            args.override,
        )

    suites = ["standard", "small_tamper", "open_set", "forgery"]
    if args.mode == "paper":
        suites.append("real_edits")
    if should_run(args.resume_from, "eval"):
        evaluate_model(
            python,
            args.config,
            None,
            args.run_name,
            "openpatch",
            final_checkpoint,
            suites,
            output_root,
            state,
            args.mode == "quick",
            args.override,
        )
        if not args.skip_baseline:
            evaluate_model(
                python,
                args.config,
                None,
                args.run_name,
                "genptw",
                None,
                suites if args.mode == "paper" else suites[:4],
                output_root,
                state,
                args.mode == "quick",
                args.override,
            )

    if args.mode == "paper":
        ablations = [
            ("no_position", "configs/ablations/no_position.yaml"),
            ("no_consistency", "configs/ablations/no_consistency.yaml"),
            ("no_status", "configs/ablations/no_status.yaml"),
            ("fixed_mask", "configs/ablations/fixed_mask.yaml"),
        ]
        for suffix, variant in ablations:
            variant_run = f"{args.run_name}_abl_{suffix}"
            checkpoint = train_variant(
                python,
                args.config,
                variant,
                variant_run,
                output_root,
                state,
                "core",
                overrides=args.override,
            )
            variant_suites = ["standard", "small_tamper", "forgery"]
            if suffix != "no_status":
                variant_suites.append("open_set")
            evaluate_model(
                python,
                args.config,
                variant,
                variant_run,
                "openpatch",
                checkpoint,
                variant_suites,
                output_root,
                state,
                False,
                args.override,
            )

    if not args.skip_package:
        run_command(
            [python, "package_results.py", "--config", args.config, "--run-prefix", args.run_name, *override_args(args.override)],
            output_root / args.run_name / "pipeline_logs" / "package.log",
            state,
            "package_results",
        )
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output_root / args.run_name / "pipeline_logs" / "pipeline_state.json", state)


if __name__ == "__main__":
    main()
