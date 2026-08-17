from pathlib import Path

from openpatch_ptw.config import load_config


def test_base_config_loads_and_resolves_paths():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "openpatch_ptw.yaml")
    assert cfg["model"]["bit_dim"] == 64
    assert Path(cfg["project"]["output_dir"]).is_absolute()
    assert abs(sum(cfg["sample_mix"][key] for key in ["valid", "unwatermarked", "forged"]) - 1.0) < 1e-6


def test_variant_and_override_merge():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(
        root / "configs" / "openpatch_ptw.yaml",
        root / "configs" / "ablations" / "no_consistency.yaml",
        ["train.batch_size=1"],
    )
    assert cfg["experiment"]["use_consistency"] is False
    assert cfg["train"]["batch_size"] == 1
