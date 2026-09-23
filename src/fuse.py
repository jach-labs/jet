"""Fuse a LoRA adapter into the base weights for faster serving.

    jet-fuse --adapter adapters/jet --out models/jet
    jet-serve --base-model models/jet

The fused model gives the same outputs without the per-layer LoRA matmuls.
calibration.json is copied along so the fused dir is self-contained.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mlx.utils import tree_unflatten
from mlx_lm import load
from mlx_lm.utils import save_config, save_model

from evaluate import adapter_base
from model import DEFAULT_BASE_MODEL


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuse a Jet LoRA adapter into its base model.")
    ap.add_argument("--adapter", type=Path, default=Path("adapters/jet"))
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--out", type=Path, default=Path("models/jet"))
    args = ap.parse_args()

    model, tokenizer, config = load(adapter_base(str(args.adapter), args.base_model), adapter_path=str(args.adapter), return_config=True)
    fused = [(name, module.fuse()) for name, module in model.named_modules() if hasattr(module, "fuse")]
    model.update_modules(tree_unflatten(fused))

    args.out.mkdir(parents=True, exist_ok=True)
    save_model(args.out, model, donate_model=True)
    save_config(config, config_path=args.out / "config.json")
    tokenizer.save_pretrained(args.out)
    if (calib := args.adapter / "calibration.json").exists():
        shutil.copy(calib, args.out / "calibration.json")
    print(f"fused {len(fused)} LoRA layers into {args.out}")


if __name__ == "__main__":
    main()
