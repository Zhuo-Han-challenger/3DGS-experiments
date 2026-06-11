from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common import psnr


def read_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PSNR from render_index.csv and plot PSNR vs encoded size per LOD.")
    parser.add_argument("--render-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.render_index)
    table["view_psnr_db"] = [
        psnr(read_image(row["original_render"]), read_image(row["decoded_render"]))
        for _, row in table.iterrows()
    ]

    group_cols = ["scheme", "quality", "lod"]
    keep_cols = [
        "input_ply",
        "encoded_file",
        "decoded_ply",
        "encoded_size_bytes",
        "incremental_size_bytes",
    ]
    rows = []
    for keys, sub in table.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for col in keep_cols:
            if col in sub.columns:
                row[col] = sub.iloc[0][col]
        row["psnr_db"] = float(sub["view_psnr_db"].mean())
        row["views"] = ",".join(sub["view"].astype(str).tolist())
        rows.append(row)
    summary = pd.DataFrame(rows)
    csv_path = args.out_dir / "psnr_size_results.csv"
    summary.to_csv(csv_path, index=False)

    plot_dir = args.out_dir / "png"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for lod in sorted(summary["lod"].unique()):
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        sub = summary[summary["lod"] == lod]
        for scheme, marker in [("independent", "o"), ("residual_lod", "s")]:
            ss = sub[sub["scheme"] == scheme].sort_values("encoded_size_bytes")
            if ss.empty:
                continue
            ax.plot(ss["encoded_size_bytes"] / 1024.0, ss["psnr_db"], marker=marker, linewidth=2, label=scheme)
            for _, row in ss.iterrows():
                ax.annotate(f"q{int(row['quality'])}", (row["encoded_size_bytes"] / 1024.0, row["psnr_db"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title(f"LOD {int(lod):02d}: PSNR vs Encoded Size")
        ax.set_xlabel("Encoded size (KiB)")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"lod_{int(lod):02d}_psnr_vs_size.png", dpi=220)
        plt.close(fig)

    print(f"Wrote {csv_path}")
    print(f"Wrote plots to {plot_dir}")


if __name__ == "__main__":
    main()
