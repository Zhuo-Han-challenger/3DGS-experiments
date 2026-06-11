from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from common import (
    load_lod_plys,
    parse_views,
    read_binary_3dgs_ply,
    render_gaussian_points,
    save_png,
    scene_center_scale,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render original and decoded LOD PLY files listed in an encode CSV.")
    parser.add_argument("--lod-dir", type=Path, required=True, help="Original five-LOD PLY directory, used for scene normalization.")
    parser.add_argument("--encode-csv", type=Path, required=True, help="CSV from reference_lod_independent.py or reference_lod_residual.py.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--views", default="iso")
    parser.add_argument("--render-size", type=int, default=512)
    args = parser.parse_args()

    _lod_files, props, lods = load_lod_plys(args.lod_dir)
    center, scale = scene_center_scale(lods)
    views = parse_views(args.views)
    table = pd.read_csv(args.encode_csv)
    rows = []

    original_rendered: set[tuple[int, str]] = set()
    for _, row in table.iterrows():
        level = int(row["lod"])
        scheme = str(row["scheme"])
        quality = int(row["quality"])
        input_ply = Path(row["input_ply"])
        decoded_ply = Path(row["decoded_ply"])

        for view in views:
            view_name = view[0]
            original_path = args.out_dir / "renders" / "original" / f"lod_{level:02d}_{view_name}.png"
            if (level, view_name) not in original_rendered:
                input_props, input_data = read_binary_3dgs_ply(input_ply)
                image = render_gaussian_points(input_data, input_props, center, scale, view, level, args.render_size)
                save_png(original_path, image)
                original_rendered.add((level, view_name))

            decoded_props, decoded_data = read_binary_3dgs_ply(decoded_ply)
            decoded_path = args.out_dir / "renders" / scheme / f"q{quality}" / f"lod_{level:02d}_{view_name}.png"
            image = render_gaussian_points(decoded_data, decoded_props, center, scale, view, level, args.render_size)
            save_png(decoded_path, image)

            rows.append(
                {
                    **row.to_dict(),
                    "view": view_name,
                    "original_render": str(original_path.resolve()),
                    "decoded_render": str(decoded_path.resolve()),
                }
            )

    index_path = args.out_dir / "render_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
