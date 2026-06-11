from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from common import load_lod_plys


DEFAULT_CODEC_ROOT = Path(__file__).resolve().parent / "reference_codec"


def parse_qualities(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def run_reference_encode(args: argparse.Namespace, ply_path: Path, save_dir: Path, scene_name: str, quality: int) -> tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        str(args.codec_root / "encode_1_0.py"),
        "--ply-path",
        str(ply_path),
        "--save-dir",
        str(save_dir),
        "--scene-name",
        scene_name,
        "--log-path",
        str(save_dir / "log.json"),
        "--quality",
        str(quality),
    ]
    if args.config:
        cmd += ["--config", str(args.config)]
    else:
        cmd += [
            "--version",
            args.version,
            "--sort-method",
            args.sort_method,
            "--video-codec",
            args.video_codec,
            "--block-size",
            str(args.block_size),
            "--pos-bitdepth",
            str(args.pos_bitdepth),
            "--rot-bitdepth",
            str(args.rot_bitdepth),
            "--sca-bitdepth",
            str(args.sca_bitdepth),
            "--op-bitdepth",
            str(args.op_bitdepth),
            "--sh0-bitdepth",
            str(args.sh0_bitdepth),
            "--shn-bitdepth",
            str(args.shn_bitdepth),
            "--sh-degree",
            str(args.sh_degree),
        ]
    subprocess.run(cmd, cwd=args.codec_root, check=True)
    encoded = save_dir / "GSCompressed_r00.egsc"
    decoded = save_dir / "GSCompressed_r00.ply"
    if not encoded.exists() or not decoded.exists():
        raise FileNotFoundError(f"Reference encoder did not produce {encoded} and {decoded}")
    return encoded, decoded


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode each LOD independently with the reference 3DGS codec.")
    parser.add_argument("--codec-root", type=Path, default=DEFAULT_CODEC_ROOT, help="Path to the reference 3DGS编码实现/python directory.")
    parser.add_argument("--lod-dir", type=Path, required=True, help="Directory containing exactly five LOD PLY files.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qualities", default="33,66,100")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--version", default="mix_d3")
    parser.add_argument("--sort-method", default="dblockn")
    parser.add_argument("--video-codec", default="x265")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--pos-bitdepth", type=int, default=15)
    parser.add_argument("--rot-bitdepth", type=int, default=8)
    parser.add_argument("--sca-bitdepth", type=int, default=8)
    parser.add_argument("--op-bitdepth", type=int, default=8)
    parser.add_argument("--sh0-bitdepth", type=int, default=8)
    parser.add_argument("--shn-bitdepth", type=int, default=8)
    parser.add_argument("--sh-degree", type=int, default=3)
    args = parser.parse_args()

    lod_files, _props, _lods = load_lod_plys(args.lod_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for quality in parse_qualities(args.qualities):
        for level, ply_path in enumerate(lod_files):
            work_dir = args.out_dir / "reference_independent" / f"q{quality}" / f"lod_{level:02d}"
            encoded, decoded = run_reference_encode(args, ply_path.resolve(), work_dir.resolve(), f"lod_{level:02d}_q{quality}", quality)
            rows.append(
                {
                    "scheme": "independent",
                    "quality": quality,
                    "lod": level,
                    "input_ply": str(ply_path.resolve()),
                    "encoded_file": str(encoded.resolve()),
                    "decoded_ply": str(decoded.resolve()),
                    "encoded_size_bytes": encoded.stat().st_size,
                    "incremental_size_bytes": encoded.stat().st_size,
                }
            )

    csv_path = args.out_dir / "independent_encoded_size.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
