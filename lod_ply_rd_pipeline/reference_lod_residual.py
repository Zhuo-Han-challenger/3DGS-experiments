from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

from common import load_lod_plys, parent_indices, read_binary_3dgs_ply, write_binary_3dgs_ply


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
    parser = argparse.ArgumentParser(description="Reference-codec residual LOD experiment.")
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

    lod_files, props, lods = load_lod_plys(args.lod_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for quality in parse_qualities(args.qualities):
        q_root = args.out_dir / "reference_residual" / f"q{quality}"
        residual_ply_dir = q_root / "residual_ply"
        recon_dir = q_root / "reconstructed_ply"
        residual_ply_dir.mkdir(parents=True, exist_ok=True)
        recon_dir.mkdir(parents=True, exist_ok=True)

        encoded_files: dict[int, Path] = {}
        incremental_sizes: dict[int, int] = {}
        reconstructed: dict[int, object] = {}

        base_work = q_root / "base_lod_04"
        base_encoded, base_decoded = run_reference_encode(args, lod_files[4].resolve(), base_work.resolve(), f"residual_base_lod_04_q{quality}", quality)
        recon_base = recon_dir / "lod_04.ply"
        shutil.copyfile(base_decoded, recon_base)
        _base_props, reconstructed[4] = read_binary_3dgs_ply(recon_base)
        encoded_files[4] = base_encoded
        incremental_sizes[4] = base_encoded.stat().st_size

        for level in range(3, -1, -1):
            parent = reconstructed[level + 1]
            idx = parent_indices(len(lods[level]), len(parent))
            residual = lods[level] - parent[idx]
            residual_ply = residual_ply_dir / f"lod_{level:02d}_residual_input.ply"
            write_binary_3dgs_ply(residual_ply, props, residual)

            work_dir = q_root / f"residual_lod_{level:02d}"
            encoded, decoded_residual_ply = run_reference_encode(args, residual_ply.resolve(), work_dir.resolve(), f"residual_lod_{level:02d}_q{quality}", quality)
            _res_props, decoded_residual = read_binary_3dgs_ply(decoded_residual_ply)
            recon = parent[idx] + decoded_residual
            recon_ply = recon_dir / f"lod_{level:02d}.ply"
            write_binary_3dgs_ply(recon_ply, props, recon)
            reconstructed[level] = recon
            encoded_files[level] = encoded
            incremental_sizes[level] = encoded.stat().st_size

        cumulative = 0
        for level in range(4, -1, -1):
            cumulative += incremental_sizes[level]
            rows.append(
                {
                    "scheme": "residual_lod",
                    "quality": quality,
                    "lod": level,
                    "input_ply": str(lod_files[level].resolve()),
                    "encoded_file": str(encoded_files[level].resolve()),
                    "decoded_ply": str((recon_dir / f"lod_{level:02d}.ply").resolve()),
                    "encoded_size_bytes": cumulative,
                    "incremental_size_bytes": incremental_sizes[level],
                }
            )

    csv_path = args.out_dir / "residual_encoded_size.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
