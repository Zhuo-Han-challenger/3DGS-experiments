from __future__ import annotations

import argparse
import json
import math
import os
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


C0 = 0.28209479177387814
QUALITY_BITS = {33: 8, 66: 12, 100: 16}
ALL_VIEWS = [
    ("front", 0.0, 8.0),
    ("right", 60.0, 8.0),
    ("back", 120.0, 8.0),
    ("left", 180.0, 8.0),
    ("top", 35.0, 68.0),
    ("iso", 45.0, 28.0),
]


def read_binary_3dgs_ply(path: Path) -> tuple[list[str], np.ndarray]:
    props: list[str] = []
    vertex_count = None
    with path.open("rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        in_vertex = False
        while True:
            line = f.readline().decode("ascii").strip()
            parts = line.split()
            if len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
            elif len(parts) >= 2 and parts[0] == "element":
                in_vertex = False
            elif len(parts) >= 3 and parts[0] == "property" and in_vertex:
                if parts[1] != "float":
                    raise ValueError(f"Only float vertex properties are supported: {line}")
                props.append(parts[2])
            elif line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"{path} has no vertex element")
        data = np.frombuffer(f.read(vertex_count * len(props) * 4), dtype="<f4")
    return props, data.reshape(vertex_count, len(props)).astype(np.float32)


def write_binary_ply(path: Path, props: list[str], data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        header = ["ply\n", "format binary_little_endian 1.0\n", f"element vertex {len(data)}\n"]
        header += [f"property float {prop}\n" for prop in props]
        header.append("end_header\n")
        f.writelines(line.encode("ascii") for line in header)
        f.write(data.astype("<f4", copy=False).tobytes())


def raw(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a).tobytes()


def quantize_minmax(values: np.ndarray, bits: int):
    levels = (1 << bits) - 1
    mn = values.min(axis=0).astype(np.float32)
    mx = values.max(axis=0).astype(np.float32)
    scale = np.maximum((mx - mn) / levels, 1e-12).astype(np.float32)
    q = np.round((values - mn) / scale).clip(0, levels)
    return q.astype(np.uint8 if bits <= 8 else np.uint16), mn, scale


def dequantize_minmax(q: np.ndarray, mn: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale.astype(np.float32) + mn.astype(np.float32)


def quantize_signed(values: np.ndarray, bits: int):
    signed_max = (1 << (bits - 1)) - 1
    max_abs = np.maximum(np.max(np.abs(values), axis=0), 1e-12).astype(np.float32)
    scale = (max_abs / signed_max).astype(np.float32)
    q = np.round(values / scale).clip(-signed_max, signed_max)
    return q.astype(np.int8 if bits <= 8 else np.int16), scale


def dequantize_signed(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale.astype(np.float32)


def zigzag(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.int64, copy=False)
    return ((values << 1) ^ (values >> 63)).astype(np.uint64)


def varint_unsigned(values: np.ndarray) -> bytes:
    out = bytearray()
    for value in values.reshape(-1).astype(np.uint64, copy=False):
        v = int(value)
        while v >= 0x80:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v)
    return bytes(out)


def varint_delta(values: np.ndarray) -> bytes:
    flat = values.reshape(-1).astype(np.int64, copy=False)
    return varint_unsigned(zigzag(np.diff(flat, prepend=0)))


def parent_indices(child_len: int, parent_len: int) -> np.ndarray:
    return np.round(np.linspace(0, parent_len - 1, child_len)).astype(np.uint32)


def encode_independent(level: int, quality: int, bits: int, data: np.ndarray, out_dir: Path):
    q, mn, scale = quantize_minmax(data, bits)
    decoded = dequantize_minmax(q, mn, scale)
    path = out_dir / "coded" / "independent" / f"q{quality}_lod_{level:02d}.gind.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scheme": "independent",
        "quality": quality,
        "bits": bits,
        "level": level,
        "shape": list(data.shape),
        "dtype": str(q.dtype),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("all_props_q.bin", raw(q))
        z.writestr("min.bin", raw(mn))
        z.writestr("scale.bin", raw(scale))
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    return path, decoded.astype(np.float32), path.stat().st_size


def encode_residual_model(quality: int, bits: int, lods: list[np.ndarray], out_dir: Path):
    decoded: dict[int, np.ndarray] = {4: lods[4].copy()}
    sizes: dict[int, int] = {}
    path = out_dir / "coded" / "residual_lod" / f"q{quality}_model.gres.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scheme": "residual_lod",
        "quality": quality,
        "bits": bits,
        "num_lods": len(lods),
        "shapes": [list(x.shape) for x in lods],
        "base_level": 4,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        z.writestr("lod_04/base_float32.bin", raw(lods[4].astype(np.float32)))
        for level in range(3, -1, -1):
            parent = decoded[level + 1]
            idx = parent_indices(len(lods[level]), len(parent))
            residual = lods[level] - parent[idx]
            q, scale = quantize_signed(residual, bits)
            decoded[level] = parent[idx] + dequantize_signed(q, scale)
            z.writestr(f"lod_{level:02d}/parent_idx_delta.varint", varint_delta(idx))
            z.writestr(f"lod_{level:02d}/residual_q.bin", raw(q))
            z.writestr(f"lod_{level:02d}/residual_scale.bin", raw(scale))

    with zipfile.ZipFile(path, "r") as z:
        info = {item.filename: item.compress_size for item in z.infolist()}
    sizes[4] = info.get("lod_04/base_float32.bin", 0)
    for level in range(3, -1, -1):
        sizes[level] = (
            info.get(f"lod_{level:02d}/parent_idx_delta.varint", 0)
            + info.get(f"lod_{level:02d}/residual_q.bin", 0)
            + info.get(f"lod_{level:02d}/residual_scale.bin", 0)
        )
    return path, decoded, sizes


def sh_to_rgb(data: np.ndarray, props: list[str]) -> np.ndarray:
    idx = [props.index("f_dc_0"), props.index("f_dc_1"), props.index("f_dc_2")]
    rgb = data[:, idx] * C0 + 0.5
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.asarray([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    return rx @ ry


def render_gaussian_points(
    data: np.ndarray,
    props: list[str],
    center: np.ndarray,
    scale: float,
    view: tuple[str, float, float],
    level: int,
    size: int,
) -> np.ndarray:
    _name, yaw, pitch = view
    pts = ((data[:, :3] - center[None, :]) / scale) @ rotation(yaw, pitch).T
    rgb = sh_to_rgb(data, props)
    screen = (pts[:, :2] * 0.86 + 0.5) * (size - 1)
    px = np.rint(screen[:, 0]).astype(np.int32)
    py = np.rint((size - 1) - screen[:, 1]).astype(np.int32)
    z = pts[:, 2]
    radius = 2 if level <= 1 else 3 if level <= 3 else 4
    valid = (px >= -radius) & (px < size + radius) & (py >= -radius) & (py < size + radius)
    px, py, z, rgb = px[valid], py[valid], z[valid], rgb[valid]
    order = np.argsort(z)
    px, py, z, rgb = px[order], py[order], z[order], rgb[order]
    image = np.ones((size, size, 3), dtype=np.float32)
    zbuf = np.full((size, size), -np.inf, dtype=np.float32)
    offsets = [
        (ox, oy)
        for oy in range(-radius, radius + 1)
        for ox in range(-radius, radius + 1)
        if ox * ox + oy * oy <= radius * radius
    ]
    for i in range(len(px)):
        for ox, oy in offsets:
            x, y = px[i] + ox, py[i] + oy
            if 0 <= x < size and 0 <= y < size and z[i] >= zbuf[y, x]:
                zbuf[y, x] = z[i]
                image[y, x] = rgb[i]
    return image


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)).save(path)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 0:
        return float("inf")
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


def render_views(
    data: np.ndarray,
    props: list[str],
    center: np.ndarray,
    scale: float,
    level: int,
    size: int,
    out_dir: Path,
    prefix: str,
    views: list[tuple[str, float, float]],
) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for view in views:
        image = render_gaussian_points(data, props, center, scale, view, level, size)
        images[view[0]] = image
        save_png(out_dir / f"{prefix}_{view[0]}.png", image)
    return images


def parse_qualities(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    for value in values:
        if value not in QUALITY_BITS:
            raise ValueError(f"Unsupported quality {value}; supported: {sorted(QUALITY_BITS)}")
    return values


def parse_views(text: str) -> list[tuple[str, float, float]]:
    by_name = {view[0]: view for view in ALL_VIEWS}
    names = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"Unsupported views {unknown}; supported: {sorted(by_name)}")
    return [by_name[name] for name in names]


def load_lods(lod_dir: Path) -> tuple[list[Path], list[str], list[np.ndarray]]:
    files = sorted(lod_dir.glob("*.ply"))
    if len(files) != 5:
        raise ValueError(f"{lod_dir} must contain exactly 5 .ply files, found {len(files)}")
    props0, data0 = read_binary_3dgs_ply(files[0])
    lods = [data0]
    for path in files[1:]:
        props, data = read_binary_3dgs_ply(path)
        if props != props0:
            raise ValueError(f"{path} has different vertex properties from {files[0]}")
        lods.append(data)
    return files, props0, lods


def plot_results(table: pd.DataFrame, out_dir: Path) -> None:
    out = out_dir / "png"
    out.mkdir(parents=True, exist_ok=True)
    for lod in sorted(table["lod"].unique()):
        sub = table[table["lod"] == lod]
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for scheme, marker in [("independent", "o"), ("residual_lod", "s")]:
            ss = sub[sub["scheme"] == scheme].sort_values("encoded_size_bytes")
            ax.plot(ss["encoded_size_bytes"] / 1024.0, ss["psnr_db"], marker=marker, linewidth=2, label=scheme)
            for _, row in ss.iterrows():
                label = f"q{int(row['quality'])}"
                ax.annotate(label, (row["encoded_size_bytes"] / 1024.0, row["psnr_db"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title(f"LOD {int(lod):02d}: PSNR vs Encoded Size")
        ax.set_xlabel("Encoded size (KiB)")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"lod_{int(lod):02d}_psnr_vs_size.png", dpi=220)
        plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lod_files, props, lods = load_lods(args.lod_dir)
    qualities = parse_qualities(args.qualities)
    views = parse_views(args.views)

    all_xyz = np.concatenate([lod[:, :3] for lod in lods], axis=0)
    center = (all_xyz.min(axis=0) + all_xyz.max(axis=0)) * 0.5
    scale = float(np.max(all_xyz.max(axis=0) - all_xyz.min(axis=0)))
    scale = max(scale, 1e-6)

    original_renders = []
    for level, data in enumerate(lods):
        original_renders.append(
            render_views(
                data,
                props,
                center,
                scale,
                level,
                args.render_size,
                out_dir / "renders" / "original",
                f"lod_{level:02d}",
                views,
            )
        )

    rows = []
    for quality in qualities:
        bits = QUALITY_BITS[quality]
        residual_path, residual_decoded, residual_sizes = encode_residual_model(quality, bits, lods, out_dir)

        for level, data in enumerate(lods):
            ind_path, ind_decoded, ind_size = encode_independent(level, quality, bits, data, out_dir)
            for scheme, decoded, encoded_size, coded_path in [
                ("independent", ind_decoded, ind_size, ind_path),
                ("residual_lod", residual_decoded[level], residual_sizes[level], residual_path),
            ]:
                decoded_ply = out_dir / "decoded_ply" / scheme / f"q{quality}" / f"lod_{level:02d}.ply"
                write_binary_ply(decoded_ply, props, decoded)
                decoded_props, decoded_data = read_binary_3dgs_ply(decoded_ply)
                decoded_renders = render_views(
                    decoded_data,
                    decoded_props,
                    center,
                    scale,
                    level,
                    args.render_size,
                    out_dir / "renders" / scheme / f"q{quality}",
                    f"lod_{level:02d}",
                    views,
                )
                psnrs = [psnr(original_renders[level][view[0]], decoded_renders[view[0]]) for view in views]
                rows.append(
                    {
                        "scheme": scheme,
                        "quality": quality,
                        "bits": bits,
                        "lod": level,
                        "input_ply": str(lod_files[level]),
                        "coded_file": str(coded_path),
                        "decoded_ply": str(decoded_ply),
                        "encoded_size_bytes": int(encoded_size),
                        "encoded_size_kib": float(encoded_size) / 1024.0,
                        "psnr_db": float(np.mean(psnrs)),
                        "view_psnr_json": json.dumps({view[0]: value for view, value in zip(views, psnrs)}),
                    }
                )

    table = pd.DataFrame(rows)
    csv_path = out_dir / "lod_rd_results.csv"
    table.to_csv(csv_path, index=False)
    plot_results(table, out_dir)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode/decode/render/PSNR/size experiment for 5 reconstructed 3DGS LOD PLY files.")
    parser.add_argument("--lod-dir", type=Path, required=True, help="Directory containing exactly five LOD .ply files.")
    parser.add_argument("--out-dir", type=Path, default=Path("output/lod_rd_pipeline_run"))
    parser.add_argument("--qualities", default="33,66,100", help="Comma-separated qualities. Supported values: 33,66,100.")
    parser.add_argument("--render-size", type=int, default=512, help="Square render resolution used for PNG and PSNR.")
    parser.add_argument("--views", default="iso", help="Comma-separated views. Supported: front,right,back,left,top,iso.")
    args = parser.parse_args()
    csv_path = run(args)
    print(f"Done. Wrote {csv_path}")


if __name__ == "__main__":
    main()
