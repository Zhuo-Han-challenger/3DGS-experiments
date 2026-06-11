from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image


C0 = 0.28209479177387814
ALL_VIEWS = {
    "front": (0.0, 8.0),
    "right": (60.0, 8.0),
    "back": (120.0, 8.0),
    "left": (180.0, 8.0),
    "top": (35.0, 68.0),
    "iso": (45.0, 28.0),
}


def parse_views(text: str) -> list[tuple[str, float, float]]:
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in ALL_VIEWS]
    if unknown:
        raise ValueError(f"Unsupported views {unknown}; supported: {sorted(ALL_VIEWS)}")
    return [(name, *ALL_VIEWS[name]) for name in names]


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


def write_binary_3dgs_ply(path: Path, props: list[str], data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        header = ["ply\n", "format binary_little_endian 1.0\n", f"element vertex {len(data)}\n"]
        header += [f"property float {prop}\n" for prop in props]
        header.append("end_header\n")
        f.writelines(line.encode("ascii") for line in header)
        f.write(data.astype("<f4", copy=False).tobytes())


def load_lod_plys(lod_dir: Path) -> tuple[list[Path], list[str], list[np.ndarray]]:
    files = sorted(lod_dir.glob("*.ply"))
    if len(files) != 5:
        raise ValueError(f"{lod_dir} must contain exactly five .ply files, found {len(files)}")
    props0, data0 = read_binary_3dgs_ply(files[0])
    lods = [data0]
    for path in files[1:]:
        props, data = read_binary_3dgs_ply(path)
        if props != props0:
            raise ValueError(f"{path} has a different property layout from {files[0]}")
        lods.append(data)
    return files, props0, lods


def parent_indices(child_len: int, parent_len: int) -> np.ndarray:
    return np.round(np.linspace(0, parent_len - 1, child_len)).astype(np.int64)


def scene_center_scale(lods: list[np.ndarray]) -> tuple[np.ndarray, float]:
    xyz = np.concatenate([lod[:, :3] for lod in lods], axis=0)
    center = (xyz.min(axis=0) + xyz.max(axis=0)) * 0.5
    scale = float(np.max(xyz.max(axis=0) - xyz.min(axis=0)))
    return center.astype(np.float32), max(scale, 1e-6)


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.asarray([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    return rx @ ry


def sh_to_rgb(data: np.ndarray, props: list[str]) -> np.ndarray:
    idx = [props.index("f_dc_0"), props.index("f_dc_1"), props.index("f_dc_2")]
    rgb = data[:, idx] * C0 + 0.5
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


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
