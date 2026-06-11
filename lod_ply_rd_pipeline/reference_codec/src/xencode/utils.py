#  Copyright [2025] [Huawei Technologies Co., Ltd.] 
#  Licensed under the Code Sharing Policy of the UHD World Association (the "Policy"); 
#  http://www.theuwa.com/UWA_Code_Sharing_Policy.pdf.
#  you may not use this file except in compliance with the Policy. 
#  Unless agreed to in writing, software distributed under the Policy is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OF ANY KIND, either express or implied. 
#  See the Policy for the specific language governing permissions and 
#  limitations under the Policy.

import torch
import os

dataType = {
    'char' : 1,
    'uchar' : 1,
    'short' : 1,
    'ushort' : 1,
    'int' : 4,
    'uint' : 4,
    'float' : 4,
    'double' : 8
}

def splitBy3(a):
    x = a & 0x1FFFFF  # we only look at the first 21 bits
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x

def mortonEncode(pos: torch.Tensor) -> torch.Tensor:
    x, y, z = pos.unbind(-1)
    answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2
    return answer

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    ir, ii, ij, ik = i * r, i * i, i * j, i * k
    jr, jj, jk = j * r, j * j, j * k
    kr, kk = k * r, k * k

    o = torch.stack((1 - two_s * (jj + kk), two_s * (ij - kr), two_s * (ik + jr),
                     two_s * (ij + kr), 1 - two_s * (ii + kk), two_s * (jk - ir),
                     two_s * (ik - jr), two_s * (jk + ir), 1 - two_s * (ii + jj)), -1, )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

import torch.nn.functional as F
def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix."""
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = quats.unbind(dim=-1)

    R = torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y - w * z),
        2 * (x * z + w * y),
        2 * (x * y + w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z - w * x),
        2 * (x * z - w * y),
        2 * (y * z + w * x),
        1 - 2 * (x * x + y * y),
    ], dim=-1)

    return R.reshape(quats.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions (w, x, y, z).
    Fully vectorized — no Python-level branching, CUDA/CPU compatible.
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    trace = m00 + m11 + m22

    # Case w: trace > 0
    Sw = torch.sqrt((trace + 1.0).clamp(min=1e-10)) * 2   # 4w
    qw_w = 0.25 * Sw
    qx_w = (m21 - m12) / Sw
    qy_w = (m02 - m20) / Sw
    qz_w = (m10 - m01) / Sw

    # Case x: m00 最大
    Sx = torch.sqrt((1.0 + m00 - m11 - m22).clamp(min=1e-10)) * 2   # 4x
    qw_x = (m21 - m12) / Sx
    qx_x = 0.25 * Sx
    qy_x = (m01 + m10) / Sx
    qz_x = (m02 + m20) / Sx

    # Case y: m11 最大
    Sy = torch.sqrt((1.0 + m11 - m00 - m22).clamp(min=1e-10)) * 2   # 4y
    qw_y = (m02 - m20) / Sy
    qx_y = (m01 + m10) / Sy
    qy_y = 0.25 * Sy
    qz_y = (m12 + m21) / Sy

    # Case z: m22 最大
    Sz = torch.sqrt((1.0 + m22 - m00 - m11).clamp(min=1e-10)) * 2   # 4z
    qw_z = (m10 - m01) / Sz
    qx_z = (m02 + m20) / Sz
    qy_z = (m12 + m21) / Sz
    qz_z = 0.25 * Sz

    cond_w = trace > 0
    cond_x = (m00 > m11) & (m00 > m22) & ~cond_w
    cond_y = (m11 > m22) & ~cond_w & ~cond_x
    # cond_z = 其余所有元素

    def select(vw, vx, vy, vz):
        return torch.where(cond_w, vw,
               torch.where(cond_x, vx,
               torch.where(cond_y, vy, vz)))

    quat = torch.stack([
        select(qw_w, qw_x, qw_y, qw_z),
        select(qx_w, qx_x, qx_y, qx_z),
        select(qy_w, qy_x, qy_y, qy_z),
        select(qz_w, qz_x, qz_y, qz_z),
    ], dim=-1)

    return F.normalize(quat, p=2, dim=-1)

def rot_scaling_to_cov(rot, scaling):
    rotation_matrix = _quat_to_rotmat(rot)
    M = rotation_matrix * scaling[..., None, :]
    covars = torch.einsum("...ij,...kj -> ...ik", M, M).reshape(-1, 9)
    return covars

def cov_to_rot_scaling(cov):
    cov = cov.view(-1, 3, 3).contiguous()
    
    if cov.is_cuda:
        R, S, _ = torch.linalg.svd(cov, full_matrices=False, driver='gesvdj')
    else:
        R, S, _ = torch.linalg.svd(cov, full_matrices=False)
    det = torch.det(R)
    flip = det.sign()
    R = R.clone()
    R[..., 2] *= flip.unsqueeze(-1)
    scaling = torch.log(torch.sqrt(S.clamp(min=1e-8)))
    rotation = matrix_to_quaternion(R)
    return rotation, scaling

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = quaternion_to_matrix(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L

def canonicalize_rot_scaling(rotation: torch.Tensor,
                              scaling: torch.Tensor) -> tuple:
    """
    输入 scaling 是 log 空间（与原始 gaussian_data["scaling"] 一致）
    输出与 cov_to_rot_scaling 等效
    """
    R = _quat_to_rotmat(rotation)            # [..., 3, 3]
    s = torch.exp(scaling)                   # [..., 3]

    # 按 scale 降序排列列
    order = torch.argsort(s, dim=-1, descending=True)   # [..., 3]
    order_R = order.unsqueeze(-2).expand_as(R)          # [..., 3, 3]
    R = torch.gather(R, dim=-1, index=order_R)
    s = torch.gather(s, dim=-1, index=order)

    # 修正每列符号 
    max_idx = R.abs().argmax(dim=-2, keepdim=True)      # [..., 1, 3]
    signs = R.gather(dim=-2, index=max_idx).sign()      # [..., 1, 3]
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    R = R * signs

    # 对齐原始 flip 逻辑：det<0 时翻转最后一列
    det = torch.linalg.det(R)
    R = R.clone()
    R[..., 2] *= det.sign().unsqueeze(-1)

    # 还原，对齐原始输出格式 
    s_out = torch.log(s.clamp(min=1e-8)) 
    q_out = matrix_to_quaternion(R)

    return q_out, s_out

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

import matplotlib.pyplot as plt
def plot_tensor_histogram(
    tensor: torch.Tensor,
    path: str,
    title: str = "Tensor Distribution Histogram",
    bins: int = 50,
    figsize: tuple = (10, 6)
):
    """
    绘制输入 PyTorch Tensor 的分布直方图，并保存到指定路径。

    Args:
        tensor (torch.Tensor): 要绘制的输入张量。
        path (str): 图像保存的完整路径（包括文件名和扩展名，如 'output/hist.png'）。
        title (str): 直方图的标题。
        bins (int): 直方图的柱子数量。
        figsize (tuple): 图像的尺寸 (宽, 高)。
    """
    # 1. 确保张量是 CPU 上的浮点或整数类型
    if not tensor.is_floating_point() and not tensor.is_integer():
        print(f"警告：张量类型 {tensor.dtype} 不支持直接绘图，尝试转换为 float。")
        tensor = tensor.float()

    # 2. 将张量展平并转换为 NumPy 数组
    # 使用 .flatten() 展平所有维度，使用 .cpu().numpy() 确保在 CPU 上并转换为 NumPy
    data = tensor.flatten().cpu().numpy()

    # 3. 检查数据是否为空
    if data.size == 0:
        print("错误：输入张量为空，无法绘制直方图。")
        return

    # 4. 创建绘图
    plt.figure(figsize=figsize)

    # 绘制直方图
    # density=True 可以使纵轴表示概率密度，方便比较不同大小数据集
    plt.hist(data, bins=bins, density=False, color='skyblue', edgecolor='black')

    # 5. 添加标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel("Tensor Values", fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)

    # 添加网格线，增强可读性
    plt.grid(axis='y', alpha=0.75)

    # 6. 保存图像
    try:
        # 确保保存路径的目录存在
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        plt.savefig(path, bbox_inches='tight')
        print(f"直方图已成功保存到：{path}")
    except Exception as e:
        print(f"保存图像时发生错误：{e}")

    # 7. 关闭图像，释放内存
    plt.close()

def find_densest_voxel(points: torch.Tensor, voxel_size: float = 1.0) -> tuple:
    """
    Find the voxel containing the most points.

    Args:
        points: Tensor of shape (N, 3) containing 3D points
        voxel_size: Size of each voxel (default: 1.0)

    Returns:
        Tuple of (voxel_center, count) where:
        - voxel_center: Tensor (x, y, z) world coordinates of the voxel's center
        - count: number of points in that voxel
    """
    # Compute voxel index for each point
    voxel_indices = torch.floor(points / voxel_size).to(torch.int64)

    # Encode (i,j,k) -> flat integer for torch.unique
    mins = voxel_indices.min(dim=0).values
    shifted = voxel_indices - mins
    dims = shifted.max(dim=0).values + 1
    flat = shifted[:, 0] * (dims[1] * dims[2]) + shifted[:, 1] * dims[2] + shifted[:, 2]

    # Count and find densest voxel
    unique, counts = torch.unique(flat, return_counts=True)
    best_flat = unique[counts.argmax()]
    count = counts.max()

    # Decode flat index back to (i, j, k)
    k = best_flat % dims[2]
    j = (best_flat // dims[2]) % dims[1]
    i = best_flat // (dims[1] * dims[2])
    densest_voxel = torch.stack([i, j, k]) + mins

    # Compute center: (index + 0.5) * voxel_size
    voxel_center = (densest_voxel.float() + 0.5) * voxel_size

    return voxel_center, count.item()