# 3DGS LOD PLY 编解码与 RD 评估流程

这个目录是一套独立运行的 3DGS LOD PLY 实验脚本。输入为已经重建好的 5 个 LOD 层级 PLY 文件，脚本会完成：

1. 独立编码每个 LOD 层级。
2. 按 LOD 层级之间的 residual 模型编码。
3. 解码后保存 reconstructed PLY。
4. 渲染编码前后的 PLY，并保存 PNG。
5. 计算原始渲染图和解码后渲染图之间的 PSNR。
6. 统计 encoded size，输出 CSV。
7. 为每个 LOD 生成 `psnr vs encoded size` PNG 图。

## 环境安装

```powershell
pip install -r requirements.txt
```

## 输入要求

输入目录中需要有 5 个 PLY 文件，按文件名排序后视为：

```text
lod_00.ply  # 最高精度
lod_01.ply
lod_02.ply
lod_03.ply
lod_04.ply  # 最低精度 / residual base
```

当前脚本面向 3DGS 常见的 binary little endian PLY，顶点属性为 float，例如：

```text
x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 ... opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3
```

渲染 PSNR 使用编码前的原始 PLY 渲染图作为 GT，和同一 LOD、同一视角下的解码 PLY 渲染图比较。

## 运行示例

```powershell
python run_lod_experiment.py `
  --lod-dir "D:\Documents\3DGS编解码\output\model_ply_lod_rd\lods" `
  --out-dir "D:\Documents\3DGS编解码\output\lod_rd_pipeline_run" `
  --qualities 33,66,100 `
  --views iso
```

如果想快速试跑，可以降低渲染分辨率：

```powershell
python run_lod_experiment.py `
  --lod-dir "D:\Documents\3DGS编解码\output\model_ply_lod_rd\lods" `
  --out-dir "D:\Documents\3DGS编解码\output\lod_rd_pipeline_smoke" `
  --qualities 33,66,100 `
  --render-size 256 `
  --views iso
```

如果需要多视角平均 PSNR，可以运行：

```powershell
python run_lod_experiment.py `
  --lod-dir "D:\path\to\five_lod_ply" `
  --out-dir "D:\path\to\out" `
  --qualities 33,66,100 `
  --views front,right,back,left,top,iso
```

## 输出目录

运行完成后，输出目录结构类似：

```text
out-dir/
  coded/
    independent/
    residual_lod/
  decoded_ply/
    independent/q33/lod_00.ply
    residual_lod/q33/lod_00.ply
  renders/
    original/lod_00_front.png
    independent/q33/lod_00_front.png
    residual_lod/q33/lod_00_front.png
  png/
    lod_00_psnr_vs_size.png
    ...
  lod_rd_results.csv
```

CSV 关键列：

- `scheme`: `independent` 或 `residual_lod`
- `quality`: 33、66、100
- `bits`: 实际量化 bit 数，默认 33->8 bit，66->12 bit，100->16 bit
- `lod`: LOD 层级编号
- `encoded_size_bytes`: 对应编码码流大小
- `psnr_db`: 多视角平均 PSNR
- `decoded_ply`: 解码后 PLY 路径

## 方法说明

独立编码：

- 对每个 LOD 的全部 float 属性做 per-property min/max 均匀量化。
- 编码数据、min、scale 和 manifest 打包为 zip。

Residual 编码：

- `lod_04` 作为 base，直接无损保存 float 属性。
- `lod_03` 到 `lod_00` 使用相邻更粗 LOD 线性 parent index 预测。
- 编码 `current - parent_prediction` 的 residual，并存储 parent index 的 delta-varint 表示。
- residual 也按 per-property signed 均匀量化。

渲染：

- 使用 CPU 点式 Gaussian 预览渲染器。
- 颜色来自 `f_dc_0..2`，采用 3DGS 常用 `rgb = f_dc * C0 + 0.5`。
- 默认渲染 `iso` 一个视角。可以用 `--views front,right,back,left,top,iso` 切换为 6 个视角平均 PSNR。
