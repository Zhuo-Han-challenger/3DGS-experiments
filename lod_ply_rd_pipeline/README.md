# 3DGS LOD Reference Codec RD Pipeline

这套脚本用于评估 5 个 LOD PLY 文件的编解码 RD 表现。**Independent LOD 编码调用《支持六自由度交互的三维图像格式标准》参考工具集里的 `encode_1_0.py`，底层编码逻辑与参考实现一致**。本目录只额外增加渲染、PSNR、CSV 和 RD 曲线生成流程。

Residual LOD 是单独设计的实验方案：`lod_04` 作为 base，`lod_03..lod_00` 编码相对更粗一级已解码 LOD 的 residual PLY。每个 base/residual PLY 也都会调用同一个参考 `encode_1_0.py` 编码，所以属性量化、transform、packing、视频/ASTC/entropy codec 仍复用参考工具集。

## 文件说明

- `reference_lod_independent.py`  
  对 5 个 LOD PLY 分别调用参考工具集编码，输出 `.egsc`、解码 PLY 和 `independent_encoded_size.csv`。默认使用同目录下 vendored 的 `reference_codec/encode_1_0.py`。

- `reference_lod_residual.py`  
  生成 LOD 间 residual PLY，并调用参考工具集编码 residual，输出重建 PLY 和 `residual_encoded_size.csv`。

- `merge_encode_csv.py`  
  合并 independent/residual 的码率统计 CSV。

- `render.py`  
  渲染编码前原始 PLY 和解码/重建 PLY，保存 PNG，并输出 `render_index.csv`。

- `psnr_report.py`  
  根据渲染图计算 PSNR，输出 `psnr_size_results.csv`，并为每个 LOD 生成 `psnr vs encoded size` PNG。

- `run_lod_experiment.py`  
  早期纯 Python zip 实验脚本，仅保留作快速原型参考；如果要求与标准参考工具集底层一致，请使用上面几个 `reference_*` 脚本。

- `reference_codec/`  
  从参考工具集复制来的 Python 编解码核心，包括 `encode_1_0.py`、`decode_1_0.py`、`src/`、`config/`、`jsonlogger.py`。没有复制第三方 `dependency/` 子模块和外部编码器二进制。

## 依赖

本目录额外需要：

```powershell
pip install -r requirements.txt
```

参考编解码器依赖可以参考：

```powershell
pip install -r reference_requirements.txt
```

另外还需要系统能调用 `ffmpeg`。建议优先使用 `--video-codec x265` 或 `--video-codec x264`，这两个路径通过 ffmpeg 管道运行。若使用 `hm` 或 `vtm`，需要按参考工具集 README 在 `reference_codec/dependency/HM-18.0` 或 `reference_codec/dependency/VTM-23.11` 下放置编译好的可执行文件。

## 输入

LOD 目录中必须有 5 个 PLY 文件，按文件名排序后视为：

```text
lod_00.ply  # 最高精度
lod_01.ply
lod_02.ply
lod_03.ply
lod_04.ply  # 最低精度 / residual base
```

## 一键式分步运行

下面假设：

- 5 个 LOD PLY 在  
  `D:\Documents\3DGS编解码\output\model_ply_lod_rd\lods`
- 输出目录为  
  `D:\Documents\3DGS编解码\output\reference_lod_rd`

### 1. Independent LOD 编码

```powershell
python reference_lod_independent.py `
  --lod-dir "D:\Documents\3DGS编解码\output\model_ply_lod_rd\lods" `
  --out-dir "D:\Documents\3DGS编解码\output\reference_lod_rd" `
  --qualities 33,66,100 `
  --video-codec x265 `
  --version mix_d3
```

### 2. Residual LOD 编码

```powershell
python reference_lod_residual.py `
  --lod-dir "D:\Documents\3DGS编解码\output\model_ply_lod_rd\lods" `
  --out-dir "D:\Documents\3DGS编解码\output\reference_lod_rd" `
  --qualities 33,66,100 `
  --video-codec x265 `
  --version mix_d3
```

### 3. 合并编码结果

```powershell
python merge_encode_csv.py `
  --inputs "D:\Documents\3DGS编解码\output\reference_lod_rd\independent_encoded_size.csv" `
           "D:\Documents\3DGS编解码\output\reference_lod_rd\residual_encoded_size.csv" `
  --output "D:\Documents\3DGS编解码\output\reference_lod_rd\merged_encoded_size.csv"
```

### 4. 渲染原始和解码 PLY

```powershell
python render.py `
  --lod-dir "D:\Documents\3DGS编解码\output\model_ply_lod_rd\lods" `
  --encode-csv "D:\Documents\3DGS编解码\output\reference_lod_rd\merged_encoded_size.csv" `
  --out-dir "D:\Documents\3DGS编解码\output\reference_lod_rd" `
  --views iso `
  --render-size 512
```

多视角平均 PSNR 可用：

```powershell
--views front,right,back,left,top,iso
```

### 5. 计算 PSNR 并画图

```powershell
python psnr_report.py `
  --render-index "D:\Documents\3DGS编解码\output\reference_lod_rd\render_index.csv" `
  --out-dir "D:\Documents\3DGS编解码\output\reference_lod_rd"
```

最终输出：

```text
reference_lod_rd/
  independent_encoded_size.csv
  residual_encoded_size.csv
  merged_encoded_size.csv
  render_index.csv
  psnr_size_results.csv
  png/
    lod_00_psnr_vs_size.png
    lod_01_psnr_vs_size.png
    lod_02_psnr_vs_size.png
    lod_03_psnr_vs_size.png
    lod_04_psnr_vs_size.png
  renders/
    original/
    independent/
    residual_lod/
```

## 码率口径

Independent 的 `encoded_size_bytes` 是对应 LOD 的单个 `.egsc` 文件大小。

Residual 的 `incremental_size_bytes` 是当前 residual/base `.egsc` 的增量大小；`encoded_size_bytes` 是从 `lod_04` base 到当前 LOD 所需的累计大小，更符合“解码该层级所需码量”的 RD 比较口径。

## 使用外部参考工具集目录

如果你不想使用 vendored 的 `reference_codec/`，也可以显式指定原始参考工具集目录：

```powershell
python reference_lod_independent.py `
  --codec-root "D:\Documents\《支持六自由度交互的三维图像格式标准》参考工具集v1.0.3\基于glTF存储三维图像格式的参考实现\glTF三维图像格式存储和解析\3DGS编码实现\python" `
  --lod-dir "D:\path\to\lods" `
  --out-dir "D:\path\to\out"
```
