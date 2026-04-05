# NeuroFlow

NeuroFlow 是一个用于小鼠脑切片图像分析的自动化流程工具，集成了 DeepSlice、Allen Brain Atlas 和 Cellpose，实现从脑切片图像到细胞计数的全自动分析。

## 功能特性

- **DeepSlice 集成**: 自动预测脑切片在 Allen Brain Atlas 中的三维空间位置
- **ROI 自动提取**: 基于艾伦脑图谱自动提取指定的感兴趣脑区
- **细胞分割与计数**: 使用 Cellpose 深度学习模型进行高精度细胞分割
- **多脑区批量分析**: 支持同时分析多个脑区，自动统计细胞数量
- **可视化输出**: 生成分割结果可视化图像和统计报表
- **GUI 界面**: 提供友好的图形用户界面，支持脑区搜索与选择
- **自动环境管理**: 自动创建和管理所需的 Conda 环境

## 项目结构

```
NeuroFlow-main/
├── main.py                 # 主程序入口（命令行）
├── gui.py                  # GUI 图形界面
├── config.yaml             # 配置文件
├── setup_env.py            # 环境自动配置脚本
├── deepslice.py            # DeepSlice 运行脚本
├── cellpose_test.py        # Cellpose 细胞分割脚本
├── allensdk_ROI.py         # Allen Atlas ROI 提取脚本
├── modules/
│   ├── run_deepslice.py    # DeepSlice 模块接口
│   ├── run_atlas.py        # Atlas 模块接口
│   └── run_cellpose.py     # Cellpose 模块接口
├── utils/
│   ├── config.py           # 配置文件读写
│   ├── atlas_utils.py      # Atlas 数据处理工具
│   ├── image_processing.py # 图像预处理工具
│   ├── analysis_utils.py   # 数据分析工具
│   └── region_selector.py  # 脑区选择工具
├── atlas/
│   └── allen_mouse_10um_java-Ontology.json  # Allen 脑区本体数据
├── annotation/
│   └── ccf_2017/           # Allen CCF 2017 注释数据
└── structures.json         # 脑区结构数据
```

## 环境要求

- Python 3.9+
- Conda (Miniconda 或 Anaconda)
- 操作系统: Windows / macOS / Linux

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/NeuroFlow.git
cd NeuroFlow-main
```

### 2. 首次运行（自动安装环境）

首次运行时，程序会自动检查并创建所需的 Conda 环境，需要通过 GUI 启动并勾选“首次启动”选项：

```bash
python gui.py
```

程序将自动创建以下三个 Conda 环境：


| 环境名称    | 用途                 | 主要依赖                |
| ----------- | -------------------- | ----------------------- |
| `deepslice` | 脑切片空间定位       | DeepSlice, TensorFlow   |
| `allensdk`  | Allen Atlas 数据处理 | allensdk, numpy, pandas |
| `cellpose`  | 细胞分割             | cellpose, torch, opencv |

## 使用方法

### 方式一：GUI 图形界面

```bash
python gui.py
```

GUI 界面功能：

1. 选择数据路径
2. 搜索并选择脑区（支持中英文关键词搜索）
3. 配置 Cellpose 参数
4. 一键运行分析流程

### 方式二：命令行

```bash
python main.py \
    --image-dir ./raw_data \
    --region-ids 382 463 525 \
    --model-type cyto3 \
    --device cuda \
    --cellprob-threshold 0.0 \
    --flow-threshold 0.4
```

### 命令行参数说明


| 参数                   | 默认值 | 说明                                   |
| ---------------------- | ------ | -------------------------------------- |
| `--image-dir`          | -      | 输入图像目录                           |
| `--region-ids`         | -      | Allen Atlas 脑区 ID（多个用空格分隔）  |
| `--model-type`         | cyto3  | Cellpose 模型类型 (cyto3/cyto2/nuclei) |
| `--device`             | cpu    | 计算设备 (cpu/cuda/mps)                |
| `--auto-diameter`      | False  | 自动估计细胞直径                       |
| `--cellprob-threshold` | 0.0    | 细胞概率阈值（弱信号可设为 -2 到 -4）  |
| `--flow-threshold`     | 0.4    | Flow 阈值（密集细胞可设为 0.3-0.5）    |
| `--enhance-contrast`   | True   | 启用 CLAHE 对比度增强                  |
| `--clahe-clip`         | 1.0    | CLAHE 对比度限制                       |
| `--clahe-grid`         | 16     | CLAHE 网格大小                         |
| `--max-tile-edge`      | 8000   | 图像分块最大边长                       |

## 分析流程

```
输入图像 (TIFF/PNG)
       ↓
[STEP 1] DeepSlice: 预测切片空间位置
       ↓
[STEP 2] Atlas ROI: 提取指定脑区掩膜
       ↓
[STEP 3] Cellpose: 细胞分割与计数
       ↓
输出结果 (CSV + 可视化图像)
```

### 输入数据要求

- 图像格式: TIFF
- 图像类型: 荧光显微镜图像
- 建议分辨率: 10-25 μm/pixel

### 输出结果

分析完成后，在图像目录下生成：

```
your_data/
├── MyResults.csv              # DeepSlice 预测结果
├── ROI_data/                  # ROI 数据
│   ├── xxx_ROI_mask.png       # ROI 彩色掩膜
│   └── xxx_color_mapping.csv  # 颜色映射表
├── predicted_mask/            # 分割可视化
│   ├── xxx_MERGED.jpg         # 合并 ROI 结果
│   └── xxx_ROI_*.jpg          # 单个 ROI 结果
└── 细胞分析结果.csv           # 最终统计结果
```

## 常见脑区 ID 参考


| 脑区名称       | 缩写 | ID  |
| -------------- | ---- | --- |
| 海马 CA1       | CA1  | 382 |
| 海马 CA3       | CA3  | 463 |
| 初级视觉皮层   | VISp | 669 |
| 初级运动皮层   | MOp  | 985 |
| 内侧前额叶皮层 | ILA  | 895 |

更多脑区 ID 请参考 [Allen Brain Atlas](https://mouse.brain-map.org/experiment/thumbnails/100048576)。

## 依赖说明

### 主要 Python 包

- **DeepSlice**: 脑切片自动配准
- **AllenSDK**: Allen Brain Atlas 数据接口
- **Cellpose**: 细胞分割深度学习模型
- **PyQt5**: GUI 界面框架
- **OpenCV**: 图像处理
- **NumPy/Pandas**: 数据处理
- **scikit-image**: 图像分析

### macOS 额外依赖

macOS 用户需要安装 Homebrew 和 HDF5：

```bash
# 安装 Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 HDF5
brew install hdf5
```

## 常见问题

### Q: DeepSlice 权重下载失败？

A: 程序会自动尝试多种下载方式。如果仍然失败，请检查网络连接或使用代理。

### Q: 内存不足？

A: 对于大图像，程序会自动分块处理。可通过 `--max-tile-edge` 调整分块大小。

### Q: 细胞分割效果不佳？

A: 尝试调整以下参数：

- 降低 `--cellprob-threshold`（如 -2.0）以检测更多细胞
- 调整 `--flow-threshold`（0.3-0.5）以处理密集细胞
- 根据染色类型选择合适的 `--model-type`（nuclei 用于核染色）

## 致谢

- [DeepSlice](https://github.com/PolarBean/DeepSlice) - 脑切片自动配准
- [Allen Institute](https://alleninstitute.org/) - Allen Brain Atlas
- [Cellpose](https://github.com/MouseLand/cellpose) - 细胞分割模型
