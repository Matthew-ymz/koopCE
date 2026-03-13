# KoopCE

KoopCE 是一个围绕 Koopman coarse-graining、稀疏识别和相关实验分支整理的研究代码仓库。当前目录已经按“核心代码 / 数据 / 实验分组 / 论文材料”重新归类，便于按数据集或任务快速定位实验入口。

## 目录结构

```text
koopCE/
├── DeepKoopCE/                       # Deep Koopman 主干代码与训练产物
│   ├── deepkoop_func.py              # 核心模型、损失函数、数据切片工具
│   ├── model_save/                   # 已保存的 checkpoint
│   ├── data/                         # DeepKoopCE 分支附带数据
│   └── experiments/
│       ├── air_quality/              # 空气质量深度 Koopman 实验
│       ├── toy/                      # toy system 深度 Koopman 实验
│       └── lora/                     # LoRA / SVD 分析与 demo
├── experiments/                      # 按数据或任务整理的经典实验
│   ├── air_quality/
│   │   └── classical_sparse/         # 空气质量稀疏基线 notebook
│   ├── kuramoto/                     # Kuramoto 系列实验
│   ├── discrete_maps/                # map / Rulkov 等离散动力系统实验
│   ├── sir/                          # SIR 实验
│   ├── toy_nonlinear/                # toy / 非线性系统 / 符号回归实验
│   ├── resdmd/                       # ResDMD 实验
│   └── synthetic_systems/
│       └── pysindy/                  # Lorenz / spring 的 PySINDy 实验
├── neural_science/                   # 神经科学数据分支
│   ├── experiments/                  # EEG/阶段建模 notebook 与预处理脚本
│   ├── data/                         # 原始 MAT 文件与已保存模型
│   └── visual_inducted_conscious/    # 预处理后的阶段数据
├── data_generators/                  # 共享数据生成器与空气质量原始数据
├── tools/                            # 通用工具函数
├── results/                          # 图像与结果产物
├── doc/                              # 报告与参考资料
├── paper/                            # 论文草稿、图表清单、证据映射
├── requirements.txt
├── requirements-dev.txt
└── pytest.ini
```

## 实验分组说明

- `experiments/air_quality/classical_sparse/`
  - 经典稀疏 coarse-graining 空气质量实验。
- `experiments/kuramoto/`
  - `kuramoto_2group.ipynb`、`kuramoto_grid_sweep_analysis.ipynb` 等 Kuramoto 相关实验集中放在这里。
- `experiments/discrete_maps/`
  - `examp_map_pll.ipynb`、`examp_rulkov_pll.ipynb`，以及对应的 `data_save/` 中间结果。
- `experiments/sir/`
  - SIR 数据上的 Koopman / 稀疏识别实验。
- `experiments/toy_nonlinear/`
  - toy system、非线性振子、符号回归和理论推导类 notebook。
- `experiments/resdmd/`
  - ResDMD 相关 notebook。
- `experiments/synthetic_systems/pysindy/`
  - Lorenz 和 spring 系统的 PySINDy 基线，配置文件位于 `configs/`。

## DeepKoopCE 分支

- `DeepKoopCE/deepkoop_func.py` 是深度 Koopman 主干实现。
- `DeepKoopCE/experiments/air_quality/air_data.ipynb` 是空气质量深度模型实验入口。
- `DeepKoopCE/experiments/toy/toy_example.ipynb` 是 toy system 深度实验入口。
- `DeepKoopCE/experiments/lora/` 集中放置 LoRA demo、SVD 分析脚本和说明文档。
- `DeepKoopCE/model_save/` 保留已有 `.pth` checkpoint，不与 notebook 混放。

## Neural Science 分支

- `neural_science/experiments/read_data_improved.py`
  - 读取 `neural_science/data/` 下的 MAT 数据并生成阶段数据。
- `neural_science/experiments/compare_stages.ipynb`
  - 阶段对比实验。
- `neural_science/experiments/exp_stage2.ipynb`
  - stage-2 相关实验。
- `neural_science/data/`
  - 原始 MAT 文件与阶段模型 `.pkl`。

## 运行示例

PySINDy 基线现在从新目录启动：

```bash
python experiments/synthetic_systems/pysindy/train_and_test.py \
  --config experiments/synthetic_systems/pysindy/configs/lorenz.json
```

```bash
python experiments/synthetic_systems/pysindy/train_and_test.py \
  --config experiments/synthetic_systems/pysindy/configs/spring.json
```

如果你要找某个数据集的实验，优先看：

- 空气质量：`experiments/air_quality/` 和 `DeepKoopCE/experiments/air_quality/`
- Kuramoto：`experiments/kuramoto/`
- SIR：`experiments/sir/`
- 离散映射：`experiments/discrete_maps/`
- Toy / 非线性系统：`experiments/toy_nonlinear/` 和 `DeepKoopCE/experiments/toy/`
- 神经科学：`neural_science/experiments/`
