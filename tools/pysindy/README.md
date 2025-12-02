# PySINDy 动力学模型训练与测试

使用 PySINDy (Sparse Identification of Nonlinear Dynamics) 算法从时间序列数据中学习动力学系统的控制方程。

## 项目结构

```
pykoop/                      # 项目根目录
├── data_generators/         # 动力学系统数据生成器（共享模块）
│   ├── __init__.py         # 包初始化
│   ├── base.py             # 动力学系统基类
│   ├── lorenz.py           # Lorenz 吸引子系统
│   └── spring.py           # 弹簧振荡器系统
├── datasource/             # 配置文件目录
│   ├── lorenz.json         # Lorenz 系统配置
│   └── spring.json         # Spring 系统配置
└── tools/pysindy/          # PySINDy 工具目录
    ├── results/            # 结果输出目录（自动创建）
    │   ├── sindy_results.png    # 可视化结果
    │   └── sindy_model_*.pkl    # 训练好的模型
    ├── train_and_test.py   # 主程序
    ├── requirements.txt    # 依赖列表
    └── README.md          # 本文件
```

**注意**: `data_generators` 模块位于项目根目录，可被多个工具共享使用。

## 功能特性

✅ **模块化设计**: 数据生成器与模型训练分离，易于扩展
✅ **完整流程**: 包含数据生成、模型训练、测试和可视化
✅ **自动发现方程**: 使用 SINDy 算法自动从数据中发现动力学方程
✅ **性能评估**: 计算 MSE 和 R² 等指标
✅ **可视化分析**: 生成 3D 轨迹、时间序列和误差分析图
✅ **模型保存**: 支持保存和加载训练好的模型

## 安装依赖

```bash
# 进入项目目录
cd tools/pysindy

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 基本使用

使用配置文件运行程序：

```bash
# 使用 Lorenz 系统配置
python train_and_test.py --config ../../datasource/lorenz.json

# 使用 Spring 系统配置
python train_and_test.py --config ../../datasource/spring.json
```

**注意**: `--config` 参数是必需的，程序通过配置文件来指定所有参数。

### 自定义配置

如需修改参数，请编辑配置文件或创建新的配置文件。配置文件格式见下文"配置文件格式"部分。

### 程序输出

程序运行时会输出以下信息：

1. **数据生成阶段**
   - 系统参数和方程
   - 数据统计信息

2. **模型训练阶段**
   - 训练参数（多项式阶数、稀疏化阈值等）
   - **发现的动力学方程**
   - 特征和系数统计

3. **模型测试阶段**
   - 性能指标（MSE、R²）
   - 各维度的详细指标

4. **可视化结果**
   - 3D 轨迹对比图
   - 各维度时间序列对比
   - 误差分析图

### 输出文件

- `results/sindy_results.png`: 可视化结果图
- `results/sindy_model.pkl`: 训练好的模型（可用于后续加载）

## 自定义使用

### 修改系统参数

```python
from data_generators import LorenzSystem

# 创建自定义参数的 Lorenz 系统
system = LorenzSystem(sigma=10.0, rho=28.0, beta=8.0/3.0)
```

### 调整训练参数

在 `train_and_test.py` 中修改以下参数：

```python
# 数据生成参数
t_span=(0, 50)        # 时间范围
n_points=5000         # 采样点数
noise_level=0.0       # 噪声水平

# 模型训练参数
poly_order=3          # 多项式阶数
threshold=0.01        # 稀疏化阈值
```

### 添加噪声测试

```python
t_train, x_train = generate_training_data(
    system,
    t_span=(0, 50),
    n_points=5000,
    noise_level=0.1  # 添加 10% 的噪声
)
```

## 扩展新的动力学系统

### 1. 创建新的系统类

在 `data_generators/` 目录下创建新文件，例如 `vanderpol.py`:

```python
"""Van der Pol 振荡器系统"""
import numpy as np
from .base import DynamicalSystem

class VanDerPolOscillator(DynamicalSystem):
    def __init__(self, mu=1.0):
        super().__init__()
        self.name = "Van der Pol Oscillator"
        self.dim = 2
        self.parameters = {'mu': mu}
    
    def _derivatives(self, t, state):
        x, y = state
        mu = self.parameters['mu']
        
        dx_dt = y
        dy_dt = mu * (1 - x**2) * y - x
        
        return np.array([dx_dt, dy_dt])
    
    def get_default_initial_conditions(self):
        return np.array([2.0, 0.0])
    
    def get_equations_text(self):
        mu = self.parameters['mu']
        return f"""Van der Pol Oscillator:
    dx/dt = y
    dy/dt = μ(1 - x²)y - x
    
Parameters:
    μ (mu) = {mu}
"""
```

### 2. 更新 `__init__.py`

```python
from .vanderpol import VanDerPolOscillator

__all__ = [
    'DynamicalSystem',
    'LorenzSystem',
    'VanDerPolOscillator',
]
```

### 3. 在主程序中使用

```python
from data_generators import VanDerPolOscillator

# 创建新系统
system = VanDerPolOscillator(mu=1.0)

# 其余代码保持不变
```

## 关于 PySINDy

PySINDy 是一个用于从数据中识别非线性动力系统的 Python 包。它实现了 SINDy (Sparse Identification of Nonlinear Dynamics) 算法，该算法可以：

- 从时间序列数据中学习动力学方程
- 使用稀疏优化找到最简洁的方程形式
- 适用于各种非线性系统

### SINDy 算法原理

1. **候选函数库**: 构建可能的函数项（如多项式、三角函数等）
2. **稀疏优化**: 使用 STLSQ 等算法选择最重要的项
3. **方程识别**: 自动发现描述系统动力学的方程

## 配置文件格式

配置文件使用 JSON 格式，包含以下字段：

```json
{
  "system_type": "系统类型 (lorenz 或 spring)",
  "system_params": {
    "参数名": "参数值"
  },
  "training": {
    "t_span": [起始时间, 结束时间],
    "n_points": 采样点数,
    "noise_level": 噪声水平
  },
  "sindy": {
    "poly_order": 多项式阶数,
    "threshold": 稀疏化阈值
  },
  "seed": 随机种子,
  "description": "配置描述"
}
```

### 配置示例

查看 `datasource/lorenz.json` 和 `datasource/spring.json` 获取完整示例。

## 支持的动力学系统

### 1. Lorenz 系统

Lorenz 吸引子是经典的混沌系统，由 Edward Lorenz 在 1963 年研究大气对流时发现。

#### 系统方程

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

#### 默认参数（产生混沌行为）

- σ = 10
- ρ = 28
- β = 8/3

#### 特点

- **混沌**: 对初始条件极度敏感
- **吸引子**: 轨迹被吸引到特定的几何结构
- **蝴蝶形状**: 在相空间中呈现标志性的双翼形状

### 2. Spring 振荡器系统

耦合弹簧振荡器系统，由多个通过弹簧连接的质点组成，可以形成不同的分组结构。

#### 系统方程

对于每个振子 i：

```
dx_i/dt = v_i
dv_i/dt = -Σ k_ij * (x_i - x_j) / m_i
```

其中：
- k_ij 是连接振子 i 和 j 的弹簧常数
- 同组内使用强弹簧常数 k_strong
- 不同组间使用弱弹簧常数 k_weak
- m_i 是振子质量（默认为 1.0）

#### 默认参数

- 振子数量: 6
- 强弹簧常数 k_strong: 50.0
- 弱弹簧常数 k_weak: 1.0
- 分组: {0:'a', 1:'b', 2:'b', 3:'c', 4:'c', 5:'c'}

#### 特点

- **模块化结构**: 可以定义不同的分组结构
- **多尺度动力学**: 组内和组间具有不同的耦合强度
- **可扩展**: 支持任意数量的振子和分组

## 常见问题

### Q: 模型性能不佳怎么办？

A: 尝试以下方法：
- 增加数据点数
- 调整多项式阶数
- 修改稀疏化阈值
- 检查数据质量（是否需要去噪）

### Q: 如何加载保存的模型？

```python
import pickle

with open('results/sindy_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 使用模型进行预测
x_pred = model.simulate(initial_conditions, t)
```

### Q: 程序运行时matplotlib不显示图形？

A: 如果使用远程服务器或无图形界面环境，可以注释掉 `plt.show()` 这行，仅保存图形文件。

## 参考资料

- [PySINDy 官方文档](https://pysindy.readthedocs.io/)
- [SINDy 原始论文](https://www.pnas.org/doi/10.1073/pnas.1517384113)
- [Lorenz 系统介绍](https://en.wikipedia.org/wiki/Lorenz_system)

## 许可证

本项目仅供学习和研究使用。

## 作者

Created by Cline AI Assistant
