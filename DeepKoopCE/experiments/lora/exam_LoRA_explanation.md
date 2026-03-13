# Koopman LoRA 实现代码详解

## 一、总体概述

这是一个使用 **LoRA (Low-Rank Approximation)** 方法学习 **Koopman 算子**的完整实现。Koopman 算子理论用于将非线性动力系统在高维函数空间中线性化表示。

### 核心思想
- **输入**: 动力系统的时间序列数据 (x_t, x_{t+1})
- **目标**: 学习特征函数 φ(x)，使得 K·φ(x_t) ≈ φ(x_{t+1})，其中 K 是线性 Koopman 算子
- **方法**: 使用神经网络学习特征函数，通过 LoRA 损失优化

---

## 二、核心组件详解

### 1. LoRA 损失函数 (LoRALoss)

```python
class LoRALoss(nn.Module):
    def __init__(self, nesting: str = None, n_modes: int = None)
```

**作用**: 实现论文中的 LoRA 优化目标

**损失公式**:
```
Loss = -2·tr(T[f,g]) + tr(M_ρ₀[f] · M_ρ₁[g])
```

其中:
- `T[f,g] = E[f·g^T]`: 互相关矩阵
- `M_ρ₀[f] = E[f·f^T]`: f 的二阶矩
- `M_ρ₁[g] = E[g·g^T]`: g 的二阶矩

**Nesting 策略**:
- `None`: 标准 LoRA
- `'seq'` (Sequential Nesting): 部分梯度停止，提高数值稳定性
- `'jnt'` (Joint Nesting): 使用加权掩码，推荐策略

**关键实现**:
```python
# 相关性项: 最大化 f 和 g 的相关性
corr_term = -2 * (f * g).mean(0).sum()

# 度量项: 正则化，确保 f 和 g 的度量结构
M_f = self._compute_second_moment(f)
M_g = self._compute_second_moment(g)
metric_term = (M_f * M_g).sum()
```

### 2. Koopman 模型架构 (KoopmanModel)

```python
class KoopmanModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_modes, center_data=True)
```

**双编码器设计**:
- `encoder_f`: 编码当前状态 x_t
- `encoder_g`: 编码未来状态 x_{t+1}

**为什么需要两个网络？**
- 学习 Koopman 算子的左右奇异函数对 (φ, ψ)
- 满足 CCA (Canonical Correlation Analysis) 的对称性要求

**Centering 机制** (center_data=True):
```python
# 显式添加常数项 φ₀=1
ones = torch.ones(x.shape[0], 1).to(x.device)
f_out = torch.cat([ones, f_out], dim=1)  # [batch, n_modes]
```
这确保了 Koopman 算子始终包含常数特征函数（对应平衡点/不变测度）。

---

## 三、数据生成与处理

### 1. 动力系统示例

**二维非线性系统**:
```python
def system_dynamics(state, lam=0.9, mu=0.5):
    x1, x2 = state[0], state[1]
    y1 = lam * x1
    y2 = mu * x2 + (lam**2 - mu) * (x1**2)
    return np.array([y1, y2])
```

这是一个**耦合的非线性映射**:
- x₁ 的演化是线性的（收缩映射，λ<1）
- x₂ 受 x₁² 的非线性项影响

**替代示例**: Logistic Map (混沌系统)
```python
x_{t+1} = 4 * x_t * (1 - x_t) + noise
```

### 2. 数据准备流程

**训练数据**:
```python
# 1. 生成长轨迹
raw_train_data = generate_long_trajectories(N_TRAIN_TRAJS, LEN_TRAIN_TRAJS)
# Shape: [100条轨迹, 100步, 2维]

# 2. 滑动窗口切片
sequences = create_sequences(raw_train_data, seq_length=31)
# Shape: [样本数, 31步, 2维]

# 3. 创建 DataLoader
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

**关键点**:
- 需要长轨迹来捕获系统动力学
- 滑动窗口增加训练样本数
- 每个样本包含 T+1 步（1 个初始状态 + T 个预测步）

---

## 四、训练过程

### 训练循环

```python
def train_koopman(model, dataloader, epochs=100, lr=1e-3, nesting='jnt'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = LoRALoss(nesting=nesting, n_modes=model.n_modes)
    
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            # 前向传播
            f, g = model(batch_x, batch_y)  # 编码为特征空间
            
            # 计算 LoRA 损失
            loss = criterion(f, g)
            
            # 反向传播优化
            loss.backward()
            optimizer.step()
```

**优化目标**:
- 最大化 f(x_t) 和 g(x_{t+1}) 之间的相关性
- 同时保持它们的度量结构（正交性和归一化）

---

## 五、频谱分析与可视化

### 1. CCA 后处理 (compute_spectral_decomposition)

训练完成后，执行以下步骤提取 Koopman 频谱:

**步骤 1: 计算经验矩阵**
```python
M_f = (f.T @ f) / N  # f 的协方差
M_g = (g.T @ g) / N  # g 的协方差
T_fg = (f.T @ g) / N  # 互相关
```

**步骤 2: 白化 (Whitening)**
```python
M_f_inv_half = M_f^(-1/2)  # 使用特征值分解计算
M_g_inv_half = M_g^(-1/2)
```

**步骤 3: 奇异值分解**
```python
O = M_f_inv_half @ T_fg @ M_g_inv_half
U, S, V = SVD(O)
```
- `S`: **Koopman 奇异值**（反映各模态的重要性）
- `U, V`: 奇异向量（用于构造正交特征函数）

**步骤 4: 构建 Koopman 矩阵**
```python
K_ols = M_f^(-1) @ T_fg  # EDMD 估计
eigvals, eigvecs = eig(K_ols)  # Koopman 特征值
```

### 2. 可视化内容 (visualize_results)

**图 1: 学习到的奇异函数**
- 绘制前 5 个模态 φ₀(x), φ₁(x), ..., φ₄(x)
- 横轴: 状态空间 x
- 纵轴: 特征函数值

**图 2: 正交性热力图**
- 显示 E[f·f^T] 矩阵
- 对角线应接近 1（归一化）
- 非对角线应接近 0（正交性）

**图 3: 奇异值频谱**
- 柱状图显示各模态的奇异值 σᵢ
- 较大的奇异值对应主导模态

---

## 六、关键技术要点

### 1. 为什么使用双编码器？

**理论依据**: Koopman 算子的 SVD 分解
```
K = Σᵢ σᵢ·ψᵢ ⊗ φᵢ
```
- `φᵢ`: 右奇异函数（作用在 x_t）
- `ψᵢ`: 左奇异函数（作用在 x_{t+1}）
- `encoder_f` 学习 φ，`encoder_g` 学习 ψ

### 2. Centering 的作用

- Koopman 算子总有特征值 1 对应常数函数
- 显式建模常数项避免网络学习退化解
- 数值上更稳定

### 3. Nesting 策略的必要性

**问题**: 直接优化 LoRA 损失可能导致:
- 模态崩塌（所有特征函数学成相同）
- 数值不稳定

**解决方案**:
- **Sequential Nesting**: 逐个模态优化，早期模态的梯度被部分停止
- **Joint Nesting**: 加权损失，高优先级模态获得更大权重

### 4. 与传统 EDMD 的区别

| 方法 | 特征函数 | 优点 | 缺点 |
|------|---------|------|------|
| EDMD | 手工选择（多项式等） | 理论保证强 | 表达能力受限 |
| LoRA | 神经网络自动学习 | 强大的非线性表达 | 需要大量数据 |

---

## 七、实验流程总结

### 完整 Pipeline

```
1. 数据生成
   ↓
2. 训练模型（优化 LoRA 损失）
   ↓
3. CCA 后处理（提取奇异值）
   ↓
4. 频谱分析（特征值/奇异值）
   ↓
5. 可视化验证（奇异函数形状、正交性）
```

### 评估指标

1. **损失曲线**: 应该单调下降并收敛
2. **奇异值分布**: 前几个模态应明显大于后续
3. **正交性**: M_f 应接近单位矩阵
4. **特征值分布**: 应在单位圆内（稳定系统）

---

## 八、代码亮点与创新

### 1. 模块化设计
- 损失函数、模型、数据处理完全解耦
- 易于替换不同的动力系统或网络架构

### 2. 数值稳定性考虑
```python
# 矩阵求逆时的正则化
vals = torch.clamp(vals, min=epsilon)
```

### 3. 灵活的 Nesting 策略
- 通过参数轻松切换不同优化策略
- 适应不同难度的问题

### 4. 完整的可视化工具
- 从原始数据到最终频谱的全流程可视化
- 便于调试和理解模型行为

---

## 九、潜在应用

1. **动力系统预测**: 学到的 Koopman 算子可用于长期预测
2. **控制设计**: 线性控制理论可应用于非线性系统
3. **模态分解**: 识别系统的主要动力学模式
4. **降维**: 高维系统投影到低维 Koopman 子空间

---

## 十、注意事项与建议

### 超参数调优
- `n_modes`: 根据系统复杂度选择（通常 5-20）
- `hidden_dim`: 网络容量，太小欠拟合，太大过拟合
- `nesting`: 对复杂系统推荐 'jnt'

### 数据要求
- 需要足够长的轨迹覆盖状态空间
- 训练集应包含系统的各种动力学行为
- 噪声数据需要更大的 batch size

### 调试技巧
1. 先在简单系统（如 Logistic Map）上验证
2. 检查奇异值是否递减
3. 可视化学习到的特征函数形状
4. 监控正交性矩阵

---

## 参考文献

本代码基于以下理论:
- **Koopman 算子理论**: Koopman, B. O. (1931)
- **LoRA 方法**: Low-Rank Approximation for Koopman Operator Learning
- **CCA 连接**: Canonical Correlation Analysis in Koopman Representation

---

## 代码运行环境

```bash
# 主要依赖
torch >= 1.9
numpy >= 1.19
matplotlib >= 3.3

# 可选（MacBook 用户）
# MPS (Metal Performance Shaders) 支持
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
