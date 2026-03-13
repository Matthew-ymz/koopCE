# LoRA 中两种奇异值的区别与联系

## 问题描述

在 LoRA Koopman 算子学习中，存在两个不同来源的奇异值：

1. **`model.svals`**: 模型中学习的奇异值参数
2. **K_matrix SVD**: Koopman 算子矩阵 K 的奇异值分解

这两者经常有显著差异，初学者容易混淆。本文档详细解释其中原因。

## 理论背景

### 1. LoRA Loss 中的学习目标

LoRA loss 最小化：
```
L(f, g) = -2·Tr(T[f, g]) + Tr(M_f · M_g)
```

其中：
- `f = encoder_f(x)`: 当前状态的编码特征
- `g = encoder_g(x')`: 未来状态的编码特征
- `M_f = E[f·f^T]`: 特征的自相关矩阵
- `T_fg = E[f·g^T]`: 特征的交叉相关矩阵

### 2. 两种奇异值的含义

#### (1) model.svals - 学习的参数

```python
# 在 KoopmanLoRAModel 中
self.svals_params = nn.Parameter(...)  # 可学习参数

@property
def svals(self):
    vals = torch.sigmoid(self.svals_params)  # 通过 sigmoid 变换
    if self.has_centering:
        vals = torch.cat([torch.ones(1), vals])  # 可选地固定第一个为 1
    return vals
```

**在训练中的作用**:
```python
# compute_koopman_spectrum 中
if model.svals is not None:
    scale = svals.sqrt().unsqueeze(0)
    f = f * scale    # 缩放特征
    g = g * scale    # 缩放特征
```

**物理意义**:
- 这些是**理想的** Koopman 算子奇异值结构
- 用来指导特征学习向某个目标方向
- 代表网络**希望学到**的奇异值

#### (2) K_matrix SVD - 经验奇异值

```python
# compute_koopman_spectrum 中
K_matrix = torch.linalg.pinv(M_f) @ T_fg
U, S, Vh = torch.linalg.svd(K_matrix)
```

**计算过程**:
```
1. M_f = E[f·f^T]           # 特征的二阶矩
2. T_fg = E[f·g^T]          # 特征交叉相关
3. K_matrix = M_f^(-1) @ T_fg   # EDMD Koopman 算子
4. K_matrix = U @ S @ V^T   # SVD 分解
```

**物理意义**:
- 这是从学习的特征中**实际获得**的 Koopman 算子
- 反映了特征空间中的真实线性动力学
- 代表网络**实际学到**的奇异值

## 为什么两者不同？

### 原因 1: 特征学习的不完全性

LoRA loss 优化的目标是让 f 和 g 满足：
```
E[f·g^T] ≈ σ·δ_ij   (近似正交和缩放)
```

但由于：
- 有限的训练轮数
- 非凸优化过程
- 网络容量的限制

特征不能完全达到理想的奇异值分解结构。

### 原因 2: 缩放应用的位置不同

**model.svals 的使用方式**:
```python
# 在损失函数中的缩放
if model.svals is not None:
    scale = svals.sqrt().unsqueeze(0)
    f = f * scale  # 损失计算前缩放
    g = g * scale
    # 然后计算 M_f, M_g, T_fg
```

**K_matrix SVD 的缩放**:
```python
# 在 K_matrix 计算中
K_matrix = M_f^(-1) @ T_fg
# 这里没有再次应用 svals 缩放
# M_f 已经包含了缩放特征的二阶矩
```

### 原因 3: 矩阵求逆的影响

```python
K_matrix = pinv(M_f) @ T_fg
```

- `pinv(M_f)` 涉及矩阵求逆，这会改变奇异值的相对关系
- 特别是当 M_f 不是完全正交时
- 条件数大的矩阵求逆会放大误差

### 原因 4: 特征空间的几何结构

学习过程中：
- `model.svals` 是*参数化的理想目标*
- `K_matrix SVD` 反映的是*实际学到的特征空间的几何*

即使两者都在"学习"，它们优化的空间和目标不同。

## 实例分析

### 典型对比

假设 `model.svals = [1.0, 0.5, 0.1]` (learn_svals=True, has_centering=True)

运行 `compute_koopman_spectrum` 后，K_matrix 的 SVD 可能给出：
```
S_K_matrix = [1.2, 0.6, 0.15]  # 与 svals 相似但不同
```

差异的原因：

1. **放大效应** (1.0 → 1.2):
   - 由 `pinv(M_f)` 的条件数引入
   - 特征空间的各向异性

2. **缩小效应** (0.5 → 0.6):
   - 特征之间的相互作用
   - 不完全正交性的补偿

3. **噪声影响** (0.1 → 0.15):
   - 小奇异值对噪声更敏感
   - 训练不足导致的波动

## 如何正确使用它们？

### 1. model.svals - 诊断特征学习

```python
if model.svals is not None:
    svals_learned = model.svals.detach().cpu().numpy()
    print("Learned singular value targets:")
    print(svals_learned)
    
    # 检查：
    # ✓ 是否单调递减？
    # ✓ 是否在 (0, 1] 范围内？
    # ✓ 是否有明显的"肘点"？
```

**用途**: 
- 理解网络的学习方向
- 检查是否学到了有意义的结构
- 调整初始化和约束

### 2. K_matrix SVD - 分析实际动力学

```python
# 从 compute_koopman_spectrum 获取
U, S, Vt = np.linalg.svd(K_matrix)

print("Actual Koopman singular values:")
print(S)

# 这些才是用来做预测/降维的真实值
cumulative_energy = np.cumsum(S) / np.sum(S)
```

**用途**:
- 进行降维和模态选择
- 进行预测和长期行为分析
- 构建宏观动力学模型

## 如何改进一致性？

### 策略 1: 增加训练轮数

```python
n_epochs = 200  # 增加到 200 或更多
# 给网络更多时间学习目标结构
```

### 策略 2: 使用正则化

```python
loss_fn = NestedLoRALoss(
    n_modes=N_MODES,
    nesting='jnt',
    reg_weight=0.1  # 增加正则化权重
)
# 让 M_f 更接近单位矩阵
# 从而使 K_matrix 更接近理想结构
```

### 策略 3: 共享编码器

```python
model = KoopmanLoRAModel(
    input_dim=INPUT_DIM,
    n_modes=N_MODES,
    shared_encoder=True,  # 使用共享编码器
    learn_svals=True
)
# 减少要学习的参数
# 更稳定的特征空间
```

### 策略 4: 调整网络架构

```python
model = KoopmanLoRAModel(
    input_dim=INPUT_DIM,
    n_modes=N_MODES,
    hidden_dims=[256, 256, 128],  # 更大的网络
    activation='ELU',              # 更好的激活函数
    use_batchnorm=True             # 添加 batch norm
)
```

## 建议的最佳实践

### ✅ 正确的分析流程

```python
# 1. 检查学习的 svals（如果启用）
if model.svals is not None:
    print("Learned singular values:")
    print(model.svals.detach().cpu().numpy())

# 2. 计算真实的 K_matrix 和其 SVD
spectrum = compute_koopman_spectrum(model, test_loader, device)
K_matrix = spectrum['K_matrix']
S_K = spectrum['singular_values']

print("\nActual K_matrix singular values:")
print(S_K)

# 3. 对比两者（如果都有）
if model.svals is not None:
    print("\nRatio (K_matrix_svals / learned_svals):")
    ratio = S_K / model.svals.detach().cpu().numpy()
    print(ratio)
    print(f"Ratio mean: {ratio.mean():.4f}, std: {ratio.std():.4f}")

# 4. 用 K_matrix SVD 进行后续分析
# （不要用 model.svals）
```

### ✅ 何时使用哪个奇异值？

| 用途 | 使用哪个 |
|------|---------|
| 理解网络学习目标 | `model.svals` |
| 评估训练进度 | `model.svals` 的变化 |
| 进行预测 | K_matrix 或其 SVD |
| 选择模态数量 | K_matrix SVD 的 S |
| 提取粗粒化系数 | K_matrix SVD 的 U |
| 分析宏观动力学 | K_matrix SVD |

## 常见错误

### ❌ 错误 1: 直接使用 model.svals 作为 K_matrix 的奇异值

```python
# 错误！
svals = model.svals.cpu().numpy()
# 把这个当作 Koopman 算子的奇异值分析

# 正确做法
spectrum = compute_koopman_spectrum(...)
S = spectrum['singular_values']
```

### ❌ 错误 2: 混淆维度缩放的顺序

```python
# 错误！
K_matrix = pinv(M_f) @ T_fg
U_wrong, S_wrong, Vh = svd(K_matrix * model.svals)  # 重复缩放

# 正确做法
U, S, Vh = svd(K_matrix)  # 直接分解，已经包含了缩放
```

### ❌ 错误 3: 期望 model.svals 和 K_matrix SVD 完全相同

```python
# 错误的期望！
# 它们代表不同的东西，不应该完全相同

# 正确的理解
# model.svals: 学习目标
# K_matrix SVD: 实际结果
# 接近但不完全相同是正常的
```

## 总结

| 特性 | model.svals | K_matrix SVD |
|------|------------|-------------|
| **性质** | 可学习参数 | 从数据计算 |
| **含义** | 理想目标 | 实际结果 |
| **来源** | 神经网络权重 | EDMD 计算 |
| **用途** | 诊断学习 | 分析动力学 |
| **应该相似？** | 大致相似 | 不要完全相同 |
| **用于降维** | ❌ 不建议 | ✅ 推荐使用 |

## 推荐阅读

1. **LoRA_SVD_Analysis_Guide.md** - 如何正确进行 SVD 分析
2. **demo_standalone_lora.ipynb** - 完整的使用示例
3. **standalone_lora.py** - 源代码中的 `compute_koopman_spectrum` 函数

---

**版本**: v1.0  
**最后更新**: 2026-02-10  
**关键词**: 奇异值、LoRA、Koopman 算子、降维
