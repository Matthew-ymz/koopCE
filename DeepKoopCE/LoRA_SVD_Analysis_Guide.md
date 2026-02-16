# LoRA Koopman 算子 SVD 分析与降维指南

## 概述

本文档详细说明如何对训练好的 LoRA Koopman 模型进行完整的 SVD 分析，包括奇异值谱分析、降维和宏观动力学可视化。

## 理论背景

### 1. Koopman 算子

Koopman 算子 K 是一个线性算子，作用在观测函数空间上：

```
K[φ](x) = φ(F(x))
```

其中 F(x) 是非线性动力系统的演化算子。

### 2. SVD 分解

对 Koopman 算子矩阵进行奇异值分解（SVD）：

```
K = U @ Σ @ V^T
```

其中：
- **U**: 左奇异向量矩阵（粗粒化函数/Coarse-graining functions）
- **Σ**: 奇异值对角矩阵（模态重要性）
- **V**: 右奇异向量矩阵

### 3. 降维的物理意义

通过截断 SVD，我们可以：
- **选择前 r 个最重要的奇异值**：保留系统的主要动力学模式
- **粗粒化（Coarse-graining）**：将高维微观状态映射到低维宏观状态
- **宏观动力学**：在低维空间中捕获系统的本质行为

## 分析流程

### 步骤 1: 获取 Koopman 矩阵

训练完成后，从模型中提取 Koopman 算子矩阵 K：

```python
# 方法 1: 通过 compute_koopman_spectrum 获取
spectrum = compute_koopman_spectrum(model, test_loader, device)
K_matrix = spectrum['K_matrix']  # shape: (n_modes, n_modes)

# 方法 2: 直接从二阶矩计算
# K = M_f^{-1} @ T_fg
```

### 步骤 2: SVD 分解

对 K 矩阵进行完整的 SVD 分解：

```python
U, S, Vt = np.linalg.svd(K_matrix)
V = Vt.T

# U: (n_modes, n_modes) - 左奇异向量
# S: (n_modes,) - 奇异值（降序排列）
# V: (n_modes, n_modes) - 右奇异向量
```

### 步骤 3: 奇异值谱分析

#### 3.1 可视化奇异值

```python
plt.figure(figsize=(10, 6))
plt.bar(range(len(S)), S, color='steelblue', edgecolor='k')
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value')
plt.title('Singular Value Spectrum of Koopman Operator')
plt.grid(True, axis='y', alpha=0.3)
plt.show()
```

#### 3.2 计算累积能量贡献

```python
cumulative_energy = np.cumsum(S) / np.sum(S)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(S)+1), cumulative_energy, 'o-', linewidth=2)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Modes')
plt.ylabel('Cumulative Energy')
plt.title('Cumulative Energy vs. Number of Modes')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 步骤 4: 选择截断 Rank

根据以下准则选择截断 rank：

1. **能量准则**：保留 90-95% 的累积能量
2. **奇异值间隙**：在奇异值有明显下降的地方截断
3. **物理意义**：根据问题的物理含义选择合适的维度

```python
# 示例：选择保留 95% 能量的 rank
threshold = 0.95
rank = np.argmax(cumulative_energy >= threshold) + 1
print(f"Selected rank: {rank} (captures {cumulative_energy[rank-1]:.2%} energy)")
```

### 步骤 5: 提取粗粒化函数

左奇异向量 U 的前 r 列即为粗粒化函数（Coarse-graining coefficients）：

```python
# 提取前 rank 个左奇异向量
coarse_grain_coeffs = U[:, :rank]  # shape: (n_modes, rank)

# 可视化粗粒化系数
import seaborn as sns
plt.figure(figsize=(8, 10))
sns.heatmap(coarse_grain_coeffs, 
            cmap='RdBu_r', 
            center=0,
            xticklabels=[f'Y{i}' for i in range(rank)],
            yticklabels=[f'φ{i}' for i in range(n_modes)])
plt.title('Coarse-Graining Coefficients (Left Singular Vectors)')
plt.xlabel('Macro Variables')
plt.ylabel('Micro Features')
plt.tight_layout()
plt.show()
```

### 步骤 6: 计算宏观变量

将微观特征映射到宏观空间：

```python
# 编码测试数据到特征空间
with torch.no_grad():
    X_test_features = model(X_test_tensor.to(device), lagged=False)
    X_test_features = X_test_features.cpu().numpy()

# 投影到宏观空间
Y_macro = X_test_features @ coarse_grain_coeffs  # shape: (n_samples, rank)
```

### 步骤 7: 宏观动力学分析

#### 7.1 时间序列可视化

```python
fig, axes = plt.subplots(rank, 1, figsize=(12, 3*rank), sharex=True)
if rank == 1:
    axes = [axes]

for i in range(rank):
    axes[i].plot(Y_macro[:500, i], linewidth=1.5, color='steelblue')
    axes[i].set_ylabel(f'Y_{i}', fontsize=12)
    axes[i].set_title(f'Macro Variable {i} (σ={S[i]:.4f})', fontsize=14)
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time Step', fontsize=12)
plt.suptitle('Macroscopic Dynamics (Reduced Dimensions)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

#### 7.2 相空间轨迹（2D/3D）

如果 rank >= 2，可以绘制相空间轨迹：

```python
if rank >= 2:
    plt.figure(figsize=(10, 8))
    plt.plot(Y_macro[:500, 0], Y_macro[:500, 1], 
             linewidth=1, alpha=0.7, color='purple')
    plt.scatter(Y_macro[0, 0], Y_macro[0, 1], 
                c='red', s=100, marker='*', label='Start', zorder=5)
    plt.xlabel(f'Y_0 (σ={S[0]:.4f})', fontsize=12)
    plt.ylabel(f'Y_1 (σ={S[1]:.4f})', fontsize=12)
    plt.title('Macroscopic Phase Space Trajectory', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

### 步骤 8: 重构与误差分析

#### 8.1 低秩重构

```python
# 使用前 rank 个模态重构 Koopman 矩阵
K_reconstructed = U[:, :rank] @ np.diag(S[:rank]) @ V[:, :rank].T

# 计算重构误差
reconstruction_error = np.linalg.norm(K_matrix - K_reconstructed, 'fro') / np.linalg.norm(K_matrix, 'fro')
print(f"Reconstruction Error (Frobenius norm): {reconstruction_error:.6f}")
```

#### 8.2 预测误差

```python
# 使用降维后的宏观动力学进行预测
K_macro = np.diag(S[:rank])  # 降维后的 Koopman 算子

# 预测下一步
Y_macro_next_pred = Y_macro @ K_macro
Y_macro_next_true = Y_test_features @ coarse_grain_coeffs

prediction_error = np.mean((Y_macro_next_pred - Y_macro_next_true)**2)
print(f"Macro Prediction Error (MSE): {prediction_error:.6f}")
```

## 降维的物理意义

### 1. 模态重要性

奇异值的大小反映了对应模态对系统动力学的贡献：
- **大奇异值**：主导模态，捕获系统的主要行为
- **小奇异值**：次要模态，对应细节或噪声

### 2. 粗粒化

粗粒化函数（左奇异向量）定义了如何将微观状态聚合为宏观变量：

```
Y_macro = X_micro @ U[:, :rank]
```

这是一个线性投影，保留了最重要的动力学信息。

### 3. 宏观动力学的简洁性

在宏观空间中，动力学变得：
- **低维**：从 n_modes 维降到 rank 维
- **近似线性**：K_macro = diag(S[:rank])
- **可解释**：每个宏观变量对应一个主导模态

### 4. 信息损失与增益

- **损失**：丢弃小奇异值对应的细节信息
- **增益**：
  - 降低计算复杂度
  - 去除噪声
  - 提取系统的本质特征
  - 更好的泛化能力

## 实际应用示例

### 案例 1: 多尺度系统

对于具有快慢变量的系统：
- 快变量通常对应小奇异值（快速衰减）
- 慢变量对应大奇异值（长期行为）
- 降维后保留慢变量，捕获长期动力学

### 案例 2: 高维时空系统

对于空间分布的系统（如气象数据）：
- 主导模态对应空间的大尺度模式
- 次要模态对应局部波动
- 降维后获得系统的"宏观天气模式"

## 常见问题

### Q1: 如何选择最优 rank？

**A**: 综合考虑：
1. 累积能量达到 90-95%
2. 奇异值谱出现"拐点"
3. 交叉验证预测误差
4. 物理可解释性

### Q2: 粗粒化系数的符号重要吗？

**A**: SVD 的符号是任意的（符号不确定性）。重要的是：
- 相对大小（绝对值）
- 空间/特征的分布模式

### Q3: 如何解释负的粗粒化系数？

**A**: 表示该微观特征对宏观变量的"反向贡献"或"竞争效应"。

### Q4: 降维后能恢复原始状态吗？

**A**: 一般不能完全恢复（有信息损失），但可以通过伪逆近似重构：
```python
X_reconstructed = Y_macro @ coarse_grain_coeffs.T
```

## 总结

通过 SVD 分析和降维，我们可以：

1. ✅ **定量评估**模态重要性（奇异值大小）
2. ✅ **提取主导模式**（前 r 个奇异向量）
3. ✅ **降维到宏观空间**（粗粒化）
4. ✅ **简化动力学**（低维线性系统）
5. ✅ **物理解释**（每个模态的意义）

这是从微观到宏观、从复杂到简洁的系统化分析过程。

## 参考文献

1. **Koopman, B. O. (1931)**. "Hamiltonian systems and transformation in Hilbert space"
2. **Williams et al. (2015)**. "A data-driven approximation of the Koopman operator"
3. **Jeong et al. (2025)**. "Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems", NeurIPS
4. **Brunton & Kutz (2019)**. "Data-Driven Science and Engineering"

---

**文档版本**: v1.0  
**最后更新**: 2026-02-10  
**作者**: LoRA Koopman Analysis Team
