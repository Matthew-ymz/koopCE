"""
LoRA Koopman 算子 SVD 分析补充代码

这些代码块应该插入到 demo_standalone_lora.ipynb 的第 9 节（预测测试）之后

包含内容：
1. 完整的 K 矩阵 SVD 分解
2. 奇异值谱分析和可视化
3. 累积能量分析和 rank 选择
4. 粗粒化系数（左奇异向量）可视化
5. 宏观变量计算和降维
6. 宏观动力学轨迹可视化
7. 物理意义解释

使用方法：
将下面的代码块复制到 Jupyter Notebook 中，按顺序执行
"""

# =============================================================================
# 第 10 节：完整的 SVD 分析与降维
# =============================================================================

"""
## 10. 完整的 SVD 分析与降维

在这一节中，我们对学习到的 Koopman 算子进行完整的奇异值分解（SVD）分析，
并通过降维展示系统的宏观动力学行为。

### 理论背景

对 Koopman 算子矩阵 K 进行 SVD 分解：

$$K = U \\Sigma V^T$$

其中：
- **U**: 左奇异向量矩阵（粗粒化函数）
- **Σ**: 奇异值对角矩阵（模态重要性）  
- **V**: 右奇异向量矩阵

通过截断 SVD，我们可以：
1. 选择前 r 个最重要的模态
2. 将高维特征空间映射到低维宏观空间
3. 在低维空间中捕获系统的本质动力学
"""

# =============================================================================
# 代码块 1: K 矩阵的 SVD 分解
# =============================================================================

import seaborn as sns

print("=" * 70)
print("SVD 分析：Koopman 算子矩阵分解")
print("=" * 70)

# 获取 Koopman 矩阵
K_matrix = spectrum['K_matrix']
print(f"\nKoopman 矩阵形状: {K_matrix.shape}")

# 执行 SVD
U, S, Vt = np.linalg.svd(K_matrix)
V = Vt.T

print(f"\n左奇异向量矩阵 U: {U.shape}")
print(f"奇异值向量 S: {S.shape}")
print(f"右奇异向量矩阵 V: {V.shape}")

print(f"\n前 5 个奇异值:")
for i, sv in enumerate(S[:5]):
    print(f"  σ_{i} = {sv:.6f}")

# =============================================================================
# 代码块 2: 奇异值谱可视化
# =============================================================================

"""
### 10.1 奇异值谱分析

奇异值的大小反映了对应模态对系统动力学的贡献。
"""

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：奇异值柱状图
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, len(S)))
bars = ax.bar(range(len(S)), S, color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Singular Value Index', fontsize=14, fontweight='bold')
ax.set_ylabel('Singular Value (σ)', fontsize=14, fontweight='bold')
ax.set_title('Singular Value Spectrum of Koopman Operator', 
             fontsize=16, fontweight='bold')
ax.grid(True, axis='y', alpha=0.4, linestyle='--')
ax.set_xticks(range(len(S)))

# 添加数值标注
for i, (bar, val) in enumerate(zip(bars, S)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 右图：奇异值对数图（突出差异）
ax = axes[1]
ax.semilogy(range(len(S)), S, 'o-', linewidth=2.5, 
            markersize=10, color='darkblue', markerfacecolor='orange')
ax.set_xlabel('Singular Value Index', fontsize=14, fontweight='bold')
ax.set_ylabel('Singular Value (σ) [log scale]', fontsize=14, fontweight='bold')
ax.set_title('Singular Value Spectrum (Log Scale)', 
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.4, linestyle='--')
ax.set_xticks(range(len(S)))

plt.tight_layout()
plt.show()

# =============================================================================
# 代码块 3: 累积能量分析与 Rank 选择
# =============================================================================

"""
### 10.2 累积能量与 Rank 选择

通过分析累积能量，我们可以确定需要保留多少个模态。
"""

# 计算累积能量
cumulative_energy = np.cumsum(S) / np.sum(S)

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：累积能量曲线
ax = axes[0]
ax.plot(range(1, len(S)+1), cumulative_energy, 
        'o-', linewidth=3, markersize=10, 
        color='darkgreen', markerfacecolor='lightgreen')

# 添加阈值线
for threshold, color, label in [(0.90, 'orange', '90%'), 
                                  (0.95, 'red', '95%'),
                                  (0.99, 'purple', '99%')]:
    ax.axhline(y=threshold, color=color, linestyle='--', 
               linewidth=2, alpha=0.7, label=f'{label} threshold')
    # 标注对应的 rank
    rank_at_threshold = np.argmax(cumulative_energy >= threshold) + 1
    ax.plot(rank_at_threshold, threshold, 'o', 
            markersize=15, color=color, markeredgecolor='black', 
            markeredgewidth=2, zorder=5)
    ax.text(rank_at_threshold, threshold - 0.03, 
            f'rank={rank_at_threshold}',
            ha='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Number of Modes (rank)', fontsize=14, fontweight='bold')
ax.set_ylabel('Cumulative Energy', fontsize=14, fontweight='bold')
ax.set_title('Cumulative Energy vs. Number of Modes', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.4)
ax.set_ylim([0, 1.05])

# 右图：每个模态的能量贡献
ax = axes[1]
energy_contribution = S / np.sum(S)
bars = ax.bar(range(len(S)), energy_contribution, 
              color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Mode Index', fontsize=14, fontweight='bold')
ax.set_ylabel('Energy Contribution', fontsize=14, fontweight='bold')
ax.set_title('Individual Mode Energy Contribution', 
             fontsize=16, fontweight='bold')
ax.grid(True, axis='y', alpha=0.4, linestyle='--')
ax.set_xticks(range(len(S)))

# 添加百分比标注
for i, (bar, val) in enumerate(zip(bars, energy_contribution)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{val*100:.1f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# 打印统计信息
print("\n" + "=" * 70)
print("累积能量统计")
print("=" * 70)
for threshold in [0.90, 0.95, 0.99]:
    rank = np.argmax(cumulative_energy >= threshold) + 1
    energy = cumulative_energy[rank-1]
    print(f"保留 {threshold*100:.0f}% 能量需要 {rank} 个模态 "
          f"(实际累积能量: {energy*100:.2f}%)")

# =============================================================================
# 代码块 4: 选择 Rank 并提取粗粒化系数
# =============================================================================

"""
### 10.3 粗粒化系数（左奇异向量）

左奇异向量 U 定义了如何将微观特征聚合为宏观变量。
"""

# 选择 rank（这里选择保留 95% 能量）
RANK_THRESHOLD = 0.95
rank = np.argmax(cumulative_energy >= RANK_THRESHOLD) + 1

print(f"\n选择的 rank: {rank}")
print(f"保留的能量: {cumulative_energy[rank-1]*100:.2f}%")

# 提取粗粒化系数
coarse_grain_coeffs = U[:, :rank]
print(f"\n粗粒化系数矩阵形状: {coarse_grain_coeffs.shape}")
print(f"含义: (n_modes={N_MODES}, rank={rank})")

# 可视化粗粒化系数
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 左图：原始系数（带符号）
ax = axes[0]
sns.heatmap(coarse_grain_coeffs, 
            cmap='RdBu_r', 
            center=0,
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Coefficient Value'},
            xticklabels=[f'Y_{i}' for i in range(rank)],
            yticklabels=[f'φ_{i}' for i in range(N_MODES)],
            ax=ax,
            linewidths=0.5,
            linecolor='gray')
ax.set_title('Coarse-Graining Coefficients (With Sign)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Macro Variables', fontsize=12, fontweight='bold')
ax.set_ylabel('Micro Features (Koopman Modes)', fontsize=12, fontweight='bold')

# 右图：绝对值（突出重要性）
ax = axes[1]
sns.heatmap(np.abs(coarse_grain_coeffs), 
            cmap='YlOrRd', 
            annot=True,
            fmt='.3f',
            cbar_kws={'label': '|Coefficient|'},
            xticklabels=[f'Y_{i}' for i in range(rank)],
            yticklabels=[f'φ_{i}' for i in range(N_MODES)],
            ax=ax,
            linewidths=0.5,
            linecolor='gray')
ax.set_title('Coarse-Graining Coefficients (Absolute Value)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Macro Variables', fontsize=12, fontweight='bold')
ax.set_ylabel('Micro Features (Koopman Modes)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# 分析每个宏观变量的主要贡献
print("\n" + "=" * 70)
print("每个宏观变量的主要贡献特征")
print("=" * 70)
for i in range(rank):
    coeffs = np.abs(coarse_grain_coeffs[:, i])
    top_indices = np.argsort(coeffs)[::-1][:3]  # 前3个最大贡献
    print(f"\n宏观变量 Y_{i} (奇异值 σ={S[i]:.4f}):")
    for idx in top_indices:
        print(f"  φ_{idx}: {coarse_grain_coeffs[idx, i]:+.4f} "
              f"(|{coeffs[idx]:.4f}|)")

# =============================================================================
# 代码块 5: 计算宏观变量
# =============================================================================

"""
### 10.4 从微观到宏观：降维映射

通过粗粒化，我们将高维特征空间映射到低维宏观空间。
"""

# 编码测试数据到特征空间
print("\n计算宏观变量...")
model.eval()
with torch.no_grad():
    X_test_features = model(X_test_tensor.to(device), lagged=False)
    X_test_features = X_test_features.cpu().numpy()

print(f"微观特征形状: {X_test_features.shape}")

# 投影到宏观空间
Y_macro = X_test_features @ coarse_grain_coeffs
print(f"宏观变量形状: {Y_macro.shape}")
print(f"降维: {N_MODES}D → {rank}D (压缩率: {rank/N_MODES*100:.1f}%)")

# 统计信息
print("\n宏观变量统计信息:")
for i in range(rank):
    print(f"Y_{i}: mean={Y_macro[:, i].mean():.4f}, "
          f"std={Y_macro[:, i].std():.4f}, "
          f"range=[{Y_macro[:, i].min():.4f}, {Y_macro[:, i].max():.4f}]")

# =============================================================================
# 代码块 6: 宏观动力学可视化 - 时间序列
# =============================================================================

"""
### 10.5 宏观动力学的时间演化

观察降维后的宏观变量如何随时间演化。
"""

n_steps_to_plot = 500

fig, axes = plt.subplots(rank, 1, figsize=(14, 4*rank), sharex=True)
if rank == 1:
    axes = [axes]

for i in range(rank):
    ax = axes[i]
    
    # 绘制时间序列
    time_steps = np.arange(n_steps_to_plot)
    ax.plot(time_steps, Y_macro[:n_steps_to_plot, i], 
            linewidth=2, color='steelblue', alpha=0.8)
    
    # 添加移动平均线（突出趋势）
    window = 20
    if n_steps_to_plot > window:
        moving_avg = np.convolve(Y_macro[:n_steps_to_plot, i], 
                                  np.ones(window)/window, mode='valid')
        ax.plot(time_steps[window-1:], moving_avg, 
                'r--', linewidth=2.5, alpha=0.7, label='Moving Average')
    
    ax.set_ylabel(f'$Y_{i}$', fontsize=14, fontweight='bold')
    ax.set_title(f'Macro Variable {i} | σ={S[i]:.4f} | '
                f'Energy Contribution: {energy_contribution[i]*100:.1f}%', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    # 标注统计信息
    ax.text(0.02, 0.95, 
            f'Mean: {Y_macro[:n_steps_to_plot, i].mean():.3f}\n'
            f'Std: {Y_macro[:n_steps_to_plot, i].std():.3f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

axes[-1].set_xlabel('Time Step', fontsize=14, fontweight='bold')
plt.suptitle(f'Macroscopic Dynamics (Reduced from {N_MODES}D to {rank}D)', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# =============================================================================
# 代码块 7: 宏观动力学可视化 - 相空间轨迹
# =============================================================================

"""
### 10.6 宏观相空间轨迹

如果 rank >= 2，我们可以在宏观相空间中可视化系统的演化轨迹。
"""

if rank >= 2:
    fig = plt.figure(figsize=(14, 6))
    
    # 2D 相空间图
    ax1 = fig.add_subplot(121)
    
    # 绘制轨迹（使用颜色编码时间）
    n_traj = min(n_steps_to_plot, len(Y_macro))
    colors_time = plt.cm.viridis(np.linspace(0, 1, n_traj))
    
    for i in range(n_traj - 1):
        ax1.plot(Y_macro[i:i+2, 0], Y_macro[i:i+2, 1], 
                color=colors_time[i], linewidth=2, alpha=0.7)
    
    # 标记起点和终点
    ax1.scatter(Y_macro[0, 0], Y_macro[0, 1], 
               c='red', s=200, marker='*', 
               edgecolors='black', linewidths=2,
               label='Start', zorder=5)
    ax1.scatter(Y_macro[n_traj-1, 0], Y_macro[n_traj-1, 1], 
               c='blue', s=200, marker='s', 
               edgecolors='black', linewidths=2,
               label='End', zorder=5)
    
    ax1.set_xlabel(f'$Y_0$ (σ={S[0]:.4f})', fontsize=14, fontweight='bold')
    ax1.set_ylabel(f'$Y_1$ (σ={S[1]:.4f})', fontsize=14, fontweight='bold')
    ax1.set_title('Macroscopic Phase Space Trajectory\n(Color: Time Evolution)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=0, vmax=n_traj))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Time Step', fontsize=12, fontweight='bold')
    
    # 如果有第3个模态，绘制另一个相平面
    if rank >= 3:
        ax2 = fig.add_subplot(122)
        
        for i in range(n_traj - 1):
            ax2.plot(Y_macro[i:i+2, 0], Y_macro[i:i+2, 2], 
                    color=colors_time[i], linewidth=2, alpha=0.7)
        
        ax2.scatter(Y_macro[0, 0], Y_macro[0, 2], 
                   c='red', s=200, marker='*', 
                   edgecolors='black', linewidths=2, zorder=5)
        ax2.scatter(Y_macro[n_traj-1, 0], Y_macro[n_traj-1, 2], 
                   c='blue', s=200, marker='s', 
                   edgecolors='black', linewidths=2, zorder=5)
        
        ax2.set_xlabel(f'$Y_0$ (σ={S[0]:.4f})', fontsize=14, fontweight='bold')
        ax2.set_ylabel(f'$Y_2$ (σ={S[2]:.4f})', fontsize=14, fontweight='bold')
        ax2.set_title('Alternative Phase Space Projection', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        sm2 = plt.cm.ScalarMappable(cmap='viridis', 
                                    norm=plt.Normalize(vmin=0, vmax=n_traj))
        sm2.set_array([])
        cbar2 = plt.colorbar(sm2, ax=ax2)
        cbar2.set_label('Time Step', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 如果 rank >= 3，绘制 3D 相空间
    if rank >= 3:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制 3D 轨迹
        for i in range(n_traj - 1):
            ax.plot(Y_macro[i:i+2, 0], Y_macro[i:i+2, 1], Y_macro[i:i+2, 2],
                   color=colors_time[i], linewidth=2, alpha=0.7)
        
        # 标记起点和终点
        ax.scatter(Y_macro[0, 0], Y_macro[0, 1], Y_macro[0, 2],
                  c='red', s=300, marker='*', 
                  edgecolors='black', linewidths=2, label='Start')
        ax.scatter(Y_macro[n_traj-1, 0], Y_macro[n_traj-1, 1], Y_macro[n_traj-1, 2],
                  c='blue', s=300, marker='s', 
                  edgecolors='black', linewidths=2, label='End')
        
        ax.set_xlabel(f'$Y_0$ (σ={S[0]:.4f})', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'$Y_1$ (σ={S[1]:.4f})', fontsize=12, fontweight='bold')
        ax.set_zlabel(f'$Y_2$ (σ={S[2]:.4f})', fontsize=12, fontweight='bold')
        ax.set_title('3D Macroscopic Phase Space', 
                     fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        plt.show()

else:
    print("\n注意: rank < 2, 无法绘制相空间轨迹图")

# =============================================================================
# 代码块 8: 重构误差分析
# =============================================================================

"""
### 10.7 重构误差与降维质量评估

评估使用前 r 个模态重构 Koopman 算子的误差。
"""

print("\n" + "=" * 70)
print("重构误差分析")
print("=" * 70)

# 低秩重构
K_reconstructed = U[:, :rank] @ np.diag(S[:rank]) @ V[:, :rank].T

# 计算各种误差度量
frobenius_error = np.linalg.norm(K_matrix - K_reconstructed, 'fro')
frobenius_relative = frobenius_error / np.linalg.norm(K_matrix, 'fro')

spectral_error = np.linalg.norm(K_matrix - K_reconstructed, 2)
spectral_relative = spectral_error / np.linalg.norm(K_matrix, 2)

print(f"\n使用前 {rank} 个模态重构:")
print(f"  Frobenius 误差: {frobenius_error:.6f}")
print(f"  相对 Frobenius 误差: {frobenius_relative*100:.2f}%")
print(f"  谱误差 (最大奇异值误差): {spectral_error:.6f}")
print(f"  相对谱误差: {spectral_relative*100:.2f}%")

# 绘制重构误差 vs rank
ranks_to_test = range(1, len(S)+1)
errors_fro = []
errors_spectral = []

for r in ranks_to_test:
    K_recon = U[:, :r] @ np.diag(S[:r]) @ V[:, :r].T
    errors_fro.append(np.linalg.norm(K_matrix - K_recon, 'fro') / 
                      np.linalg.norm(K_matrix, 'fro'))
    errors_spectral.append(np.linalg.norm(K_matrix - K_recon, 2) / 
                           np.linalg.norm(K_matrix, 2))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：相对误差 vs rank
ax = axes[0]
ax.plot(ranks_to_test, errors_fro, 'o-', linewidth=2.5, 
        markersize=10, label='Frobenius Norm', color='darkblue')
ax.plot(ranks_to_test, errors_spectral, 's--', linewidth=2.5, 
        markersize=10, label='Spectral Norm', color='darkred')
ax.axvline(x=rank, color='green', linestyle=':', linewidth=2.5, 
           label=f'Selected rank={rank}', alpha=0.7)
ax.set_xlabel('Rank', fontsize=14, fontweight='bold')
ax.set_ylabel('Relative Reconstruction Error', fontsize=14, fontweight='bold')
ax.set_title('Reconstruction Error vs. Rank', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.4)
ax.set_yscale('log')

# 右图：误差减少的边际效益
ax = axes[1]
error_reduction = np.diff([1.0] + errors_fro)  # 每增加一个模态的误差减少量
ax.bar(ranks_to_test, -error_reduction, color='steelblue', 
       edgecolor='black', linewidth=1.5, alpha=0.7)
ax.axvline(x=rank, color='green', linestyle=':', linewidth=2.5, 
           label=f'Selected rank={rank}', alpha=0.7)
ax.set_xlabel('Mode Added', fontsize=14, fontweight='bold')
ax.set_ylabel('Error Reduction (Marginal Benefit)', fontsize=14, fontweight='bold')
ax.set_title('Marginal Benefit of Each Mode', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, axis='y', alpha=0.4)

plt.tight_layout()
plt.show()

# =============================================================================
# 代码块 9: 降维的物理意义总结
# =============================================================================

"""
### 10.8 降维的物理意义与总结

通过 SVD 分析和降维，我们实现了：
"""

print("\n" + "=" * 70)
print("降维分析总结")
print("=" * 70)

print(f"\n1. 维度压缩:")
print(f"   原始维度: {N_MODES} (Koopman modes)")
print(f"   降维后: {rank} (macro variables)")
print(f"   压缩率: {rank/N_MODES*100:.1f}%")
print(f"   信息保留: {cumulative_energy[rank-1]*100:.2f}%")

print(f"\n2. 模态重要性:")
for i in range(min(rank, 3)):
    print(f"   模态 {i}: σ={S[i]:.4f}, "
          f"能量贡献={energy_contribution[i]*100:.1f}%")

print(f"\n3. 重构质量:")
print(f"   相对 Frobenius 误差: {frobenius_relative*100:.2f}%")
print(f"   说明: 使用 {rank} 个模态可以以 {frobenius_relative*100:.2f}% 的误差重构完整动力学")

print(f"\n4. 物理意义:")
print(f"   - 大奇异值模态: 捕获系统的主导行为（慢变量）")
print(f"   - 小奇异值模态: 对应快速衰减或噪声（快变量）")
print(f"   - 粗粒化: 将微观 Koopman 特征聚合为宏观变量")
print(f"   - 宏观动力学: 在低维空间中近似线性演化")

print(f"\n5. 应用价值:")
print(f"   - 计算效率: 降维后的系统更容易计算和预测")
print(f"   - 可解释性: 宏观变量通常有清晰的物理意义")
print(f"   - 噪声抑制: 小奇异值对应的噪声被过滤")
print(f"   - 长期预测: 主导模态控制长期行为")

print("\n" + "=" * 70)
print("SVD 分析与降维完成！")
print("=" * 70)
