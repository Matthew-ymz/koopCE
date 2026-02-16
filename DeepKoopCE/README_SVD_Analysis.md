# LoRA Koopman 算子 SVD 分析补充使用说明

## 📋 文件说明

本目录包含 LoRA Koopman 算子学习的完整 SVD 分析补充材料：

### 核心文件

1. **`demo_standalone_lora.ipynb`** 
   - 原始的 LoRA 训练和基础分析 notebook
   - 包含数据生成、模型训练、基础可视化

2. **`LoRA_SVD_Analysis_Supplement.py`**
   - SVD 分析补充代码（9个代码块）
   - 包含完整的奇异值分析、降维和宏观动力学可视化

3. **`LoRA_SVD_Analysis_Guide.md`**
   - 详细的理论指南和使用说明
   - 解释每个步骤的数学背景和物理意义

4. **`SVD_Singular_Values_Explanation.md`** ⭐ **新增 - 重要！**
   - **解释 model.svals 与 K_matrix SVD 奇异值的区别**
   - 为什么两个奇异值来源有显著差异
   - 如何正确选择使用哪个奇异值
   - 常见错误和最佳实践

5. **`standalone_lora.py`**
   - 独立的 LoRA 实现（不依赖原项目）

## 🚀 快速开始

### 方法 1: 在现有 Notebook 中添加补充分析（推荐）

1. 打开 `demo_standalone_lora.ipynb`

2. 运行所有单元格直到第 9 节结束（模型预测测试）

3. 在第 9 节之后，依次复制粘贴 `LoRA_SVD_Analysis_Supplement.py` 中的代码块

4. 按顺序执行每个代码块

### 方法 2: 创建新的分析 Notebook

```python
# 1. 首先运行完整的训练流程
%run demo_standalone_lora.ipynb

# 2. 然后执行补充分析
%run LoRA_SVD_Analysis_Supplement.py
```

## 📊 补充分析包含的内容

### 代码块 1: K 矩阵的 SVD 分解
- 对 Koopman 算子矩阵执行奇异值分解
- 输出 U, Σ, V 矩阵的形状和前 5 个奇异值

### 代码块 2: 奇异值谱可视化
- 柱状图：显示所有奇异值的大小
- 对数图：突出奇异值之间的差异
- **物理意义**: 奇异值的大小反映模态的重要性

### 代码块 3: 累积能量分析
- 计算累积能量贡献
- 标注 90%、95%、99% 能量阈值对应的 rank
- **作用**: 帮助选择合适的降维 rank

### 代码块 4: 粗粒化系数可视化
- 热力图显示左奇异向量（coarse-graining coefficients）
- 分析每个宏观变量的主要贡献特征
- **物理意义**: 定义如何将微观特征聚合为宏观变量

### 代码块 5: 计算宏观变量
- 将测试数据投影到低维宏观空间
- 输出降维前后的维度对比
- **结果**: Y_macro = X_features @ U[:, :rank]

### 代码块 6: 宏观动力学时间序列
- 可视化每个宏观变量随时间的演化
- 添加移动平均线突出趋势
- **物理意义**: 展示降维后系统的时间行为

### 代码块 7: 宏观相空间轨迹
- 2D 相空间图（如果 rank >= 2）
- 3D 相空间图（如果 rank >= 3）
- 使用颜色编码时间演化
- **物理意义**: 在低维空间中可视化系统的整体行为

### 代码块 8: 重构误差分析
- 计算不同 rank 的重构误差
- 边际效益分析：每增加一个模态的误差减少量
- **评估**: 量化降维的质量

### 代码块 9: 降维总结
- 维度压缩统计
- 模态重要性排序
- 物理意义解释
- 应用价值总结

## 🔬 典型分析流程

```
训练模型 (Sections 1-9)
    ↓
获取 Koopman 矩阵 K
    ↓
SVD 分解: K = U @ Σ @ V^T
    ↓
奇异值谱分析 → 选择 rank
    ↓
提取粗粒化系数: U[:, :rank]
    ↓
计算宏观变量: Y = X @ U[:, :rank]
    ↓
可视化宏观动力学
    ↓
评估降维质量
```

## 📈 预期输出

运行完整分析后，你将获得：

### 1. 定量结果
- 奇异值谱（数值和图表）
- 累积能量曲线
- 选定的 rank 和对应的能量保留率
- 重构误差统计

### 2. 可视化图表
- 奇异值柱状图和对数图
- 累积能量曲线和阈值标注
- 粗粒化系数热力图（带符号和绝对值）
- 宏观变量时间序列图
- 宏观相空间轨迹图（2D/3D）
- 重构误差 vs rank 曲线

### 3. 物理洞察
- 哪些模态最重要
- 如何从微观特征聚合到宏观变量
- 降维后系统的动力学行为
- 信息损失与计算效率的权衡

## 🎯 关键参数

### 可调参数

在代码块 4 中：

```python
RANK_THRESHOLD = 0.95  # 能量阈值（90%, 95%, 99%）
```

- **0.90**: 保留 90% 能量，更激进的降维
- **0.95**: 保留 95% 能量，平衡的选择（推荐）
- **0.99**: 保留 99% 能量，保守的降维

在代码块 6 中：

```python
n_steps_to_plot = 500  # 可视化的时间步数
```

调整此参数以显示更长或更短的时间窗口。

## 💡 使用技巧

### 1. 选择合适的 Rank

参考以下准则：

- **能量准则**: 保留 90-95% 的累积能量
- **视觉准则**: 奇异值谱出现明显"拐点"
- **误差准则**: 重构误差在可接受范围内
- **应用准则**: 根据具体问题的维度需求

### 2. 解读粗粒化系数

- **大的正系数**: 该微观特征对宏观变量有强正贡献
- **大的负系数**: 该微观特征有强反向贡献（竞争效应）
- **接近零**: 该微观特征对此宏观变量不重要

### 3. 分析宏观动力学

- **时间序列**: 观察是否有周期性、趋势或混沌行为
- **相空间**: 查看是否有吸引子、极限环或其他结构
- **与原始系统对比**: 验证宏观动力学是否捕获了系统的本质

### 4. 评估降维质量

- **重构误差 < 5%**: 优秀的降维
- **重构误差 5-10%**: 良好的降维
- **重构误差 > 10%**: 可能需要增加 rank

## 🔍 常见问题

### Q1: 运行补充代码时出现变量未定义错误

**A**: 确保先运行完整的 `demo_standalone_lora.ipynb` 直到第 9 节，需要以下变量：
- `model`: 训练好的 Koopman 模型
- `spectrum`: 包含 K 矩阵的字典
- `X_test_tensor`: 测试数据
- `device`: PyTorch 设备
- `N_MODES`: 模态数量

### Q2: 如何解释负的粗粒化系数？

**A**: 负系数表示该微观特征对宏观变量有"反向贡献"。例如：
- 在生态系统中，捕食者与猎物可能有负系数（竞争关系）
- 在物理系统中，可能表示相位相反的振荡

### Q3: rank = 1 时无法绘制相空间图

**A**: 这是正常的。相空间图需要至少 2 个维度。你仍然可以查看：
- 时间序列图
- 与原始数据的对比

### Q4: 奇异值谱没有明显间隙

**A**: 可能的原因：
- 系统本身就是高维的，没有明显的主导模态
- 需要更多的训练数据
- n_modes 设置得太大或太小
- 尝试调整模型架构或训练参数

### Q5: model.svals 与 K_matrix SVD 的奇异值为什么差异这么大？

**A**: 这是一个常见的问题！**这不是 bug，而是设计如此**。详见 `SVD_Singular_Values_Explanation.md`。

简要解释：
- **model.svals**: 学习的理想目标（参数化的奇异值）
- **K_matrix SVD**: 实际从数据计算的经验奇异值
- 两者应该相似但不完全相同
- **用于降维必须使用 K_matrix SVD**，不能用 model.svals

## 📚 参考文献

1. **Jeong et al. (2025)**. "Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems", NeurIPS
   
2. **Brunton & Kutz (2019)**. "Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control"
   
3. **Williams et al. (2015)**. "A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition"
   
4. **Koopman (1931)**. "Hamiltonian Systems and Transformation in Hilbert Space"

## 🤝 支持与反馈

如遇到问题或有改进建议，请：

1. 检查 `LoRA_SVD_Analysis_Guide.md` 中的详细说明
2. 确认所有依赖包已安装：`numpy`, `torch`, `matplotlib`, `seaborn`
3. 查看示例 notebooks: `toy_examp_analysis.ipynb`, `air_data.ipynb`

---

**版本**: v1.0  
**最后更新**: 2026-02-10  
**作者**: Koopman Analysis Team  
**许可**: MIT License
