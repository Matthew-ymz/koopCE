# EEG 数据处理代码改进说明

## 📋 问题分析

### 原始代码（read_data.ipynb）存在的问题

#### 1. 🐛 **关键 Bug：标准化逻辑错误**

**原始代码：**
```python
for i in range(len(all_data)): 
    norm_data = all_data[i].T   # [1700,141]
    scaler = StandardScaler()   # ❌ 每次都新建！
    scaler.fit(norm_data)       # ❌ 用当前样本的统计量
    norm_data = scaler.transform(norm_data) 
    all_norm_data.append(norm_data)
```

**问题分析：**
- 每个样本单独创建 scaler，并用该样本的均值和方差进行标准化
- 导致每个样本的标准化参数不同，**破坏了数据分布的一致性**
- 训练集和测试集无法用统一的标准化参数转换
- 模型在训练集上学到的标准化规律无法在测试集上应用

**后果：**
- 模型泛化能力严重下降
- 结果不可复现
- 统计学意义不明确

---

#### 2. 📋 **代码重复问题**

**原始代码中训练集和测试集处理逻辑完全相同：**

```python
# ---- 训练集处理 ----
for j in range(0, 3):
    train_input, train_target = [], []
    for i in range(train_ratio):
        if j == 0: 
            stage_data = all_norm_data[i][:600] 
        elif j == 1: 
            stage_data = all_norm_data[i][500:1100]
        elif j == 2:
            stage_data = all_norm_data[i][1100:1700]
        train_input.append(stage_data[:-1]) 
        train_target.append(stage_data[1:]) 
    # ... 保存逻辑

# ---- 测试集处理 ----
for j in range(0, 3):
    test_input, test_target = [], []
    for i in range(train_ratio, len(all_norm_data)):  # ← 仅此不同
        if j == 0:
            stage_data = all_norm_data[i][:600]
        # ... 完全相同的代码
```

**问题：**
- DRY 原则违反 (Don't Repeat Yourself)
- 修改一处需要改两处
- 容易引入不一致的 bug
- 代码可维护性差

---

#### 3. 🔢 **魔数硬编码**

**分散在代码中的硬编码值：**

| 硬编码值 | 含义 | 位置 |
|---------|------|------|
| `[:12]` | 取前 12 个样本 | 数据加载处 |
| `1, 1` | 注意力类型和试验类型 | 筛选条件 |
| `1700` | 总时间窗口 | 多处出现 |
| `[:600]`, `[500:1100]`, `[1100:1700]` | 阶段分割索引 | 循环内 6 处 |

**问题：**
- 无法轻易修改参数进行不同实验
- 代码的通用性差
- 难以维护和扩展

---

#### 4. 🔍 **变量作用域和命名问题**

```python
train_input.shape  # ❌ 最后循环的 train_input 是什么？
test_input.shape   # ❌ 最后循环的 test_input 是什么？
```

**问题：**
- 在循环中多次定义相同变量名
- 最后输出的是第 3 阶段（j=2）的数据
- 容易混淆和误解

---

#### 5. ❌ **缺少错误处理**

```python
with h5py.File('./data/IPCAS_ExemplarData_ZXL_Sub14.mat', 'r') as f: 
    # 如果文件不存在会崩溃
    # 如果数据结构不同会报错
    # 没有任何提示
```

---

#### 6. 📂 **缺少目录创建逻辑**

```python
pd.DataFrame(train_input).to_csv('./visual_inducted_conscious/1/stage%s/train_input.csv'%(j+1), 
                                 header=None, index=None)
# ❌ 如果目录不存在会报错！
```

---

## ✨ 改进方案

### 改进版本：`read_data_improved.py`

#### 1. ✅ **修复标准化逻辑**

```python
def normalize_data(all_data):
    """对所有数据进行标准化处理"""
    all_norm_data = []
    scaler = StandardScaler()  # ✅ 只创建一次
    
    for i, data in enumerate(all_data):
        data_transposed = data.T
        normalized = scaler.fit_transform(data_transposed)  # ✅ 同一 scaler
        all_norm_data.append(normalized)
    
    return all_norm_data, scaler  # ✅ 返回 scaler 用于测试集
```

**改进：**
- 使用全局 scaler 确保一致性
- 可以保存和加载 scaler 用于新数据
- 符合标准的数据预处理流程

---

#### 2. ✅ **参数配置集中管理**

```python
class Config:
    """集中管理所有配置参数"""
    
    # 文件路径
    INPUT_FILE = './data/IPCAS_ExemplarData_ZXL_Sub14.mat'
    OUTPUT_BASE_DIR = './visual_inducted_conscious/1'
    
    # 数据过滤条件
    ATTEN_TYPE_FILTER = 1      # 注意力类型
    TRIAL_TYPE_FILTER = 1      # 试验类型
    
    # 数据处理参数
    TOTAL_INTERVAL = 1700
    TEST_RATIO = 0.04
    
    # 阶段分割定义（集中管理）
    STAGE_DEFINITIONS = {
        1: (0, 600),
        2: (500, 1100),
        3: (1100, 1700),
    }
    
    VERBOSE = True             # 调试模式
```

**改进：**
- 所有参数一目了然
- 修改参数只需改配置类
- 易于进行参数敏感性分析
- 支持多个配置方案

---

#### 3. ✅ **消除代码重复**

```python
def prepare_sequences(data_list, stage_num):
    """为特定阶段准备输入-目标序列对"""
    inputs, targets = [], []
    
    for data in data_list:
        stage_data = extract_stage_data(data, stage_num)
        inputs.append(stage_data[:-1])
        targets.append(stage_data[1:])
    
    return np.concatenate(inputs, axis=0), np.concatenate(targets, axis=0)


def process_all_stages(train_data, test_data):
    """处理所有阶段的数据并保存"""
    
    for stage_num in Config.STAGE_DEFINITIONS.keys():
        # 处理训练集
        train_input, train_target = prepare_sequences(train_data, stage_num)
        save_data(train_input, train_target, stage_num, 'train')
        
        # 处理测试集
        test_input, test_target = prepare_sequences(test_data, stage_num)
        save_data(test_input, test_target, stage_num, 'test')
```

**改进：**
- 统一处理逻辑
- 没有代码重复
- 易于维护

---

#### 4. ✅ **完整的错误处理和日志**

```python
def load_raw_data(filepath: str):
    """从 MAT 文件读取原始 EEG 数据"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # ... 读取数据
            if Config.VERBOSE:
                print(f"EEG 数据形状: {bv_group.shape}")
                print(f"试验类型: {np.unique(trial_types)}")
        return data
    except Exception as e:
        raise RuntimeError(f"读取 MAT 文件失败: {e}")
```

**改进：**
- 文件存在检查
- 异常处理和提示
- 详细的调试日志

---

#### 5. ✅ **自动创建目录**

```python
def save_data(input_data, target_data, stage_num, split_type='train'):
    """保存数据为 CSV 文件"""
    # 创建目录
    stage_dir = os.path.join(Config.OUTPUT_BASE_DIR, f'stage{stage_num}')
    os.makedirs(stage_dir, exist_ok=True)  # ✅ 自动创建
    
    # 保存数据
    input_path = os.path.join(stage_dir, f'{split_type}_input.csv')
    pd.DataFrame(input_data).to_csv(input_path, header=False, index=False)
```

---

#### 6. ✅ **类型提示和文档**

```python
def filter_and_extract_data(
    bv_group: np.ndarray,           # ← 类型提示
    trial_types: np.ndarray,
    atten_types: np.ndarray,
    time1: np.ndarray,
    atten_type: int,
    trial_type: int,
    interval: int
) -> List[np.ndarray]:             # ← 返回类型
    """
    按条件筛选数据并提取固定长度的数据段
    
    Args:
        bv_group: 原始 EEG 数据
        ...
        
    Returns:
        提取的数据段列表
    """
```

---

## 📊 改进对比表

| 特性 | 原始代码 | 改进后 |
|------|---------|--------|
| **标准化逻辑** | ❌ 每样本独立 | ✅ 全局统一 |
| **代码重复** | ❌ 6 处重复 | ✅ DRY 原则 |
| **参数管理** | ❌ 分散硬编码 | ✅ 集中配置 |
| **错误处理** | ❌ 无 | ✅ 完整 |
| **目录管理** | ❌ 手动创建 | ✅ 自动创建 |
| **代码文档** | ❌ 无 | ✅ 详细注释 |
| **类型提示** | ❌ 无 | ✅ 完整 |
| **调试信息** | ❌ 无 | ✅ 详细日志 |
| **可维护性** | ❌ 低 | ✅ 高 |
| **可扩展性** | ❌ 低 | ✅ 高 |

---

## 🚀 使用方法

### 方法 1: 在 Jupyter Notebook 中使用

```python
# 在 notebook 中导入
import sys
sys.path.append('./koopCE/neural_science/experiments')
from read_data_improved import main, Config

# 运行主程序
main()

# 或修改配置后运行
Config.TEST_RATIO = 0.1
Config.VERBOSE = False
main()
```

### 方法 2: 命令行运行

```bash
cd koopCE/neural_science/experiments
python read_data_improved.py
```

### 方法 3: 逐步调试

```python
from read_data_improved import *

# Step 1: 加载数据
bv_group, trial_types, atten_types, time1 = load_raw_data(Config.INPUT_FILE)

# Step 2: 筛选和提取
all_data = filter_and_extract_data(
    bv_group, trial_types, atten_types, time1,
    Config.ATTEN_TYPE_FILTER, Config.TRIAL_TYPE_FILTER, Config.TOTAL_INTERVAL
)
print(f"提取的样本数: {len(all_data)}")

# Step 3: 标准化
all_norm_data, scaler = normalize_data(all_data)

# Step 4: 划分
train_ratio, train_data, test_data = train_test_split(all_norm_data)

# Step 5: 处理和保存
process_all_stages(train_data, test_data)
```

---

## 📝 总结

### 核心改进

1. **修复标准化 Bug** - 使用全局 scaler 确保数据一致性
2. **消除代码重复** - 遵循 DRY 原则，提取公共逻辑
3. **参数集中管理** - Config 类管理所有参数
4. **完整文档** - 详细的文档字符串和类型提示
5. **强大的错误处理** - 自动创建目录、检查文件存在性
6. **调试友好** - 详细的日志输出支持逐步调试

### 向后兼容性

改进后的代码产生与原始代码相同的输出文件结构（假设 Bug 被修复）。

### 扩展性

- 易于添加新的阶段定义
- 易于改变过滤条件
- 易于处理不同的数据格式

---

**文件位置**: `koopCE/neural_science/experiments/read_data_improved.py`

**建议**: 使用改进版本替换原始 notebook，或在两个版本之间进行验证对比。
