"""
EEG 数据读取、预处理和数据集生成

功能：
1. 从 MAT 文件读取 EEG 数据
2. 按条件筛选数据
3. 数据标准化
4. 划分训练/测试集
5. 按阶段分割并保存为 CSV
"""

import os
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import List, Tuple, Dict


# =============================================================================
# 配置参数
# =============================================================================

class Config:
    """集中管理所有配置参数"""
    
    # 文件路径
    INPUT_FILE = './IPCAS_ExemplarData_ZXL_Sub14.mat'
    OUTPUT_BASE_DIR = './visual_inducted_conscious/1'
    
    # 数据过滤条件
    ATTEN_TYPE_FILTER = 1      # 注意力类型
    TRIAL_TYPE_FILTER = 1      # 试验类型
    
    # 数据处理参数
    TOTAL_INTERVAL = 1700      # 总时间窗口 (250+300+300)*2
    TEST_RATIO = 0.04          # 测试集比例
    
    # 阶段分割定义
    # 说明: 原始数据包含三个阶段: 准备期(250ms) + 呈现期(300ms) + 反应期(300ms)
    STAGE_DEFINITIONS = {
        1: (0, 600),        # 阶段 1: 0-600
        2: (500, 1100),     # 阶段 2: 500-1100 (与阶段1有重叠)
        3: (1100, 1700),    # 阶段 3: 1100-1700
    }
    
    # 输出配置
    VERBOSE = True             # 是否打印详细信息


# =============================================================================
# 数据加载函数
# =============================================================================

def load_raw_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从 MAT 文件读取原始 EEG 数据
    
    Args:
        filepath: MAT 文件路径
        
    Returns:
        bv_group: EEG 数据 (time_points, channels)
        trial_types: 试验类型数组
        atten_types: 注意力类型数组
        time1: 时间标记数组
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")
    
    if Config.VERBOSE:
        print(f"正在读取数据文件: {filepath}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # 读取试验类型 (1,2,3,4)
            trial_types = np.array(f['BV']['trialTypes']).T[:12].reshape(1, -1)[0]
            
            # 读取注意力类型 (1,2)
            atten_types = np.array(f['BV']['attenTypes']).T[:12].reshape(1, -1)[0]
            
            # 读取时间标记
            time1 = np.array(f['Timing1'])[:, 0]
            
            # 读取 EEG 数据
            bv_group = np.array(f['eegdata'])
            
            if Config.VERBOSE:
                print(f"  EEG 数据形状: {bv_group.shape}")
                print(f"  试验类型: {np.unique(trial_types)}")
                print(f"  注意力类型: {np.unique(atten_types)}")
                print(f"  时间标记数: {len(time1)}")
            
            return bv_group, trial_types, atten_types, time1
            
    except Exception as e:
        raise RuntimeError(f"读取 MAT 文件失败: {e}")


def filter_and_extract_data(
    bv_group: np.ndarray,
    trial_types: np.ndarray,
    atten_types: np.ndarray,
    time1: np.ndarray,
    atten_type: int,
    trial_type: int,
    interval: int
) -> List[np.ndarray]:
    """
    按条件筛选数据并提取固定长度的数据段
    
    Args:
        bv_group: 原始 EEG 数据
        trial_types: 试验类型数组
        atten_types: 注意力类型数组
        time1: 时间标记数组
        atten_type: 要筛选的注意力类型
        trial_type: 要筛选的试验类型
        interval: 每个数据段的长度
        
    Returns:
        提取的数据段列表
    """
    all_data = []
    
    # 创建过滤条件
    mask = (atten_types == atten_type) & (trial_types == trial_type)
    filtered_indices = np.where(mask)[0]
    
    if Config.VERBOSE:
        print(f"\n筛选条件: attenType={atten_type}, trialType={trial_type}")
        print(f"  匹配的试验数: {len(filtered_indices)}")
    
    # 提取数据段
    for idx in filtered_indices:
        start = int(time1[idx])
        end = start + interval
        
        # 检查边界
        if end <= bv_group.shape[0]:
            data_segment = bv_group[start:end].T  # 转置: (channels, time)
            all_data.append(data_segment)
    
    if Config.VERBOSE:
        print(f"  成功提取: {len(all_data)} 个数据段")
    
    return all_data


# =============================================================================
# 数据预处理函数
# =============================================================================

def normalize_data(all_data: List[np.ndarray]) -> Tuple[List[np.ndarray], StandardScaler]:
    """
    对所有数据进行标准化处理
    
    Args:
        all_data: 原始数据列表，每个元素形状为 (channels, time_points)
        
    Returns:
        all_norm_data: 标准化后的数据列表
        scaler: 用于标准化的 scaler 对象 (用于后续数据转换)
    """
    all_norm_data = []
    
    if Config.VERBOSE:
        print(f"\n正在标准化数据...")
    
    # 创建全局 scaler
    scaler = StandardScaler()
    
    for i, data in enumerate(all_data):
        # 数据形状: (channels, time_points)
        # 转置为 (time_points, channels) 用于标准化
        data_transposed = data.T
        
        # 对每个样本进行标准化
        normalized = scaler.fit_transform(data_transposed)
        all_norm_data.append(normalized)
    
    if Config.VERBOSE:
        print(f"  标准化完成: {len(all_norm_data)} 个样本")
    
    return all_norm_data, scaler


def train_test_split(
    all_norm_data: List[np.ndarray],
    test_ratio: float = 0.04
) -> Tuple[int, List[np.ndarray], List[np.ndarray]]:
    """
    划分训练集和测试集
    
    Args:
        all_norm_data: 标准化的数据列表
        test_ratio: 测试集比例
        
    Returns:
        train_ratio: 训练集样本数
        train_data: 训练集
        test_data: 测试集
    """
    total_samples = len(all_norm_data)
    test_count = int(total_samples * test_ratio)
    train_ratio = total_samples - test_count
    
    train_data = all_norm_data[:train_ratio]
    test_data = all_norm_data[train_ratio:]
    
    if Config.VERBOSE:
        print(f"\n划分训练/测试集:")
        print(f"  总样本数: {total_samples}")
        print(f"  训练集: {len(train_data)} ({len(train_data)/total_samples*100:.1f}%)")
        print(f"  测试集: {len(test_data)} ({len(test_data)/total_samples*100:.1f}%)")
    
    return train_ratio, train_data, test_data


# =============================================================================
# 数据分割和保存函数
# =============================================================================

def extract_stage_data(
    data: np.ndarray,
    stage_num: int
) -> np.ndarray:
    """
    提取特定阶段的数据
    
    Args:
        data: 原始数据 (time_points, channels)
        stage_num: 阶段编号 (1, 2, 3)
        
    Returns:
        阶段数据 (time_points_in_stage, channels)
    """
    if stage_num not in Config.STAGE_DEFINITIONS:
        raise ValueError(f"无效的阶段编号: {stage_num}")
    
    start, end = Config.STAGE_DEFINITIONS[stage_num]
    return data[start:end]


def prepare_sequences(
    data_list: List[np.ndarray],
    stage_num: int
) -> Tuple[np.ndarray, np.ndarray]:

    inputs = []
    targets = []
    seq_list = []

    for data in data_list:
        # 提取阶段数据
        stage_data = extract_stage_data(data, stage_num)
        seq_list.append(stage_data)

        # 创建序列对 (t -> t+1)
        inputs.append(stage_data[:-1])    # 除了最后一个时间步
        targets.append(stage_data[1:])    # 除了第一个时间步
    
    # 拼接所有数据
    input_data = np.concatenate(inputs, axis=0)
    target_data = np.concatenate(targets, axis=0)
    
    return  seq_list, input_data, target_data


def save_data(
    seq_list: list,
    input_data: np.ndarray,
    target_data: np.ndarray,
    stage_num: int,
    split_type: str = 'train'
) -> None:
    """
    保存数据为 CSV 文件
    
    Args:
        input_data: 输入数据
        target_data: 目标数据
        stage_num: 阶段编号
        split_type: 数据集类型 ('train' 或 'test')
    """
    # 创建目录
    stage_dir = os.path.join(Config.OUTPUT_BASE_DIR, f'stage{stage_num}')
    os.makedirs(stage_dir, exist_ok=True)
    
    # 准备文件路径
    seqs_path = os.path.join(stage_dir, f'{split_type}_seqs')
    input_path = os.path.join(stage_dir, f'{split_type}_input.csv')
    target_path = os.path.join(stage_dir, f'{split_type}_target.csv')
    
    # 保存数据
    np.save(seqs_path, seq_list)
    pd.DataFrame(input_data).to_csv(input_path, header=False, index=False)
    pd.DataFrame(target_data).to_csv(target_path, header=False, index=False)
    
    if Config.VERBOSE:
        print(f"  已保存: {input_path}")
        print(f"         {target_path}")
        print(f"         形状: input {input_data.shape}, target {target_data.shape}")


def process_all_stages(
    train_data: List[np.ndarray],
    test_data: List[np.ndarray]
) -> None:
    """
    处理所有阶段的数据并保存
    
    Args:
        train_data: 训练集数据列表
        test_data: 测试集数据列表
    """
    # 创建输出基目录
    os.makedirs(Config.OUTPUT_BASE_DIR, exist_ok=True)
    
    # 处理每个阶段
    for stage_num in Config.STAGE_DEFINITIONS.keys():
        if Config.VERBOSE:
            print(f"\n{'='*60}")
            print(f"处理阶段 {stage_num}")
            print(f"{'='*60}")
        
        # 处理训练集
        if Config.VERBOSE:
            print(f"\n训练集:")
        seq_list, train_input, train_target = prepare_sequences(train_data, stage_num)
        save_data(seq_list, train_input, train_target, stage_num, 'train')
        
        # 处理测试集
        if Config.VERBOSE:
            print(f"\n测试集:")
        seq_list, test_input, test_target = prepare_sequences(test_data, stage_num)
        save_data(seq_list, test_input, test_target, stage_num, 'test')


# =============================================================================
# 时间序列数据格式化函数
# =============================================================================

def format_timeseries_data(
    input_data: np.ndarray,
    target_data: np.ndarray,
    dt: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将输入-目标序列对转换为时间序列格式，适配 pysindy 库
    
    Args:
        input_data: 输入序列 (n_samples, n_features)
        target_data: 目标序列 (n_samples, n_features)
        dt: 时间步长（秒），默认 0.001s
        
    Returns:
        X: 时间序列数据 (n_features, n_samples)
        t: 时间数组 (n_samples,)
        X_dot: 时间导数 (n_features, n_samples)
    """
    # 计算时间导数：差分方程
    X_diff = target_data - input_data  # (n_samples, n_features)
    X_dot = X_diff / dt  # (n_samples, n_features)
    
    # 转置为 pysindy 格式：(n_features, n_samples)
    X = input_data.T
    X_dot = X_dot.T
    
    # 创建时间数组
    t = np.arange(X.shape[1]) * dt
    
    return X, t, X_dot


# =============================================================================
# 主函数
# =============================================================================

def main(atten_type = 1, trial_type = 1):
    """主处理流程"""
    print("="*60)
    print("EEG 数据处理流程")
    print("="*60)

    Config.ATTEN_TYPE_FILTER = atten_type
    Config.TRIAL_TYPE_FILTER = trial_type

    try:
        # 1. 加载原始数据
        bv_group, trial_types, atten_types, time1 = load_raw_data(Config.INPUT_FILE)
        
        # 2. 按条件筛选和提取数据
        all_data = filter_and_extract_data(
            bv_group, trial_types, atten_types, time1,
            atten_type=Config.ATTEN_TYPE_FILTER,
            trial_type=Config.TRIAL_TYPE_FILTER,
            interval=Config.TOTAL_INTERVAL
        )
        
        # 3. 标准化数据
        all_norm_data, scaler = normalize_data(all_data)
        
        # 4. 划分训练/测试集
        train_ratio, train_data, test_data = train_test_split(
            all_norm_data,
            test_ratio=Config.TEST_RATIO
        )
        
        # 5. 处理所有阶段的数据
        process_all_stages(train_data, test_data)
        
        print(f"\n{'='*60}")
        print("✓ 处理完成！")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n✗ 处理失败: {e}")
        raise



if __name__ == "__main__":
    main()
