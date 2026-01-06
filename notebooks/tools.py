import pysindy as ps
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import math
from typing import Union
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import odeint
import seaborn as sns
from numpy.polynomial.legendre import Legendre
from scipy.special import legendre # legendre(n)用于生成n阶勒让德多项式。
from scipy.integrate import fixed_quad 
from scipy.linalg import eig
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import warnings

def compute_koopman_matrices(XH, W, YH):
    """
    Parameters:
    XH+YH
    Returns:
    G: Gram矩阵<psi_j, psi_i>
    A: 第一个Galerkin矩阵 <K psi_j, psi_i>
    L: 第二个Galerkin矩阵 <K psi_j, K psi_i>
    """
    M = XH.shape[0] # 数据点数量
    N = XH.shape[1]  # 基函数数量
    
    # 初始化矩阵
    G = np.zeros((N, N))
    A = np.zeros((N, N))
    L = np.zeros((N, N))
    
    # 计算矩阵元素
    for j in range(M):
        G += np.outer(XH[j, :] * W[j], XH[j, :])
        A += np.outer(XH[j, :] * W[j], YH[j, :])
        L += np.outer(YH[j, :] * W[j], YH[j, :])
    
    return G, A, L

def compute_edmd_eigenpairs(G, A, L):
    """
    根据算法2步骤2，求解广义特征值问题 A g = λ G g
    返回特征值和对应的特征向量（系数向量）

    Parameters:
    G: Gram矩阵 Ψ_X^* W Ψ_X
    A: 矩阵 Ψ_X^* W Ψ_Y
    L: 矩阵 Ψ_Y^* W Ψ_Y

    Returns:
    eigenvalues: 特征值数组
    eigenvectors: 特征向量矩阵，每一列是一个特征向量 g
    """
    eigenvalues, eigenvectors = eig(A, G)
    return eigenvalues, eigenvectors

def compute_residual(lambda_val, g, G, A, L):
    """
    根据公式(4.6)计算给定特征对(λ, g)的残差

    Parameters:
    lambda_val: 特征值 λ
    g: 特征向量（系数向量）
    G, A, L: ResDMD矩阵

    Returns:
    res: 残差值
    """
    g = g.reshape(-1, 1) # 确保是列向量
    gH = g.T.conj() # 行向量，g的共轭转置

    # 计算分子: g^* (L - λ A^H - \bar{λ} A + |λ|^2 G) g
    numerator_matrix = L - lambda_val * A.T.conj() - np.conj(lambda_val) * A + (np.abs(lambda_val)**2) * G
    numerator = (gH @ numerator_matrix @ g).item()
    numerator = np.real(numerator) # 确保是实数

    # 计算分母: g^* G g
    denominator = (gH @ G @ g).item()
    denominator = np.real(denominator)

    # 避免除零，计算残差
    if denominator <= 0:
        return np.inf
    else:
        res_squared = numerator / denominator
        # 由于数值误差，结果可能为负，取绝对值再开方
        return np.sqrt(np.abs(res_squared))
    
def koop_pseudo_spec(G, A, L, z_pts, **kwargs):
    """
    计算K的伪谱
    
    Parameters:
    G: Gram矩阵<psi_j, psi_i>
    A: 第一个Galerkin矩阵 <K psi_j, psi_i>
    L: 第二个Galerkin矩阵 <K psi_j, K psi_i>
    z_pts: 复数点向量，用于计算伪谱
    
    Optional:
    reg_param: G的正则化参数
    
    Returns:
    RES: z_pts处的残差
    """
    # 解析可选参数
    reg_param = kwargs.get('reg_param', 1e-14)
    
    # 确保矩阵是Hermitian的
    G = (G + G.T.conj()) / 2
    L = (L + L.T.conj()) / 2
    
    # 正则化G并计算SQ矩阵
    G_reg = G + np.linalg.norm(G) * reg_param * np.eye(G.shape[0])
    w, V = eigh(G_reg)
    
    # 避免除零和负值
    w = np.maximum(w, 1e-15)
    w_sqrt = np.sqrt(1.0 / np.abs(w))
    SQ = V @ np.diag(w_sqrt) @ V.T.conj()
    
    z_pts = z_pts.flatten()
    LL = len(z_pts)
    RES = np.zeros(LL, dtype=float)
    
    if LL > 0:
        warnings.filterwarnings('ignore', category=UserWarning)
        
        for jj in range(LL):
            z = z_pts[jj]
            try:
                # 构造该z对应的矩阵
                M_z = (L - z * A.T.conj() - np.conj(z) * A + (np.abs(z)**2) * G)
                M_transformed = SQ @ M_z @ SQ
                
                # 确保矩阵是Hermitian的以用于eigvalsh
                M_transformed = (M_transformed + M_transformed.T.conj()) / 2
                
                # 使用稠密计算找到最小特征值（更稳定）
                eigvals = np.linalg.eigvalsh(M_transformed)
                min_eigval = np.min(np.real(eigvals))
                
                # 避免由于数值误差导致的负值
                RES[jj] = np.sqrt(max(0, min_eigval))
                
            except Exception as e:
                print(f"Warning: Error at z={z}: {e}")
                RES[jj] = np.nan
    
    warnings.resetwarnings()
    
    return RES

def plot_pseudospectra(G, A, L, step=0.05, residual_threshold=0.01, padding_ratio=0.2, fixed_range=None):
    """
    绘制伪谱图，并根据残差阈值区分可靠和虚假特征值

    Parameters:
    G, A, L: Koopman矩阵
    x_range, y_range: 绘图范围
    step: 网格步长
    residual_threshold: 算法2中的残差阈值ε
    padding_ratio: 在特征值范围基础上添加的边距比例
    fixed_range: 如果提供，则使用固定的绘图范围 (x_min, x_max, y_min, y_max)
    """
    # 1. 计算EDMD特征对 (算法2步骤2)
    eigenvalues, eigenvectors = compute_edmd_eigenpairs(G, A, L)
    # 特征向量在矩阵eigenvectors的列中

    # 2. 为每个特征对计算残差 (算法2步骤3的逻辑)
    residuals = []
    reliable_indices = []
    spurious_indices = []

    for i in range(len(eigenvalues)):
        lambda_i = eigenvalues[i]
        g_i = eigenvectors[:, i]
        res_i = compute_residual(lambda_i, g_i, G, A, L)
        residuals.append(res_i)

        if res_i <= residual_threshold:
            reliable_indices.append(i)
        else:
            spurious_indices.append(i)

    print(f"总特征值数量: {len(eigenvalues)}")
    print(f"res阈值: {residual_threshold}")
    print(f"可靠特征值数量 (res <= {residual_threshold}): {len(reliable_indices),reliable_indices}")
    print(f"虚假特征值数量 (res > {residual_threshold}): {len(spurious_indices)}")

    # 3. 确定绘图范围
    if fixed_range is not None:
        # 使用固定的绘图范围
        x_range = (fixed_range[0], fixed_range[1])
        y_range = (fixed_range[2], fixed_range[3])
    else:
        # 自适应范围：基于所有特征值的分布
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        
        # 计算特征值的范围
        real_min, real_max = np.min(real_parts), np.max(real_parts)
        imag_min, imag_max = np.min(imag_parts), np.max(imag_parts)
        
        # 使用相同的范围，保持纵横比一致
        overall_min = min(real_min, imag_min)
        overall_max = max(real_max, imag_max)
        overall_range = overall_max - overall_min
        
        # 如果范围太小（如所有特征值都集中在一点），设置最小范围
        if overall_range < 0.1:
            overall_range = 1.0
            center = (overall_min + overall_max) / 2
            overall_min = center - 0.5
            overall_max = center + 0.5
        
        # 添加边距
        padding = overall_range * padding_ratio
        x_range = (overall_min - padding, overall_max + padding)
        y_range = (overall_min - padding, overall_max + padding)
    
    # 3. 计算伪谱网格（算法3/原有伪谱计算逻辑）
    x_pts = np.arange(x_range[0], x_range[1] + step, step)
    y_pts = np.arange(y_range[0], y_range[1] + step, step)
    X_grid, Y_grid = np.meshgrid(x_pts, y_pts)
    z_pts = X_grid + 1j * Y_grid
    z_flat = z_pts.flatten()

    RES = koop_pseudo_spec(G, A, L, z_flat)
    RES = RES.reshape(z_pts.shape)
    RES = np.nan_to_num(RES, nan=np.max(RES[~np.isnan(RES)]))
    # 计算完RES并reshape后，添加这行
    print(f"残差RES的范围：{np.min(RES):.6f} ~ {np.max(RES):.6f}")

    # 4. 绘图
    plt.figure(figsize=(8, 8))

    # 伪谱等高线
    #levels = [0.001, 0.01, 0.1, 0.3]
    levels = [0.001, 0.01, 0.1]
    contour = plt.contour(X_grid, Y_grid, np.real(RES), levels=levels,
                         colors='black', linewidths=2)
    plt.clabel(contour, inline=True, fontsize=11, fmt='%.3f')

    # 绘制特征值
    # 可靠特征值 (残差小) - 蓝色十字
    reliable_eigs = eigenvalues[reliable_indices]
    plt.plot(np.real(reliable_eigs), np.imag(reliable_eigs), 'x',
             markersize=8, color='blue', markeredgewidth=2,
             label=f'Reliable eigenvalues (res $\leq$ {residual_threshold})')

    # 虚假特征值 (残差大) - 洋红点
    spurious_eigs = eigenvalues[spurious_indices]
    plt.plot(np.real(spurious_eigs), np.imag(spurious_eigs), '.',
             markersize=10, color='magenta',
             label='Spurious eigenvalues')

    # 格式化
    plt.gca().set_aspect('equal')
    plt.xlabel('Real', fontsize=14)
    plt.ylabel('Imaginary', fontsize=14)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.grid(True, alpha=0.3)
    #plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.legend(fontsize=12)
    plt.title(f'Reliable num={len(reliable_indices)}, Residual Threshold={residual_threshold}', fontsize=14)
    plt.tight_layout()
    plt.show()
    return eigenvalues, residuals, reliable_indices

def plot_main(kp, X_embed):
    M = len(X_embed)-1
    W = np.ones(M) / M
    X = X_embed[:-1]
    Y = X_embed[1:]
    XH = kp.transform(X)
    YH = kp.transform(Y)
    G, A, L = compute_koopman_matrices(XH, W, YH)
    eigenvalues, residuals, reliable_indices = plot_pseudospectra(G, A, L, residual_threshold=0.01,fixed_range=(0.8,1.20,-0.15,0.15))
    return eigenvalues, residuals, reliable_indices

def matrix_l1_norm_manual(matrix):
    """
    手动计算矩阵的L1范数（不使用numpy）
    """
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    # 计算每列的绝对值之和
    column_sums = []
    for j in range(cols):
        col_sum = 0
        for i in range(rows):
            col_sum += abs(matrix[i][j])
        column_sums.append(col_sum)
    
    # 返回最大的列和
    return max(column_sums)

def matrix_l0_norm_corrected(matrix, threshold=1e-10):
    """
    计算矩阵的L0范数（各列非零元素数量的最大值）
    
    参数:
    matrix: numpy数组或可以转换为numpy数组的矩阵
    threshold: 阈值，绝对值小于此值的元素视为零
    
    返回:
    l0_norm: 矩阵的L0范数（整数）
    column_norms: 各列的L0范数
    """
    matrix = np.array(matrix, dtype=float)
    
    # 应用阈值：将接近零的元素视为零
    matrix_thresholded = np.where(np.abs(matrix) < threshold, 0, matrix)
    
    # 计算每列的非零元素数量
    column_norms = []
    for col in range(matrix_thresholded.shape[1]):
        non_zero_count = np.count_nonzero(matrix_thresholded[:, col])
        column_norms.append(non_zero_count)
    
    # 矩阵的L0范数是各列L0范数的最大值
    l0_norm = max(column_norms)
    
    return l0_norm

def get_positive_contributions(sing_values):  
    ave_sig = []
    for i in range(len(sing_values)):
        ave_sig.append(np.mean(np.log(sing_values[0:i+1])))

    output = []
    for id in range(len(ave_sig)-1):
        diff = ave_sig[id] - ave_sig[id+1]
        output.append(diff)
    return output

def compute_entropy(increments):
    if not increments:
        return 0.0
    
    total = sum(increments)
    # If total is 0, there's no variation => 0.0 entropy
    if total == 0:
        return 0.0
    
    # Normalize to probabilities
    probabilities = [x / total for x in increments]

    # Compute Shannon entropy (base 2)
    entropy = 0.0
    for p in probabilities:
        # Only compute for p > 0 to avoid math domain errors
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy

def print_equations(coefficient_matrix, feature_names, target_names, threshold=1e-5):
    """
    将系数矩阵转换为数学方程并打印。
    
    Args:
        coefficient_matrix (np.array): 系数矩阵 (行数=特征数, 列数=目标变量数)
        feature_names (list): 纵轴标签列表 (对应矩阵的行, 如 ['x0', 'sin(x0)', ...])
        target_names (list): 横轴标签列表 (对应矩阵的列, 如 ['y0', 'y1', 'y2'])
        threshold (float): 忽略系数绝对值小于此阈值的项 (默认 1e-5)
    """
    
    rows, cols = coefficient_matrix.shape
    
    # 检查维度匹配
    if len(feature_names) != rows:
        print(f"错误: feature_names 长度 ({len(feature_names)}) 与矩阵行数 ({rows}) 不一致")
        return
    if len(target_names) != cols:
        print(f"错误: target_names 长度 ({len(target_names)}) 与矩阵列数 ({cols}) 不一致")
        return

    # 遍历每一列 (即每一个 y0, y1, y2...)
    for col_idx in range(cols):
        lhs = target_names[col_idx] # 等号左边
        rhs_parts = []
        
        # 遍历该列的每一行，寻找非零系数
        for row_idx in range(rows):
            coef = coefficient_matrix[row_idx, col_idx]
            
            # 如果系数绝对值大于阈值，则认为该项存在
            if abs(coef) > threshold:
                term_name = feature_names[row_idx]
                
                # 格式化系数，保留4位小数
                formatted_coef = f"{coef:.4f}"
                
                # 拼接项：例如 "0.5234 * sin(x0)"
                rhs_parts.append(f"{formatted_coef} * {term_name}")
        
        # 组装整个方程
        if not rhs_parts:
            equation = f"{lhs} = 0"
        else:
            # 用 " + " 连接所有项
            equation_str = " + ".join(rhs_parts)
            # 简单的美化：处理 "+ -" 为 "- "
            equation_str = equation_str.replace("+ -", "- ")
            equation = f"{lhs} = {equation_str}"
            
        print(equation)
        print("-" * 30) # 分隔线

def draw_fft(data: pd.DataFrame, dt: Union[str, float] = 'index', 
             remove_dc: bool = True, max_f: float = 0, normalize: bool = True) -> pd.DataFrame:
    """
    对 DataFrame 数据进行 FFT 分析并绘图
    
    Parameters:
    data: pd.DataFrame - 输入数据
    dt: Union[str, float] - 采样间隔设置
        - 'index': 使用 data.index 计算间隔（支持 datetime 类型）
        - 0: 使用 cycle=1
        - 非零实数: 使用 cycle=dt 作为采样周期
    remove_dc: bool - 是否去除直流分量（默认 True）
    max_f: float - 最大显示频率（默认 0 表示不限制）
    normalize: bool - 是否对数据进行归一化（默认 True）
        归一化方法：对每列减去均值并除以标准差
    
    Returns:
    pd.DataFrame - FFT 结果，index 为频率，columns 与输入相同
    """
    # 步骤0: 归一化数据（如果需要）
    if normalize:
        data_processed = (data - data.mean()) / data.std()
    else:
        data_processed = data.copy()
    
    # 步骤1: 去除直流分量
    if remove_dc:
        data_processed = data_processed - data_processed.mean()
    
    # 获取数据点数
    N = len(data_processed)
    
    # 步骤2: 对各列进行 FFT 变换
    fft_result = np.fft.fft(data_processed.values, axis=0)
    # 归一化并只保留正频率部分
    fft_data = np.abs(fft_result / N)[:N//2]
    
    # 步骤3: 计算频率轴
    if dt == 'index':
        # 使用 index 计算采样周期
        cycle = data.index[1] - data.index[0]
        # 如果是 datetime 类型，转换为秒
        if isinstance(cycle, pd.Timedelta):
            cycle = cycle.total_seconds()
    elif dt == 0:
        cycle = 1
    else:
        # dt 是非零实数，作为采样周期
        cycle = dt
    
    # 使用 fftfreq 生成频率轴，只取正频率部分
    freq = np.fft.fftfreq(N, cycle)[:N//2]
    
    # 步骤4: 构建结果 DataFrame
    result_df = pd.DataFrame(fft_data, index=freq, columns=data.columns)
    
    # 如果指定了最大频率，进行过滤
    if max_f > 0:
        result_df = result_df[result_df.index <= max_f]
    
    # 步骤5: 绘制频谱图
    plt.figure(figsize=(10, 6))
    for col in result_df.columns:
        plt.plot(result_df.index, result_df[col], label=col)
    
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('FFT Spectrum', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return result_df

def fit_sindy_sr3_robust(X, lib, feature_names,
                         penalty='l0',       
                         dt=1,
                         discrete_time=True,
                         thresholds=None,
                         nu=1.0,
                         max_iter=3000,
                         tol=1e-6,
                         test_size=0.2,
                         metric='aic'):
    
    # --- 1. 针对不同范数调整默认扫描范围 ---
    if thresholds is None:
        if penalty == 'l0':
            thresholds = np.logspace(-5, -1, 20)  # L0: 物理意义的系数截断值
        elif penalty == 'l1':
            thresholds = np.logspace(-5, -1, 20)  # L1: 物理意义的软阈值截断值
        else:
            thresholds = np.logspace(-2, 4, 20)   # L2: 直接是正则化权重，范围通常较大

    # --- 2. 数据拆分 ---
    split_idx = int(len(X) * (1 - test_size))
    X_train = X[:split_idx]
    X_test = X[split_idx:]

    best_score = float('inf')
    best_model = None
    history = []

    print(f"开始使用 {penalty.upper()} 范数扫描 {len(thresholds)} 个参数...")

    for thr in tqdm(thresholds):
        # --- 核心修改：根据范数计算 lambda ---
        if penalty == 'l0':
            # L0: 硬阈值换算
            lam = ps.SR3.calculate_l0_weight(thr, nu)
            reg_type = "l0"
        elif penalty == 'l1':
            # L1: 软阈值换算 (lambda = nu * threshold)
            lam = thr * nu 
            reg_type = "l1"
        elif penalty == 'l2':
            # L2: 没有截断概念，thr 直接作为 lambda
            lam = thr
            reg_type = "l2"
        else:
            raise ValueError("penalty must be 'l0', 'l1', or 'l2'")

        # 配置 SR3 优化器
        opt = ps.SR3(
            reg_weight_lam=lam,
            regularizer=reg_type,
            relax_coeff_nu=nu,
            normalize_columns=True, 
            unbias=True,  # 对 L1 非常重要！因为 L1 会压缩系数，必须由 unbias 修正回真值
            max_iter=max_iter,
            tol=tol,
        )

        model = ps.SINDy(feature_library=lib, optimizer=opt, discrete_time=discrete_time)
        model.fit(X_train, t=dt, feature_names=feature_names)

        # --- 评估部分 (同前) ---
        X_test_pred = model.predict(X_test)
        
        if not discrete_time:
            mse = -model.score(X_test, t=dt) 
        else:
            mse = mean_squared_error(X_test[1:], X_test_pred[:-1])

        coef = model.coefficients()
        # 统计非零项：对于 L1/L2，由于浮点误差，可能不会严格为 0，需设置一个小门槛
        k_params = np.sum(np.abs(coef) > 1e-5) 

        if k_params == 0:
            score = float('inf')
        else:
            n_samples = len(X_test)
            if mse <= 0: mse = 1e-10
            log_likelihood = n_samples * np.log(mse)
            
            if metric == 'aic':
                score = log_likelihood + 2 * k_params
            elif metric == 'bic':
                score = log_likelihood + k_params * np.log(n_samples)
            else: 
                score = mse

        history.append({'thr': thr, 'lam': lam, 'mse': mse, 'k': k_params, 'score': score, 'model': model})

        if score < best_score:
            best_score = score
            best_model = model


    # --- 结果展示 ---
    history.sort(key=lambda x: x['score'])
    
    if best_model:
        top = history[0]
        print(f"\n最佳模型 ({metric}) | Penalty: {penalty}")
        print(f"Param (Thr/Lam): {top['thr']:.3e}")
        print(f"Test MSE: {top['mse']:.4e}")
        print(f"Complexity: {top['k']}")
        best_model.print()
    
    return best_model, history

def lift_time_delay(X, feature_names=None, n_delays=1, delay_interval=1):
    """
    将时间序列 X 提升到时间延迟坐标系，并自动生成对应的新变量名。
    
    参数:
    X : np.ndarray, shape (n_samples, n_dim)
        原始状态数据。
    feature_names : list of str, optional
        原始变量的名字，例如 ['x', 'y', 'z']。
        如果不提供，默认生成 ['x0', 'x1', ...]。
    n_delays : int
        向后看的步数 (delay count)。
    delay_interval : int
        延迟的间隔步长 (tau)。
        
    返回:
    H : np.ndarray
        提升后的 Hankel 矩阵。
    new_feature_names : list of str
        对应的变量名列表。
        格式示例：['x', 'y', 'x_tau1', 'y_tau1', 'x_tau2', 'y_tau2']
    """
    n_samples, n_dim = X.shape
    
    # 1. 处理默认变量名
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_dim)]
    
    if len(feature_names) != n_dim:
        raise ValueError(f"feature_names 长度 ({len(feature_names)}) 与数据维度 ({n_dim}) 不匹配")

    # 2. 计算有效样本数
    window_size = n_delays * delay_interval
    n_valid_samples = n_samples - window_size
    
    if n_valid_samples <= 0:
        raise ValueError(f"数据太短，无法进行 {n_delays} 次延迟")

    # 3. 构建数据矩阵和名字列表
    shifted_features = []
    new_names = []
    
    # --- 第0层：当前时刻 t (无后缀或标记为 t) ---
    # 取从 window_size 到最后的点
    shifted_features.append(X[window_size : ])
    new_names.extend(feature_names) # 保持原名，表示 x(t)
    
    # --- 后续层：延迟时刻 t - k*tau ---
    for k in range(1, n_delays + 1):
        # 计算偏移
        offset = window_size - k * delay_interval
        
        # 数据切片
        if offset == 0:
            shifted_data = X[: -window_size]
        else:
            shifted_data = X[offset : offset + n_valid_samples]
        shifted_features.append(shifted_data)
        
        # 名字生成：统一添加后缀，例如 _d1 表示 delay 1, _d2 表示 delay 2
        # 或者使用 _tau1, _tau2
        suffix = f"_d{k}" if delay_interval == 1 else f"_d{k*delay_interval}"
        current_names = [f"{name}{suffix}" for name in feature_names]
        new_names.extend(current_names)

    # 4. 拼接矩阵
    H = np.column_stack(shifted_features)
    
    return H, new_names

def plot_station(df, coarse_grain_coff, delay=0):
    if isinstance(coarse_grain_coff, np.ndarray):
        coff_df = pd.DataFrame(coarse_grain_coff)
    else:
        coff_df = coarse_grain_coff.reset_index(drop=True)

    # 2. 遍历每一列进行绘图
    for col_idx in coff_df.columns:
        # 将当前列的数据合并到原始 df 中，命名为一个临时列名，例如 'value_to_plot'
        df_plot = df.copy()
        df_plot['value_to_plot'] = coff_df[col_idx].values
        
        # 绘制图形
        fig = px.scatter(
            df_plot,
            x="lon",
            y="lat",
            text="station_id",
            color="value_to_plot",  # 核心修改：颜色映射到当前列的数值
            hover_data=["station_name", "city"],
            color_continuous_scale='Viridis', # 设置颜色条，可选 'Plasma', 'Inferno', 'Turbo' 等
            title=f"y{col_idx}_d{delay}"
        )
        
        # 调整标注位置和点的大小
        fig.update_traces(
            textposition='top center',
            marker=dict(size=10, opacity=0.8) # 稍微调大点的大小以看清颜色
        )
        
        # 优化布局：保持经纬度比例，以免地图变形
        fig.update_layout(
            yaxis_scaleanchor="x", 
            yaxis_scaleratio=1,
            height=600,
            width=800
        )
        
        fig.show()
