import pykoop
import sklearn
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import seaborn as sns
from numpy.polynomial.legendre import Legendre
from scipy.special import legendre # legendre(n)用于生成n阶勒让德多项式。
from scipy.integrate import fixed_quad 
from scipy.linalg import eig
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import warnings
import matplotlib as mpl

import scipy
import sklearn
import pysindy as ps 
from sklearn.linear_model import Lasso

from sklearn.exceptions import ConvergenceWarning
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, fixed
import ipywidgets as widgets

# Lorenz model
def lorenz(t, x, sigma=10, beta=2.66667, rho=28):
    """
    Lorenz系统的微分方程
    
    参数:
        t: 时间
        x: 状态向量 [x, y, z]
        sigma: 参数 (默认 10)
        beta: 参数 (默认 2.66667)
        rho: 参数 (默认 28)
    
    返回:
        dx/dt: 状态导数
    """
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]

# Nonlinear pendulum
def npendulum(t, x):
    """
    nonlinear pendulum 的 Docstring
    
    参数
    :param t: 时间
    :param x: 状态向量 [x, y]

    返回
    :return dx/dt: 状态导数
    """
    return [
        x[1],
        -np.sin(x[0])
    ]

def double_osc(t, x):
    """
    a simple nonlinear system
    """
    w1 = 1.
    w2 = 1.618
    return [
        - x[1] * w1,
        x[0] * w1,
        x[0]**2 - x[3] * w2,
        x[2] * w2
    ]

def kuramoto_ode_cluster(theta, omega, K_matrix):
    """Kuramoto ODE with custom coupling matrix K_ij."""
    N = len(theta)
    dtheta = np.zeros(N)
    for i in range(N):
        dtheta[i] = omega[i] + np.sum(
            K_matrix[i, :] * np.sin(theta - theta[i])
        ) / N
    return dtheta

def generate_kuramoto_cluster_data_sin_cos(
    N=12, n_clusters=3, K_intra=2.0, K_inter=0.2,
    dt=0.01, T=30, noise=0.0, random_seed1=0, random_seed2=0
):
    """
    生成带‘团结构’的Kuramoto振子数据。
    团内耦合K_intra > 团间耦合K_inter。
    """
    np.random.seed(random_seed1)
    t_steps = int(T / dt)
    t = np.arange(0, T, dt)
    omega = 2 * np.pi * (0.2 + 0.05 * np.random.randn(N))
    np.random.seed(random_seed2)
    theta = np.random.uniform(0, 2 * np.pi, N)

    # --- 构造耦合矩阵 ---
    cluster_size = N // n_clusters
    K_matrix = np.full((N, N), K_inter)
    for c in range(n_clusters):
        start = c * cluster_size
        end = N if c == n_clusters - 1 else (c + 1) * cluster_size
        K_matrix[start:end, start:end] = K_intra

    # --- 时间积分 ---
    theta_hist = np.zeros((t_steps, N))
    theta_hist[0] = theta
    for i in range(1, t_steps):
        dtheta = kuramoto_ode_cluster(theta, omega, K_matrix)
        theta = np.mod(theta + dtheta * dt, 2 * np.pi)
        if noise > 0:
            theta += noise * np.random.randn(N)
        theta_hist[i] = theta

    X = np.hstack([np.cos(theta_hist), np.sin(theta_hist)])  # sin-cos embedding
    return X, theta_hist, t, K_matrix


def compute_order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta), axis=1))

def compute_cluster_order_parameters(theta, n_clusters):
    """计算每个团的序参量"""
    N = theta.shape[1]
    cluster_size = N // n_clusters
    group_r = []
    for c in range(n_clusters):
        start = c * cluster_size
        end = N if c == n_clusters - 1 else (c + 1) * cluster_size
        r_c = compute_order_parameter(theta[:, start:end])
        group_r.append(r_c)
    return group_r

def plot_clustered_kuramoto(N=12, n_clusters=3, K_intra=2.0, K_inter=0.2, noise=0.0, T=30, dt=0.01, random_seed1=0, random_seed2=0):
    X_embed, theta_hist, t, K_matrix = generate_kuramoto_cluster_data_sin_cos(
        N=N, n_clusters=n_clusters, K_intra=K_intra, K_inter=K_inter, dt=dt, T=T, noise=noise, random_seed1=random_seed1, random_seed2=random_seed2
    )

    r_total = compute_order_parameter(theta_hist)
    r_groups = compute_cluster_order_parameters(theta_hist, n_clusters)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    colors = plt.cm.tab10(np.arange(n_clusters))
    N_cluster = N // n_clusters

    # (2) 每个振子的相位随时间
    ax2 = axes[0]
    for g in range(n_clusters):
        for i in range(g * N_cluster, (g + 1) * N_cluster):
            ax2.plot(t, X_embed[:, i], lw=0.8, color=colors[g])
    ax2.set_title("Phase Evolution θ_i(t)")
    ax2.set_xlabel("Time")

    # (3) 各团与总体序参量
    ax3 = axes[1]
    ax3.plot(t, r_total, "k", lw=2, label="Overall r(t)")
    for g, r_c in enumerate(r_groups):
        ax3.plot(t, r_c, lw=2, color=colors[g], label=f"Group {g+1}")
    ax3.set_ylim(0, 1.05)
    ax3.set_title("Order parameters")
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return X_embed, theta_hist, t, K_matrix

# SIR model
def sir_model_normalized(y, t, beta, gamma):
    """
    归一化SIR模型的微分方程。
    y: 一个包含s, i, r比例的列表或数组
    t: 时间点
    beta: 传染率
    gamma: 康复率
    """
    s, i, r = y
    ds_dt = -beta * s * i
    di_dt = beta * s * i - gamma * i
    dr_dt = gamma * i
    return [ds_dt, di_dt, dr_dt]

def gen_sir_data(initial_infected_ratio=0.5,
                 initial_recovered_ratio=0.0,beta=0.3,gamma=0.05,
                 total_days=200,dt=0.01,noise_mean=0.0,
                 noise_std=0.001,random_seed=None):
    """
    生成带噪声的SIR模型数据
    
    参数:
    --------
    initial_infected_ratio : float初始感染比例
    initial_recovered_ratio : float初始康复比例
    beta : float传染率
    gamma : float康复率
    total_days : int模拟总天数
    dt : int间距
    noise_mean : float高斯噪声均值
    noise_std : float高斯噪声标准差
    random_seed : int or None随机种子，用于复现结果
    
    返回:
    --------
    data : np.ndarray包含噪声的SIR数据，列顺序：s, i,r
    t : np.ndarray时间数组
    """
    # 设置随机种子
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 计算初始易感比例
    initial_susceptible_ratio = 1.0 - initial_infected_ratio - initial_recovered_ratio
    y0 = [initial_susceptible_ratio, initial_infected_ratio, initial_recovered_ratio]
    
    # 生成时间点
    t = np.arange(0, total_days, dt)
    
    # 求解SIR模型
    solution = odeint(sir_model_normalized, y0, t, args=(beta, gamma))
    
    # 构造无噪声数据（重复s和i列）
    s_col = solution[:, 0, np.newaxis]
    i_col = solution[:, 1, np.newaxis]
    r_col = solution[:, 2, np.newaxis]
    data_noiseless = np.hstack([s_col, i_col, r_col])
    
    # 添加高斯噪声
    noise = noise_mean + np.random.randn(*data_noiseless.shape) * noise_std
    x = data_noiseless + noise
    
    return x, t

def plot_sir_results(x, t, figsize=(12, 4), title_prefix="SIR Model"):
    if isinstance(x, np.ndarray) and isinstance(t, np.ndarray):
        x = [x]
        t = [t]
    # ========== 初始化画布 ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ========== 多序列样式配置（自动渐变颜色+透明度） ==========
    n_sequences = len(x)
    # 生成渐变颜色（基于序列数量自动分配）
    cmap = plt.cm.viridis  # 美观的渐变色谱（蓝→绿→黄→红）
    colors = cmap(np.linspace(0, 1, n_sequences))
    # 透明度梯度（避免多序列重叠过密）
    alphas = np.linspace(0.2, 0.8, n_sequences)

    # ========== 遍历绘图 ==========
    for idx in range(n_sequences):
        data = x[idx]
        t_seq = t[idx]
        color = colors[idx]
        alpha = alphas[idx]
        
        # 时间序列图：单序列展示s/i/r，多序列只展示i（带渐变颜色+透明度）
        if n_sequences == 1:  # 单序列：保留原始样式
            ax1.plot(t_seq, data[:,0], color='blue', label='Susceptible (s)', linewidth=1.5)
            ax1.plot(t_seq, data[:,1], color='red', label='Infected (i)', linewidth=1.5)
            ax1.plot(t_seq, 1-data[:,0]-data[:,1], color='green', label='Recovered (r)', linewidth=1.5)
            ax1.legend()  # 单序列显示图例
        else:  # 多序列：渐变颜色+透明度，仅展示i列
            ax1.plot(t_seq, data[:,1], color=color, alpha=alpha, linewidth=0.5)
        
        # 相平面图：单/多序列差异化样式
        ax2.plot(
            data[:,0], data[:,1], 
            color=color, alpha=alpha,
            linewidth=1.5 if n_sequences == 1 else 0.5
        )

    # ========== 统一图表配置 ==========
    # 时间序列图
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Proportion')
    ax1.set_title(f'{title_prefix} - Time Series')
    ax1.grid(alpha=0.3)
    
    # 相平面图（多序列添加颜色条，直观区分轨迹）
    ax2.set_xlabel('Susceptible (s)')
    ax2.set_ylabel('Infected (i)')
    ax2.set_title(f'{title_prefix} - Phase Plane')
    ax2.grid(alpha=0.3)
    
    # 多序列添加颜色条（无需额外参数，自动生成）
    if n_sequences > 1:
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, n_sequences-1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax2, shrink=0.8)
        cbar.set_label('Sequence Index', fontsize=8)

    plt.tight_layout()
    plt.show()