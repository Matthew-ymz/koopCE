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