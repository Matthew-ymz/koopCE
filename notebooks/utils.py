import numpy as np


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


def lift_double_osc(x):
    """
    lift x from 4 dim to 7 dim
    """
    return [
        x[0],x[1],x[2],x[3],x[0]**2,x[0]*x[1],x[1]**2
    ]


def lift_double_osc_dot(y):
    """
    lift x from 4 dim to 7 dim
    """
    w1 = 1.
    w2 = 1.618
    return [
        - y[1] * w1,
        y[0] * w1,
        - y[3] * w2 + y[4],
        y[2] * w2,
        - 2 * y[5] * w1,
        (y[4] - y[6]) * w1,
        2 * w1 * y[5]
    ]
