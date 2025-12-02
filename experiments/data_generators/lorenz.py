"""
Lorenz 吸引子系统数据生成器
"""
import numpy as np
from .base import DynamicalSystem


class LorenzSystem(DynamicalSystem):
    """
    Lorenz 混沌吸引子系统
    
    系统方程:
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz
    
    默认参数（混沌参数）:
        σ = 10
        ρ = 28
        β = 8/3
    """
    
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        """
        初始化 Lorenz 系统
        
        Parameters:
        -----------
        sigma : float
            Prandtl 数
        rho : float
            Rayleigh 数
        beta : float
            几何因子
        """
        super().__init__()
        self.name = "Lorenz Attractor"
        self.dim = 3
        self.parameters = {
            'sigma': sigma,
            'rho': rho,
            'beta': beta
        }
    
    def _derivatives(self, t, state):
        """
        计算 Lorenz 系统的导数
        
        Parameters:
        -----------
        t : float
            时间（在自治系统中不使用）
        state : array-like
            当前状态 [x, y, z]
            
        Returns:
        --------
        derivatives : ndarray
            导数 [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        sigma = self.parameters['sigma']
        rho = self.parameters['rho']
        beta = self.parameters['beta']
        
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        
        return np.array([dx_dt, dy_dt, dz_dt])
    
    def get_default_initial_conditions(self):
        """
        获取默认初始条件
        
        Returns:
        --------
        initial_state : ndarray
            初始状态 [x0, y0, z0]
        """
        # 使用经典的初始条件
        return np.array([1.0, 1.0, 1.0])
    
    def get_equations_text(self):
        """
        获取 Lorenz 系统方程的文本描述
        
        Returns:
        --------
        equations : str
            方程的文本描述
        """
        sigma = self.parameters['sigma']
        rho = self.parameters['rho']
        beta = self.parameters['beta']
        
        return f"""Lorenz System Equations:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    
Parameters:
    σ (sigma) = {sigma}
    ρ (rho) = {rho}
    β (beta) = {beta:.4f}
"""
