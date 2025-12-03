"""
多振子弹簧系统数据生成器
"""
import numpy as np
from .base import DynamicalSystem


class SpringOscillatorSystem(DynamicalSystem):
    """
    多振子弹簧系统
    
    系统包含N个振子，通过弹簧连接。振子可以分组，组内使用强弹簧连接，
    组间使用弱弹簧连接。
    
    系统方程（对于第i个振子）:
        dx_i/dt = v_xi
        dy_i/dt = v_yi
        dv_xi/dt = Σ F_ij_x / m
        dv_yi/dt = Σ F_ij_y / m
    
    其中 F_ij 是振子i和振子j之间的弹簧力
    """
    
    def __init__(self, n_oscillators=6, mass=1.0, k_strong=50.0, k_weak=1.0,
                 groups=None, intra_group_length=0.5, inter_group_length=3.0):
        """
        初始化弹簧振子系统
        
        Parameters:
        -----------
        n_oscillators : int
            振子数量
        mass : float
            每个振子的质量
        k_strong : float
            组内弹簧常数
        k_weak : float
            组间弹簧常数
        groups : dict, optional
            振子分组，格式为 {振子索引: 组标识}
            如果为None，则所有振子在同一组
        intra_group_length : float
            组内弹簧的自然长度
        inter_group_length : float
            组间弹簧的自然长度
        """
        super().__init__()
        self.name = "Spring Oscillator System"
        self.n_oscillators = n_oscillators
        self.dim = 4 * n_oscillators  # x, y, vx, vy for each oscillator
        
        # 默认分组
        if groups is None:
            groups = {i: 'a' for i in range(n_oscillators)}
        
        self.parameters = {
            'n_oscillators': n_oscillators,
            'mass': mass,
            'k_strong': k_strong,
            'k_weak': k_weak,
            'groups': groups,
            'intra_group_length': intra_group_length,
            'inter_group_length': inter_group_length
        }
        
        # 构建弹簧连接
        self.springs = []  # [(i, j, k_ij)]
        self.natural_lengths = {}
        
        for i in range(n_oscillators):
            for j in range(i + 1, n_oscillators):  # 只考虑 i < j
                if groups[i] == groups[j]:
                    # 组内连接
                    self.springs.append((i, j, k_strong))
                    self.natural_lengths[(i, j)] = intra_group_length
                else:
                    # 组间连接
                    self.springs.append((i, j, k_weak))
                    self.natural_lengths[(i, j)] = inter_group_length
    
    def _derivatives(self, t, state):
        """
        计算弹簧振子系统的导数
        
        Parameters:
        -----------
        t : float
            时间
        state : array-like
            当前状态 [x1, y1, x2, y2, ..., xN, yN, vx1, vy1, ..., vxN, vyN]
            
        Returns:
        --------
        derivatives : ndarray
            状态导数
        """
        N = self.n_oscillators
        m = self.parameters['mass']
        
        # 解析状态
        positions = state[:2*N].reshape(N, 2)  # (N, 2) - (x, y) for each
        velocities = state[2*N:].reshape(N, 2)  # (N, 2) - (vx, vy) for each
        
        # 计算加速度
        accelerations = np.zeros((N, 2))
        
        for i, j, k_ij in self.springs:
            diff = positions[i] - positions[j]
            distance = np.linalg.norm(diff)
            
            if distance > 1e-10:  # 避免除零
                # 弹簧力: F = -k * (d - L0) * (diff/d)
                L0 = self.natural_lengths[(i, j)]
                force = -k_ij * (distance - L0) * (diff / distance)
                
                # 牛顿第三定律
                accelerations[i] += force / m
                accelerations[j] -= force / m
        
        # 组装导数
        derivatives = np.concatenate([
            velocities.flatten(),
            accelerations.flatten()
        ])
        
        return derivatives
    
    def get_default_initial_conditions(self):
        """
        获取默认初始条件
        
        Returns:
        --------
        initial_state : ndarray
            初始状态向量
        """
        N = self.n_oscillators
        groups = self.parameters['groups']
        
        # 根据分组设置初始位置
        unique_groups = sorted(set(groups.values()))
        n_groups = len(unique_groups)
        
        positions = np.zeros((N, 2))
        
        # 为每个组分配位置
        for idx, (osc_idx, group) in enumerate(groups.items()):
            group_idx = unique_groups.index(group)
            # 在圆周上分布
            angle = 2 * np.pi * group_idx / n_groups
            radius = 2.0
            
            # 组内偏移
            group_members = [i for i, g in groups.items() if g == group]
            member_idx = group_members.index(osc_idx)
            offset_angle = 2 * np.pi * member_idx / len(group_members)
            offset_radius = 0.5
            
            positions[osc_idx, 0] = radius * np.cos(angle) + offset_radius * np.cos(offset_angle)
            positions[osc_idx, 1] = radius * np.sin(angle) + offset_radius * np.sin(offset_angle)
        
        # 添加小扰动
        positions += np.random.randn(N, 2) * 0.1
        
        # 初始速度为零
        velocities = np.zeros((N, 2))
        
        # 组装状态向量
        return np.concatenate([positions.flatten(), velocities.flatten()])
    
    def get_equations_text(self):
        """
        获取系统方程的文本描述
        
        Returns:
        --------
        equations : str
            方程的文本描述
        """
        N = self.parameters['n_oscillators']
        m = self.parameters['mass']
        k_strong = self.parameters['k_strong']
        k_weak = self.parameters['k_weak']
        
        return f"""Spring Oscillator System:
    For each oscillator i (i = 1, ..., {N}):
        dx_i/dt = vx_i
        dy_i/dt = vy_i
        dvx_i/dt = Σ F_ij_x / m
        dvy_i/dt = Σ F_ij_y / m
    
    Where F_ij is the spring force between oscillators i and j:
        F_ij = -k_ij * (d_ij - L_ij) * (r_i - r_j) / d_ij
    
Parameters:
    Number of oscillators: {N}
    Mass: {m}
    Strong spring constant (intra-group): {k_strong}
    Weak spring constant (inter-group): {k_weak}
    Number of springs: {len(self.springs)}
    Number of groups: {len(set(self.parameters['groups'].values()))}
"""
