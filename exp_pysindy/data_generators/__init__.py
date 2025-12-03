"""
动力学系统数据生成器包

包含多种动力学系统的数据生成器，用于测试和训练 SINDy 模型。
"""

from .base import DynamicalSystem
from .lorenz import LorenzSystem
from .spring import SpringOscillatorSystem

# 导出所有可用的生成器
__all__ = [
    'DynamicalSystem',
    'LorenzSystem',
    'SpringOscillatorSystem',
]

# 版本信息
__version__ = '1.0.0'
