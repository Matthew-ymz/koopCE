"""
全面的库版本检查脚本
检查所有与 EDMD、Koopman 算子相关的库及其版本兼容性
"""

import sys
import subprocess
from packaging import version
import importlib

print("=" * 80)
print("EDMD & Koopman 算子相关库版本检查")
print("=" * 80)
print(f"\n当前 Python 版本: {sys.version}")
print(f"Python 版本号: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# ==================== 核心库版本检查 ====================

def check_package(package_name, display_name=None, critical=False):
    """
    检查单个包的版本
    
    参数：
    ------
    package_name : str
        包的导入名称（可能与 pip 名称不同）
    display_name : str, optional
        显示名称（如果不同于 package_name）
    critical : bool
        是否是关键库（会影响 EDMD）
    """
    if display_name is None:
        display_name = package_name
    
    try:
        pkg = importlib.import_module(package_name)
        ver = pkg.__version__
        status = "✓" if not critical else "✓ [关键]"
        print(f"{status} {display_name:<30} {ver:<15}")
        return ver
    except ImportError:
        print(f"✗ {display_name:<30} {'未安装':<15}")
        return None
    except AttributeError:
        print(f"? {display_name:<30} {'无法获取版本':<15}")
        return "unknown"

print("\n" + "-" * 80)
print("1. 核心科学计算库")
print("-" * 80)

numpy_ver = check_package('numpy', 'NumPy', critical=True)
scipy_ver = check_package('scipy', 'SciPy', critical=True)
sklearn_ver = check_package('sklearn', 'Scikit-learn', critical=True)

print("\n" + "-" * 80)
print("2. EDMD 和动态系统相关库")
print("-" * 80)

pysindy_ver = check_package('pysindy', 'PySINDy', critical=True)
dmdc_ver = check_package('DMDc', 'DMDc', critical=False)

# 检查其他可能的 DMD 库
try:
    import pydmd
    print(f"✓ PyDMD (pydmd)           {pydmd.__version__:<15}")
except ImportError:
    print(f"✗ PyDMD (pydmd)           {'未安装':<15}")
except AttributeError:
    print(f"? PyDMD (pydmd)           {'无法获取版本':<15}")

print("\n" + "-" * 80)
print("3. 数据处理和可视化库")
print("-" * 80)

pandas_ver = check_package('pandas', 'Pandas', critical=False)
matplotlib_ver = check_package('matplotlib', 'Matplotlib', critical=False)
seaborn_ver = check_package('seaborn', 'Seaborn', critical=False)

print("\n" + "-" * 80)
print("4. 矩阵计算和线性代数扩展")
print("-" * 80)

sympy_ver = check_package('sympy', 'SymPy', critical=False)
cvxpy_ver = check_package('cvxpy', 'CVXPY', critical=False)

print("\n" + "-" * 80)
print("5. 优化和并行计算库")
print("-" * 80)

numba_ver = check_package('numba', 'Numba', critical=False)
joblib_ver = check_package('joblib', 'Joblib', critical=False)

print("\n" + "-" * 80)
print("6. 深度学习 / 神经网络库（可选）")
print("-" * 80)

torch_ver = check_package('torch', 'PyTorch', critical=False)
tf_ver = check_package('tensorflow', 'TensorFlow', critical=False)

print("\n" + "-" * 80)
print("7. 其他工具库")
print("-" * 80)

check_package('tqdm', 'tqdm', critical=False)
check_package('pytest', 'pytest', critical=False)
check_package('jupyter', 'Jupyter', critical=False)

# ==================== 版本兼容性检查 ====================

print("\n" + "=" * 80)
print("版本兼容性检查")
print("=" * 80)

compatibility_issues = []

# NumPy - SciPy 兼容性
if numpy_ver and scipy_ver:
    numpy_major, scipy_major = numpy_ver.split('.')[0], scipy_ver.split('.')[0]
    print(f"\n✓ NumPy 主版本: {numpy_major}")
    print(f"✓ SciPy 主版本: {scipy_major}")
    
    # 通常需要 NumPy 和 SciPy 版本接近
    if abs(int(numpy_major) - int(scipy_major)) > 1:
        compatibility_issues.append(
            f"⚠ NumPy ({numpy_ver}) 和 SciPy ({scipy_ver}) 版本差异较大"
        )

# NumPy - Scikit-learn 兼容性
if numpy_ver and sklearn_ver:
    try:
        np_version = version.parse(numpy_ver)
        sklearn_version = version.parse(sklearn_ver)
        
        if np_version < version.parse("1.14.6"):
            compatibility_issues.append(
                f"⚠ NumPy 版本过旧 ({numpy_ver})，建议升级到 1.14.6 或以上"
            )
        if sklearn_version < version.parse("0.20"):
            compatibility_issues.append(
                f"⚠ Scikit-learn 版本过旧 ({sklearn_ver})，建议升级到 0.20 或以上"
            )
    except:
        pass

# PySINDy 和 NumPy/SciPy 兼容性
if pysindy_ver and numpy_ver and scipy_ver:
    print(f"\n✓ PySINDy 版本: {pysindy_ver}")
    print(f"  依赖库: NumPy {numpy_ver}, SciPy {scipy_ver}")
    
    try:
        pysindy_v = version.parse(pysindy_ver)
        scipy_v = version.parse(scipy_ver)
        numpy_v = version.parse(numpy_ver)
        
        # PySINDy 1.7+ 对新版本的 SciPy/NumPy 可能有问题
        if pysindy_v >= version.parse("1.7.0"):
            if scipy_v >= version.parse("1.12.0"):
                compatibility_issues.append(
                    f"⚠ PySINDy {pysindy_ver} 与 SciPy {scipy_ver} 可能存在 ODE 求解器兼容性问题"
                )
            if numpy_v >= version.parse("1.26.0"):
                compatibility_issues.append(
                    f"⚠ PySINDy {pysindy_ver} 与 NumPy {numpy_ver} 可能存在兼容性问题"
                )
    except:
        pass

# ==================== 推荐版本组合 ====================

print("\n" + "=" * 80)
print("推荐的库版本组合（已验证的稳定配置）")
print("=" * 80)

recommendations = {
    "Python 3.10 + 稳定组合": [
        "python==3.10",
        "numpy==1.24.3",
        "scipy==1.11.4",
        "scikit-learn==1.3.2",
        "pysindy==1.7.2",
        "matplotlib==3.8.0",
        "pandas==2.0.3",
    ],
    "Python 3.11 + 最新组合": [
        "python==3.11",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "pysindy==1.7.2",
        "matplotlib>=3.8.0",
        "pandas>=2.0.0",
    ],
    "轻量级 EDMD 组合（无 ODE 求解）": [
        "python>=3.9",
        "numpy>=1.20",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
    ],
}

for combo_name, packages in recommendations.items():
    print(f"\n{combo_name}:")
    for pkg in packages:
        print(f"  - {pkg}")

# ==================== 问题诊断和建议 ====================

print("\n" + "=" * 80)
print("问题诊断和建议")
print("=" * 80)

if compatibility_issues:
    print("\n检测到的潜在兼容性问题:")
    for i, issue in enumerate(compatibility_issues, 1):
        print(f"{i}. {issue}")
else:
    print("\n✓ 未检测到明显的版本兼容性问题")

# 根据检测到的库提供建议
print("\n" + "-" * 80)
print("针对你的环境的建议:")
print("-" * 80)

if pysindy_ver is None:
    print("\n⚠ PySINDy 未安装，建议安装以支持 EDMD：")
    print("  pip install pysindy")
elif pysindy_ver:
    print(f"\n✓ PySINDy 已安装（版本 {pysindy_ver}）")
    print("  注意：如果遇到 Kernel 崩溃问题，可以：")
    print("  1. 使用 EDMD 代码（不依赖 PySINDy 的 ODE 求解器）")
    print("  2. 重新安装 PySINDy: pip install --upgrade --force-reinstall pysindy")
    print("  3. 使用推荐的版本组合")

if numba_ver:
    print(f"\n✓ Numba 已安装（版本 {numba_ver}），可用于加速 ODE 求解")
else:
    print("\n推荐安装 Numba 用于 JIT 编译和加速：")
    print("  pip install numba")

# ==================== 环境修复建议 ====================

print("\n" + "=" * 80)
print("环境修复步骤（如遇到问题）")
print("=" * 80)

print("""
1. 完整清理并重新安装（最彻底的方法）：
   
   # 使用 conda
   conda remove --name edmd_env --all -y
   conda create -n edmd_env python=3.10 -y
   conda activate edmd_env
   conda install pysindy scipy numpy scikit-learn matplotlib -c conda-forge
   
   # 或使用 pip
   pip install --upgrade --force-reinstall pysindy scipy numpy scikit-learn

2. 如果只是想修复 PySINDy ODE 求解器问题：
   
   pip uninstall pysindy scipy -y
   pip install pysindy==1.7.2 scipy==1.11.4 numpy==1.24.3

3. 升级所有关键库（谨慎操作）：
   
   pip install --upgrade numpy scipy scikit-learn matplotlib pandas

4. 验证安装是否成功：
   
   python -c "import pysindy; print(pysindy.__version__)"
   python -c "from scipy.integrate import odeint; print('ODE 求解器正常')"
""")

# ==================== 生成版本报告文件 ====================

print("\n" + "=" * 80)
print("生成版本报告")
print("=" * 80)

# 生成可导出的版本报告
report_filename = "library_versions.txt"
try:
    with open(report_filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EDMD & Koopman 算子相关库版本报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"生成时间: {__import__('datetime').datetime.now()}\n")
        f.write(f"Python 版本: {sys.version}\n\n")
        
        f.write("库版本信息:\n")
        f.write("-" * 80 + "\n")
        
        # 重新收集信息写入文件
        libs_to_check = [
            ('numpy', 'NumPy', True),
            ('scipy', 'SciPy', True),
            ('sklearn', 'Scikit-learn', True),
            ('pysindy', 'PySINDy', True),
            ('pandas', 'Pandas', False),
            ('matplotlib', 'Matplotlib', False),
        ]
        
        for pkg_name, display_name, is_critical in libs_to_check:
            try:
                pkg = importlib.import_module(pkg_name)
                ver = pkg.__version__
                critical_flag = " [关键]" if is_critical else ""
                f.write(f"{display_name:<30} {ver:<15}{critical_flag}\n")
            except ImportError:
                f.write(f"{display_name:<30} {'未安装':<15}\n")
        
        if compatibility_issues:
            f.write("\n潜在的兼容性问题:\n")
            f.write("-" * 80 + "\n")
            for issue in compatibility_issues:
                f.write(f"- {issue}\n")
        else:
            f.write("\n✓ 未检测到明显的版本兼容性问题\n")
    
    print(f"\n✓ 版本报告已保存到: {report_filename}")
except Exception as e:
    print(f"\n✗ 无法保存版本报告: {e}")

print("\n" + "=" * 80)
print("检查完成！")
print("=" * 80)