import numpy as np


def lorenz96_dynamics(x, F):
    """
    计算每个状态的变化速度（核心公式）
    对应公式：ẋᵢ = (xᵢ₊₁ - xᵢ₋₂) * xᵢ₋₁ - xᵢ + F
    np.roll 自动处理首尾相连的环形边界
    """
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def rk4_step(dynamics_func, x, dt, F):
    """
    单步四阶龙格-库塔（RK4）积分
    作用：知道当前状态x和变化速度，算出dt时间之后的新状态
    这是连续公式转离散计算的标准方法
    """
    k1 = dynamics_func(x, F)
    k2 = dynamics_func(x + 0.5 * dt * k1, F)
    k3 = dynamics_func(x + 0.5 * dt * k2, F)
    k4 = dynamics_func(x + dt * k3, F)
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_lorenz96(n=40, F=8.0, dt=0.05, total_steps=1000, seed=42):
    """
    生成完整的Lorenz-96轨迹（纯确定性，无任何噪声）
    参数：
        n: 状态变量的个数
        F: 外部强迫项（固定8）
        dt: 每一步的时间间隔
        total_steps: 总共生成多少步数据
        seed: 随机种子，保证初始扰动固定，结果可复现
    返回：
        trajectory: 形状为 (total_steps, n) 的轨迹数组
    """
    np.random.seed(seed)

    # 1. 初始状态：所有变量都等于F（这是系统的平衡点）
    # 加一丢丢随机扰动，让系统离开平衡点，触发混沌
    x = F * np.ones(n) + 0.01 * np.random.randn(n)

    # 2. 预积分（预热）：先跑1000步，丢掉过渡过程
    # 原因：初始状态不在真正的混沌轨道上，前面是“启动阶段”，不算数
    for _ in range(1000):
        x = rk4_step(lorenz96_dynamics, x, dt, F)

    # 3. 正式保存轨迹
    trajectory = []
    for _ in range(total_steps):
        trajectory.append(x.copy())  # 保存当前时刻的状态
        x = rk4_step(lorenz96_dynamics, x, dt, F)  # 往前推进一步

    return np.array(trajectory)