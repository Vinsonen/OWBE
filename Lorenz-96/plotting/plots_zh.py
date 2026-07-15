import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_lorenz96_trajectories(
    true_states: np.ndarray,
    measurements: np.ndarray,
    n: int,
    F: float
) -> None:
    """
    绘制Lorenz-96前3个状态变量的真实轨迹与测量值对比图

    参数:
        true_states: 真实状态轨迹
        measurements: 测量值序列
        n: 状态维度
        F: 强迫项参数
    """
    plt.figure(figsize=(12, 8))

    max_range = 3
    for i in range(max_range):
        plt.subplot(max_range, 1, i+1)
        plt.plot(true_states[:, i], 'b-', linewidth=1.5, label=f'真实状态 true state $x_{i+1}$')
        plt.plot(measurements[:, i], 'r.', markersize=2, alpha=0.6, label=f'测量值 measurement $y_{i+1}$')
        plt.ylabel(f'$x_{i+1}$')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

    plt.xlabel('时间步 t (time step)')
    plt.suptitle(f'Lorenz-96系统前{max_range}个状态变量轨迹 (n={n}, F={F})', y=0.95)
    plt.tight_layout()
    # plt.show()


def plot_lorenz96_snapshot(
    true_states: np.ndarray,
    time_step: int = 500
) -> None:
    """
    绘制单时刻的状态空间快照（snapshot），体现系统的空间结构

    参数:
        true_states: 真实状态轨迹
        time_step: 选择的时刻索引
    """
    plt.figure(figsize=(10, 5))
    plt.plot(true_states[time_step, :], 'bo-', linewidth=1.5, markersize=4)
    plt.xlabel('状态变量索引 i (state index)')
    plt.ylabel('$x_i$')
    plt.title(f't={time_step} 时刻的状态快照')
    plt.grid(True, alpha=0.3)
    # plt.show()