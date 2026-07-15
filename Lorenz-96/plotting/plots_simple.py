import matplotlib.pyplot as plt


def plot_lorenz96_trajectory(trajectory, n, F):
    """画出前3个状态变量随时间变化的轨迹"""
    plt.figure(figsize=(10, 6))

    # 只画前3个，太多了看不清
    range_max = 10
    for i in range(range_max):
        plt.plot(trajectory[:, i], linewidth=1.2, label=f'x_{i + 1}')

    plt.xlabel('时间步')
    plt.ylabel('状态值')
    plt.title(f'Lorenz-96 系统前{range_max}个状态轨迹 (n={n}, F={F})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()