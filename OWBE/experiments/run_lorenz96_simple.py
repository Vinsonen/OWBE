from models.lorenz96_simple import simulate_lorenz96
from plotting.plots_simple import plot_lorenz96_trajectory


if __name__ == "__main__":
    # ========== 所有参数都在这里改 ==========
    n = 40          # 状态变量个数
    F = 8.0         # 强迫项（固定8，混沌状态）
    dt = 0.05       # 时间步长
    total_steps = 1000  # 总模拟步数
    # =======================================

    # 运行模拟，得到完整轨迹
    trajectory = simulate_lorenz96(
        n=n,
        F=F,
        dt=dt,
        total_steps=total_steps
    )

    # 输出基本信息
    print(f"模拟完成！")
    print(f"轨迹形状：{trajectory.shape}  (时间步数 × 状态维度)")
    print(f"状态值范围：[{trajectory.min():.2f}, {trajectory.max():.2f}]")

    # 画图
    plot_lorenz96_trajectory(trajectory, n, F)