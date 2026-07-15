import numpy as np
from models.lorenz96_zh import generate_lorenz96_data
from plotting.plots_zh import plot_lorenz96_trajectories, plot_lorenz96_snapshot
import matplotlib.pyplot as plt


def run_lorenz96_simulation():
    """
    Lorenz-96模拟实验主函数
    完成：参数配置 → 数据生成 → 信息输出 → 结果可视化
    """
    # ===================== 实验参数集中配置区 =====================
    # 系统核心参数（system core parameters）
    n_state = 40  # 状态维度（state dimension）
    F_forcing = 8.0  # 外部强迫项（external forcing term）
    dt_step = 0.05  # 采样时间步长（sampling time step）
    T_steps = 1000  # 总模拟时间步数（total simulation steps）
    burn_in = 1000  # 预积分步数（burn-in steps）

    # 噪声参数（noise parameters）
    sigma_process = 0.01  # 过程噪声标准差（process noise std）
    sigma_measure = 0.1  # 测量噪声标准差（measurement noise std）

    # 随机种子（random seed）
    random_seed = 42
    # ============================================================

    # 1. 打印实验配置
    print("=" * 50)
    print("=== Lorenz-96 系统模拟实验 ===")
    print(f"状态维度 n: {n_state}")
    print(f"强迫项 F: {F_forcing}")
    print(f"采样时间 dt: {dt_step}")
    print(f"总时间步数: {T_steps}")
    print(f"随机种子: {random_seed}")
    print("=" * 50)

    # 2. 调用模型模块生成数据
    true_states, measurements, inputs = generate_lorenz96_data(
        n=n_state,
        F=F_forcing,
        dt=dt_step,
        T_total=T_steps,
        burn_in_steps=burn_in,
        sigma_w=sigma_process,
        sigma_v=sigma_measure,
        seed=random_seed
    )

    # 3. 输出数据基本信息
    print(f"真实状态轨迹形状: {true_states.shape}  (时间步 × 状态维度)")
    print(f"测量值序列形状:   {measurements.shape}  (时间步 × 测量维度)")

    # 4. 调用可视化函数
    plot_lorenz96_trajectories(true_states, measurements, n_state, F_forcing)
    plot_lorenz96_snapshot(true_states, time_step=500)
    plt.show()


if __name__ == "__main__":
    run_lorenz96_simulation()