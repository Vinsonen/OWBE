import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# 导入 Lorenz-96 系统模型：统一版动力学、RK4积分器、数据生成函数
from models.lorenz96 import lorenz96_dynamics, rk4_step, generate_lorenz96_data
# 导入绘图工具：轨迹对比图、状态快照图、估计结果对比图
from plotting.plots import (
    plot_lorenz96_trajectories,
    plot_lorenz96_snapshot,
    plot_lorenz96_estimation
)


def full_batch_estimation_casadi(
    measurements: np.ndarray,
    inputs: np.ndarray,
    dt: float,
    sigma_w: float = 0.01,
    sigma_v: float = 0.1
) -> tuple[np.ndarray, dict]:
    """
    基于 CasADi + IPOPT 实现的全批量状态估计器
    严格对应指南 Problem Setting 中的公式 (3)(4)(5)，采用加性噪声消元后的等价形式

    代价函数（消去w后的无约束优化形式）：
    J = Σ_{j=0}^{T-1} ||x̂_{j+1} - f(x̂_j, u_j)||_Q²  +  Σ_{j=0}^{T} ||y_j - x̂_j||_R²

    参数:
        measurements: 测量值序列 y_{0:T}，形状 (总时间步数, 状态维度)，对应指南中的观测数据 d_{0:T}
        inputs: 控制输入序列 u_{0:T-1}，形状 (总时间步数,)，这里即时变强迫项 F(t)
        dt: 采样时间步长
        sigma_w: 过程噪声标准差，用于构造动力学项的加权矩阵 Q
        sigma_v: 测量噪声标准差，用于构造测量项的加权矩阵 R

    返回:
        x_hat: 最优状态估计轨迹 x̂_{0:T}，形状 (总时间步数, 状态维度)
        info: 求解器信息字典，包含收敛状态、最终代价值、迭代次数
    """
    # 获取总时间步数和状态维度
    T_total, n = measurements.shape

    # ========== 第1步：定义优化决策变量 ==========
    # X 是符号矩阵，每一列对应一个时刻的 n 维状态估计值
    # 共 T_total 个时刻，是整个优化问题要求解的未知量
    X = ca.MX.sym('X', n, T_total)

    # ========== 第2步：构造总代价函数 ==========
    J = 0.0  # 总代价初始化为0

    # 加权系数：噪声越小，权重越大（加权最小二乘的标准设定）
    q_weight = 1.0 / (sigma_w ** 2)  # Q 矩阵对角元：过程噪声项的权重
    r_weight = 1.0 / (sigma_v ** 2)  # R 矩阵对角元：测量残差项的权重

    # 遍历每一步状态转移（共 T_total-1 步）
    for j in range(T_total - 1):
        x_j = X[:, j]       # 第 j 时刻的状态估计值
        x_j1 = X[:, j+1]    # 第 j+1 时刻的状态估计值
        u_j = inputs[j]     # 第 j 步的控制输入（强迫项 F 值）

        # --- 过程噪声惩罚项：对应指南公式(5)的 ||ŵ_j||_Q² ---
        # 加性噪声下 ŵ_j = x̂_{j+1} - f(x̂_j, u_j)
        # 用 RK4 计算确定性状态转移 f(x_j, u_j)
        x_pred = rk4_step(lorenz96_dynamics, x_j, dt, u_j)
        # 动力学残差 = 实际估计状态 - 模型预测状态
        dyn_residual = x_j1 - x_pred
        # 加权平方和加入总代价
        J += q_weight * ca.dot(dyn_residual, dyn_residual)

        # --- 测量残差惩罚项：对应指南公式(5)的 ||y_j - h(x̂_j)||_R² ---
        # 全观测场景下观测函数 h(x)=x，直接计算测量值与估计值的偏差
        meas_residual = measurements[j] - x_j
        # 加权平方和加入总代价
        J += r_weight * ca.dot(meas_residual, meas_residual)

    # --- 终端时刻测量惩罚项：对应指南公式(5)的 ||y_T - h(x̂_T)||_G² ---
    # 最后一个时刻没有状态转移，只有测量约束，单独补充一项
    # 这里取终端权重 G = R，与中间时刻保持一致
    meas_res_last = measurements[-1] - X[:, -1]
    J += r_weight * ca.dot(meas_res_last, meas_res_last)

    # ========== 第3步：构建非线性规划(NLP)问题，调用IPOPT求解 ==========
    # 标准NLP格式：x 是展平的决策变量，f 是目标函数
    nlp = {
        'x': ca.vec(X),    # 将二维状态矩阵展平成一维向量，作为优化变量
        'f': J             # 待最小化的总代价函数
    }

    # IPOPT 求解器参数配置
    opts = {
        'ipopt.print_level': 0,   # 求解器输出等级，0为静默模式，不打印迭代细节
        'ipopt.max_iter': 500,    # 最大迭代次数，防止无限循环
        'print_time': 0,          # 不打印求解耗时
        'ipopt.tol': 1e-6         # 收敛精度阈值
    }
    # 创建求解器实例
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # 优化初始猜测值：全观测场景下直接用测量值作为初值，收敛快且稳定
    x0 = ca.vec(measurements.T)

    # 执行求解
    sol = solver(x0=x0)

    # ========== 第4步：提取并整理求解结果 ==========
    # 将展平的结果向量还原为 (状态维度, 时间步数) 的矩阵，再转置为 (时间步数, 状态维度)
    x_hat = sol['x'].full().reshape((n, T_total)).T

    # 整理求解信息
    info = {
        'success': solver.stats()['success'],    # 是否成功收敛
        'cost': float(sol['f']),                 # 最终收敛的代价值
        'iterations': solver.stats()['iter_count']  # 实际迭代次数
    }

    return x_hat, info


def run_full_batch_casadi():
    """
    全批量估计实验主入口
    完整流程：参数配置 → 生成仿真数据 → 执行全批量估计 → 结果打印 → 可视化对比
    """
    # ===================== 实验参数配置区 =====================
    n_state = 10          # 状态维度 n
    T_steps = 50          # 总时间步数 T
    F_forcing = 8.0       # 强迫项 F（标量=固定值，数组=时变）
    dt_step = 0.05        # 采样时间步长
    burn_in = 500         # 预积分步数（让系统进入混沌吸引子）
    sigma_process = 0.01  # 过程噪声标准差
    sigma_measure = 0.1   # 测量噪声标准差
    random_seed = 42      # 随机种子，保证实验可复现
    # ==========================================================

    # 1. 生成 Lorenz-96 仿真数据（真实状态、带噪测量值、控制输入序列）
    true_states, measurements, inputs = generate_lorenz96_data(
        n=n_state, F=F_forcing, dt=dt_step, T_total=T_steps,
        burn_in_steps=burn_in, sigma_w=sigma_process,
        sigma_v=sigma_measure, seed=random_seed
    )

    # 2. 执行全批量状态估计
    x_hat, info = full_batch_estimation_casadi(
        measurements=measurements, inputs=inputs,
        dt=dt_step, sigma_w=sigma_process, sigma_v=sigma_measure
    )

    # 3. 打印结果统计信息
    print(f"求解状态: {'成功 ✓' if info['success'] else '失败 ✗'}")
    print(f"最终代价值: {info['cost']:.4f}")
    print(f"迭代次数: {info['iterations']}")
    # 计算均方根误差 RMSE，量化估计效果
    rmse_meas = np.sqrt(np.mean((true_states - measurements) ** 2))
    rmse_est = np.sqrt(np.mean((true_states - x_hat) ** 2))
    print(f"原始测量值 RMSE: {rmse_meas:.6f}")
    print(f"全批量估计 RMSE: {rmse_est:.6f}")
    print(f"相对精度提升: {(1 - rmse_est/rmse_meas)*100:.1f}%")

    # 4. 可视化：真实状态、测量值、估计值三者对比
    plot_lorenz96_estimation(true_states, measurements, x_hat, n_state, F_forcing)
    plt.show()


if __name__ == "__main__":
    # 直接运行本文件即可执行完整的全批量估计实验
    run_full_batch_casadi()