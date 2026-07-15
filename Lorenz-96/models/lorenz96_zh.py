import numpy as np


def lorenz96_dynamics(x: np.ndarray, F: float) -> np.ndarray:
    """
    计算Lorenz-96系统的状态时间导数（time derivative of state）
    公式：ẋᵢ = (xᵢ₊₁ - xᵢ₋₂)xᵢ₋₁ - xᵢ + F

    参数:
        x: 形状(n,)，当前时刻的状态向量（state vector）
        F: 外部强迫项（external forcing term），标准值取8

    返回:
        dx: 形状(n,)，状态的时间导数
    """
    dx = (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    return dx


def rk4_step(f, x: np.ndarray, dt: float, *args) -> np.ndarray:
    """
    单步四阶龙格-库塔积分器（4th-order Runge-Kutta integrator, RK4）

    参数:
        f: 动力学函数（dynamics function），输入状态输出导数
        x: 当前状态向量
        dt: 积分步长（integration time step）
        *args: 传递给动力学函数的额外参数（此处为强迫项F）

    返回:
        x_next: 下一时刻的状态向量
    """
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next


def generate_lorenz96_data(
        n: int = 40,
        F: float = 8.0,
        dt: float = 0.05,
        T_total: int = 1000,
        burn_in_steps: int = 1000,
        sigma_w: float = 0.01,
        sigma_v: float = 0.1,
        seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成Lorenz-96系统的完整模拟数据集（simulation dataset）
    参数:
        n: 系统状态维度（state dimension）
        F: 外部强迫项
        dt: 采样时间步长（sampling time step）
        T_total: 总模拟时间步数（total simulation steps）
        burn_in_steps: 预积分步数（burn-in steps）
        sigma_w: 过程噪声标准差（process noise standard deviation）
        sigma_v: 测量噪声标准差（measurement noise standard deviation）
        seed: 随机种子（random seed），保证实验可复现性（reproducibility）

    返回:
        true_states: 真实状态轨迹（true state trajectory），形状(T_total, n)
        measurements: 测量值序列（measurements），形状(T_total, n)
        inputs: 控制输入序列（control inputs），Lorenz-96无控制，形状(T_total, 0)
    """
    np.random.seed(seed)

    # 1. 初始化状态：平衡态叠加微小扰动，触发混沌（chaos）
    x = F * np.ones(n) + 0.01 * np.random.randn(n)

    # 2. 预积分阶段：丢弃暂态过程（transient process），进入混沌吸引子（chaotic attractor）
    for _ in range(burn_in_steps):
        x = rk4_step(lorenz96_dynamics, x, dt, F)

    # 3. 正式生成数据
    true_states = []
    measurements = []
    inputs = []
    # C = np.eye(n)

    for _ in range(T_total):
        true_states.append(x.copy())

        # 全观测模型（full observation model）：y_t = C * x_t + v_t
        y = x + sigma_v * np.random.randn(n)
        measurements.append(y)

        # 无控制输入，保留空数组保持格式统一
        inputs.append(np.array([]))

        # 状态更新，叠加过程噪声（模拟模型误差 model error）
        x = rk4_step(lorenz96_dynamics, x, dt, F) + sigma_w * np.random.randn(n)

    true_states = np.array(true_states)
    measurements = np.array(measurements)
    inputs = np.array(inputs)

    return true_states, measurements, inputs