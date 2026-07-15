import numpy as np
import casadi as ca


def lorenz96_dynamics(x: np.ndarray, F: float) -> np.ndarray:
    """
    Calculate the time derivative of the Lorenz-96 system state
    Formula: ẋᵢ = (xᵢ₊₁ - xᵢ₋₂)xᵢ₋₁ - xᵢ + F

    Parameters:
        x: Shape (n,), state vector at current time step
        F: External forcing term, standard value is 8

    Returns:
        dx: Shape (n,), time derivative of the state
    """
    x_left1 = ca.vertcat(x[1:], x[0])    # equivalent to np.roll(x, -1)
    x_right1 = ca.vertcat(x[-1], x[:-1]) # equivalent to np.roll(x, 1)
    x_right2 = ca.vertcat(x[-2:], x[:-2])# equivalent to np.roll(x, 2)

    dx = (x_left1 - x_right2) * x_right1 - x + F
    return dx


def rk4_step(f, x: np.ndarray, dt: float, *args) -> np.ndarray:
    """
    Single-step 4th-order Runge-Kutta integrator (RK4)

    Parameters:
        f: Dynamics function, input state and output derivative
        x: Current state vector
        dt: Integration time step
        *args: Additional parameters passed to the dynamics function (here is forcing term F)

    Returns:
        x_next: State vector at next time step
    """
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next


def generate_lorenz96_data(
        n: int = 40,
        F: float | np.ndarray = 8.0,
        dt: float = 0.05,
        T_total: int = 1000,
        burn_in_steps: int = 1000,
        sigma_w: float = 0.01,
        sigma_v: float = 0.1,
        seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate complete simulation dataset for Lorenz-96 system

    Parameters:
        n: System state dimension
        F: External forcing term
        dt: Sampling time step
        T_total: Total number of simulation time steps
        burn_in_steps: Burn-in steps (pre-integration steps)
        sigma_w: Process noise standard deviation
        sigma_v: Measurement noise standard deviation
        seed: Random seed to ensure experiment reproducibility

    Returns:
        true_states: True state trajectory, shape (T_total, n)
        measurements: Measurement sequence, shape (T_total, n)
        inputs: Control input sequence (no control for Lorenz-96), shape (T_total, 0)
    """
    np.random.seed(seed)
    if np.isscalar(F):
        u_sequence = np.full(T_total, F)
        burn_in_F = F
    else:
        u_sequence = np.asarray(F)
        if u_sequence.shape != (T_total,):
            raise ValueError(f"The length of F must equal to T_total={T_total}")
        burn_in_F = u_sequence[0]

    # Initialization + Pre-integration
    x = burn_in_F * np.ones(n) + 0.01 * np.random.randn(n)
    for _ in range(burn_in_steps):
        x = rk4_step(lorenz96_dynamics, x, dt, burn_in_F)

    # Formal data generation
    true_states = []
    measurements = []
    # C =

    for t in range(T_total):
        true_states.append(np.array(x).flatten())

        # Full observation model: y_t = x_t + v_t
        y = np.array(x).flatten() + sigma_v * np.random.randn(n)
        measurements.append(y)

        # State update with process noise (simulate model error)
        current_F = u_sequence[t]
        x = rk4_step(lorenz96_dynamics, x, dt, current_F) + sigma_w * np.random.randn(n)

    true_states = np.array(true_states)
    measurements = np.array(measurements)
    inputs = u_sequence
    # print(measurements)

    return true_states, measurements, inputs
