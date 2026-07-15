import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from models.lorenz96 import lorenz96_dynamics, rk4_step, generate_lorenz96_data
from plotting.plots import (
    plot_lorenz96_trajectories,
    plot_lorenz96_snapshot,
    plot_lorenz96_estimation
)


def full_batch_estimation(
    measurements: np.ndarray,
    inputs: np.ndarray,
    dt: float,
    sigma_w: float = 0.01,
    sigma_v: float = 0.1
) -> tuple[np.ndarray, dict]:
    """
    Cost function:
    J = Σ_{j=0}^{N-1} ( ||ŵ_j||_Q² + ||y_j - h(x̂_j, u_j)||_R² ) + ||y_N - h(x̂_N, u_N)||_R²
    J = Σ_{j=0}^{T-1} ||x̂_{j+1} - f(x̂_j, u_j)||_Q²  +  Σ_{j=0}^{T} ||y_j - x̂_j||_R²

    Parameters:
        measurements: Measurement sequence y_{0:T}, shape (T_total, n)
        inputs: Control input sequence u_{0:T-1}, shape (T_total,)
        dt: Sampling time step
        sigma_w: Process noise std, used to construct weighting matrix Q
        sigma_v: Measurement noise std, used to construct weighting matrix R

    Returns:
        x_hat: Optimal state estimation trajectory, shape (T_total, n)
        info: Solver information dict (success, cost, iterations)
    """
    T_total, n = measurements.shape

    #  1. Define decision variables: state at all time steps =========
    X = ca.MX.sym('X', n, T_total)  # each column = state at one time step

    # 2. Build cost function ==========
    J = 0.0
    q_weight = 1.0 / (sigma_w ** 2)  # Q matrix
    r_weight = 1.0 / (sigma_v ** 2)  # Diag R matrix

    # T_total-1 steps==
    for j in range(T_total - 1):
        x_j = X[:, j]
        x_j1 = X[:, j+1]    # X_{j+1}
        u_j = inputs[j]

        # --- Process noise term: ||ŵ_j||_Q²
        # --with noise: ŵ_j = x̂_{j+1} - f(x̂_j, u_j)
        x_pre = rk4_step(lorenz96_dynamics, x_j, dt, u_j)  # f(x_j, u_j)
        dyn_residual = x_j1 - x_pre
        J += q_weight * ca.dot(dyn_residual, dyn_residual)

        # --- Measurement residual term: ||y_j - h(x̂_j)||_R²  ---
        meas_residual = measurements[j] - x_j
        J += r_weight * ca.dot(meas_residual, meas_residual)

    # --- Terminal measurement term: ||y_T - h(x̂_T)||_G² ---
    meas_res_last = measurements[-1] - X[:, -1]
    J += r_weight * ca.dot(meas_res_last, meas_res_last)

    # 3. Build NLP problem and solve with IPOPT ==========
    nlp = {'x': ca.vec(X), 'f': J}
    opts = {
        'ipopt.print_level': 0,
        'ipopt.max_iter': 500,
        'print_time': 0,
        'ipopt.tol': 1e-6
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Initial guess: use measurements directly
    x0 = ca.vec(measurements.T)
    sol = solver(x0=x0)

    # 4. Extract results ==========
    x_hat = sol['x'].full().reshape((n, T_total), order='F').T
    info = {
        'success': solver.stats()['success'],
        'cost': float(sol['f']),
        'iterations': solver.stats()['iter_count']
    }
    return x_hat, info


def run_full_batch_estimation():
    """Main entry for full batch estimation experiment"""
    # Experiment Parameters =====================
    n_state = 10
    T_steps = 150
    F_forcing = 8.0
    dt_step = 0.05
    burn_in = 500
    sigma_process = 0.01
    sigma_measure = 0.1
    random_seed = 42

    # 1. Generate data Lorenz96
    true_states, measurements, inputs = generate_lorenz96_data(
        n=n_state, F=F_forcing, dt=dt_step, T_total=T_steps,
        burn_in_steps=burn_in, sigma_w=sigma_process,
        sigma_v=sigma_measure, seed=random_seed
    )

    # 2. Run full batch estimation
    x_hat, info = full_batch_estimation(
        measurements=measurements, inputs=inputs,
        dt=dt_step, sigma_w=sigma_process, sigma_v=sigma_measure
    )

    # 3. Print results
    print(f"Solver status: {'Success' if info['success'] else 'Failed'}")
    print(f"Final cost: {info['cost']:.4f}")
    print(f"Iterations: {info['iterations']}")
    # RMSE
    rmse_meas = np.sqrt(np.mean((true_states - measurements) ** 2))
    rmse_est = np.sqrt(np.mean((true_states - x_hat) ** 2))
    print(f"Measurement RMSE: {rmse_meas:.6f}")
    print(f"Estimation RMSE: {rmse_est:.6f}")
    print(f"Accuracy improvement: {(1 - rmse_est/rmse_meas)*100:.1f}%")

    # 4. Visualization  plotting
    plot_lorenz96_estimation(true_states, measurements, x_hat, n_state, F_forcing)
    plt.show()


if __name__ == "__main__":
    run_full_batch_estimation()
