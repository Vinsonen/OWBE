import numpy as np
from models.lorenz96 import generate_lorenz96_data
from plotting.plots import plot_lorenz96_trajectories, plot_lorenz96_snapshot
import matplotlib.pyplot as plt


def run_lorenz96_simulation():
    """
    Main function for Lorenz-96 simulation experiment
    Process: Parameter configuration → Data generation → Information output → Result visualization
    """
    # ===================== Experiment Parameter Configuration Area =====================
    # System core parameters
    n_state = 20  # State dimension   20 40 80 100
    T_steps = 1000  # Total number of simulation time steps
    # External forcing term ,const or array,  # 8.0 | np.concatenate([np.full(500, 8.0), np.full(500, 10.0)])
    F_forcing = np.concatenate([np.full(T_steps // 2, 8.0), np.full(T_steps // 2, 10.0)])
    dt_step = 0.05  # Sampling time step
    burn_in = 1000  # Burn-in steps (pre-integration steps)

    # Noise parameters
    sigma_process = 0.01  # Process noise standard deviation
    sigma_measure = 0.1  # Measurement noise standard deviation

    # Random seed
    random_seed = 42
    # ============================================================

    # 1. Print experiment configuration
    print("=" * 50)
    print("=== Lorenz-96 System Simulation Experiment ===")
    print(f"State dimension n: {n_state}")
    print(f"Forcing term F: {F_forcing}")
    print(f"Sampling time dt: {dt_step}")
    print(f"Total time steps: {T_steps}")
    print(f"Random seed: {random_seed}")
    print("=" * 50)

    # 2. Call model module to generate data
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

    # 3. Output basic data information
    print(f"True state trajectory shape: {true_states.shape}  (time steps × state dimension)")
    print(f"Measurement sequence shape:   {measurements.shape}  (time steps × measurement dimension)")

    # 4. Call visualization functions
    plot_lorenz96_trajectories(true_states, measurements, n_state, F_forcing)
    plot_lorenz96_snapshot(true_states, time_step=T_steps // 2)
    plt.show()


if __name__ == "__main__":
    run_lorenz96_simulation()
