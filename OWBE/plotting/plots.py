import numpy as np
import matplotlib.pyplot as plt

# Set English font configuration
plt.rcParams['font.size'] = 12


def plot_lorenz96_trajectories(
    true_states: np.ndarray,
    measurements: np.ndarray,
    n: int,
    F: float | np.ndarray
) -> None:
    """
    Plot comparison of true trajectories and measurements for the first 3 state variables of Lorenz-96

    Parameters:
        true_states: True state trajectory
        measurements: Measurement sequence
        n: State dimension
        F: Forcing term parameter
    """
    plt.figure(figsize=(12, 8))
    max_range = 3

    # F text
    if np.isscalar(F):
        f_display = f"{F}"
    else:
        f_display = f"{F.min():.1f}~{F.max():.1f} (time-varying $u_F$)"  # Label when array

    for i in range(max_range):
        plt.subplot(max_range, 1, i+1)
        plt.plot(true_states[:, i], 'b-', linewidth=1.5, label=f'True state $x_{i+1}$')
        plt.plot(measurements[:, i], 'r.', markersize=2, alpha=0.6, label=f'Measurement $y_{i+1}$')
        plt.ylabel(f'$x_{i+1}$')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

    plt.xlabel('Time step t')
    plt.suptitle(f'Lorenz-96 System - First {max_range} State Variable Trajectories (n={n}, F={f_display})', y=0.95)
    plt.tight_layout()
    # plt.show()


def plot_lorenz96_snapshot(
    true_states: np.ndarray,
    time_step: int = 500
) -> None:
    """
    Plot state space snapshot at a single time step to reflect the spatial structure of the system

    Parameters:
        true_states: True state trajectory
        time_step: Index of the selected time step
    """
    plt.figure(figsize=(10, 5))
    plt.plot(true_states[time_step, :], 'bo-', linewidth=1.5, markersize=4)
    plt.xlabel('State variable index i')
    plt.ylabel('$x_i$')
    plt.title(f'State Snapshot at Time Step t={time_step}')
    plt.grid(True, alpha=0.3)
    # plt.show()


def plot_lorenz96_estimation(
    true_states: np.ndarray,
    measurements: np.ndarray,
    estimated_states: np.ndarray,
    n: int,
    F: float | np.ndarray
) -> None:
    """
    Plot comparison of true state, measurements and estimated state
    for the first 3 state variables of Lorenz-96
    """
    plt.figure(figsize=(12, 8))
    max_range = 3

    # Format F display text
    if np.isscalar(F):
        f_display = f"{F}"
    else:
        f_display = f"{F.min():.1f}~{F.max():.1f} (time-varying $u_F$)"

    for i in range(max_range):
        plt.subplot(max_range, 1, i+1)
        plt.plot(true_states[:, i], 'b-', linewidth=1.5, label=f'True $x_{i+1}$', alpha=0.8)
        plt.plot(measurements[:, i], 'r.', markersize=3, label=f'Measurement $y_{i+1}$', alpha=0.5)
        plt.plot(estimated_states[:, i], 'g--', linewidth=1.2, label=f'Estimated $\\hat{{x}}_{i+1}$', alpha=0.8)
        plt.ylabel(f'$x_{i+1}$')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

    plt.xlabel('Time step t')
    plt.suptitle(f'Lorenz-96 Full-Batch Estimation (n={n}, F={f_display})', y=0.95)
    plt.tight_layout()
