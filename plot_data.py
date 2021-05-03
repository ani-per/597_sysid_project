# Logistic packages
import itertools as it  # Readable nested for loops
import typing  # Argument / output type checking
import warnings  # Ignore user warnings
from pathlib import Path  # Filepaths

# Numeric packages
import numpy as np  # N-dim arrays + math
import scipy.linalg as spla  # Complex linear algebra
import scipy.signal as spsg  # Signal processing
import sympy  # Symbolic math + pretty printing

# Plotting packages
import matplotlib.figure as figure  # Figure documentation
import matplotlib.pyplot as plt  # Plots

# Other packages
import control
import textwrap


def plot_data(train_file, test_file):
    start_index = 0
    train_data = np.load(Path.cwd() / "data" / train_file)
    t_train = train_data["t"].squeeze()[start_index:]
    U_train = train_data["U"][:, start_index:]
    Z_train = train_data["Z"][:, start_index:]

    test_data = np.load(Path.cwd() / "data" / test_file)
    t_test = test_data["t"].squeeze()[start_index:]
    U_test = test_data["U"][:, start_index:]
    Z_test = test_data["Z"][:, start_index:]

    # Simulation dimensions
    r = U_train.shape[0]  # Number of inputs
    m = Z_train.shape[0]  # Number of measurements
    t_max = t_train[-1]  # Total simulation time

    wrapper = textwrap.TextWrapper(width=10)
    input_labels = [wrapper.fill(x) for x in ["Throttle", "$\delta$ (Steering Angle)"]]
    obs_labels = [
        wrapper.fill(x)
        for x in [
            "$x$ (Northwards Position)",
            "$y$ (Eastwards Position)",
            "$v_{x}$ (Body-Frame Forward Velocity)",
            "$v_{y}$ (Body-Frame Eastward Velocity)",
            "$\psi$ (Yaw Angle)",
            "$r$ (Yaw Rate)",
        ]
    ]
    # Observation plots
    ms = 0.5  # Marker size
    fig, axs = plt.subplots(
        (r + m) // 2, 2, sharex="col", constrained_layout=True
    )  # type:figure.Figure
    fig.suptitle(f"Inputs / Observations", fontweight="bold")

    for i in range(r):
        axs[0][i].plot(t_train, U_train[i, :])
        axs[0][i].plot(t_train, U_test[i, :], "o", ms=ms, mfc="None")
        plt.setp(axs[0][i], ylabel=f"{input_labels[i]}", xlim=[0, t_max])

    for i, obs_ax in zip(range(m), axs[1:].flat):
        obs_ax.plot(t_train, Z_train[i, :])
        obs_ax.plot(t_train, Z_test[i, :], "o", ms=ms, mfc="None")
        plt.setp(obs_ax, ylabel=f"{obs_labels[i]}", xlim=[0, t_max])
        if i > (m - 2):
            plt.setp(obs_ax, xlabel=f"Time")
    fig.legend(labels=["Train", "Test"], bbox_to_anchor=(1, 0.5), loc=6)
    fig.savefig(Path.cwd() / "figs" / f"input_obs.pdf", bbox_inches="tight")


plot_data("data_train.npz", "data_test.npz")