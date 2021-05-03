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


def plot_data_bulk(train_file, test_file):
    start_index = 0
    train_data = np.load(Path.cwd() / "data" / train_file)
    t_train = train_data["t"].squeeze()[:, start_index:]
    U_train = train_data["U"][:, :, start_index:]
    Z_train = train_data["Z"][:, :, start_index:]

    test_data = np.load(Path.cwd() / "data" / test_file)
    t_test = test_data["t"].squeeze()[:, start_index:]
    U_test = test_data["U"][:, :, start_index:]
    Z_test = test_data["Z"][:, :, start_index:]

    # Simulation dimensions
    n_sim = t_train.shape[0]
    r = U_train.shape[1]  # Number of inputs
    m = Z_train.shape[1]  # Number of measurements
    t_max = t_train[0, -1]  # Total simulation time

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
    fs = 4  # Font size
    fig, axs = plt.subplots(
        (r + m) // 2, 2, sharex="col", constrained_layout=True
    )  # type:figure.Figure
    fig.suptitle(f"Inputs / Observations", fontweight="bold")

    for i in range(r):
        for j in range(n_sim):
            axs[0][i].plot(t_train[j, :], U_train[j, i, :])
            axs[0][i].annotate(
                f"{j + 1}",
                xy=(t_train[j, -1], U_train[j, i, -1]),
                xytext=(0.75, 0),
                textcoords="offset points",
                fontsize=fs,
            )
            # axs[0][i].plot(t_test[j, :], U_test[j, i, :], "o", ms=ms, mfc="None")
        plt.setp(axs[0][i], ylabel=f"{input_labels[i]}", xlim=[0, t_max])

    for i, obs_ax in zip(range(m), axs[1:].flat):
        for j in range(n_sim):
            obs_ax.plot(t_train[j, :], Z_train[j, i, :])
            obs_ax.annotate(
                f"{j + 1}",
                xy=(t_train[j, -1], Z_train[j, i, -1]),
                xytext=(0.75, 0),
                textcoords="offset points",
                fontsize=fs,
            )
            # obs_ax.plot(t_test[j, :], Z_test[j, i, :], "o", ms=ms, mfc="None")
        plt.setp(obs_ax, ylabel=f"{obs_labels[i]}", xlim=[0, t_max])
        if i > (m - 2):
            plt.setp(obs_ax, xlabel=f"Time")
    # fig.legend(labels=["Train", "Test"], bbox_to_anchor=(1, 0.5), loc=6)
    fig.savefig(Path.cwd() / "figs" / f"input_obs_bulk.pdf", bbox_inches="tight")


plot_data_bulk("data_train_bulk.npz", "data_test_bulk.npz")