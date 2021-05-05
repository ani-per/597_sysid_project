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


def plot_data_bulk(fp):
    max_sim = 10
    start_index = 0
    airsim_data = np.load(Path.cwd() / "data" / fp)
    t = airsim_data["t"].squeeze()[:max_sim, start_index:]
    U = airsim_data["U"][:max_sim, :, start_index:]
    Z = airsim_data["Z"][:max_sim, :, start_index:]

    # Simulation dimensions
    n_sim = t.shape[0]
    r = U.shape[1]  # Number of inputs
    m = Z.shape[1]  # Number of measurements
    t_max = t[0, -1]  # Total simulation time

    wrapper = textwrap.TextWrapper(width=20)
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
    fs = 7.5  # Font size
    sim_fs = 4  # Sim. label font size
    fig, axs = plt.subplots(
        (r + m) // 2, 2, sharex="col", constrained_layout=True
    )  # type:figure.Figure
    fig.suptitle(f"Inputs / Observations", fontweight="bold")

    for i in range(r):
        for j in range(n_sim):
            axs[0][i].plot(t[j, :], U[j, i, :])
            axs[0][i].annotate(
                f"{j + 1}",
                xy=(t[j, -1], U[j, i, -1]),
                xytext=(0.75, 0),
                textcoords="offset points",
                fontsize=sim_fs,
            )
        plt.setp(axs[0][i], xlim=[0, t_max])
        axs[0][i].set_ylabel(f"{input_labels[i]}", fontsize=fs)
        axs[0][i].tick_params(axis="y", labelsize=fs)

    for i, obs_ax in zip(range(m), axs[1:].flat):
        for j in range(n_sim):
            obs_ax.plot(t[j, :], Z[j, i, :])
            obs_ax.annotate(
                f"{j + 1}",
                xy=(t[j, -1], Z[j, i, -1]),
                xytext=(0.75, 0),
                textcoords="offset points",
                fontsize=sim_fs,
            )
        plt.setp(obs_ax, xlim=[0, t_max])
        obs_ax.set_ylabel(f"{obs_labels[i]}", fontsize=fs)
        obs_ax.tick_params(axis="y", labelsize=fs)
        if i > (m - 2):
            plt.setp(obs_ax, xlabel=f"Time")
    fig.savefig(Path.cwd() / "figs" / f"input_obs_bulk.pdf", bbox_inches="tight")


plot_data_bulk("data_bulk.npz")