# Logistic packages
import itertools as it  # Readable nested for loops
import typing  # Argument / output type checking
from pathlib import Path  # Filepaths
import time  # Timing/sleeping
import json

# Numeric packages
import numpy as np  # N-dim arrays + math
import scipy.linalg as spla  # Complex linear algebra
import scipy.signal as spsg  # Signal processing
import scipy.io as spio  # Read/write numeric data

# Plotting packages
import matplotlib.figure as figure  # Figure documentation
import matplotlib.pyplot as plt  # Plots
import pandas as pd  # Dataframes

# Other packages
import airsim  # Airsim APIs

# Logistics
figs_dir = Path.cwd() / "figs"
data_dir = Path.cwd() / "data"
start_time = time.time()


def reset_client(client: airsim.client.CarClient):
    client.reset()
    client = airsim.CarClient()
    client.enableApiControl(True)
    time.sleep(1.5)
    pass


def offset_client(client: airsim.client.CarClient, offset: np.array):
    pose = client.simGetVehiclePose()
    pose.position.x_val -= offset[0]
    pose.position.y_val -= offset[1]
    client.simSetVehiclePose(pose, True)
    time.sleep(2)
    pass


def collect_data(
    client: airsim.client.CarClient,
    dt: float,
    t_max: float,
    rng: np.random.Generator,
    offset: np.array = np.array([0, 0]),
    traintest: int = 0,
) -> typing.Tuple[np.array, np.array, np.array, pd.DataFrame]:
    reset_client(client)
    offset_client(client, offset)

    car_controls = airsim.CarControls()
    car_state = client.getCarState()
    start_time = car_state.timestamp

    i_max = int(np.ceil(t_max / dt))
    t = np.zeros([i_max])
    U = np.zeros([2, i_max])
    Z = np.zeros([6, i_max])

    driving_states = []
    throttle_mag = rng.uniform(0.2, 0.35)
    steering_mag = rng.uniform(-0.15, 0.15)
    steering_freq = rng.uniform(0, 1)

    for i in range(i_max):

        # car_controls.throttle = throttle_mag
        # car_controls.steering = steering_mag
        car_controls.throttle = throttle_mag
        car_controls.steering = steering_mag * (
            np.sin(2 * np.pi * steering_freq * dt * i)
        )
        client.setCarControls(car_controls)

        current_state = client.getCarState()
        current_time = np.around((current_state.timestamp - start_time) / 1e9, 3)
        current_controls = client.getCarControls()

        cyaw = np.cos(
            airsim.to_eularian_angles(current_state.kinematics_estimated.orientation)[2]
        )
        syaw = np.sin(
            airsim.to_eularian_angles(current_state.kinematics_estimated.orientation)[2]
        )
        dcm_nb = np.array([[cyaw, syaw], [-syaw, cyaw]])

        t[i] = dt * i
        U[:, i] = np.array([current_controls.throttle, current_controls.steering])
        Z[:2, i] = (
            current_state.kinematics_estimated.position.to_numpy_array()[:2] + offset
        )
        Z[2:4, i] = (
            dcm_nb
            @ current_state.kinematics_estimated.linear_velocity.to_numpy_array()[:2]
        )
        Z[4, i] = np.array(
            airsim.to_eularian_angles(current_state.kinematics_estimated.orientation)
        )[2]
        Z[5, i] = (
            current_state.kinematics_estimated.angular_velocity.to_numpy_array()
        )[2]

        current_state_dict = {
            **{
                "t": dt * i,
                "throttle": current_controls.throttle,
                "steering": current_controls.steering,
            },
            **dict(
                zip(
                    ["x", "y"],
                    current_state.kinematics_estimated.position.to_numpy_array()[:2]
                    + offset,
                )
            ),
            **dict(
                zip(
                    ["v_x", "v_y"],
                    current_state.kinematics_estimated.linear_velocity.to_numpy_array()[
                        :2
                    ],
                )
            ),
            **dict(
                zip(
                    ["U", "V"],
                    dcm_nb
                    @ current_state.kinematics_estimated.linear_velocity.to_numpy_array()[
                        :2
                    ],
                )
            ),
            **dict(
                zip(
                    ["roll", "pitch", "yaw"],
                    np.degrees(
                        np.array(
                            airsim.to_eularian_angles(
                                current_state.kinematics_estimated.orientation
                            )
                        )[[1, 0, 2]]
                    ),
                )
            ),
            **dict(
                zip(
                    ["p", "q", "r"],
                    np.degrees(
                        current_state.kinematics_estimated.angular_velocity.to_numpy_array()
                    ),
                )
            ),
        }
        driving_states.append(current_state_dict)

        time.sleep(dt)
    driving_df = pd.DataFrame(driving_states)

    car_controls.throttle = 0
    car_controls.steering = 0
    client.setCarControls(car_controls)
    return t, U, Z, driving_df


# Connect to the AirSim simulator
client = airsim.CarClient()
offset = np.array([40.5, -7.5])
client.ping()
client.enableApiControl(True)

dt = 0.1
t_max = 30
i_max = int(np.ceil(t_max / dt))
n_sim = 500

t = np.zeros([n_sim, 1, i_max])
U = np.zeros([n_sim, 2, i_max])
Z = np.zeros([n_sim, 6, i_max])

rng = np.random.default_rng(seed=4)
for sim in range(n_sim):
    print(f"Sim: {sim}/{n_sim}")
    (t[sim, :, :], U[sim, :, :], Z[sim, :, :], driving_df,) = collect_data(
        client=client,
        rng=rng,
        dt=dt,
        t_max=t_max,
        offset=offset,
        traintest=0,
    )
    # driving_df.to_csv(data_dir / f"data.csv", index=False)

np.savez(data_dir / f"data_bulk.npz", t=t, U=U, Z=Z)
spio.savemat(data_dir / f"data_bulk.mat", {"t": t, "U": U, "Z": Z})

reset_client(client)
print(f"Elapsed time: {(time.time() - start_time)/60:0.4f} minutes.")