# Import necessary packages
from pathlib import Path  # Filepaths
import typing  # Argument / output type checking
import numpy as np  # N-dim arrays + math
import scipy.linalg as spla  # Complex linear algebra
import pandas as pd  # Dataframes
import airsim  # Airsim APIs
import time  # Timing/sleeping

# Logistics
figs_dir = Path.cwd() / "figs"
data_dir = Path.cwd() / "data"


def reset_client(client: airsim.client.CarClient):
    client.reset()
    client = airsim.CarClient()
    client.enableApiControl(True)
    time.sleep(2)
    pass


def offset_client(client: airsim.client.CarClient, offset: np.array):
    pose = client.simGetVehiclePose()
    pose.position.x_val -= offset[0]
    pose.position.y_val -= offset[1]
    client.simSetVehiclePose(pose, True)
    time.sleep(3)
    pass


def collect_data(
    client: airsim.client.CarClient,
    rng: np.random.Generator,
    dt: float,
    t_max: float,
    offset,
) -> typing.Tuple[np.array, np.array, np.array, pd.DataFrame]:

    car_controls = airsim.CarControls()
    car_state = client.getCarState()
    start_time = car_state.timestamp

    i_max = int(np.ceil(t_max / dt))
    t = np.zeros([1, i_max])
    U = np.zeros([2, i_max])
    Z = np.zeros([6, i_max])
    driving_states = []

    for i in range(i_max):
        current_state = client.getCarState()
        current_controls = client.getCarControls()
        current_time = np.around((current_state.timestamp - start_time) / 1e9, 3)

        cyaw = np.cos(
            airsim.to_eularian_angles(current_state.kinematics_estimated.orientation)[2]
        )
        syaw = np.sin(
            airsim.to_eularian_angles(current_state.kinematics_estimated.orientation)[2]
        )
        dcm_nb = np.array([[cyaw, syaw], [-syaw, cyaw]])

        t[:, i] = dt * i
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

        car_controls.throttle = rng.uniform(0, 1)
        car_controls.steering = rng.uniform(-1, 1)
        client.setCarControls(car_controls)

        time.sleep(dt)
    driving_df = pd.DataFrame(driving_states)

    car_controls.throttle = 0
    car_controls.steering = 0
    client.setCarControls(car_controls)
    return t, U, Z, driving_df


# Connect to the AirSim simulator
client = airsim.CarClient()
client.enableApiControl(True)
reset_client(client)

offset = np.array([40, 0])

offset_client(client, offset)
t, U, Z, driving_df = collect_data(
    client=client,
    rng=np.random.default_rng(seed=100),
    dt=0.1,
    t_max=20,
    offset=offset,
)
driving_df.to_csv(data_dir / f"data_train.csv", index=False)

reset_client(client)

offset_client(client, offset)
t, U, Z, driving_df = collect_data(
    client=client,
    rng=np.random.default_rng(seed=10),
    dt=0.1,
    t_max=20,
    offset=offset,
)
driving_df.to_csv(data_dir / f"data_test.csv", index=False)