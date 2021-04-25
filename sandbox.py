# Import necessary packages
from pathlib import Path  # Filepaths
import numpy as np  # N-dim arrays + math
import scipy.linalg as spla  # Complex linear algebra
import pandas as pd  # Dataframes
import airsim  # Airsim APIs
import time  # Timing/sleeping

# Logistics
figs_dir = Path.cwd() / "figs"
data_dir = Path.cwd() / "data"

# Set seed for consistent results
rng = np.random.default_rng(seed=100)

# Connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
print(f"API Control enabled: {client.isApiControlEnabled()}")

client.reset()

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)

time.sleep(2)
# pose = client.simGetVehiclePose()
# pose.position.x_val -= 1
# client.simSetVehiclePose(pose, True)
# time.sleep(3)

car_controls = airsim.CarControls()
car_state = client.getCarState()
start_time = car_state.timestamp

dt = 0.5
t_max = 3
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
    Z[:2, i] = current_state.kinematics_estimated.position.to_numpy_array()[:2]
    Z[2:4, i] = (
        dcm_nb @ current_state.kinematics_estimated.linear_velocity.to_numpy_array()[:2]
    )
    Z[4, i] = np.degrees(
        np.array(
            airsim.to_eularian_angles(current_state.kinematics_estimated.orientation)
        )[2]
    )

    Z[5, i] = np.degrees(
        current_state.kinematics_estimated.angular_velocity.to_numpy_array()[2]
    )
    current_state_dict = {
        **{
            "t": dt * i,
            "throttle": current_controls.throttle,
            "steering": current_controls.steering,
        },
        **dict(
            zip(
                ["x", "y"],
                current_state.kinematics_estimated.position.to_numpy_array()[:2],
            )
        ),
        **dict(
            zip(
                ["v_x", "v_y"],
                current_state.kinematics_estimated.linear_velocity.to_numpy_array()[:2],
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

car_controls.throttle = 0
car_controls.steering = 0
client.setCarControls(car_controls)

driving_df = pd.DataFrame(driving_states)
car_pose = client.simGetVehiclePose()
time.sleep(5)

client.armDisarm(False)
client.reset()
client.enableApiControl(False)