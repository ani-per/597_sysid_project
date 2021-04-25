# Import necessary packages
import warnings  # Ignore user warnings
import itertools as it  # Readable nested for loops
from pathlib import Path  # Filepaths
import typing  # Argument / output type checking
import numpy as np  # N-dim arrays + math
import scipy.linalg as spla  # Complex linear algebra
import matplotlib.pyplot as plt  # Plots
import matplotlib.figure as figure  # Figure documentation
import scipy.signal as spsg  # Signal processing
import sympy  # Symbolic math + pretty printing
import pandas as pd  # Dataframes
import setup_path  # Airsim client setup
import airsim  # Airsim APIs
import time  # Timing/sleeping

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
pose = client.simGetVehiclePose()
pose.position.x_val -= 1
client.simSetVehiclePose(pose, True)
time.sleep(3)

car_controls = airsim.CarControls()
car_state = client.getCarState()
start_time = car_state.timestamp
# print(f"{car_state=}")
# print(f"{client.getHomeGeoPoint()=}")
# print(f"{client.simGetGroundTruthEnvironment()}")

car_controls.throttle = 0.5
car_controls.steering = -0.75
client.setCarControls(car_controls)

dt = 0.5
t_max = 6
i_max = int(np.ceil(t_max / dt))
t = np.zeros([1, i_max])
U = np.zeros([2, i_max])
Z = np.zeros([4, i_max])
driving_states = []

for i in range(i_max):
    current_state = client.getCarState()
    current_controls = client.getCarControls()
    current_time = np.around((current_state.timestamp - start_time) / 1e9, 3)

    print(f"{client.simGetGroundTruthEnvironment().position}")
    t[:, i] = current_time
    U[:, i] = np.array([current_controls.throttle, current_controls.steering])
    Z[:2, i] = current_state.kinematics_estimated.position.to_numpy_array()[:2]
    Z[2:, i] = current_state.kinematics_estimated.linear_velocity.to_numpy_array()[:2]
    cyaw = np.cos(
        airsim.to_eularian_angles(current_state.kinematics_estimated.orientation)[2]
    )
    syaw = np.sin(
        airsim.to_eularian_angles(current_state.kinematics_estimated.orientation)[2]
    )
    dcm_nb = np.array([[cyaw, syaw], [-syaw, cyaw]])
    driving_states.append(
        {
            "t": current_time,
            "x": current_state.kinematics_estimated.position.x_val,
            "y": current_state.kinematics_estimated.position.y_val,
            "z": current_state.kinematics_estimated.position.z_val,
            "v_x": current_state.kinematics_estimated.linear_velocity.x_val,
            "v_y": current_state.kinematics_estimated.linear_velocity.y_val,
            "v_z": current_state.kinematics_estimated.linear_velocity.z_val,
            "v_x": current_state.kinematics_estimated.linear_velocity.x_val,
            "v_y": current_state.kinematics_estimated.linear_velocity.y_val,
            "v_z": current_state.kinematics_estimated.linear_velocity.z_val,
            "u": (
                dcm_nb
                @ current_state.kinematics_estimated.linear_velocity.to_numpy_array()[
                    :2
                ]
            )[0],
            "v": (
                dcm_nb
                @ current_state.kinematics_estimated.linear_velocity.to_numpy_array()[
                    :2
                ]
            )[1],
            "roll": np.degrees(
                airsim.to_eularian_angles(
                    current_state.kinematics_estimated.orientation
                )[1]
            ),
            "pitch": np.degrees(
                airsim.to_eularian_angles(
                    current_state.kinematics_estimated.orientation
                )[0]
            ),
            "yaw": np.degrees(
                airsim.to_eularian_angles(
                    current_state.kinematics_estimated.orientation
                )[2]
            ),
            "p": np.degrees(
                current_state.kinematics_estimated.angular_velocity.to_numpy_array()[0]
            ),
            "q": np.degrees(
                current_state.kinematics_estimated.angular_velocity.to_numpy_array()[1]
            ),
            "r": np.degrees(
                current_state.kinematics_estimated.angular_velocity.to_numpy_array()[2]
            ),
            "throttle": current_controls.throttle,
            "steering": current_controls.steering,
        }
    )
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