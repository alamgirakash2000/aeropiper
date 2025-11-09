import mujoco
import mujoco.viewer
import numpy as np
import atexit
import signal
import time

# Load the dual scene (right + left)
model = mujoco.MjModel.from_xml_path("assets/scene_dual.xml")
data = mujoco.MjData(model)
ctrlrange = np.array(model.actuator_ctrlrange)

def noop_cleanup():
    pass

atexit.register(noop_cleanup)

def signal_handler(signum, frame):
    # Quiet exit
    raise SystemExit

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_right_arm_control():
    return False

def get_left_arm_control():
    return np.random.uniform(ctrlrange[13:19, 0], ctrlrange[13:19, 1])

def get_right_hand_control():
    return np.random.uniform(ctrlrange[6:13, 0], ctrlrange[6:13, 1])

def get_left_hand_control():
    return False

# Control update interval (seconds)
CONTROL_UPDATE_INTERVAL = 2.0
ARM_VELOCITY_SCALE = 0.2  # 20% of maximum velocity

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Initialize control values (current positions)
        right_arm_current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        right_arm_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        right_hand_control = np.concatenate(([ctrlrange[6, 0]], ctrlrange[7:13, 1]))
        
        left_arm_current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        left_arm_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        left_hand_control = np.concatenate(([ctrlrange[19, 0]], ctrlrange[20:26, 1]))
        
        last_update_time = time.time()
        
        while viewer.is_running():
            # Update control targets at specified interval
            current_time = time.time()
            if current_time - last_update_time >= CONTROL_UPDATE_INTERVAL:
                right_arm_target= (rc if (rc:=get_right_arm_control()) is not False else np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
                right_hand_control= (rhc if (rhc:=get_right_hand_control()) is not False else np.concatenate(([ctrlrange[6, 0]], ctrlrange[7:13, 1])))
                left_arm_target= (lac if (lac:=get_left_arm_control()) is not False else np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
                left_hand_control= (lhc if (lhc:=get_left_hand_control()) is not False else np.concatenate(([ctrlrange[19, 0]], ctrlrange[20:26, 1])))
                last_update_time = current_time
            
            # Smoothly interpolate arm positions toward targets at reduced velocity
            right_arm_current += (right_arm_target - right_arm_current) * ARM_VELOCITY_SCALE * 0.1
            left_arm_current += (left_arm_target - left_arm_current) * ARM_VELOCITY_SCALE * 0.1

            system_control=np.concatenate([right_arm_current, right_hand_control, left_arm_current, left_hand_control])
            data.ctrl[:] = system_control
            mujoco.mj_step(model, data)
            viewer.sync()
except SystemExit:
    pass
