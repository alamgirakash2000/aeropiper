import mujoco
import mujoco.viewer
import numpy as np
import atexit
import signal

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

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Random controls for all actuators within their limits
            data.ctrl[:] = np.random.uniform(ctrlrange[:, 0], ctrlrange[:, 1])
            mujoco.mj_step(model, data)
            viewer.sync()
except SystemExit:
    pass
