#!/usr/bin/env python3
import mujoco
import mujoco.viewer
import numpy as np
import os
import sys
import signal
import atexit

# Allow importing from scripts/module
sys.path.append(os.path.join(os.path.dirname(__file__), "module"))
from physical_to_mujoco import physical_to_mujoco

# Load the dual scene (right + left)
model = mujoco.MjModel.from_xml_path("assets/scene_dual.xml")
data = mujoco.MjData(model)
ctrlrange = np.array(model.actuator_ctrlrange)


# Actuator index layout (include order: right first, then left)
# right: [0..5]=arm(6), [6..12]=hand(7)
# left:  [13..18]=arm(6), [19..25]=hand(7)
RIGHT_ARM_SLICE = slice(0, 6)
RIGHT_HAND_SLICE = slice(6, 13)
LEFT_ARM_SLICE = slice(13, 19)
LEFT_HAND_SLICE = slice(19, 26)

# RESTING POSITION FOR ARMS (all joints at 0)
ARM_RESTING = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Physical trajectory format: ([7 DOF values], duration_seconds)
# Physical/MuJoCo order (matched): [thumb_abd, thumb_flex, thumb_tendon, index, middle, ring, pinky]
# Values in [0..100] where 0=open/straight, 100=closed/bent
PHYSICAL_TRAJECTORY_NAMED = [
    ("Open Palm",                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0),
    # Pinch fingers one by one
    ("Touch Pinkie",                           [100.0, 35.0, 23.0, 0.0, 0.0, 0.0, 50.0], 2),
    ("Touch Pinkie - Hold",                    [100.0, 35.0, 23.0, 0.0, 0.0, 0.0, 50.0], 2),
    ("Touch Ring",                             [100.0, 42.0, 23.0, 0.0, 0.0, 52.0, 0.0], 2),
    ("Touch Ring - Hold",                      [100.0, 42.0, 23.0, 0.0, 0.0, 52.0, 0.0], 2),
    ("Touch Middle",                           [83.0, 42.0, 23.0, 0.0, 50.0, 0.0, 0.0], 2),
    ("Touch Middle - Hold",                    [83.0, 42.0, 23.0, 0.0, 50.0, 0.0, 0.0], 2),
    ("Touch Index",                            [75.0, 25.0, 30.0, 50.0, 0.0, 0.0, 0.0], 0.5),
    ("Touch Index - Hold",                     [75.0, 25.0, 30.0, 50.0, 0.0, 0.0, 0.0], 0.25),
    ("Open Palm",                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
    ("Open Palm - Hold",                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
    # Peace Sign
    ("Peace Sign",                             [90.0, 0.0, 0.0, 0.0, 0.0, 90.0, 90.0], 0.5),
    ("Peace Sign - Thumb Flex",                [90.0, 45.0, 60.0, 0.0, 0.0, 90.0, 90.0], 0.5),
    ("Peace Sign - Hold",                      [90.0, 45.0, 60.0, 0.0, 0.0, 90.0, 90.0], 1.0),
    ("Open Palm",                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
    ("Open Palm - Hold",                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
    # Rockstar Sign
    ("Rockstar Sign - Close Middle & Ring",    [0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 0.0], 0.5),
    ("Rockstar Sign - Hold",                   [0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 0.0], 1.0),
    ("Open Palm",                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
]

# Use the physical trajectory as-is (orders already matched)
# If you want to sanitize pinch steps, add a transform here.
ACTIVE_TRAJECTORY_NAMED = PHYSICAL_TRAJECTORY_NAMED

def get_trajectory_control(t: float, trajectory_named, transition_time=0.3):
    """
    Get control values for a hand following a trajectory at time t.
    
    Args:
        t: Current time in seconds
        trajectory_named: List of (name, physical_pose, duration) tuples; pose is in physical order
        transition_time: Time to smoothly transition between poses
    
    Returns:
        Control values for the 7 hand DOF
    """
    # Calculate total trajectory time
    total_time = sum(duration for _, _, duration in trajectory_named)
    
    # Loop the trajectory
    t_loop = t % total_time
    
    # Find current segment
    accumulated_time = 0.0
    for idx, (_, pose, duration) in enumerate(trajectory_named):
        if accumulated_time + duration > t_loop:
            # We're in this segment
            time_in_segment = t_loop - accumulated_time
            
            # Get current and next pose
            current_pose = np.array(pose, dtype=float)
            next_idx = (idx + 1) % len(trajectory_named)
            next_pose = np.array(trajectory_named[next_idx][1], dtype=float)
            
            # Convert physical values to MuJoCo control values
            current_ctrl = physical_to_mujoco(current_pose)
            next_ctrl = physical_to_mujoco(next_pose)
            
            # Check if we're in transition phase (end of segment)
            if duration - time_in_segment < transition_time:
                # Smoothly transition to next pose
                trans = (transition_time - (duration - time_in_segment)) / transition_time
                trans = min(1.0, max(0.0, trans))  # Clamp to [0, 1]
                # Smooth interpolation
                smooth_trans = 3 * trans**2 - 2 * trans**3
                return current_ctrl + (next_ctrl - current_ctrl) * smooth_trans
            else:
                # Hold current pose
                return current_ctrl
        
        accumulated_time += duration
    
    # Fallback (shouldn't reach here)
    return physical_to_mujoco(trajectory_named[0][1])


# Signal handling
def noop_cleanup():
    pass

atexit.register(noop_cleanup)

def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}.")
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# Calculate total trajectory time
total_traj_time = sum(duration for _, _, duration in ACTIVE_TRAJECTORY_NAMED)

#

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0
        last_waypoint_idx = -1
        
        while viewer.is_running():
            # Keep arms at resting position
            data.ctrl[RIGHT_ARM_SLICE] = ARM_RESTING
            data.ctrl[LEFT_ARM_SLICE] = ARM_RESTING
            
            # Get hand controls from trajectory
            right_hand_ctrl = get_trajectory_control(t, ACTIVE_TRAJECTORY_NAMED)
            left_hand_ctrl = get_trajectory_control(t, ACTIVE_TRAJECTORY_NAMED)
            
            # Apply to both hands (mirrored behavior)
            data.ctrl[RIGHT_HAND_SLICE] = right_hand_ctrl
            data.ctrl[LEFT_HAND_SLICE] = left_hand_ctrl
            
            # Track waypoint internally (no printing)
            t_in_cycle = t % total_traj_time
            waypoint_idx = 0
            acc_time = 0.0
            for idx, (_, _, duration) in enumerate(ACTIVE_TRAJECTORY_NAMED):
                if acc_time + duration > t_in_cycle:
                    waypoint_idx = idx
                    break
                acc_time += duration
            if waypoint_idx != last_waypoint_idx:
                last_waypoint_idx = waypoint_idx

            mujoco.mj_step(model, data)
            viewer.sync()
            t += model.opt.timestep

except KeyboardInterrupt:
    print("\nInterrupted by user")

