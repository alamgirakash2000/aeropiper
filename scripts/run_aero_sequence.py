#!/usr/bin/env python3
"""
Aero Hand Trajectory Demo
==========================
Run standalone: python3 run_aero_sequence.py

This script will automatically handle serial port permissions if needed.
"""

import os
import sys
import subprocess
import glob
import time
from aero_open_sdk.aero_hand import AeroHand

# ============================================
# CONFIGURATION
# ============================================

# Try common port names (will auto-detect) - prefer by-id for stability
POSSIBLE_PORTS = [
    "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_*",
    "/dev/ttyACM0",
    "/dev/ttyUSB0",
]
HAND_BAUDRATE = 921600

# ============================================
# PORT DETECTION & SETUP
# ============================================

def find_hand_port():
    """Auto-detect the hand's serial port"""
    print("\n→ Detecting hand port...")
    
    # Check each possible port
    for port_pattern in POSSIBLE_PORTS:
        # Handle wildcards
        if '*' in port_pattern:
            matches = glob.glob(port_pattern)
            if matches:
                port = matches[0]
                print(f"  ✓ Found: {port}")
                return port
        elif os.path.exists(port_pattern):
            print(f"  ✓ Found: {port_pattern}")
            return port_pattern
    
    # If nothing found, list available ports
    print("\n✗ No hand port found!")
    print("\nAvailable serial ports:")
    
    available = []
    for pattern in ["/dev/ttyACM*", "/dev/ttyUSB*", "/dev/serial/by-id/*Espressif*"]:
        available.extend(glob.glob(pattern))
    
    if available:
        for p in available:
            print(f"  - {p}")
        print(f"\nTip: Update POSSIBLE_PORTS in this script if needed")
    else:
        print("  (none found)")
        print("\nPlease check:")
        print("  1. Is the hand connected via USB?")
        print("  2. Try: ls /dev/ttyACM* /dev/ttyUSB*")
    
    return None

def setup_serial_permissions(port):
    """Setup serial port permissions automatically"""
    print("→ Checking permissions...")
    
    try:
        # Test if we can access it
        try:
            with open(port, 'r') as f:
                pass
            print("  ✓ Port accessible")
            return True
        except PermissionError:
            # Need to fix permissions
            print("  → Fixing permissions (sudo required)...")
            
            # Fix permissions with sudo
            subprocess.run(["sudo", "chmod", "666", port], check=True)
            
            # Verify
            try:
                with open(port, 'r') as f:
                    pass
                print("  ✓ Permissions fixed")
                return True
            except:
                print("✗ Permission fix failed")
                return False
                
    except subprocess.CalledProcessError as e:
        print(f"✗ Permission setup failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

# ============================================
# INITIALIZE HAND
# ============================================

def init_hand(port):
    """Initialize the Aero hand"""
    print("→ Connecting to hand...")
    
    try:
        hand = AeroHand(port=port, baudrate=HAND_BAUDRATE)
        print("  ✓ Connected")
        time.sleep(1.0)  # Give hand time to be ready
        print("  ✓ Initialized")
        return hand
    except Exception as e:
        print(f"✗ Failed to initialize hand: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================
# TRAJECTORY DEFINITION
# ============================================

def get_trajectory():
    """Define the hand trajectory sequence"""
    return [
        ## Open Palm
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0),

        ## Pinch fingers one by one
        ([100.0, 35.0, 23.0, 0.0, 0.0, 0.0, 50.0], 0.5), # Touch Pinkie
        ([100.0, 35.0, 23.0, 0.0, 0.0, 0.0, 50.0], 0.25), # Hold
        ([100.0, 42.0, 23.0, 0.0, 0.0, 52.0, 0.0], 0.5), # Touch Ring
        ([100.0, 42.0, 23.0, 0.0, 0.0, 52.0, 0.0], 0.25), # Hold
        ([83.0, 42.0, 23.0, 0.0, 50.0, 0.0, 0.0], 0.5), # Touch Middle
        ([83.0, 42.0, 23.0, 0.0, 50.0, 0.0, 0.0], 0.25), # Hold
        ([75.0, 25.0, 30.0, 50.0, 0.0, 0.0, 0.0], 0.5), # Touch Index
        ([75.0, 25.0, 30.0, 50.0, 0.0, 0.0, 0.0], 0.25), # Hold

        ## Open Palm
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5), # Hold

        ## Peace Sign
        ([90.0, 0.0, 0.0, 0.0, 0.0, 90.0, 90.0], 0.5),
        ([90.0, 45.0, 60.0, 0.0, 0.0, 90.0, 90.0], 0.5),
        ([90.0, 45.0, 60.0, 0.0, 0.0, 90.0, 90.0], 1.0),

        ## Open Palm
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5), # Hold

        ## Rockstar Sign
        ([0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 0.0], 0.5), # Close Middle and Ring Fingers
        ([0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 0.0], 1.0), # Hold

        ## Open Palm
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
    ]

# ============================================
# MAIN
# ============================================

def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("AERO HAND TRAJECTORY DEMO")
    print("=" * 70)
    
    try:
        # Find the hand port
        port = find_hand_port()
        if not port:
            return False
        
        # Setup permissions if needed
        if not setup_serial_permissions(port):
            return False
        
        # Initialize hand
        hand = init_hand(port)
        if not hand:
            return False
        
        # Quick test to verify hand responds
        print("→ Testing hand...")
        try:
            test_open = hand.convert_seven_joints_to_sixteen([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            hand.set_joint_positions(test_open)
            time.sleep(0.5)
            print("  ✓ Hand responding")
        except Exception as e:
            print(f"✗ Hand test failed: {e}")
            return False
        
        
        trajectory = get_trajectory()
        print(f"→ Running trajectory ({len(trajectory)} waypoints)...")
        
        # Execute trajectory manually using set_joint_positions (like robot_connection.py)
        for pose_7dof, duration in trajectory:
            # Convert 7 DOF to 16 joint positions
            pose_16 = hand.convert_seven_joints_to_sixteen(pose_7dof)
            
            # Set the position
            hand.set_joint_positions(pose_16)
            
            # Wait for the duration
            time.sleep(duration)
        
        print("✓ Trajectory complete!")
        
        # Cleanup
        print("→ Closing connection...")
        hand.close()
        print("✓ Done\n")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  STOPPED BY USER!")
        try:
            if 'hand' in locals() and hand:
                # Emergency open
                open_pos = hand.convert_seven_joints_to_sixteen([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                hand.set_joint_positions(open_pos)
                hand.close()
        except:
            pass
        return False
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)