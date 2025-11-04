# PiPER Arm + Aero Hand Control

## Quick Start

### Installation:

  - `pip install piper_sdk`
  - `pip install aero-open-sdk`
Just plug in both devices and run:

```
python robot_connection.py
python reach_task.py
```

The script will automatically:
- Setup CAN interface for PiPER arm (requires sudo, will prompt for password)
- Fix serial port permissions for Aero Hand (requires sudo, will prompt for password)
- Initialize both devices
- Run test movements
This takes ~3 minutes and calibrates the hand to its zero positions.

### Troubleshooting

**Serial port issues:**
- Script now handles permissions automatically
- If needed, manually: `sudo chmod 666 /dev/ttyACM0`

**Hand not moving:**
- Check external power supply is connected
- Try running: `python3 test_hand_power.py`

**CAN interface issues:**
- Script handles this automatically
- Manual setup: `sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up`

