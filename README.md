# ArUco Marker Detection and Tracking Pipeline

A robust, real-time ArUco marker detection and tracking system using OpenCV with ZMQ streaming and data logging capabilities.

## Features

- Real-time ArUco marker detection from camera feed
- Marker ID display and square outline visualization
- ZMQ streaming for real-time detection data (ID, timestamp, pose)
- Automatic detection logging to JSON files
- Robust detection with multiple preprocessing strategies
- Camera calibration support
- Cross-platform support (Windows, macOS, Linux)

## Requirements

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (fast Python package installer)
- OpenCV with contrib modules
- NumPy
- PyZMQ (for streaming)
- Camera/webcam

## Installation

### Install uv

#### Linux/macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using pip:
```bash
pip install uv
```

#### Windows

Using PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or using pip:
```powershell
pip install uv
```

### Install Project Dependencies

**Recommended: Using `uv sync`** (creates virtual environment automatically):

```bash
uv sync
```

This will:
- Create a virtual environment (`.venv/`)
- Install all dependencies from `pyproject.toml`
- Make the project ready to use

**Alternative: Manual installation**

**Linux/macOS:**
```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

**Windows:**
```powershell
# Create virtual environment
uv venv

# Activate it
.venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

## Usage

### Basic Usage

Run the tracker with default settings (4x4_100 dictionary, marker size 0.001m):

**Linux/macOS:**
```bash
uv run python aruco_tracker.py
```

**Windows:**
```powershell
uv run python aruco_tracker.py
```

Or if you have activated the virtual environment:
```bash
python aruco_tracker.py
```

### Command Line Options

```bash
uv run python aruco_tracker.py [OPTIONS]
```

**Options:**
- `--device DEVICE`: Camera device ID (int) or device path (str, e.g., /dev/video4). Default: 0
- `--dict DICT`: ArUco dictionary name (e.g., 4x4_100, 6x6_250). Default: 4x4_100
- `--marker-size SIZE`: Physical marker size in meters. Default: 0.001
- `--calib FILE`: Path to camera calibration JSON file
- `--zmq-port PORT`: Enable ZMQ streaming on specified port (e.g., 5555)
- `--save FILE`: Save detections to JSON file (e.g., detections.json)
- `--pose`: Enable pose estimation axes (requires camera calibration)
- `--fast`: Use faster but less robust detection

### Examples

**Use a different camera:**
```bash
uv run python aruco_tracker.py --device 1
```

**Use device path (Linux):**
```bash
uv run python aruco_tracker.py --device /dev/video4
```

**Use camera calibration file:**
```bash
uv run python aruco_tracker.py --calib calib_3937__0c45_6366__1280.json
```

**Enable ZMQ streaming:**
```bash
uv run python aruco_tracker.py --zmq-port 5555
```

**Save detections to file:**
```bash
uv run python aruco_tracker.py --save detections.json
```

**Combined usage:**
```bash
uv run python aruco_tracker.py --calib calib.json --zmq-port 5555 --save detections.json --marker-size 0.002
```

## Camera Calibration

### Using CalibDB

[CalibDB](https://www.calibdb.net/) provides a database of pre-calibrated camera parameters for many common cameras. This is the easiest way to get accurate calibration data:

1. Visit [https://www.calibdb.net/](https://www.calibdb.net/)
2. Search for your camera model or USB vendor/product ID
3. Download the calibration JSON file
4. Use it with the `--calib` option:
   ```bash
   uv run python aruco_tracker.py --calib calib_*.json
   ```

The calibration board image is available at: [https://www.calibdb.net/board.png](https://www.calibdb.net/board.png)

### Manual Calibration

If your camera is not in CalibDB, you can calibrate it manually:

1. Print the calibration board from [https://www.calibdb.net/board.png](https://www.calibdb.net/board.png)
2. Capture multiple images of the board from different angles
3. Use OpenCV's camera calibration tools to compute the camera matrix and distortion coefficients
4. Save the results in JSON format matching the CalibDB format

## ZMQ Streaming

The tracker can stream detection data via ZMQ for real-time integration with other applications.

**Start the tracker with ZMQ:**
```bash
uv run python aruco_tracker.py --zmq-port 5555
```

**Subscribe to the stream (Python example):**
```python
import zmq
import json

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

while True:
    data = socket.recv_json()
    print(f"Marker ID: {data['id']}, Timestamp: {data['timestamp']}")
    if data['pose']:
        print(f"Translation: {data['pose']['translation']}")
        print(f"Rotation: {data['pose']['rotation']}")
```

**Data Format:**
```json
{
  "id": 0,
  "timestamp": "2024-01-25T12:34:56.789",
  "pose": {
    "rotation": [0.1, 0.2, 0.3],
    "translation": [0.01, 0.02, 0.15]
  }
}
```

## Saving Detections

Detections are automatically saved to a JSON file when using the `--save` option:

```bash
uv run python aruco_tracker.py --save detections.json
```

**Output Format:**
```json
[
  {
    "id": 0,
    "timestamp": "2024-01-25T12:34:56.789",
    "pose": {
      "rotation": [0.1, 0.2, 0.3],
      "translation": [0.01, 0.02, 0.15]
    }
  },
  ...
]
```

Detections are auto-saved every 5 seconds and on program exit.

## Controls

- **'q'**: Quit the application
- **'s'**: Save current frame as image

## ArUco Dictionary Types

Available dictionaries:
- `4x4_50`, `4x4_100` (default), `4x4_250`, `4x4_1000`
- `5x5_50`, `5x5_100`, `5x5_250`, `5x5_1000`
- `6x6_50`, `6x6_100`, `6x6_250`, `6x6_1000`
- `7x7_50`, `7x7_100`, `7x7_250`, `7x7_1000`

## Generating ArUco Markers

Generate markers using the provided script:

```bash
uv run python generate_markers.py
```

**With options:**
```bash
# Generate specific markers
uv run python generate_markers.py --ids 0 1 2 3 4 --size 300

# Use different dictionary
uv run python generate_markers.py --dict 4x4_100 --ids 0 1 2
```

## Project Structure

```
Aruco Hangar/
├── aruco_tracker.py    # Main tracking application
├── generate_markers.py # Marker generation utility
├── pyproject.toml      # Project configuration and dependencies
├── requirements.txt    # Python dependencies (alternative)
└── README.md          # This file
```

## Platform-Specific Notes

### Linux

- Device paths like `/dev/video0`, `/dev/video1` are supported
- May need to install `v4l-utils` for camera access:
  ```bash
  sudo apt-get install v4l-utils  # Debian/Ubuntu
  ```

### macOS

- Camera access may require permissions in System Preferences > Security & Privacy
- Use device IDs (0, 1, 2, etc.) instead of device paths

### Windows

- Use device IDs (0, 1, 2, etc.) for cameras
- May need to install camera drivers
- PowerShell is recommended for running commands

## Tips for Best Results

- Use good lighting conditions for better detection
- Print markers on flat, non-reflective surfaces
- Larger markers are easier to detect from greater distances
- Use camera calibration from CalibDB for accurate pose estimation
- Default marker size is 0.001m (1mm) - adjust with `--marker-size` if needed
- Enable robust mode (default) for better detection reliability

## Troubleshooting

**Camera not detected:**
- Check camera permissions (macOS/Windows)
- Verify camera is connected and not in use by another application
- Try different device IDs or paths

**Poor detection:**
- Ensure good lighting
- Check marker is flat and not wrinkled
- Try different dictionary sizes
- Use `--fast` flag to disable robust mode if performance is an issue

**ZMQ connection issues:**
- Ensure port is not in use by another application
- Check firewall settings
- Verify pyzmq is installed: `uv pip install pyzmq`

## License

This project is open source and available for use.
