# ArUco Gully

**ArUco Gully** (ಗಲ್ಲಿ - "gully" means alley/corridor in Kannada and many other Indian languages) is a robust, real-time ArUco marker detection and tracking system designed for tracking markers on objects moving through narrow corridors or confined spaces. Features ZMQ streaming and CSV data logging capabilities for real-time monitoring and analysis.

## Features

- Real-time ArUco marker detection from camera feed
- Marker ID display and square outline visualization
- ZMQ streaming for real-time detection data (ID, timestamp, pose)
- Automatic detection logging to CSV files
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
- `--save FILE`: Save detections to CSV file (default: detections.csv, use 'none' to disable)
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

**Save detections to custom file:**
```bash
uv run python aruco_tracker.py --save my_detections.csv
```

**Combined usage:**
```bash
  uv run python aruco_tracker.py --calib calib.json --zmq-port 5555 --save detections.csv --marker-size 0.002
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
    # Data contains multiple detections in a single message
    for detection in data['detections']:
        print(f"Marker ID: {detection['id']}, Timestamp: {detection['timestamp']}")
        if detection['pose']:
            print(f"Translation: {detection['pose']['translation']}")
            print(f"Rotation: {detection['pose']['rotation']}")
```

**ZMQ Data Format (supports multiple tags per message):**
```json
{
  "detections": [
    {
      "id": 0,
      "timestamp": "2024-01-25T12:34:56.789",
      "pose": {
        "rotation": [0.1, 0.2, 0.3],
        "translation": [0.01, 0.02, 0.15]
      }
    },
    {
      "id": 1,
      "timestamp": "2024-01-25T12:34:56.789",
      "pose": {
        "rotation": [0.2, 0.3, 0.4],
        "translation": [0.02, 0.03, 0.16]
      }
    }
  ]
}
```

Multiple markers detected in the same frame are sent together in a single ZMQ message for efficiency.

## Saving Detections

Detections are automatically saved to a CSV file by default (`detections.csv`). You can specify a different filename or disable saving:

```bash
# Use default filename (detections.csv)
uv run python aruco_tracker.py

# Specify custom filename
uv run python aruco_tracker.py --save my_detections.csv

# Disable saving
uv run python aruco_tracker.py --save none
```

**CSV Output Format:**
```csv
timestamp,id,tx,ty,tz,rx,ry,rz
2024-01-25T12:34:56.789,0,0.010000,0.020000,0.150000,0.100000,0.200000,0.300000
2024-01-25T12:34:56.890,1,0.015000,0.025000,0.160000,0.110000,0.210000,0.310000
```

**Columns:**
- `timestamp`: ISO format timestamp
- `id`: Marker ID
- `tx, ty, tz`: Translation (position) in meters
- `rx, ry, rz`: Rotation vector components

Detections are auto-saved every 5 seconds and on program exit.

### Plotting Data with PlotJuggler

The saved detection data can be visualized and analyzed using [PlotJuggler](https://www.plotjuggler.io/), a powerful time-series data visualization tool.

**Steps to plot ArUco Gully data:**

1. **Install PlotJuggler:**
   - Download from [https://www.plotjuggler.io/](https://www.plotjuggler.io/)
   - Available for Windows, macOS, and Linux

2. **Load CSV directly in PlotJuggler:**
   - Open PlotJuggler
   - File → Load Data → Select `detections.csv`
   - The CSV is already in the correct format with timestamp, id, and pose data
   - Visualize marker trajectories, velocities, and pose over time
   - Filter by marker ID to track individual markers

**Alternative: Direct ZMQ streaming to PlotJuggler**

PlotJuggler can also receive data directly via ZMQ. Configure PlotJuggler's ZMQ subscriber to connect to the same port as ArUco Gully's publisher for real-time visualization.

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
Aruco Gully/
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

## Use Case: Tracking in Narrow Corridors

ArUco Gully is specifically designed for tracking objects moving through narrow corridors or confined spaces. The system is optimized for:

- **Small markers** (default 1mm) suitable for compact objects
- **Robust detection** in challenging lighting conditions
- **Real-time streaming** for monitoring moving objects
- **Precise pose estimation** for trajectory analysis

**Typical applications:**
- Assembly line tracking
- Conveyor belt monitoring
- Small object tracking in confined spaces
- Quality control in narrow production lines

## Tips for Best Results

- Use good lighting conditions for better detection
- Print markers on flat, non-reflective surfaces
- For narrow corridors, use smaller markers (0.001m default) for compact objects
- Use camera calibration from CalibDB for accurate pose estimation
- Enable robust mode (default) for better detection reliability in challenging environments
- Position camera to minimize occlusion in tight spaces

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
