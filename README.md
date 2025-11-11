# ArUco Marker Detection and Tracking Pipeline

A simple, real-time ArUco marker detection and tracking system using OpenCV.

## Features

- Real-time ArUco marker detection from camera feed
- Marker ID display and tracking
- Visual marker boundaries and corner points
- Optional 3D pose estimation (requires camera calibration)
- Save frames with detected markers

## Requirements

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (fast Python package installer)
- OpenCV with contrib modules
- NumPy
- Camera/webcam

## Quick Start

1. Install `uv` (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies and run:
```bash
uv sync
uv run python aruco_tracker.py
```

## Installation

### Install uv

If you don't have `uv` installed, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using pip:
```bash
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

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

Or install directly without virtual environment:
```bash
uv pip install -e .
```

## Usage

### Basic Usage

Run the tracker with default settings:

If using `uv`:
```bash
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

Options:
- `--camera ID`: Specify camera device ID (default: 0)
- `--dictionary ID`: Specify ArUco dictionary type (default: DICT_6X6_250)
- `--pose`: Enable pose estimation (requires camera calibration)

### Examples

Use a different camera:
```bash
uv run python aruco_tracker.py --camera 1
```

Use a different dictionary (e.g., 4x4):
```bash
uv run python aruco_tracker.py --dictionary 4
```

Enable pose estimation:
```bash
uv run python aruco_tracker.py --pose
```

## Controls

- **'q'**: Quit the application
- **'s'**: Save current frame as image

## ArUco Dictionary Types

Common dictionary IDs:
- `4`: DICT_4X4_50
- `5`: DICT_4X4_100
- `6`: DICT_4X4_250
- `7`: DICT_4X4_1000
- `8`: DICT_5X5_50
- `9`: DICT_5X5_100
- `10`: DICT_5X5_250
- `11`: DICT_5X5_1000
- `12`: DICT_6X6_50
- `13`: DICT_6X6_100
- `14`: DICT_6X6_250 (default)
- `15`: DICT_6X6_1000
- `16`: DICT_7X7_50
- `17`: DICT_7X7_100
- `18`: DICT_7X7_250
- `19`: DICT_7X7_1000

## Generating ArUco Markers

You can generate ArUco markers using OpenCV:

```python
import cv2

# Generate a marker
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id=0, side_pixels=200)
cv2.imwrite(f"marker_{marker_id}.png", marker_image)
```

Or use the provided marker generator script:
```bash
uv run python generate_markers.py
```

With options:
```bash
# Generate specific markers
uv run python generate_markers.py --ids 0 1 2 3 4 --size 300

# Use different dictionary
uv run python generate_markers.py --dictionary 4 --ids 0 1 2
```

## Camera Calibration (for Pose Estimation)

To enable pose estimation, you need to calibrate your camera. This involves:
1. Capturing multiple images of a calibration pattern (chessboard)
2. Computing camera matrix and distortion coefficients
3. Loading them in the tracker

See OpenCV's camera calibration tutorial for details.

## Project Structure

```
Aruco Hangar/
├── aruco_tracker.py    # Main tracking application
├── generate_markers.py # Marker generation utility
├── pyproject.toml      # Project configuration and dependencies
├── requirements.txt    # Python dependencies (alternative)
└── README.md          # This file
```

## Notes

- The tracker works best with good lighting and clear marker visibility
- Markers should be printed on flat, non-reflective surfaces
- Larger markers are easier to detect from greater distances
- For pose estimation, you need to know the physical size of your markers

