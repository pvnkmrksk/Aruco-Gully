#!/usr/bin/env python3
"""
Simple ArUco Marker Detection and Tracking Pipeline
Detects and tracks ArUco markers in real-time from camera feed
"""

import cv2
import numpy as np
import argparse
import json
import os


class ArucoTracker:
    def __init__(self, dictionary_id=cv2.aruco.DICT_6X6_250, camera_id=0, robust=True):
        """
        Initialize ArUco tracker

        Args:
            dictionary_id: ArUco dictionary type (default: DICT_6X6_250)
            camera_id: Camera device ID (int) or device path (str) (default: 0)
            robust: Use robust detection parameters (default: True)
        """
        self.dictionary_id = dictionary_id
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.robust_mode = robust
        if robust:
            self.parameters = self._create_robust_parameters()
        else:
            self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.camera_id = camera_id
        self.cap = None
        self.marker_size = 0.05  # Default marker size in meters

    def _create_robust_parameters(self):
        """Create detector parameters optimized for robustness"""
        params = cv2.aruco.DetectorParameters()

        # Adaptive thresholding - more robust to lighting changes
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7

        # Corner refinement - improves accuracy
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.01

        # Marker detection parameters - more lenient
        params.minMarkerPerimeterRate = 0.03  # Smaller markers allowed
        params.maxMarkerPerimeterRate = 4.0  # Larger markers allowed
        params.polygonalApproxAccuracyRate = 0.03  # More lenient polygon approximation
        params.minCornerDistanceRate = 0.05  # Allow closer corners

        # Error correction - more tolerant
        params.errorCorrectionRate = 0.6  # More error correction

        # Border bits - important for detection
        params.minOtsuStdDev = 5.0  # Lower threshold for Otsu

        # Perspective removal
        params.perspectiveRemovePixelPerCell = 4  # Standard
        params.perspectiveRemoveIgnoredMarginPerCell = 0.13

        # Max error ratio
        params.maxErroneousBitsInBorderRate = 0.35  # More tolerant of border errors

        return params

    def initialize_camera(self):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def get_default_camera_matrix(self, frame_width, frame_height):
        """
        Get default camera matrix for ArduCam or similar cameras
        Uses typical ArduCam calibration parameters or identity-like matrix

        Args:
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            camera_matrix: Default camera intrinsic matrix
            dist_coeffs: Default distortion coefficients (zeros)
        """
        # Typical ArduCam calibration parameters for common resolutions
        # For 1280x720 or similar, typical focal length is around 600-800 pixels
        # Using a more standard approach based on sensor size

        # Common ArduCam focal length approximation (in pixels)
        # For most ArduCam modules, fx ≈ fy ≈ width * 0.7 to 0.9
        focal_length = frame_width * 0.8

        center_x = frame_width / 2.0
        center_y = frame_height / 2.0

        camera_matrix = np.array(
            [
                [focal_length, 0, center_x],
                [0, focal_length, center_y],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # ArduCam typically has minimal distortion, but slight radial distortion
        # Using small values typical for ArduCam modules
        dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(4, 1)

        return camera_matrix, dist_coeffs

    def detect_markers(self, frame):
        """
        Detect ArUco markers in a frame with robust preprocessing

        Args:
            frame: Input image frame

        Returns:
            corners: Detected marker corners
            ids: Detected marker IDs
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Fast mode: simple detection
        if not self.robust_mode:
            corners, ids, _ = self.detector.detectMarkers(gray)
            return corners, ids

        # Robust mode: multiple preprocessing strategies
        # 1. Noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # 2. Try multiple preprocessing strategies
        processed_images = [
            denoised,  # Original denoised
            cv2.equalizeHist(denoised),  # Histogram equalization
            cv2.GaussianBlur(denoised, (5, 5), 0),  # Gaussian blur
        ]

        # Try detection on each processed image
        best_corners = None
        best_ids = None
        max_detections = 0

        for processed in processed_images:
            corners, ids, _ = self.detector.detectMarkers(processed)

            # Count valid detections
            if ids is not None:
                num_detections = len(ids)
                if num_detections > max_detections:
                    max_detections = num_detections
                    best_corners = corners
                    best_ids = ids

        # If no detections, try with adaptive threshold
        if best_ids is None or len(best_ids) == 0:
            adaptive = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            corners, ids, _ = self.detector.detectMarkers(adaptive)
            if ids is not None and len(ids) > 0:
                best_corners = corners
                best_ids = ids

        return best_corners, best_ids

    def draw_markers(self, frame, corners, ids):
        """
        Draw detected markers on the frame with square outline and ID

        Args:
            frame: Input frame
            corners: Detected marker corners
            ids: Detected marker IDs

        Returns:
            frame: Frame with drawn markers
        """
        if ids is not None:
            # Draw marker boundaries (square outline)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Draw ID for each marker
            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                corner = corner[0]

                # Calculate center
                center_x = int(np.mean(corner[:, 0]))
                center_y = int(np.mean(corner[:, 1]))

                # Draw ID text with background
                label = f"ID: {marker_id}"
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    frame,
                    (center_x - text_width // 2 - 5, center_y - text_height - 30),
                    (center_x + text_width // 2 + 5, center_y - 5),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (center_x - text_width // 2, center_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        return frame

    def estimate_pose(
        self, corners, ids, camera_matrix=None, dist_coeffs=None, marker_size=0.05
    ):
        """
        Estimate pose of detected markers (requires camera calibration)

        Args:
            corners: Detected marker corners
            ids: Detected marker IDs
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            marker_size: Physical size of marker in meters

        Returns:
            rvecs: Rotation vectors
            tvecs: Translation vectors
        """
        if camera_matrix is None or dist_coeffs is None:
            return None, None

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, dist_coeffs
        )
        return rvecs, tvecs

    def draw_pose(self, frame, corners, ids, rvecs, tvecs, camera_matrix, dist_coeffs):
        """
        Draw 3D pose axes on markers

        Args:
            frame: Input frame
            corners: Detected marker corners (unused but kept for API consistency)
            ids: Detected marker IDs
            rvecs: Rotation vectors
            tvecs: Translation vectors
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        if rvecs is None or tvecs is None:
            return frame

        for i in range(len(ids)):
            cv2.drawFrameAxes(
                frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03
            )
        return frame

    def get_dictionary_name(self):
        """Get human-readable dictionary name"""
        dict_names = {
            cv2.aruco.DICT_4X4_50: "DICT_4X4_50",
            cv2.aruco.DICT_4X4_100: "DICT_4X4_100",
            cv2.aruco.DICT_4X4_250: "DICT_4X4_250",
            cv2.aruco.DICT_4X4_1000: "DICT_4X4_1000",
            cv2.aruco.DICT_5X5_50: "DICT_5X5_50",
            cv2.aruco.DICT_5X5_100: "DICT_5X5_100",
            cv2.aruco.DICT_5X5_250: "DICT_5X5_250",
            cv2.aruco.DICT_5X5_1000: "DICT_5X5_1000",
            cv2.aruco.DICT_6X6_50: "DICT_6X6_50",
            cv2.aruco.DICT_6X6_100: "DICT_6X6_100",
            cv2.aruco.DICT_6X6_250: "DICT_6X6_250",
            cv2.aruco.DICT_6X6_1000: "DICT_6X6_1000",
            cv2.aruco.DICT_7X7_50: "DICT_7X7_50",
            cv2.aruco.DICT_7X7_100: "DICT_7X7_100",
            cv2.aruco.DICT_7X7_250: "DICT_7X7_250",
            cv2.aruco.DICT_7X7_1000: "DICT_7X7_1000",
        }
        return dict_names.get(self.dictionary_id, f"Unknown ({self.dictionary_id})")

    def run(
        self,
        show_pose=False,
        camera_matrix=None,
        dist_coeffs=None,
        use_default_camera_matrix=True,
    ):
        """
        Main tracking loop

        Args:
            show_pose: Whether to show pose estimation axes
            camera_matrix: Camera intrinsic matrix for pose estimation
            dist_coeffs: Distortion coefficients for pose estimation
            use_default_camera_matrix: Use default camera matrix for pose estimation if no calibration
        """
        self.initialize_camera()

        # Get frame dimensions for default camera matrix
        ret, test_frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read test frame from camera")
        frame_height, frame_width = test_frame.shape[:2]

        # Use default camera matrix if not provided
        if camera_matrix is None and use_default_camera_matrix:
            camera_matrix, dist_coeffs = self.get_default_camera_matrix(
                frame_width, frame_height
            )
            print("Using default camera matrix (approximation)")

        print("ArUco Tracker Started")
        print(f"Dictionary: {self.get_dictionary_name()}")
        print(f"Camera device: {self.camera_id}")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")

        frame_count = 0

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Detect markers
                corners, ids = self.detect_markers(frame)

                # Draw markers with square outline and ID
                frame = self.draw_markers(frame, corners, ids)

                # Estimate and draw pose axes if requested
                if show_pose and camera_matrix is not None and dist_coeffs is not None:
                    rvecs, tvecs = self.estimate_pose(
                        corners, ids, camera_matrix, dist_coeffs, self.marker_size
                    )
                    if rvecs is not None:
                        frame = self.draw_pose(
                            frame,
                            corners,
                            ids,
                            rvecs,
                            tvecs,
                            camera_matrix,
                            dist_coeffs,
                        )

                # Display marker count and dictionary
                marker_count = len(ids) if ids is not None else 0
                info_text = (
                    f"Markers: {marker_count} | Dict: {self.get_dictionary_name()}"
                )
                cv2.putText(
                    frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                # Display frame
                cv2.imshow("ArUco Tracker", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    filename = f"aruco_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved as {filename}")

                frame_count += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Tracker stopped")


def load_calibration(calib_file):
    """
    Load camera calibration from JSON file

    Args:
        calib_file: Path to calibration JSON file

    Returns:
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
    """
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")

    with open(calib_file, "r", encoding="utf-8") as f:
        calib_data = json.load(f)

    # Extract camera matrix
    camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float32)

    # Extract distortion coefficients
    dist_coeffs = np.array(calib_data["distortion_coefficients"], dtype=np.float32)

    # Reshape distortion coefficients to (4, 1) or (5, 1) format
    if len(dist_coeffs) == 5:
        dist_coeffs = dist_coeffs.reshape(5, 1)
    else:
        dist_coeffs = dist_coeffs.reshape(-1, 1)

    print(f"Loaded calibration from: {calib_file}")
    print(f"Camera: {calib_data.get('camera', 'Unknown')}")
    print(f"Image size: {calib_data.get('img_size', 'Unknown')}")
    print(
        f"Avg reprojection error: {calib_data.get('avg_reprojection_error', 'Unknown'):.4f}"
    )

    return camera_matrix, dist_coeffs


def get_dictionary_from_name(name):
    """Convert dictionary name string to OpenCV constant"""
    dict_map = {
        "4x4_50": cv2.aruco.DICT_4X4_50,
        "4x4_100": cv2.aruco.DICT_4X4_100,
        "4x4_250": cv2.aruco.DICT_4X4_250,
        "4x4_1000": cv2.aruco.DICT_4X4_1000,
        "5x5_50": cv2.aruco.DICT_5X5_50,
        "5x5_100": cv2.aruco.DICT_5X5_100,
        "5x5_250": cv2.aruco.DICT_5X5_250,
        "5x5_1000": cv2.aruco.DICT_5X5_1000,
        "6x6_50": cv2.aruco.DICT_6X6_50,
        "6x6_100": cv2.aruco.DICT_6X6_100,
        "6x6_250": cv2.aruco.DICT_6X6_250,
        "6x6_1000": cv2.aruco.DICT_6X6_1000,
        "7x7_50": cv2.aruco.DICT_7X7_50,
        "7x7_100": cv2.aruco.DICT_7X7_100,
        "7x7_250": cv2.aruco.DICT_7X7_250,
        "7x7_1000": cv2.aruco.DICT_7X7_1000,
    }
    return dict_map.get(name.lower())


def main():
    parser = argparse.ArgumentParser(
        description="ArUco Marker Detection and Tracking with 3D Cube Overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default camera (0) and dictionary (6x6_250)
  python aruco_tracker.py

  # Use camera 1 with 4x4_250 dictionary
  python aruco_tracker.py --device 1 --dict 4x4_250

  # Use device path /dev/video4
  python aruco_tracker.py --device /dev/video4 --dict 4x4_250

  # Use 5x5_100 dictionary
  python aruco_tracker.py --dict 5x5_100

  # Use camera calibration file
  python aruco_tracker.py --calib calib_3937__0c45_6366__1280.json

Available dictionaries:
  4x4_50, 4x4_100, 4x4_250, 4x4_1000
  5x5_50, 5x5_100, 5x5_250, 5x5_1000
  6x6_50, 6x6_100, 6x6_250, 6x6_1000 (default: 6x6_250)
  7x7_50, 7x7_100, 7x7_250, 7x7_1000
        """,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Camera device ID (int) or device path (str, e.g., /dev/video4). Default: 0",
    )
    parser.add_argument(
        "--dict",
        type=str,
        default="6x6_250",
        help="ArUco dictionary name (e.g., 6x6_250, 4x4_100). Default: 6x6_250",
    )
    parser.add_argument(
        "--pose",
        action="store_true",
        help="Enable pose estimation axes (requires camera calibration)",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.05,
        help="Physical marker size in meters (default: 0.05)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster but less robust detection (disables multi-strategy detection). Default: robust mode enabled",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default=None,
        help="Path to camera calibration JSON file (e.g., calib_*.json)",
    )

    args = parser.parse_args()

    # Convert dictionary name to constant
    dictionary_id = get_dictionary_from_name(args.dict)
    if dictionary_id is None:
        print(f"Error: Unknown dictionary '{args.dict}'")
        print(
            "Available: 4x4_50, 4x4_100, 4x4_250, 4x4_1000, 5x5_50, 5x5_100, 5x5_250, 5x5_1000, 6x6_50, 6x6_100, 6x6_250, 6x6_1000, 7x7_50, 7x7_100, 7x7_250, 7x7_1000"
        )
        return

    # Convert device argument: try int first, otherwise use as string path
    try:
        camera_device = int(args.device)
    except ValueError:
        # Not a number, use as device path
        camera_device = args.device

    # Determine robust mode (default is robust, unless --fast is specified)
    robust_mode = not args.fast

    # Create tracker
    tracker = ArucoTracker(
        dictionary_id=dictionary_id, camera_id=camera_device, robust=robust_mode
    )
    tracker.marker_size = args.marker_size

    # Load camera calibration if provided
    camera_matrix = None
    dist_coeffs = None

    if args.calib:
        try:
            camera_matrix, dist_coeffs = load_calibration(args.calib)
        except Exception as e:
            print(f"Error loading calibration file: {e}")
            print("Falling back to default camera matrix")
            camera_matrix = None
            dist_coeffs = None

    # Run tracker
    tracker.run(
        show_pose=args.pose, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs
    )


if __name__ == "__main__":
    main()
