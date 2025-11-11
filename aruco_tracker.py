#!/usr/bin/env python3
"""
Simple ArUco Marker Detection and Tracking Pipeline
Detects and tracks ArUco markers in real-time from camera feed
"""

import cv2
import numpy as np
import argparse


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
        Get default camera matrix (approximation without calibration)

        Args:
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            camera_matrix: Default camera intrinsic matrix
            dist_coeffs: Default distortion coefficients (zeros)
        """
        # Approximate focal length (assuming ~60 degree FOV)
        focal_length = frame_width
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

        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

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

    def draw_markers(self, frame, corners, ids, camera_matrix=None, dist_coeffs=None):
        """
        Draw detected markers on the frame with 3D cube overlay

        Args:
            frame: Input frame
            corners: Detected marker corners
            ids: Detected marker IDs
            camera_matrix: Camera intrinsic matrix for 3D drawing
            dist_coeffs: Distortion coefficients

        Returns:
            frame: Frame with drawn markers
        """
        if ids is not None:
            # Draw marker boundaries
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Draw 3D cube for each marker
            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                corner = corner[0]

                # Calculate center
                center_x = int(np.mean(corner[:, 0]))
                center_y = int(np.mean(corner[:, 1]))

                # Draw center point
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

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

                # Draw 3D cube if camera matrix is available
                if camera_matrix is not None:
                    self.draw_cube(frame, corner, camera_matrix, dist_coeffs)

        return frame

    def draw_cube(self, frame, corner, camera_matrix, dist_coeffs=None):
        """
        Draw a 3D cube around the marker

        Args:
            frame: Input frame
            corner: Marker corner points (4x2 array)
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        # Define 3D points of a cube (in marker coordinate system)
        # Cube extends above the marker plane
        cube_size = self.marker_size * 0.5  # Half the marker size
        object_points = np.array(
            [
                [-cube_size, -cube_size, 0],  # Bottom front-left
                [cube_size, -cube_size, 0],  # Bottom front-right
                [cube_size, cube_size, 0],  # Bottom back-right
                [-cube_size, cube_size, 0],  # Bottom back-left
                [-cube_size, -cube_size, cube_size * 2],  # Top front-left
                [cube_size, -cube_size, cube_size * 2],  # Top front-right
                [cube_size, cube_size, cube_size * 2],  # Top back-right
                [-cube_size, cube_size, cube_size * 2],  # Top back-left
            ],
            dtype=np.float32,
        )

        # Estimate pose from marker corners
        # Use the corner points to estimate the pose
        marker_points = np.array(
            [
                [-self.marker_size / 2, -self.marker_size / 2, 0],
                [self.marker_size / 2, -self.marker_size / 2, 0],
                [self.marker_size / 2, self.marker_size / 2, 0],
                [-self.marker_size / 2, self.marker_size / 2, 0],
            ],
            dtype=np.float32,
        )

        # Estimate pose
        if dist_coeffs is None:
            dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            marker_points, corner, camera_matrix, dist_coeffs
        )

        if success:
            # Project 3D cube points to 2D
            image_points, _ = cv2.projectPoints(
                object_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            image_points = np.int32(image_points).reshape(-1, 2)

            # Draw cube edges
            # Bottom face
            cv2.line(
                frame, tuple(image_points[0]), tuple(image_points[1]), (255, 0, 0), 2
            )
            cv2.line(
                frame, tuple(image_points[1]), tuple(image_points[2]), (255, 0, 0), 2
            )
            cv2.line(
                frame, tuple(image_points[2]), tuple(image_points[3]), (255, 0, 0), 2
            )
            cv2.line(
                frame, tuple(image_points[3]), tuple(image_points[0]), (255, 0, 0), 2
            )

            # Top face
            cv2.line(
                frame, tuple(image_points[4]), tuple(image_points[5]), (0, 0, 255), 2
            )
            cv2.line(
                frame, tuple(image_points[5]), tuple(image_points[6]), (0, 0, 255), 2
            )
            cv2.line(
                frame, tuple(image_points[6]), tuple(image_points[7]), (0, 0, 255), 2
            )
            cv2.line(
                frame, tuple(image_points[7]), tuple(image_points[4]), (0, 0, 255), 2
            )

            # Vertical edges
            cv2.line(
                frame, tuple(image_points[0]), tuple(image_points[4]), (0, 255, 0), 2
            )
            cv2.line(
                frame, tuple(image_points[1]), tuple(image_points[5]), (0, 255, 0), 2
            )
            cv2.line(
                frame, tuple(image_points[2]), tuple(image_points[6]), (0, 255, 0), 2
            )
            cv2.line(
                frame, tuple(image_points[3]), tuple(image_points[7]), (0, 255, 0), 2
            )

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
            use_default_camera_matrix: Use default camera matrix for cube drawing if no calibration
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

                # Draw markers with 3D cube overlay
                frame = self.draw_markers(
                    frame, corners, ids, camera_matrix, dist_coeffs
                )

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

    # For pose estimation, you would need to load camera calibration data
    # Example (uncomment and provide your calibration file):
    # camera_matrix = np.load('camera_matrix.npy')
    # dist_coeffs = np.load('dist_coeffs.npy')
    camera_matrix = None
    dist_coeffs = None

    # Run tracker
    tracker.run(
        show_pose=args.pose, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs
    )


if __name__ == "__main__":
    main()
