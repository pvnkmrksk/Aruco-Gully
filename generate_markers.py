#!/usr/bin/env python3
"""
Generate ArUco markers for printing
"""

import cv2
import argparse
import os


def generate_markers(dictionary_id, marker_ids, output_dir="markers", size=200):
    """
    Generate ArUco markers

    Args:
        dictionary_id: ArUco dictionary type
        marker_ids: List of marker IDs to generate
        output_dir: Output directory for markers
        size: Marker size in pixels
    """
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating markers using dictionary {dictionary_id}")
    print(f"Output directory: {output_dir}")
    print(f"Marker size: {size}x{size} pixels\n")

    for marker_id in marker_ids:
        try:
            marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, size)

            filename = os.path.join(output_dir, f"marker_{marker_id}.png")
            cv2.imwrite(filename, marker_image)
            print(f"Generated marker ID {marker_id}: {filename}")

        except cv2.error as e:
            print(f"Error generating marker {marker_id}: {e}")

    print(f"\nAll markers saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate ArUco markers")
    parser.add_argument(
        "--dictionary",
        type=int,
        default=cv2.aruco.DICT_6X6_250,
        help="ArUco dictionary ID (default: DICT_6X6_250)",
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Marker IDs to generate (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="markers",
        help="Output directory (default: markers)",
    )
    parser.add_argument(
        "--size", type=int, default=200, help="Marker size in pixels (default: 200)"
    )

    args = parser.parse_args()

    generate_markers(args.dictionary, args.ids, args.output, args.size)


if __name__ == "__main__":
    main()
