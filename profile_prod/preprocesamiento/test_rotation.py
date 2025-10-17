"""
Simple test script to verify rotation functionality
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
from app.utils.rotation_utils import FaceRotationAligner

def test_rotation_aligner():
    """Test the rotation aligner with a simple test"""
    print("="*80)
    print("Testing Face Rotation Aligner")
    print("="*80)

    # Check if model exists
    model_path = "models/best_point_detection_model_v2.pth"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return False

    print(f"✓ Model found at {model_path}")

    try:
        # Initialize aligner
        print("\n1. Initializing FaceRotationAligner...")
        aligner = FaceRotationAligner(model_path, device='cpu')
        print("✓ FaceRotationAligner initialized successfully")

        # Create a simple test image (white rectangle on black background)
        print("\n2. Creating test image...")
        test_image = np.zeros((600, 600, 3), dtype=np.uint8)
        test_image[200:400, 250:350] = [255, 255, 255]  # White rectangle
        print("✓ Test image created")

        print("\n3. Testing point detection...")
        points = aligner.detect_points(test_image)
        print(f"✓ Detected {len(points)} points: {list(points.keys())}")

        # Test rotation angle calculation (if points 34 and 10 are found)
        if '34' in points and '10' in points:
            print("\n4. Testing rotation angle calculation...")
            angle = aligner.calculate_rotation_angle(points['34'], points['10'])
            print(f"✓ Calculated rotation angle: {angle:.2f}°")

            print("\n5. Testing image rotation...")
            rotated = aligner.rotate_image(test_image, angle)
            print(f"✓ Image rotated successfully. New size: {rotated.shape[:2]}")

            print("\n6. Testing full alignment pipeline...")
            aligned_image, metadata = aligner.align_face(test_image)
            if aligned_image is not None:
                print(f"✓ Face aligned successfully")
                print(f"  Rotation angle: {metadata.get('rotation_angle', 'N/A'):.2f}°")
                print(f"  Points detected: {list(metadata.get('points_detected', {}).keys())}")
            else:
                print(f"⚠ Face alignment failed: {metadata.get('error', 'Unknown error')}")
        else:
            print("\n⚠ Points 34 and 10 not detected in test image")
            print(f"  Detected points: {list(points.keys())}")

        print("\n" + "="*80)
        print("✓ All tests passed successfully!")
        print("="*80)
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rotation_aligner()
    sys.exit(0 if success else 1)
