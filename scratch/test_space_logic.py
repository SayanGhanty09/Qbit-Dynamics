import cv2
import numpy as np
import sys
import os

# Add the project dir to path
sys.path.append(os.getcwd())

from vision import get_path_space

def test_space():
    # Create fake line mask (white path from pixel 100 to 500)
    # 640x480 gray image
    line_mask = np.zeros((240, 640), dtype=np.uint8)
    line_mask[:, 100:540] = 255
    
    # Case 1: Obstacle at center (300 in 640)
    # Obstacle rect: (x, y, w, h)
    # Let's put it at x=320, w=40
    # Obstacle is from 320 to 360.
    # Path is from 100 to 540.
    # Left space: 320 - 100 = 220
    # Right space: 540 - 360 = 180
    obs_rect = (320, 100, 40, 40)
    ls, rs, start, end = get_path_space(line_mask, obs_rect)
    print(f"Test 1 (Center-Right bias): L={ls}, R={rs}, Path=[{start}, {end}]")
    assert ls == 220, f"Expected 220, got {ls}"
    assert rs == 180, f"Expected 180, got {rs}"
    
    # Case 2: Obstacle shifted left
    # x=200, w=40 (200-240)
    # Left space: 200 - 100 = 100
    # Right space: 540 - 240 = 300
    obs_rect_2 = (200, 100, 40, 40)
    ls2, rs2, start2, end2 = get_path_space(line_mask, obs_rect_2)
    print(f"Test 2 (Left shift): L={ls2}, R={rs2}, Path=[{start2}, {end2}]")
    assert ls2 == 100, f"Expected 100, got {ls2}"
    assert rs2 == 300, f"Expected 300, got {rs2}"

    print("All logic tests passed!")

if __name__ == "__main__":
    test_space()
