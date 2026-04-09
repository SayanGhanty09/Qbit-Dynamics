import cv2
import numpy as np
from vision import detect_lane, detect_obstacle

# Mock a black track (0, 0, 0)
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Add a white line in the middle
# White in BGR is (255, 255, 255)
frame[100:480, 300:340] = (255, 255, 255)

# Add a dark blue obstacle on the logic's "left" 
# Blue in BGR is (255, 0, 0)
frame[300:360, 100:160] = (200, 0, 0)

# Add a red obstacle on the logic's "right" 
# Red in BGR is (0, 0, 255)
frame[300:360, 500:560] = (0, 0, 200)

lane_deviation, lane_detected, foreground_mask, line_mask = detect_lane(frame)
print(f"Lane deviation: {lane_deviation:.2f}, detected: {lane_detected}")

obstacle_detected, obstacle_pos, candidates = detect_obstacle(frame, foreground_mask, line_mask)
print(f"Obstacle detected: {obstacle_detected}, Pos: {obstacle_pos}")
if obstacle_detected:
    for cand in candidates:
        print("Candidate:", cand)
