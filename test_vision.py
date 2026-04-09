import cv2
import numpy as np
from vision import detect_obstacle

# Create a mock track mask
# Track mask is 640x480, let's make it entirely white (track), with a black rectangle (obstacle)
mask = np.zeros((480, 640), dtype=np.uint8)

# Draw track (polygon or rectangle)
# Let's just make the bottom half white
mask[240:480, 100:540] = 255

# Draw an obstacle (black hole inside the white track)
# the track mask roi_top is 240. So y must be > 240
# Obstacle at x=300, y=300, width=50, height=50
mask[300:350, 300:350] = 0

# Mock frame
frame = np.zeros((480, 640, 3), dtype=np.uint8)
frame[240:480, 100:540] = (200, 200, 200) # white track
frame[300:350, 300:350] = (0, 0, 0) # black obstacle

obstacle_found, pos, cands = detect_obstacle(frame, mask)
print(f"Detected: {obstacle_found}, Position: {pos}")
for c in cands:
    print(c)
