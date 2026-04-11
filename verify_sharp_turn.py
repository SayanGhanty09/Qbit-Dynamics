import cv2
import numpy as np
from vision import detect_lane
from inference import AutonomousCar
import os

def test_sharp_turn():
    print("--- [VERIFICATION] Sharp 90-Degree Turn Logic ---")
    
    # Pre-requisite: model file exists
    model_path = "autonomous_car_model.tflite"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    car = AutonomousCar(model_path)
    
    # 1. Create a synthetic frame (640x480)
    # Background: black
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a 90-degree right turn (white dotted line equivalent)
    # The ROI in vision.detect_lane is the bottom half: [240:480, :]
    # Let's draw in the ROI area.
    
    # Vertical segment (entering the turn)
    cv2.line(frame, (320, 480), (320, 380), (255, 255, 255), 10)
    cv2.line(frame, (320, 360), (320, 300), (255, 255, 255), 10)
    
    # Horizontal segment (the sharp 90 deg turn to the right)
    cv2.line(frame, (320, 300), (450, 300), (255, 255, 255), 10)
    cv2.line(frame, (470, 300), (600, 300), (255, 255, 255), 10)
    
    # 2. Run Hybrid Control
    result = car.hybrid_control(frame)
    speed, dir, obs, cands, dodge, space, tl, stop, status = result
    
    print(f"\n[SCENARIO: 90-DEGREE RIGHT TURN]")
    print(f"Status:       {status}")
    print(f"Speed:        {speed}")
    print(f"Direction:    {dir:.4f} (Expect close to +1.0 for right turn)")
    
    # Let's do a Left turn too
    frame_l = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(frame_l, (320, 480), (320, 380), (255, 255, 255), 10)
    cv2.line(frame_l, (320, 360), (320, 300), (255, 255, 255), 10)
    cv2.line(frame_l, (320, 300), (190, 300), (255, 255, 255), 10)
    cv2.line(frame_l, (170, 300), (40, 300), (255, 255, 255), 10)
    
    result_l = car.hybrid_control(frame_l)
    print(f"\n[SCENARIO: 90-DEGREE LEFT TURN]")
    print(f"Status:       {result_l[8]}")
    print(f"Speed:        {result_l[0]}")
    print(f"Direction:    {result_l[1]:.4f} (Expect close to -1.0 for left turn)")

if __name__ == "__main__":
    test_sharp_turn()
