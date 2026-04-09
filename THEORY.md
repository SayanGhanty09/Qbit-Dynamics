# AI Car: Theoretical Foundation

This document explains the technical implementation and design choices for the autonomous driving system.

## 1. AI Model Architecture (CNN)
The core "brain" of the car is a **Convolutional Neural Network (CNN)** inspired by the NVIDIA DAVE-2 architecture.

*   **Input Layer**: Accepts $60 \times 160$ normalized RGB images (preprocessed from $640 \times 480$ capture).
*   **Convolutional Layers**:
    *   3 layers of $5 \times 5$ kernels with strides for spatial reduction and feature extraction (lane edges, curves).
    *   1 layer of $3 \times 3$ kernels for higher-level pattern recognition.
*   **Activation**: **ReLU** is used for hidden layers to ensure fast convergence. The output layer uses **Tanh** to normalize control vectors to the $[-1, 1]$ range.
*   **Flatten & Dense**: High-dimensional features are flattened and passed through multiple fully connected layers (100, 50, 10 neurons) to arrive at the final controls.

## 2. Robust Vision Algorithms (Computer Vision)
While the AI handles the primary driving, we use classical CV for high-reliability safety overrides.

### Adaptive Lane Detection
To handle **Lighting Variations**, we avoid fixed thresholds.
*   **Algorithm**: **Otsu’s Binarization**.
*   **Process**: The system analyzes the grayscale histogram of the frame to automatically find the optimal separation point between the white line and the black track.
*   **Filtering**: Gaussian blurring is applied to remove "salt-and-pepper" noise from surface reflections.

### Color-Agnostic Obstacle Detection
Since obstacles can be any color, we don't rely on color segmentation (HSV).
*   **Algorithm**: **ROI Contour Analysis**.
*   **Logic**: Anything on the black surface that isn't the white line is an obstacle.
*   **Verification**: We filter detected objects by **Aspect Ratio** and **Area**. Vertical, thin objects are treated as lanes; blob-like objects are treated as obstacles.

## 3. PID-Controlled Steering
To ensure smooth steering, we implemented a **Proportional-Integral-Derivative (PID)** controller.

*   **Proportional (P)**: Corrects steering based on current distance from the line.
*   **Integral (I)**: Corrects for accumulated bias (e.g., if the car is consistently pulled left).
*   **Derivative (D)**: Acts as a damper to prevent oversteering and "wobbling."

## 4. Active Dodging & Sensor Fusion
Instead of stopping, the car now uses **Setpoint Shifting** to navigate around obstacles.

1.  **Track-Masking (The "Safe Zone")**: To prevent the car from reacting to floor glares or track edges (like A4 paper boundaries), we use the detected road area as a binary mask. We only detect edges **inside** this mask.
2.  **Road-Aware Swerving**: When an obstacle is detected in the center, the car checks its `lane_deviation`. It intelligently swerves toward the side of the track with the most available space.
3.  **Sensor Fusion**: The final steering command is a weighted blend:
    *   **40% CNN Prediction**: Provides intuitive "vision" for upcoming curves.
    *   **60% PID Correction**: Provides industrial precision for centering and dodging.

---

