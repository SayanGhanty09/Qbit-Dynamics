import cv2
import numpy as np
import tensorflow as tf
import time
from vision import preprocess_image, detect_lane, detect_obstacle

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return output

class AutonomousCar:
    def __init__(self, model_path):
        """
        Initialize the AI Car with a TFLite model and PID controller.
        """
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Steering PID Controller
            # Tune these gains: P (Primary), I (Bias), D (Damping)
            self.pid = PIDController(kp=0.5, ki=0.01, kd=0.1)
            self.last_time = time.time()
            
            print("AI Model & PID Controller loaded successfully.")
        except Exception as e:
            print(f"Failed to initialize car: {e}")
            self.interpreter = None

    def predict_cnn(self, img_input):
        """
        Runs CNN inference.
        """
        if self.interpreter is None:
            return [0.0, 0.0]
            
        try:
            # Add batch dimension
            img_input = np.expand_dims(img_input, axis=0)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], img_input)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # prediction is [speed, direction]
            # Flatten to 1D if necessary
            return prediction.flatten()
        except Exception as e:
            print(f"CNN Inference error: {e}")
            return [0.0, 0.0]

    def hybrid_control(self, frame):
        """
        Fuses CNN, Lane PID, and Road-Aware Obstacle Dodging.
        """
        try:
            current_time = time.time()
            dt = max(current_time - self.last_time, 0.001)
            self.last_time = current_time

            # 1. Preprocess
            processed_img = preprocess_image(frame)
            
            # 2. Get CNN prediction (baseline behavior)
            prediction = self.predict_cnn(processed_img)
            cnn_speed, cnn_dir = prediction[0], prediction[1]
            
            # 3. Lane/Track Detection (Separates bright objects and white line)
            lane_deviation, lane_detected, bright_mask, line_mask = detect_lane(frame)
            
            # 4. Obstacle Detection (Uses bright_mask and line_mask to find non-line objects)
            obstacle_detected, obstacle_pos, candidates = detect_obstacle(frame, bright_mask, line_mask)
            
            # --- DECISION LOGIC ---
            final_speed = cnn_speed
            setpoint = 0.0  # Default: Stay in the center of the track
            
            # TRACK-GUIDED DODGING
            dodge_action = ""
            if obstacle_detected and obstacle_pos is not None:
                final_speed = 0.35  # Safety speed during dodge
                
                # OBSTACLE AVOIDANCE: Move opposite to obstacle position
                if obstacle_pos == 'left':
                    # Obstacle on LEFT → Move RIGHT
                    setpoint = -0.65
                    dodge_action = "SWERVE RIGHT (OBSTACLE LEFT)"
                elif obstacle_pos == 'right':
                    # Obstacle on RIGHT → Move LEFT
                    setpoint = 0.65
                    dodge_action = "SWERVE LEFT (OBSTACLE RIGHT)"
                elif obstacle_pos == 'center':
                    # Obstacle in CENTER -> Dodge slowly instead of stopping
                    final_speed = 0.2  # Move slowly
                    setpoint = -0.65  # Default to swerving right
                    dodge_action = "SWERVE RIGHT (OBSTACLE CENTER)"
            
            # PID CALCULATION
            error = lane_deviation - setpoint
            pid_correction = self.pid.compute(error, dt)
            
            # Fail-safe: If the entire track is lost, emergency stop
            if not lane_detected:
                final_speed = 0.0
                pid_correction = 0.0
            
            # SENSOR FUSION
            # Using 60% PID for precision lane-following and dodging
            final_dir = (0.4 * cnn_dir) + (0.6 * pid_correction)
            final_dir = max(min(final_dir, 1.0), -1.0)
                
            return [float(final_speed), float(final_dir), obstacle_detected, candidates, dodge_action]
            
        except Exception as e:
            print(f"Decision logic error: {e}")
            return [0.0, 0.0, False, [], ""]

def main():
    import argparse
    parser = argparse.ArgumentParser(description='AI Car Inference')
    parser.add_argument('--image', type=str, help='Path to an image to test on')
    args = parser.parse_args()

    # Path to your TFLite model
    MODEL_PATH = "autonomous_car_model.tflite"
    
    car = AutonomousCar(MODEL_PATH)

    if args.image:
        print(f"Testing on image: {args.image}")
        frame = cv2.imread(args.image)
        if frame is None:
            print("Could not read image.")
            return
            
        control_vector = car.hybrid_control(frame)
        speed, direction, obstacle_detected, candidates, dodge_action = control_vector
        print("\n--- INFERENCE RESULT ---")
        print(f"AI Speed: {speed:.4f}")
        print(f"AI Direction: {direction:.4f}")
        print(f"Obstacle: {obstacle_detected} ({dodge_action})")
        print(f"Candidates Found: {len(candidates)}")
        print("------------------------\n")
        return

    # Camera mode
    cap = cv2.VideoCapture(0) # Open Raspberry Pi Camera
    
    # Set Resolution to 640x480 as per requirement
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting Inference Loop (640x480)... Press 'q' to quit.")
    
    try:
        while cap.isOpened():
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Fail-safe: [0, 0]")
                continue
                
            # Run Hybrid Decision Logic
            control_vector = car.hybrid_control(frame)
            speed, direction, obstacle_detected, candidates, dodge_action = control_vector
            
            # Calculate Latency
            latency = (time.time() - start_time) * 1000
            
            # Visual Debugging: Draw boxes for all candidates
            for cand in candidates:
                x, y, w_box, h_box = cand['rect']
                color = (0, 255, 0) # Green for candidate
                if obstacle_detected:
                    color = (0, 0, 255) # Red if it triggers dodge
                cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), color, 2)

            # Output final speed and direction to terminal
            print(f"RES: 640x480 | AI Speed: {speed:.2f} | AI Dir: {direction:.2f} | Latency: {latency:.1f}ms")
            
            # --- HUD (Head-Up Display) Overlay ---
            # Speed
            cv2.putText(frame, f"Speed: {speed:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Direction
            dir_text = f"Dir: {direction:.2f}"
            if direction > 0.1:
                dir_text += " (RIGHT)"
            elif direction < -0.1:
                dir_text += " (LEFT)"
            else:
                dir_text += " (CENTER)"
            cv2.putText(frame, dir_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Latency
            cv2.putText(frame, f"Latency: {latency:.1f} ms", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Visualization Warning (Shifted down to avoid HUD)
            if obstacle_detected:
                cv2.putText(frame, "DODGING OBSTACLE", (50, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # Show the AI's Logic/Prediction
                cv2.putText(frame, dodge_action, (50, 180), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow("Original Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping car...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
