import cv2
import numpy as np
import tensorflow as tf
import time
from vision import preprocess_image, detect_lane, detect_obstacle, detect_traffic_light, detect_stop_sign, get_path_space


# ─────────────────────────────────────────────
#  PID Controller
# ─────────────────────────────────────────────
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral   = 0

    def compute(self, error, dt):
        self.integral     += error * dt
        derivative         = (error - self.prev_error) / dt
        output             = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error    = error
        # Anti-windup: clamp integral
        self.integral      = max(min(self.integral, 5.0), -5.0)
        return output


# ─────────────────────────────────────────────
#  Traffic Light State Machine
# ─────────────────────────────────────────────
class TrafficLightStateMachine:
    """
    Manages debouncing for traffic light transitions.
    - Requires DEBOUNCE_FRAMES consecutive matching frames to commit a new state.
    - Resets to None after RESET_FRAMES frames of 'no detection', so the
      car always resumes when the light is out of view.
    """
    DEBOUNCE_FRAMES = 5   # Frames needed to confirm a new state
    RESET_FRAMES    = 10  # Frames of no-detection before clearing state

    def __init__(self):
        self.state           = None   # None = no traffic light in view
        self.candidate       = None
        self.candidate_count = 0
        self.none_count      = 0      # Counts consecutive frames with no detection

    def update(self, raw_state):
        """
        Update with a raw detection result each frame.
        Returns the committed (debounced) state, or None if no light visible.
        """
        if raw_state is None:
            # Nothing detected this frame
            self.candidate_count = 0
            self.none_count += 1
            if self.none_count >= self.RESET_FRAMES:
                # Light has been absent long enough → clear state
                self.state     = None
                self.candidate = None
            return self.state

        # A light is visible this frame
        self.none_count = 0
        if raw_state == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate       = raw_state
            self.candidate_count = 1

        if self.candidate_count >= self.DEBOUNCE_FRAMES:
            self.state = self.candidate

        return self.state


# ─────────────────────────────────────────────
#  Main Car Class
# ─────────────────────────────────────────────
class AutonomousCar:
    # ------- Tunable constants -------
    STOP_SIGN_AREA_THRESHOLD  = 8000   # Pixel area: stop sign is truly close
    STOP_SIGN_CONFIRM_FRAMES  = 15     # Must see a large stop sign this many frames in a row
    SHARP_TURN_THRESHOLD      = 0.65   # |deviation| above this → steep turn mode
    SHARP_TURN_GAIN           = 1.6    # Extra multiplier for steep-turn PID correction
    CRUISE_SPEED              = 0.55   # Normal speed (normalised 0-1)
    DODGE_SPEED               = 0.35   # Speed while dodging an obstacle
    SLOW_DODGE_SPEED          = 0.25   # Speed during centre-obstacle dynamic dodge

    def __init__(self, model_path):
        try:
            self.interpreter    = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details  = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # PID: kp (proportional gain), ki (integral), kd (derivative / damping)
            self.pid            = PIDController(kp=0.6, ki=0.01, kd=0.12)
            self.last_time      = time.time()

            # Traffic light debouncer
            self.tl_state_machine = TrafficLightStateMachine()

            # Stop-sign: latch ONLY after N consecutive confirmed frames
            self.permanently_stopped   = False
            self.stop_sign_frame_count = 0   # Consecutive frames with a large stop sign

            print("AI Model & PID Controller loaded successfully.")
        except Exception as e:
            print(f"Failed to initialize car: {e}")
            self.interpreter           = None
            self.tl_state_machine      = TrafficLightStateMachine()
            self.permanently_stopped   = False
            self.stop_sign_frame_count = 0

    # ─── CNN Inference ────────────────────────
    def predict_cnn(self, img_input):
        if self.interpreter is None:
            return [0.0, 0.0]
        try:
            img_input = np.expand_dims(img_input, axis=0)
            self.interpreter.set_tensor(self.input_details[0]['index'], img_input)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
            return prediction.flatten()
        except Exception as e:
            print(f"CNN Inference error: {e}")
            return [0.0, 0.0]

    # ─── Main Decision Logic ──────────────────
    def hybrid_control(self, frame):
        """
        Fuses CNN, Lane PID, Traffic Light, Stop Sign, and Road-Aware Obstacle Dodging.

        Returns:
            [speed, direction, obstacle_detected, candidates,
             dodge_action, space_info, tl_state, stop_sign_detected, status_text]
        """
        try:
            current_time = time.time()
            dt           = max(current_time - self.last_time, 0.001)
            self.last_time = current_time

            # ── PERMANENT STOP (triggered by Stop Sign) ────────────────────
            if self.permanently_stopped:
                return [0.0, 0.0, False, [], "STOPPED (STOP SIGN)", None,
                        self.tl_state_machine.state, False, "PERMANENT STOP"]

            # ── VISION PIPELINE ────────────────────────────────────────────
            processed_img = preprocess_image(frame)
            prediction    = self.predict_cnn(processed_img)
            cnn_speed, cnn_dir = prediction[0], prediction[1]

            lane_deviation, lane_detected, bright_mask, line_mask = detect_lane(frame)
            obstacle_detected, obstacle_pos, candidates            = detect_obstacle(frame, bright_mask, line_mask)

            # Traffic light detection
            raw_tl_state, tl_confidence = detect_traffic_light(frame)
            tl_state = self.tl_state_machine.update(raw_tl_state)

            # Stop sign detection
            stop_sign_found, stop_sign_area = detect_stop_sign(frame)

            # ── 1. TRAFFIC LIGHT: RED only stops; GREEN resumes ───────────
            # YELLOW is ignored (treated as no-light = keep going)
            if tl_state == 'RED':
                print(f"[TRAFFIC LIGHT] RED → STOPPING")
                return [0.0, 0.0, False, [], "TRAFFIC LIGHT: RED", None,
                        tl_state, False, "STOP – RED LIGHT"]
            # GREEN: car is explicitly allowed to go (no action needed, falls through)

            # ── 2. STOP SIGN: Proximity + Debounce Check ──────────────────
            # Require STOP_SIGN_CONFIRM_FRAMES consecutive frames of a large stop sign
            if stop_sign_found and stop_sign_area >= self.STOP_SIGN_AREA_THRESHOLD:
                self.stop_sign_frame_count += 1
                if self.stop_sign_frame_count >= self.STOP_SIGN_CONFIRM_FRAMES:
                    print(f"[STOP SIGN] Confirmed after {self.stop_sign_frame_count} frames → PERMANENT STOP")
                    self.permanently_stopped = True
                    return [0.0, 0.0, False, [], "STOP SIGN REACHED", None,
                            tl_state, True, "PERMANENT STOP"]
                else:
                    print(f"[STOP SIGN] Candidate frame {self.stop_sign_frame_count}/{self.STOP_SIGN_CONFIRM_FRAMES}")
            else:
                # Reset counter if stop sign not visible / not close enough
                self.stop_sign_frame_count = 0

            # ── 3. OBSTACLE AVOIDANCE ──────────────────────────────────────
            final_speed = self.CRUISE_SPEED
            setpoint    = 0.0
            dodge_action = ""
            space_info   = None

            if obstacle_detected and obstacle_pos is not None:
                final_speed = self.DODGE_SPEED

                if obstacle_pos == 'left':
                    setpoint     = -0.65
                    dodge_action = "SWERVE RIGHT (OBSTACLE LEFT)"
                elif obstacle_pos == 'right':
                    setpoint     = 0.65
                    dodge_action = "SWERVE LEFT (OBSTACLE RIGHT)"
                elif obstacle_pos == 'center':
                    # Dynamic: choose side with most space
                    largest_cand = max(candidates, key=lambda c: c['area'])
                    obs_rect     = largest_cand['rect']
                    l_space, r_space, p_start, p_end = get_path_space(line_mask, obs_rect)
                    space_info   = (l_space, r_space, p_start, p_end, obs_rect)

                    if l_space > r_space:
                        setpoint     = 0.7
                        dodge_action = f"DODGE L (space: {l_space} > {r_space})"
                    else:
                        setpoint     = -0.7
                        dodge_action = f"DODGE R (space: {r_space} > {l_space})"
                    final_speed = self.SLOW_DODGE_SPEED

            # ── 4. SHARP / STEEP TURN HANDLING ────────────────────────────
            # When the lane deviation is very large (>65%), amplify the PID correction
            # to punch through 90-degree bends without drifting off-track.
            is_sharp_turn = abs(lane_deviation) > self.SHARP_TURN_THRESHOLD
            if is_sharp_turn and not obstacle_detected:
                setpoint    = 0.0   # Trust the lane reading, aim for centre
                # Speed reduction on steep turn to prevent cutting the corner
                final_speed = min(final_speed, 0.4)

            # ── 5. PID CALCULATION ─────────────────────────────────────────
            error          = lane_deviation - setpoint
            pid_correction = self.pid.compute(error, dt)

            if is_sharp_turn and not obstacle_detected:
                pid_correction *= self.SHARP_TURN_GAIN  # Boost steering into the bend

            # Fail-safe: total track loss → emergency stop
            if not lane_detected:
                final_speed    = 0.0
                pid_correction = 0.0

            # ── 6. SENSOR FUSION ───────────────────────────────────────────
            # 35% CNN baseline + 65% PID precision
            final_dir = (0.35 * cnn_dir) + (0.65 * pid_correction)
            final_dir = max(min(final_dir, 1.0), -1.0)

            # Determine status for HUD
            if obstacle_detected:
                status_text = f"DODGING: {dodge_action}"
            elif is_sharp_turn:
                turn_dir    = "LEFT" if lane_deviation < 0 else "RIGHT"
                status_text = f"SHARP TURN {turn_dir}"
            else:
                status_text = "FOLLOWING LINE"

            return [float(final_speed), float(final_dir), obstacle_detected, candidates,
                    dodge_action, space_info, tl_state, stop_sign_found, status_text]

        except Exception as e:
            print(f"Decision logic error: {e}")
            return [0.0, 0.0, False, [], "", None, "GREEN", False, "ERROR"]


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description='AI Car Inference')
    parser.add_argument('--image', type=str, help='Path to an image to test on')
    args = parser.parse_args()

    MODEL_PATH = "autonomous_car_model.tflite"
    car = AutonomousCar(MODEL_PATH)

    # ── Static Image Test Mode ───────────────
    if args.image:
        print(f"Testing on image: {args.image}")
        frame = cv2.imread(args.image)
        if frame is None:
            print("Could not read image.")
            return
        result = car.hybrid_control(frame)
        speed, direction, obs, cands, dodge, space, tl, stop, status = result
        print("\n--- INFERENCE RESULT ---")
        print(f"Status:          {status}")
        print(f"AI Speed:        {speed:.4f}")
        print(f"AI Direction:    {direction:.4f}")
        print(f"Traffic Light:   {tl}")
        print(f"Obstacle:        {obs} ({dodge})")
        print(f"Stop Sign:       {stop}")
        print(f"Candidates:      {len(cands)}")
        print("------------------------\n")
        return

    # ── Camera Live Mode ─────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Starting Inference Loop (640x480)... Press 'q' to quit.")

    try:
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Fail-safe: [0, 0]")
                continue

            result = car.hybrid_control(frame)
            speed, direction, obs, cands, dodge, space_info, tl_state, stop_sign, status = result

            latency = (time.time() - start_time) * 1000

            # ── Draw Obstacle Boxes ──────────────
            for cand in cands:
                x, y, w_box, h_box = cand['rect']
                color = (0, 0, 255) if obs else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)

            # ── HUD Text ─────────────────────────
            # Status bar background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            cv2.putText(frame, f"Speed: {speed:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            dir_label = "RIGHT" if direction > 0.1 else ("LEFT" if direction < -0.1 else "CENTER")
            cv2.putText(frame, f"Dir: {direction:.2f} ({dir_label})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Latency: {latency:.1f}ms",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Status overlay
            status_color = (0, 255, 0)
            if tl_state in ('RED', 'YELLOW') or stop_sign:
                status_color = (0, 0, 255)
            elif obs:
                status_color = (0, 165, 255)  # Orange

            cv2.putText(frame, status,
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

            # Traffic Light Indicator – only show when a light is visible
            tl_colors = {'RED': (0, 0, 255), 'YELLOW': (0, 200, 255), 'GREEN': (0, 255, 0)}
            if tl_state in tl_colors:
                tl_color = tl_colors[tl_state]
                cv2.circle(frame, (600, 30), 20, tl_color, -1)
                cv2.putText(frame, tl_state,
                            (570, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tl_color, 1)
            else:
                # No traffic light detected – draw a hollow grey ring
                cv2.circle(frame, (600, 30), 20, (80, 80, 80), 2)
                cv2.putText(frame, "---",
                            (578, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

            # Space Visualisation
            if space_info is not None:
                l_space, r_space, p_start, p_end, obs_rect = space_info
                ox, oy, ow, oh = obs_rect

                cv2.line(frame, (p_start, oy), (p_start, oy + oh), (255, 255, 0), 2)
                cv2.line(frame, (p_end,   oy), (p_end,   oy + oh), (255, 255, 0), 2)

                if l_space > 0:
                    cv2.rectangle(frame, (p_start, oy), (ox, oy + oh), (255, 255, 0), 1)
                    cv2.putText(frame, f"L:{l_space}", (p_start + 5, oy + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if r_space > 0:
                    cv2.rectangle(frame, (ox + ow, oy), (p_end, oy + oh), (255, 0, 255), 1)
                    cv2.putText(frame, f"R:{r_space}", (ox + ow + 5, oy + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # Terminal log
            tl_label = tl_state if tl_state else "---"
            print(f"{latency:.1f}ms | Speed: {speed:.2f} | Dir: {direction:.2f} | TL: {tl_label} | {status}")


            cv2.imshow("QBit Dynamics – AI Car", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping car...")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
