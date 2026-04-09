import cv2
import os
import csv
import time

def main():
    # Setup directories and files
    img_dir = "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        
    csv_file = "dataset.csv"
    
    # Check if we need to write headers
    write_header = not os.path.exists(csv_file)
    
    # Open webcam (0 is usually the built-in laptop webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("--- Data Collection Started ---")
    print("Click on the video window and use keys to drive:")
    print("  'w' : Forward")
    print("  'a' : Left")
    print("  'd' : Right")
    print("  's' : Backward")
    print("  'q' : Quit and stop collecting")
    print("Only frames where you are actively 'driving' will be saved.")

    image_counter = 0

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['image_path', 'speed', 'direction'])
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame to our target 160x120 to save exactly what the model expects
            frame_resized = cv2.resize(frame, (160, 120))
            
            # Show the original frame so you can see what you are doing
            cv2.imshow("Webcam - Driving Simulator", frame)
            
            # Wait for key press (50 ms)
            key = cv2.waitKey(50) & 0xFF
            
            # Initialize variables outside the loop if they don't exist
            if 'current_steering' not in locals():
                current_steering = 0.0
                current_speed = 0.0
                
            speed = 0.0
            record_frame = False
            
            # Simulated analog properties
            STEERING_STEP = 0.2    # How fast it turns when key is held
            STEERING_DECAY = 0.1   # How fast it returns to center when released
            
            if key == ord('a'):
                current_steering -= STEERING_STEP
                current_speed = 0.5
                record_frame = True
            elif key == ord('d'):
                current_steering += STEERING_STEP
                current_speed = 0.5
                record_frame = True
            elif key == ord('w'):
                # Approaching center smoothly if just going forward
                if current_steering > 0: current_steering -= STEERING_DECAY
                if current_steering < 0: current_steering += STEERING_DECAY
                current_speed = 0.5
                record_frame = True
            elif key == ord('s'):
                current_speed = -0.5
                record_frame = True
            elif key == ord('q'):
                print("Stopping data collection.")
                break
            else:
                # Decay steering to 0 when no keys are pressed
                if current_steering > 0: current_steering -= STEERING_DECAY
                if current_steering < 0: current_steering += STEERING_DECAY
                
            # Clamp steering between -1.0 and 1.0
            current_steering = max(-1.0, min(1.0, current_steering))
            
            # Snap to 0 if it's very close to prevent floating point drift
            if abs(current_steering) < 0.05:
                current_steering = 0.0
                
            # Save the image and label only if an action was taken
            if record_frame:
                timestamp = int(time.time() * 1000)
                img_name = f"img_{timestamp}.jpg"
                img_path = os.path.join(img_dir, img_name)
                
                # Save the resized frame
                cv2.imwrite(img_path, frame_resized)
                
                # Write to CSV using relative path (e.g., images/img_xxxxx.jpg)
                # We round 'current_steering' to 2 decimal places so it looks clean like -0.80 or 0.40
                writer.writerow([img_path.replace('\\', '/'), current_speed, round(current_steering, 2)])
                image_counter += 1
                
                print(f"Recorded: {img_path} | Speed: {current_speed} | Dir: {current_steering:.2f} (Total: {image_counter})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"--- Data Collection Complete. Total images saved: {image_counter} ---")

if __name__ == "__main__":
    main()
