import cv2
import numpy as np

def preprocess_image(image, target_size=(160, 120)):
    """
    Standard preprocessing: Resize, ROI, Normalization.
    """
    # Resize
    image = cv2.resize(image, target_size)
    
    # ROI: Keep the bottom half
    h, w = target_size[1], target_size[0]
    roi = image[h//2:h, 0:w]
    
    # Normalization (0-1) for CNN input
    normalized = roi.astype(np.float32) / 255.0
    return normalized

def detect_lane(image):
    """
    Detect white center line on a black track.
    Returns: deviation, detected_status, foreground_mask, line_mask.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's Thresholding (Adaptive)
    _, bright_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert to HSV to isolate colors
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    
    # Any colored object will have substantial saturation > 60
    _, high_sat_mask = cv2.threshold(s_channel, 60, 255, cv2.THRESH_BINARY)
    _, low_sat_mask = cv2.threshold(s_channel, 60, 255, cv2.THRESH_BINARY_INV)
    
    # The actual white line is bright AND low saturation
    line_mask = cv2.bitwise_and(bright_mask, low_sat_mask)
    
    # Foreground includes anything bright PLUS anything saturated/colored (even if it's dark blue/red)
    foreground_mask = cv2.bitwise_or(bright_mask, high_sat_mask)
    
    # Optional morphological opening to remove small noise from the line mask
    kernel = np.ones((5, 5), np.uint8)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel)
    
    # Calculate moments to find centroid of the white line
    M = cv2.moments(line_mask)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        w_center = foreground_mask.shape[1] / 2
        deviation = (cx - w_center) / w_center  # Normalize to -1 to 1
        return deviation, True, foreground_mask, line_mask
    else:
        return 0, False, foreground_mask, line_mask

def detect_obstacle(image, foreground_mask=None, line_mask=None):
    """
    Detect ANY color obstacle on the black track.
    Obstacles are bright items that do NOT belong to the white line.
    Returns the MOST PROMINENT obstacle (largest by area) and its position.
    """
    h, w = image.shape[:2]
    roi_top = h // 2
    
    if foreground_mask is None or line_mask is None:
        return False, None, []
        
    current_bright = foreground_mask[roi_top:h, :]
    current_line = line_mask[roi_top:h, :]
    
    # Obstacles are just bright things minus the white line
    obstacles_mask = cv2.subtract(current_bright, current_line)
    
    # Clean up minor edge artifact noise
    kernel_mask = np.ones((3, 3), np.uint8)
    obstacles_mask = cv2.erode(obstacles_mask, kernel_mask, iterations=1)
    obstacles_mask = cv2.dilate(obstacles_mask, kernel_mask, iterations=2)
    
    # Find contours
    obs_contours, _ = cv2.findContours(obstacles_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    obstacle_found = False
    final_position = None
    largest_obstacle = None
    largest_area = 0

    for cnt in obs_contours:
        area = cv2.contourArea(cnt)
        if area > 80: # Filter small noise
            x, y, w_obj, h_obj = cv2.boundingRect(cnt)
            y_full = y + roi_top
            
            candidates.append({'rect': (x, y_full, w_obj, h_obj), 'area': area})
            obstacle_found = True
            
            # Track the largest (most prominent) obstacle
            if area > largest_area:
                largest_area = area
                largest_obstacle = {
                    'x': x,
                    'w': w_obj,
                    'area': area
                }
    
    # Determine position based on the LARGEST obstacle
    if largest_obstacle is not None:
        # Use the CENTER of the obstacle to determine position
        cx = largest_obstacle['x'] + largest_obstacle['w'] // 2
        w_roi = current_bright.shape[1]  # Width of ROI
        
        # Determine which third of the frame the obstacle is in
        if cx < w_roi // 3:
            final_position = 'left'
        elif cx > 2 * w_roi // 3:
            final_position = 'right'
        else:
            final_position = 'center'
            
        print(f"[DEBUG] Obstacle found! area={largest_obstacle['area']}, cx={cx}, position={final_position}")
    else:
        print("[DEBUG] No obstacle found passing area threshold.")
                
    return obstacle_found, final_position, candidates
