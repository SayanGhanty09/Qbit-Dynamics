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
    Detect white line/track using adaptive thresholding.
    Returns: deviation, detected_status, and track_mask (binary).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's Thresholding (Adaptive)
    # Finds the best split between dark floor and bright track/lane
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate moments to find centroid
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        w_center = thresh.shape[1] / 2
        deviation = (cx - w_center) / w_center  # Normalize to -1 to 1
        return deviation, True, thresh
    else:
        return 0, False, thresh

def detect_obstacle(image, track_mask=None):
    """
    Detect ANY color obstacle on the track.
    Uses Hole-Filling logic to find dark objects entirely enclosed by the bright track.
    Returns the MOST PROMINENT obstacle (largest by area) and its position.
    """
    h, w = image.shape[:2]
    roi_top = h // 2
    
    if track_mask is None:
        return False, None, []
        
    current_mask = track_mask[roi_top:h, :]
    
    # 1. Find the external boundary of the track (ignoring holes)
    contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 2. Create a solid road layout (fill the holes where obstacles might be)
    road_layout = np.zeros_like(current_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400: # Substantial piece of paper
            cv2.drawContours(road_layout, [cnt], -1, 255, thickness=cv2.FILLED)
            
    # 3. Erode the road layout to create a safe zone inside the paper boundaries
    kernel_mask = np.ones((3, 3), np.uint8) # Smaller kernel to preserve edge obstacles
    safe_zone = cv2.erode(road_layout, kernel_mask, iterations=1)
    
    # 4. Isolate the objects
    # Obstacles are dark (0) on the track mask. Invert to make them 255.
    inv_mask = cv2.bitwise_not(current_mask)
    
    # And it with the safe zone. This completely eliminates the floor!
    obstacles_mask = cv2.bitwise_and(inv_mask, safe_zone)
    
    # 5. Find contours of the perfectly isolated obstacles
    obs_contours, _ = cv2.findContours(obstacles_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    obstacle_found = False
    final_position = None
    largest_obstacle = None
    largest_area = 0

    for cnt in obs_contours:
        area = cv2.contourArea(cnt)
        if area > 80: # Lower threshold to detect smaller obstacles like the car model
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
        # Use the LEFT EDGE of obstacle to determine position (more accurate than center)
        x_left = largest_obstacle['x']
        w_roi = current_mask.shape[1]  # Width of ROI
        
        # Determine which third of the frame the obstacle is in
        if x_left < w_roi // 3:
            final_position = 'left'
        elif x_left > 2 * w_roi // 3:
            final_position = 'right'
        else:
            final_position = 'center'
                
    return obstacle_found, final_position, candidates
