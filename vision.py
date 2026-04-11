import cv2
import numpy as np

def preprocess_image(image, target_size=(160, 120)):
    """
    Standard preprocessing: Resize, ROI, Normalization.
    """
    # Resize
    image = cv2.resize(image, target_size)
    
    # ROI: Keep the bottom half (road ahead, not sky/hood)
    h, w = target_size[1], target_size[0]
    roi = image[h//2:h, 0:w]
    
    # Normalization (0-1) for CNN input
    normalized = roi.astype(np.float32) / 255.0
    return normalized


def detect_lane(image):
    """
    Detect white center line on a black track.
    Robust to dotted lines by using morphological closing to bridge gaps.
    Returns: deviation, detected_status, foreground_mask, line_mask.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's Thresholding (Adaptive)
    _, bright_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # HSV saturation separation: white line has low saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    _, high_sat_mask = cv2.threshold(s_channel, 60, 255, cv2.THRESH_BINARY)
    _, low_sat_mask  = cv2.threshold(s_channel, 60, 255, cv2.THRESH_BINARY_INV)

    # The actual white line is bright AND low saturation
    line_mask = cv2.bitwise_and(bright_mask, low_sat_mask)

    # Foreground: anything bright OR colored
    foreground_mask = cv2.bitwise_or(bright_mask, high_sat_mask)

    # --- DOTTED LINE HANDLING ---
    # Use morphological CLOSING to connect the dots into a solid line
    # This makes centroid detection stable even between dashes
    close_kernel = np.ones((15, 15), np.uint8)  # Square kernel: bridges both vertical and horizontal gaps
    line_mask_closed = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, close_kernel)

    # Remove small noise with opening
    open_kernel = np.ones((5, 5), np.uint8)
    line_mask_closed = cv2.morphologyEx(line_mask_closed, cv2.MORPH_OPEN, open_kernel)

    # Calculate centroid from the 'closed' mask for stability
    contours, _ = cv2.findContours(line_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, False, foreground_mask, line_mask_closed

    main_cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(main_cnt)
    
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        w_center = image.shape[1] / 2
        deviation = (cx - w_center) / w_center   # Normalized: -1 (left) to +1 (right)
        
        # --- SHARP TURN / 90-DEGREE CORNER DETECTION ---
        x, y, w, h = cv2.boundingRect(main_cnt)
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # If the contour stretches broadly across the screen horizontally...
        if aspect_ratio > 1.5 and w > line_mask_closed.shape[1] * 0.35:
            # Check the horizontal shift in the TOP section of this contour
            # This identifies which way the "L-shape" is bending
            top_h = max(int(h * 0.4), 1)
            top_half_mask = np.zeros_like(line_mask_closed)
            top_half_mask[y:y+top_h, x:x+w] = line_mask_closed[y:y+top_h, x:x+w]
            
            M_top = cv2.moments(top_half_mask)
            if M_top["m00"] > 0:
                cx_top = int(M_top["m10"] / M_top["m00"])
                if cx_top > w_center + 15:
                    deviation = 1.0   # HARD RIGHT 
                elif cx_top < w_center - 15:
                    deviation = -1.0  # HARD LEFT
        
        return deviation, True, foreground_mask, line_mask_closed
    else:
        return 0, False, foreground_mask, line_mask_closed


def detect_obstacle(image, foreground_mask=None, line_mask=None):
    """
    Detect ANY color obstacle on the black track (box, surprise object, etc.).
    Obstacles are bright items that do NOT belong to the white line.
    Returns the MOST PROMINENT obstacle (largest by area) and its position.
    """
    h, w = image.shape[:2]
    roi_top = h // 2

    if foreground_mask is None or line_mask is None:
        return False, None, []

    current_bright = foreground_mask[roi_top:h, :]
    current_line   = line_mask[roi_top:h, :]

    # Obstacles = bright things minus the white line
    obstacles_mask = cv2.subtract(current_bright, current_line)

    # Clean up minor noise
    kernel_mask = np.ones((3, 3), np.uint8)
    obstacles_mask = cv2.erode(obstacles_mask, kernel_mask, iterations=1)
    obstacles_mask = cv2.dilate(obstacles_mask, kernel_mask, iterations=2)

    obs_contours, _ = cv2.findContours(obstacles_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates      = []
    obstacle_found  = False
    final_position  = None
    largest_obstacle = None
    largest_area     = 0

    for cnt in obs_contours:
        area = cv2.contourArea(cnt)
        if area > 80:
            x, y, w_obj, h_obj = cv2.boundingRect(cnt)
            y_full = y + roi_top
            candidates.append({'rect': (x, y_full, w_obj, h_obj), 'area': area})
            obstacle_found = True
            if area > largest_area:
                largest_area = area
                largest_obstacle = {'x': x, 'w': w_obj, 'area': area}

    if largest_obstacle is not None:
        cx    = largest_obstacle['x'] + largest_obstacle['w'] // 2
        w_roi = current_bright.shape[1]
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


def detect_traffic_light(image):
    """
    Detects a real physical traffic light in the frame.
    Detects RED (stop) and GREEN (go) only. YELLOW is intentionally ignored.
    Only fires when a sufficiently large, roughly circular blob is found.
    Returns:
        state (str): 'RED', 'GREEN', or None if nothing detected.
        confidence (float): area of the detected blob.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- RED (wraps around 180 in HSV) ---
    red_lower1 = np.array([0,   150, 80])
    red_upper1 = np.array([10,  255, 255])
    red_lower2 = np.array([165, 150, 80])
    red_upper2 = np.array([180, 255, 255])
    red_mask   = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)

    # --- GREEN (traffic-lamp green, avoids leafy greens) ---
    green_lower = np.array([50, 120, 80])
    green_upper = np.array([80, 255, 255])
    green_mask  = cv2.inRange(hsv, green_lower, green_upper)

    # A genuine traffic-light lamp must be at least this many pixels
    MIN_AREA        = 1500
    # Circularity: rejects elongated blobs (track lines, wires, sign boards)
    MIN_CIRCULARITY = 0.3

    best_state      = None
    best_confidence = 0.0

    for mask, label in [(red_mask, 'RED'), (green_mask, 'GREEN')]:
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue

            # Circularity check: rejects elongated blobs (track lines, wires, etc.)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity < MIN_CIRCULARITY:
                continue

            # Safety priority: RED > YELLOW > GREEN
            priority = {'RED': 3, 'YELLOW': 2, 'GREEN': 1}
            if area > best_confidence or \
               priority.get(label, 0) > priority.get(best_state, 0):
                best_confidence = area
                best_state = label

    return best_state, best_confidence


def detect_stop_sign(image):
    """
    Detects a red STOP sign in the frame.
    Uses the red color mask and a proximity check (larger area = closer = stop now).

    Returns:
        detected (bool): True if a stop sign is recognized.
        area (float): Contour area - use this to decide proximity threshold.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red in HSV wraps around 180
    red_lower1 = np.array([0,   120, 70])
    red_upper1 = np.array([10,  255, 255])
    red_lower2 = np.array([160, 120, 70])
    red_upper2 = np.array([180, 255, 255])
    red_mask   = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)

    # Clean up the mask
    cleaned = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    STOP_MIN_AREA = 2000  # A stop sign close enough to trigger a stop

    largest_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > STOP_MIN_AREA:
            # Approximate the contour shape: stop signs are roughly octagonal
            perimeter  = cv2.arcLength(cnt, True)
            approx     = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            num_sides  = len(approx)

            # Accept 6-12 sides (octagon-ish) or a large circular red region
            if 6 <= num_sides <= 12:
                if area > largest_area:
                    largest_area = area

    return largest_area > 0, largest_area


def get_path_space(line_mask, obstacle_rect):
    """
    Calculates the available path space (white paper) to the left and right
    of the obstacle relative to the detected path boundaries.
    """
    x, y, w_obj, h_obj = obstacle_rect
    h_mask, w_mask = line_mask.shape

    sample_y = y + h_obj // 2
    if sample_y >= h_mask:
        sample_y = h_mask - 1

    line_row     = line_mask[sample_y, :]
    white_indices = np.where(line_row > 0)[0]

    if len(white_indices) == 0:
        return x, w_mask - (x + w_obj), 0, w_mask

    path_start = white_indices[0]
    path_end   = white_indices[-1]

    left_space  = max(0, x - path_start)
    right_space = max(0, path_end - (x + w_obj))

    return left_space, right_space, path_start, path_end
