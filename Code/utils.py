import cv2
import numpy as np

def preprocess_frame(frame, resize_width=960):
    """
    Resize frame for faster processing, blur to reduce noise,
    and convert to HSV color space.
    """
    h, w = frame.shape[:2]
    if w != resize_width:
        scale = resize_width / float(w)
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return frame, hsv


def get_ball_mask(hsv, ball_color="white"):
    """
    Return a binary mask for the ball.
    For this assessment we focus on a white ball: low S, high V.[web:98][web:124]
    """
    if ball_color == "white":
        # Relaxed range for white ball under floodlights
        # Allow slightly darker and more saturated pixels so the ball is not missed.
        lower = (0, 0, 190)      # H, S, V
        upper = (180, 90, 255)
        mask = cv2.inRange(hsv, lower, upper)

    elif ball_color == "red":
        # Generic red ball ranges (not used for your current test video)
        lower1 = (0, 120, 70)
        upper1 = (10, 255, 255)
        lower2 = (170, 120, 70)
        upper2 = (180, 255, 255)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = mask1 | mask2

    else:
        raise ValueError("Unsupported ball_color")

    # Morphology to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def find_ball_centroid(mask, min_area=10, max_area=8000, min_circularity=0.3):
    """
    Find the most likely ball contour and return its centroid (x, y, visible).
    Relaxed thresholds so slightly blurred/small balls are still detected.[web:2][web:9]
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_center = None
    best_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4.0 * np.pi * (area / (perimeter * perimeter))

        if circularity < min_circularity:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Prefer blobs that are larger and more circular
        score = circularity * area
        if score > best_score:
            best_score = score
            best_center = (cx, cy)

    if best_center is None:
        return -1, -1, 0  # not visible

    return best_center[0], best_center[1], 1
