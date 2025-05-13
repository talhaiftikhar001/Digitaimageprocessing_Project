import cv2
import numpy as np

previous_frame = None

def region_selection(image):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        ignore_mask_color = (255,) * image.shape[2]
    else:
        ignore_mask_color = 255
    rows, cols = image.shape[:2]

    bottom_left = [cols * 0.05, rows * 0.90]
    top_left = [cols * 0.45, rows * 0.55]
    top_right = [cols * 0.55, rows * 0.55]
    bottom_right = [cols * 0.95, rows * 0.90]

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(image, mask)

def hough_transform(image):
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=20, minLineLength=30, maxLineGap=250)

def average_slope_intercept(image, lines):
    left_lines, right_lines = [], []
    left_weights, right_weights = [], []
    mid_x = image.shape[1] / 2

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.7:
                continue
            intercept = y1 - slope * x1
            length = np.hypot(x2 - x1, y2 - y1)
            if slope < 0 and x1 < mid_x and x2 < mid_x:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            elif slope > 0 and x1 > mid_x and x2 > mid_x:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None or line[0] == 0:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, int(y1)), (x2, int(y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(image, lines)
    y1 = image.shape[0]
    y2 = int(y1 * 0.45)
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=10):
    line_image = np.zeros_like(image)

    left_line, right_line = lines
    if left_line is not None and right_line is not None:
        pts = np.array([left_line[0], left_line[1], right_line[1], right_line[0]], dtype=np.int32)
        cv2.fillPoly(line_image, [pts], (0, 255, 0))  

    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)

    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def detect_objects_on_road(frame):
    global previous_frame
    height, width = frame.shape[:2]
    roi_polygon = np.array([[(100, height), (2200, height), (1000, 260), (700, 260)]])
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi_mask, roi_polygon, 255)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if previous_frame is None:
        previous_frame = blur
        return frame, False

    diff = cv2.absdiff(previous_frame, blur)
    _, thresh = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000 or area > 20000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w + 1)

        if aspect_ratio < 1.5 or aspect_ratio > 3.0:
            continue

        cx, cy = x + w // 2, y + h // 2
        if roi_mask[cy, cx] == 255:
            box_mask = np.zeros_like(roi_mask)
            cv2.rectangle(box_mask, (x, y), (x + w, y + h), 255, -1)
            intersection = cv2.bitwise_and(box_mask, roi_mask)
            overlap_area = cv2.countNonZero(intersection)
            box_area = w * h

            if overlap_area / box_area > 0.7:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                detected = True

    previous_frame = blur
    return frame, detected


def detect_turn_direction(frame, lines):
    left_lane, right_lane = average_slope_intercept(frame, lines)

    turn_direction = "Straight"

    if left_lane is not None and right_lane is not None:
        left_slope = left_lane[0]
        right_slope = right_lane[0]

        if abs(left_slope) < 0.3 and abs(right_slope) < 0.3:
            turn_direction = "Curve Ahead"
        elif left_slope > -0.3 and right_slope > 0.7:
            turn_direction = "Right Turn"
        elif right_slope < 0.3 and left_slope < -0.7:
            turn_direction = "Left Turn"

    elif left_lane is not None:
        if left_lane[0] > -0.5:
            turn_direction = "Right Turn"
    elif right_lane is not None:
        if right_lane[0] < 0.5:
            turn_direction = "Left Turn"

    return turn_direction


def process_frame(frame):
    frame_with_objects, detected = detect_objects_on_road(frame)

    # Convert to HSV and apply yellow mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 13, 13])
    upper_yellow = np.array([80, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    blurred = cv2.GaussianBlur(yellow_mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    edges = cv2.dilate(edges, kernel, iterations=1)

    roi = region_selection(edges)

    hough = hough_transform(roi)

    final_output = frame_with_objects
    turn = "Unknown"
    if hough is not None:
        lines = lane_lines(frame, hough)
        turn = detect_turn_direction(frame, hough) 
        final_output = draw_lane_lines(final_output, lines)

    # ➕ Overlay text at top-left corner
    status_text = "Object Detected: Yes" if detected else "Object Detected: No"
    text_color = (0, 255, 0) if detected else (0, 0, 255)
    cv2.putText(final_output, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
    cv2.putText(final_output, f"Turn: {turn}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    return final_output


def main():
    cap = cv2.VideoCapture("C:/Users\Administrator\Downloads\DIP Project Videos\IMG_9992.mp4")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 != 0:
            continue

        processed = process_frame(frame)
        cv2.imshow("Lane + Human Detection", processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
