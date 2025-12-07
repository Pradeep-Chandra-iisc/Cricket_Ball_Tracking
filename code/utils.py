import cv2
import numpy as np
import math


# ---------------------------------------------------
# 1. STRONG BALL COLOR DETECTION (PER VIDEO ONLY)
# ---------------------------------------------------
def detect_ball_color(frame):
    """
    Detect whether the ball in this VIDEO is RED or WHITE.
    Called only once per video (first frame).
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_mean = np.mean(hsv[..., 0])
    s_mean = np.mean(hsv[..., 1])
    v_mean = np.mean(hsv[..., 2])

    # Strong red ball detection using hue mask
    red_mask = cv2.inRange(hsv, (0, 70, 50), (15, 255, 255)) + \
               cv2.inRange(hsv, (160, 70, 50), (179, 255, 255))

    red_ratio = np.count_nonzero(red_mask) / float(frame.shape[0] * frame.shape[1])

    if red_ratio > 0.08:
        return "RED"

    # White ball detection
    if v_mean > 160 and s_mean < 80:
        return "WHITE"

    # fallback
    return "WHITE"


# ---------------------------------------------------
# 2. PARSE YOLO BBOX
# ---------------------------------------------------
def box_to_ints(box):
    xy = box.xyxy[0].cpu().numpy()
    return map(int, xy.astype(int))


# ---------------------------------------------------
# 3. PICK BEST CANDIDATE
# ---------------------------------------------------
def pick_best_candidate(frame, boxes, last_point, min_area=20, max_area=2000):

    best = None
    best_score = -1
    h, w = frame.shape[:2]

    for b in boxes:
        try:
            x1, y1, x2, y2 = box_to_ints(b)
        except:
            continue

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        area = (x2 - x1) * (y2 - y1)
        if not (min_area <= area <= max_area):
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        roi = frame[y1:y2, x1:x2]

        rf = 0
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
            m2 = cv2.inRange(hsv, (160, 60, 60), (179, 255, 255))
            rf = np.count_nonzero(m1 | m2) / float(roi.size)
        except:
            pass

        conf = float(b.conf[0].cpu().numpy())
        score = conf * (1 + rf)

        if last_point:
            dist = math.dist(last_point, (cx, cy))
            if dist > 400:
                score *= 0.25
            else:
                score *= (1 - min(0.9, dist / 400))

        if score > best_score:
            best_score = score
            best = ((x1, y1, x2, y2), (cx, cy), conf)

    return best


# ---------------------------------------------------
# 4. REDETECTION CROP
# ---------------------------------------------------
def re_detect_crop(model, frame, last_point, halfsize=80, scale=2.0):

    h, w = frame.shape[:2]
    lx, ly = last_point

    x1 = max(0, lx - halfsize)
    y1 = max(0, ly - halfsize)
    x2 = min(w - 1, lx + halfsize)
    y2 = min(h - 1, ly + halfsize)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    up = cv2.resize(crop, None, fx=scale, fy=scale)
    res = model.predict(up, conf=0.12, max_det=6, verbose=False)

    if not res or len(res[0].boxes) == 0:
        return None

    candidates = []

    for b in res[0].boxes:
        xy = b.xyxy[0].cpu().numpy()
        ax1, ay1, ax2, ay2 = xy

        rx1 = int(ax1 / scale) + x1
        ry1 = int(ay1 / scale) + y1
        rx2 = int(ax2 / scale) + x1
        ry2 = int(ay2 / scale) + y1

        class TB:
            def __init__(self, xy, conf):
                self.xyxy = [np.array(xy)]
                self.conf = [np.array(conf)]

        candidates.append(TB([rx1, ry1, rx2, ry2], float(b.conf[0].cpu().numpy())))

    return pick_best_candidate(frame, candidates, last_point)


# ---------------------------------------------------
# 5. KALMAN FILTER CLASS
# ---------------------------------------------------
class BallKalman:

    def __init__(self, white_mode=False):

        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)

        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)

        q = 0.30 * (1.4 if white_mode else 1)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * q
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.12

        self.initialized = False

    def update(self, x, y):
        if not self.initialized:
            self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True
        return self.kf.correct(np.array([[x], [y]], dtype=np.float32))

    def predict(self):
        p = self.kf.predict()
        return int(p[0]), int(p[1])


# ---------------------------------------------------
# 6. SMOOTH BOX INTERPOLATION
# ---------------------------------------------------
def lerp(a, b, alpha):
    return a + (b - a) * alpha

def lerp_box(curr_box, target_box, alpha):
    cx = lerp(curr_box[0], target_box[0], alpha)
    cy = lerp(curr_box[1], target_box[1], alpha)
    w = lerp(curr_box[2], target_box[2], alpha)
    h = lerp(curr_box[3], target_box[3], alpha)
    return (cx, cy, w, h)
