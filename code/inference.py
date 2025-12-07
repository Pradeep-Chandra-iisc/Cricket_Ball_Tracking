# ----------------- inference.py (Original EdgeFleet Batch Version — SINGLE FILE) -----------------

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import os
import math
import traceback

# ---------------- DIRECTORIES ----------------
INPUT_DIR = "input_videos"
ANNOT_DIR = "annotations"
OUTPUT_DIR = "results"

os.makedirs(ANNOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- PARAMETERS ----------------
CONF_THRESH_RED = 0.45
CONF_THRESH_WHITE = 0.35
RE_DETECT_CONF = 0.12
MAX_DETS = 6

RE_DETECT_AFTER = 4
RE_DETECT_HALF = 80
RE_DETECT_SCALE = 2.0

TRACE_MAX = 1000
KF_PROCESS_Q = 0.30
KF_MEAS_R = 0.12

BOX_LERP_ALPHA = 0.35
PATH_INTERP_POINTS = 200

# -----------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------

def detect_ball_color_from_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # RED detection (small but strongly red areas)
    mask_red1 = cv2.inRange(hsv, (0,70,50), (10,255,255))
    mask_red2 = cv2.inRange(hsv, (160,70,50), (179,255,255))
    red_pixels = np.count_nonzero(mask_red1 | mask_red2)

    # WHITE detection (bright circular highlights)
    mask_white = cv2.inRange(hsv, (0,0,180), (180,40,255))
    white_pixels = np.count_nonzero(mask_white)

    # Compare strengths
    if white_pixels > red_pixels:
        return "WHITE"
    else:
        return "RED"



class BallKalman:
    def __init__(self, white_mode=False):
        self.kf = cv2.KalmanFilter(4,2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)

        q = KF_PROCESS_Q * (1.4 if white_mode else 1.0)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32)*q
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32)*KF_MEAS_R

        self.initialized = False

    def update(self, x, y):
        if not self.initialized:
            self.kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
            self.initialized = True
        return self.kf.correct(np.array([[x],[y]], np.float32))

    def predict(self):
        p = self.kf.predict()
        return int(p[0]), int(p[1])


def box_to_ints(box):
    xy = box.xyxy[0]
    xy = xy.cpu().numpy() if hasattr(xy,"cpu") else np.array(xy)
    return tuple(map(int, xy))


def pick_best_candidate(frame, boxes, last_point, min_area=20, max_area=2000):
    """Pick best YOLO detection using area, color fraction, and motion proximity."""
    best, best_score = None, -1
    h,w = frame.shape[:2]

    for b in boxes:
        try:
            x1,y1,x2,y2 = box_to_ints(b)
        except:
            continue

        x1 = max(0,min(w-1,x1))
        x2 = max(0,min(w-1,x2))
        y1 = max(0,min(h-1,y1))
        y2 = max(0,min(h-1,y2))

        if x2<=x1 or y2<=y1: continue

        area = (x2-x1)*(y2-y1)
        if not(min_area <= area <= max_area):
            continue

        cx = (x1+x2)//2
        cy = (y1+y2)//2

        roi = frame[y1:y2, x1:x2]

        rf = 0
        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, (0,80,60), (10,255,255))
            m2 = cv2.inRange(hsv, (160,80,60), (179,255,255))
            rf = np.count_nonzero(m1|m2) / max(1, roi.size//3)
        except:
            rf = 0

        conf = float(b.conf[0].cpu().numpy()) if hasattr(b.conf[0],"cpu") else float(b.conf[0])
        score = conf*(0.5 + rf)

        if last_point is not None:
            d = math.dist(last_point, (cx,cy))
            if d > 400:
                score *= 0.25
            else:
                score *= (1 - min(0.9, d/400))

        if score > best_score:
            best_score = score
            best = ((x1,y1,x2,y2),(cx,cy),conf)

    return best


def re_detect_crop(model, frame, last_point, halfsize, scale):
    h,w = frame.shape[:2]
    lx,ly = last_point

    x1 = max(0,lx-halfsize)
    y1 = max(0,ly-halfsize)
    x2 = min(w-1,lx+halfsize)
    y2 = min(h-1,ly+halfsize)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    up = cv2.resize(crop, None, fx=scale, fy=scale)
    res = model.predict(up, conf=RE_DETECT_CONF, max_det=MAX_DETS, verbose=False)

    if len(res)==0 or len(res[0].boxes)==0:
        return None

    cand = []
    for b in res[0].boxes:
        xy = b.xyxy[0]
        xy = xy.cpu().numpy() if hasattr(xy,"cpu") else np.array(xy)
        ax1,ay1,ax2,ay2 = map(float, xy)

        rx1 = int(ax1/scale) + x1
        ry1 = int(ay1/scale) + y1
        rx2 = int(ax2/scale) + x1
        ry2 = int(ay2/scale) + y1

        class B:
            def __init__(self,box,conf):
                self.xyxy=[np.array(box)]
                self.conf=[np.array(conf)]

        confval = float(b.conf[0].cpu().numpy() if hasattr(b.conf[0],"cpu") else b.conf[0])
        cand.append(B([rx1,ry1,rx2,ry2], confval))

    return pick_best_candidate(frame, cand, last_point)


# -----------------------------------------------------------------
# PROCESS SINGLE VIDEO (Original Flow)
# -----------------------------------------------------------------
def process_video(video_path, model):

    name = os.path.splitext(os.path.basename(video_path))[0]
    print("\nProcessing:", name)

    ann_file = os.path.join(ANNOT_DIR, f"{name}.csv")
    out_file = os.path.join(OUTPUT_DIR, f"{name}_processed.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open", video_path)
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    fcsv = open(ann_file, "w")
    fcsv.write("frame,x,y,visible\n")

    # detect ball color
    ret, sample = cap.read()
    if not ret:
        cap.release()
        return
    ball_color = detect_ball_color_from_frame(sample)
    print("Ball color:", ball_color)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    kf = BallKalman(white_mode=(ball_color=="WHITE"))
    pts = deque(maxlen=TRACE_MAX)

    frame_id = 0
    missed = 0
    locked = False
    curr_box = None
    last_wh = (24,24)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            frame_det = frame.copy()

            # color-based preprocessing
            if ball_color=="WHITE":
                lab = cv2.cvtColor(frame_det, cv2.COLOR_BGR2LAB)
                l,a,b = cv2.split(lab)
                l = cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8)).apply(l)
                frame_det = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
            else:
                blur = cv2.GaussianBlur(frame_det,(3,3),0)
                frame_det = cv2.addWeighted(frame_det,1.05,blur,-0.05,0)

            conf_th = CONF_THRESH_RED if ball_color=="RED" else CONF_THRESH_WHITE
            results = model.predict(frame_det, conf=conf_th, max_det=MAX_DETS, verbose=False)

            last_pt = (pts[-1][1],pts[-1][2]) if len(pts)>0 else None

            best = None
            if len(results) and len(results[0].boxes)>0:
                best = pick_best_candidate(frame, results[0].boxes, last_pt)

            cx=cy=None
            visible=0

            if best:
                (x1,y1,x2,y2),(cx,cy),conf = best
                visible=1
                missed=0
            else:
                missed+=1

                if last_pt and missed>=RE_DETECT_AFTER:
                    rd = re_detect_crop(model, frame_det, last_pt,
                                        halfsize=RE_DETECT_HALF,
                                        scale=RE_DETECT_SCALE)
                    if rd:
                        (x1,y1,x2,y2),(cx,cy),conf = rd
                        visible=1
                        missed=0

            # ---------------- UPDATE STATE ----------------
            if visible==1 and cx is not None:
                fcsv.write(f"{frame_id},{cx},{cy},1\n")
                pts.append((frame_id,cx,cy))

                det_w = max(8,x2-x1)
                det_h = max(8,y2-y1)
                last_wh = (det_w,det_h)

                if not kf.initialized:
                    kf.update(cx,cy)
                else:
                    kf.update(cx,cy)

                target = (float(cx),float(cy),float(det_w),float(det_h))

                if not locked:
                    curr_box = target
                    locked = True
                else:
                    # smooth interpolation
                    cx0 = curr_box[0] + (target[0]-curr_box[0])*BOX_LERP_ALPHA
                    cy0 = curr_box[1] + (target[1]-curr_box[1])*BOX_LERP_ALPHA
                    w0  = curr_box[2] + (target[2]-curr_box[2])*BOX_LERP_ALPHA
                    h0  = curr_box[3] + (target[3]-curr_box[3])*BOX_LERP_ALPHA
                    curr_box = (cx0,cy0,w0,h0)

            else:
                fcsv.write(f"{frame_id},-1,-1,0\n")

                if locked and kf.initialized:
                    px,py = kf.predict()
                    target = (float(px),float(py),float(last_wh[0]),float(last_wh[1]))
                    cx0 = curr_box[0] + (target[0]-curr_box[0])*BOX_LERP_ALPHA
                    cy0 = curr_box[1] + (target[1]-curr_box[1])*BOX_LERP_ALPHA
                    w0  = curr_box[2] + (target[2]-curr_box[2])*BOX_LERP_ALPHA
                    h0  = curr_box[3] + (target[3]-curr_box[3])*BOX_LERP_ALPHA
                    curr_box = (cx0,cy0,w0,h0)

            # ---------------- DRAW ----------------
            vis = frame.copy()

            # green path
            if len(pts)>1:
                arr = np.array([[p[1],p[2]] for p in pts])
                for i in range(1,len(arr)):
                    cv2.line(vis, tuple(arr[i-1]), tuple(arr[i]), (0,255,0), 3)

            # red locked box
            if locked:
                cx,cy,wf,hf = curr_box
                x1=int(cx-wf/2); y1=int(cy-hf/2)
                x2=int(cx+wf/2); y2=int(cy+hf/2)

                x1=max(0,min(w-1,x1))
                x2=max(0,min(w-1,x2))
                y1=max(0,min(h-1,y1))
                y2=max(0,min(h-1,y2))

                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,0,255),2)

            out.write(vis)

        fcsv.close()
        cap.release()
        out.release()
        print("✔ Done:", name)

    except Exception:
        traceback.print_exc()
        try:
            fcsv.close()
        except:
            pass


# -----------------------------------------------------------------
# RUN ALL VIDEOS (Batch Mode)
# -----------------------------------------------------------------
if __name__ == "__main__":
    model = YOLO("model/best.pt")

    videos = sorted([v for v in os.listdir(INPUT_DIR)
                     if v.lower().endswith((".mp4",".mov"))])

    print("Found videos:", videos)

    for v in videos:
        process_video(os.path.join(INPUT_DIR, v), model)

    print("\nALL DONE.")
