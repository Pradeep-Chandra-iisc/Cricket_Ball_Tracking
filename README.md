# 🏏 Cricket Ball Detection & Tracking System
### Author: Pradeep Chandra  
### Institute ID: 25173  

This repository contains a complete cricket ball detection and tracking pipeline implemented using:

- **YOLOv11s (custom-trained)**  
- **Re-detection crop mechanism**  
- **Kalman Filter for motion prediction**  
- **Trajectory visualization and per-frame CSV annotations**  

The system takes cricket match videos recorded from a fixed camera and produces:
- A processed output video with bounding box + trajectory overlay  
- A CSV file containing `frame, x, y, visible` annotations  

---

# 📌 Project Overview

The tracking pipeline consists of the following stages:

1. **Ball Color Detection (red/white)**  
   Using HSV-based pixel heuristics to identify whether the ball is red or white.

2. **Color-Specific Preprocessing**  
   - **White ball** → Enhanced using CLAHE in LAB color space  
   - **Red ball** → Light contrast sharpening  

3. **Primary YOLO Detection**  
   The ball is detected using a custom YOLOv11s model trained specifically for cricket ball imagery.

4. **Re-detection Crop (Fallback)**  
   If YOLO misses for a few frames, a zoomed crop around the last known position is evaluated again.

5. **Kalman Filter Prediction**  
   When detection fails completely, future positions are predicted using a constant-velocity Kalman model.

6. **Red Bounding Box Smoothing**  
   Linear interpolation ensures stable bounding box movement.

7. **Green Trajectory Drawing**  
   The green path is drawn using actual detections (visible=1).

8. **Outputs**  
   - Processed video saved under `results/`  
   - Annotation CSV saved under `annotations/`  

---
---

Running the Tracker

Place your cricket videos inside:

input_videos/


Then run:

python code/inference.py


The system automatically:

Detects ball color

Processes all videos one by one

Saves:

Processed videos in results/

Annotation CSVs in annotations/





