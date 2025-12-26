# Cricket Ball Detection and Tracking

This project implements a classical computer-vision pipeline to detect and track a **white cricket ball** from a single static camera, as required in the EdgeFleet.AI NIT Trichy assessment.
---
## Repository Structure
cricket_ball_tracking/
code/
detect_ball.py
utils.py
extract_frames.py
track_and_annotate.py
interpolate_trajectory.py
annotations/
results/
requirements.txt
README.md
report.pdf

- `code/`: all scripts for detection, tracking, and trajectory interpolation.
- `annotations/`: CSV files with `frame,x,y,visible`.  
- `results/`: processed videos and debug frames.  

---

## Setup

1. Install Python 3.8+.
2. From the project root:
pip install -r requirements.txt


`requirements.txt` includes `opencv-python`, `numpy`, `pandas`, and `tqdm` for video I/O, processing, and CSV handling.

Place the test video(s) (e.g. `1.mp4`) in the project root, as specified in the assessment.

---

## Running the Pipeline

From the project root:

cd code
python detect_ball.py
--video "../1.mp4"
--out_video "../results/1_traj_final.mp4"
--out_csv "../annotations/1_final.csv"
--ball_color white


Outputs:

- `results/1_traj_final.mp4`: video with ball centroid (red circle) and trajectory (green line).  
- `annotations/1_final.csv`: per-frame annotations:
frame,x,y,visible
0,-1,-1,0
1,945.0,338.0,1
...

matching the required format.

---

## Optional Utilities

### Extract frames

cd code
python extract_frames.py
--video "../1.mp4"
--out_dir "../results/frames_1"
--step 20


Useful for visually checking ball appearance and region priors.

### Interpolate full trajectory

1. Build an interpolated / parabolic trajectory from a detection CSV:

cd code
python interpolate_trajectory.py
--csv_in "../annotations/1_final.csv"
--csv_out "../annotations/1_final_parabola.csv"
--back_extend 12


2. Re-render using that CSV:
python track_and_annotate.py
--video "../1.mp4"
--csv "../annotations/1_final_parabola.csv"
--out_video "../results/1_traj_parabola.mp4"

This produces a visually smooth, downward-facing ball trajectory from bowler to stumps while preserving the true detections around the stumps region.

---

## Method Overview (Short)

- Convert frames to HSV, blur, and resize.  
- Threshold for white ball pixels (low saturation, high value) using tuned ranges, then clean with morphology.
- Find contours, filter by area and circularity, and take the best centroid as the ball.
- Restrict detections to an upper central “pitch corridor” and exclude a fixed box around the non-striker’s gloves/bat.  
- Apply a jump filter to remove sudden jumps to wrong objects and accumulate the accepted centroids into a trajectory.
- Optionally interpolate a parabolic trajectory from the bowler’s end to the reliable detections near the stumps.

For detailed modelling decisions, assumptions, and failure cases, see `Cricket Ball Tracking report.pdf`.


