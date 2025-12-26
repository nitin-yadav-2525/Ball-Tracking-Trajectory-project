import cv2
import numpy as np
import pandas as pd
import argparse
import os
from utils import preprocess_frame, get_ball_mask, find_ball_centroid


def process_video(
    video_path,
    out_video_path,
    out_csv_path,
    ball_color="white"
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    records = []
    trajectory_points = []

    frame_idx = 0
    max_jump = 150  # pixels, tune if needed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- detection on resized frame ---
        proc_frame, hsv = preprocess_frame(frame)
        mask = get_ball_mask(hsv, ball_color=ball_color)
        x, y, visible = find_ball_centroid(mask)

        ph, pw = proc_frame.shape[:2]
        scale_x = width / float(pw)
        scale_y = height / float(ph)

        # -------- map to original coordinates --------
        if visible:
            x_orig = int(x * scale_x)
            y_orig = int(y * scale_y)
        else:
            x_orig, y_orig = -1, -1

        # ------------------------------------------------
        # CENTRAL PITCH REGION ONLY  (camera-specific)
        # ------------------------------------------------
        if visible:
            # A) vertical crop: keep upper ~75% of frame
            if y_orig > int(height * 0.75):
                visible = 0
                x_orig, y_orig = -1, -1

        if visible:
            # B) horizontal crop: central lane (adjust 0.30 / 0.70 if needed)
            x_min = int(width * 0.30)
            x_max = int(width * 0.70)
            if not (x_min <= x_orig <= x_max):
                visible = 0
                x_orig, y_orig = -1, -1

        if visible:
            # C) extra box around non-striker gloves/bat on right side
            glove_x_min = int(width * 0.60)
            glove_x_max = int(width * 0.90)
            glove_y_min = int(height * 0.35)
            glove_y_max = int(height * 0.80)
            if (glove_x_min <= x_orig <= glove_x_max and
                    glove_y_min <= y_orig <= glove_y_max):
                visible = 0
                x_orig, y_orig = -1, -1

        # -------- jump filter --------
        if visible:
            if trajectory_points:
                last_x, last_y = trajectory_points[-1]
                dist = ((x_orig - last_x) ** 2 + (y_orig - last_y) ** 2) ** 0.5
                if dist > max_jump:
                    visible = 0
                    x_orig, y_orig = -1, -1

        if visible:
            trajectory_points.append((x_orig, y_orig))
        else:
            x_orig, y_orig = -1, -1

        # -------- draw detection & trajectory --------
        if visible:
            cv2.circle(frame, (x_orig, y_orig), 10, (0, 0, 255), 2)

        for i in range(1, len(trajectory_points)):
            cv2.line(
                frame,
                trajectory_points[i - 1],
                trajectory_points[i],
                (0, 255, 0),
                2,
            )

        out_writer.write(frame)
        records.append(
            {
                "frame": frame_idx,
                "x": float(x_orig),
                "y": float(y_orig),
                "visible": int(visible),
            }
        )
        frame_idx += 1

    cap.release()
    out_writer.release()

    df = pd.DataFrame(records)
    df.to_csv(out_csv_path, index=False)
    print(f"Saved CSV to {out_csv_path}")
    print(f"Saved video to {out_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out_video", required=True, help="Output processed video path")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument("--ball_color", default="white", choices=["red", "white"])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    process_video(
        args.video,
        args.out_video,
        args.out_csv,
        ball_color=args.ball_color,
    )
