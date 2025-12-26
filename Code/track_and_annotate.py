import cv2
import pandas as pd
import argparse
import os

def redraw_from_csv(video_path, csv_path, out_video_path):
    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    trajectory_points = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(df):
            break

        row = df.iloc[frame_idx]
        x, y, visible = int(row["x"]), int(row["y"]), int(row["visible"])

        if visible and x >= 0 and y >= 0:
            trajectory_points.append((x, y))
            cv2.circle(frame, (x, y), 10, (0, 0, 255), 2)

        for i in range(1, len(trajectory_points)):
            cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 255, 0), 2)

        out_writer.write(frame)
        frame_idx += 1

    cap.release()
    out_writer.release()
    print(f"Redrawn video saved to {out_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_video", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)

    redraw_from_csv(args.video, args.csv, args.out_video)
