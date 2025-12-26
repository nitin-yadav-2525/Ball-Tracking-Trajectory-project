import cv2
import os
import argparse

def extract_frames(video_path, output_dir, every_n=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"Saved {saved_idx} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out_dir", required=True, help="Output directory for frames")
    parser.add_argument("--step", type=int, default=1, help="Save every n-th frame")
    args = parser.parse_args()

    extract_frames(args.video, args.out_dir, args.step)
