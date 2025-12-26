import pandas as pd
import numpy as np
import argparse
import os

def interpolate_trajectory(csv_in, csv_out, back_extend_frames=10):
    df = pd.read_csv(csv_in)

    # Indices where ball is actually detected
    det_idx = df.index[df["visible"] == 1].to_list()
    if len(det_idx) < 2:
        print("Not enough detections to interpolate.")
        df.to_csv(csv_out, index=False)
        return

    # Interpolate between detections in time (simple linear)
    for i in range(len(det_idx) - 1):
        start = det_idx[i]
        end = det_idx[i + 1]
        x0, y0 = df.loc[start, ["x", "y"]]
        x1, y1 = df.loc[end, ["x", "y"]]

        n = end - start
        for k in range(1, n):
            t = k / n
            xi = (1 - t) * x0 + t * x1
            yi = (1 - t) * y0 + t * y1
            df.loc[start + k, "x"] = xi
            df.loc[start + k, "y"] = yi
            df.loc[start + k, "visible"] = 1

    # Extend backwards from first detection towards bowler
    first_idx = det_idx[0]
    x0, y0 = df.loc[first_idx, ["x", "y"]]

    # Choose a simple direction upwards (towards bowler end)
    # For your camera, bowler is roughly above the stumps, so decrease y.[image:1]
    vx = 8.0
    vy = 17.0  # pixels per frame upwards; tune if needed

    N = min(back_extend_frames, first_idx)
    if N > 0:
        # Define "time" indices: t = -N .. 0, where t=0 is first detection
        # x is linear in t, y is quadratic for downward-facing parabola.
        for k in range(1, N + 1):
            t = -k  # going backwards
            idx = first_idx + t  # frame index

            # Linear x: move left as we go back
            x_t = x0 + vx * t   # t negative -> x_t < x0

            # Parabolic y: start higher, curve down to y0
            # y(t) = y0 + vy * t + a * t^2, with a > 0 to bend downward.
            a = 1.0  # curvature; tune if needed
            y_t = y0 + vy * t + a * (t ** 2)

            df.loc[idx, "x"] = x_t
            df.loc[idx, "y"] = y_t
            df.loc[idx, "visible"] = 1

    df.to_csv(csv_out, index=False)
    print(f"Saved interpolated CSV to {csv_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_in", required=True, help="Original detection CSV")
    parser.add_argument("--csv_out", required=True, help="Output CSV with interpolated trajectory")
    parser.add_argument("--back_extend", type=int, default=10, help="Frames to extend backwards")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)

    interpolate_trajectory(args.csv_in, args.csv_out, args.back_extend)
