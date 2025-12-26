import cv2
import argparse

def nothing(x):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a frame image with ball visible")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    # Create the windows
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

    # Create the trackbars in the 'Trackbars' window
    cv2.createTrackbar("LH", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("LS", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("LV", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("UH", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("US", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("UV", "Trackbars", 255, 255, nothing)

    while True:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lh = cv2.getTrackbarPos("LH", "Trackbars")
        ls = cv2.getTrackbarPos("LS", "Trackbars")
        lv = cv2.getTrackbarPos("LV", "Trackbars")
        uh = cv2.getTrackbarPos("UH", "Trackbars")
        us = cv2.getTrackbarPos("US", "Trackbars")
        uv = cv2.getTrackbarPos("UV", "Trackbars")

        lower = (lh, ls, lv)
        upper = (uh, us, uv)

        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow("Original", img)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", res)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Lower:", lower)
            print("Upper:", upper)
            break

    cv2.destroyAllWindows()
