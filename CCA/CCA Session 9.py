
import cv2
from ugot import ugot
from ultralytics import YOLO


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROBOT_IP = "192.168.88.1"
MODEL_PATH = "yolo26n-pose.pt"

CONF_THRESHOLD = 0.3
EXIT_FRAMES = 10  # X Pose must be detected for 10 consecutive frames


# ==========================================
# 2. POSE DETECTION
# ==========================================


# ==========================================
# 3. ROBOT ACTIONS
# ==========================================


# ==========================================
# 4. MAIN PROGRAM
# ==========================================

def main():

    # Initialize YOLO and UGOT
    model = YOLO(MODEL_PATH)

    got = ugot.UGOT()
    got.initialize(ROBOT_IP)
    got.show_light_rgb([0, 1, 2, 3], 255, 255, 255)

    cap = cv2.VideoCapture(0)

    previous_pose = None
    x_pose_count = 0

    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Run YOLO Pose
            results = model(frame, verbose=False)
            result = results[0]

            # Draw detected skeletons
            annotated_frame = result.plot()

            # TODO: PROGRAM LOGIC HERE

            cv2.imshow("YOLO Hand Pose Detection", annotated_frame)

            # Press Q to exit manually
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Manual exit requested.")
                break

    finally:
        # Always attempt to stop robot and release resources
        try:
            got.balance_stop_balancing()
        finally:
            cap.release()
            cv2.destroyAllWindows()

    print("Program ended.")


# ==========================================
# 5. START APPLICATION
# ==========================================

if __name__ == "__main__":
    main()
