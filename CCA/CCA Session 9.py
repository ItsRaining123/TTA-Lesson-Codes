
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

def detect_hands(keypoints, conf):
    threshold = CONF_THRESHOLD

    # Get shoulder, elbow, and wrist coordinates
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    left_elbow = keypoints[7]
    right_elbow = keypoints[8]

    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    # Check for X Pose FIRST
    x_pose = (
        all(conf[i] >= threshold for i in [5, 6, 7, 8, 9, 10])
        and left_elbow[1] > left_wrist[1] > left_shoulder[1]
        and right_elbow[1] > right_wrist[1] > right_shoulder[1]
        and left_wrist[0] < right_wrist[0]
    )

    if x_pose:
        return "X Pose"

    # Check if left arm is raised
    if conf[5] < threshold:
        left_up = None
    elif conf[9] >= threshold:
        left_up = left_wrist[1] < left_shoulder[1]
    elif conf[7] >= threshold:
        left_up = left_elbow[1] < left_shoulder[1]
    else:
        left_up = None

    # Check if right arm is raised
    if conf[6] < threshold:
        right_up = None
    elif conf[10] >= threshold:
        right_up = right_wrist[1] < right_shoulder[1]
    elif conf[8] >= threshold:
        right_up = right_elbow[1] < right_shoulder[1]
    else:
        right_up = None

    # Determine pose
    if left_up is None or right_up is None:
        return "Pose Uncertain"
    elif left_up and right_up:
        return "Both Hands Up"
    elif left_up:
        return "Left Hand Up"
    elif right_up:
        return "Right Hand Up"
    else:
        return "Both Hands Down"


# ==========================================
# 3. ROBOT ACTIONS
# ==========================================

def run_robot_action(pose, got):
    if pose == "Both Hands Up":
        got.balance_move_speed(0, 20)
    elif pose == "Left Hand Up":
        got.balance_turn_speed(2, 45)
    elif pose == "Right Hand Up":
        got.balance_turn_speed(3, 45)
    elif pose in ("Both Hands Down", "Pose Uncertain", "X Pose"):
        got.balance_stop_balancing()


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

            # Default: stop if no person is detected
            pose = "Pose Uncertain"

            # Process first detected person
            if (
                result.keypoints is not None
                and result.keypoints.conf is not None
                and len(result.keypoints.xy) > 0
            ):
                keypoints = result.keypoints.xy[0].cpu().numpy()
                conf = result.keypoints.conf[0].cpu().numpy()

                pose = detect_hands(keypoints, conf)

            # Track consecutive X Pose frames
            if pose == "X Pose":
                x_pose_count += 1
            else:
                x_pose_count = 0

            # Execute robot action only when pose changes
            if pose != previous_pose:
                run_robot_action(pose, got)
                previous_pose = pose

            # Display detected gesture
            cv2.putText(
                annotated_frame,
                pose,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.imshow("YOLO Hand Pose Detection", annotated_frame)

            # Exit if X Pose is held for enough frames
            if x_pose_count >= EXIT_FRAMES:
                print("X Pose detected! Exiting program...")
                break

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
