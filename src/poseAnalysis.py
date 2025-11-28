import os
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv
from poseComputation import computeAngle


load_dotenv()

model = YOLO(model="yolo11n-pose.pt", task="pose", verbose=False)
cap = cv2.VideoCapture("samples/soccerKick.mp4")

RIGHT_KNEE_ANGLES = []
LEFT_KNEE_ANGLES = []

def extractKeyFrames() -> list:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        if results[0].keypoints is not None:
            kp = results[0].keypoints.xy[0]

            left_hip = kp[11]
            left_knee = kp[13]
            left_ankle = kp[15]

            right_hip = kp[12]
            right_knee = kp[14]
            right_ankle = kp[16]

            left_angle = computeAngle(left_hip, left_knee, left_ankle)
            right_angle = computeAngle(right_hip, right_knee, right_ankle)
            if left_angle is not None:
                LEFT_KNEE_ANGLES.append(left_angle)
            if right_angle is not None:
                RIGHT_KNEE_ANGLES.append(right_angle)
        

        cv2.imshow("Pose", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return [LEFT_KNEE_ANGLES, RIGHT_KNEE_ANGLES]