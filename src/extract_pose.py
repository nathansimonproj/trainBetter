from ultralytics import YOLO
import cv2
from analyze import computeAngle
from llm import formatJson, createLLMPrompt
import os
from dotenv import load_dotenv

load_dotenv()

model = YOLO(model="yolo11n-pose.pt", task="pose", verbose=False)
cap = cv2.VideoCapture("samples/soccerKick.mp4")

RIGHT_KNEE_ANGLES = []
LEFT_KNEE_ANGLES = []
OPEN_AI_API = os.getenv("API_KEY")

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

metrics = formatJson(focus="Shooting", kneeAngles=RIGHT_KNEE_ANGLES)
createLLMPrompt(keyMetrics=metrics)


"""
print("Right Knee Angles")
for angle in RIGHT_KNEE_ANGLES:
    print(angle)

print("Left Knee Angles")
for angle in LEFT_KNEE_ANGLES:
    print(angle)

"""




