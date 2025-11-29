import cv2
from ultralytics import YOLO

model = YOLO(model="yolo11n-pose.pt", task="pose", verbose=False)
cap = cv2.VideoCapture("samples/sane.mp4")

def multiPlayerClass():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        cv2.imshow("Pose", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def testYoloTracking():
    model2 = YOLO("yolo11n.pt")
    results = model2.track(
        source="samples/soccerKick.mp4",
        show = True,
        tracker="bytetrack.yaml",
        save=True,
        stream=True
    )


    #TODO: FIX THIS LOOP TO CREATE A DICT FOR EACH PLAYER ALONG WITH THEIR POSITION, NUMBER, ETC
    playerTracks = {}
    for frame in results: #each frame in the video
        for player in frame.boxes: #each high confidence player box in the frame
            if not playerTracks[player.id]:
                playerTracks[player.id] = []
            playerTracks[player.id].append(player.xyxy)
        

if __name__ == "__main__":
    testYoloTracking()
