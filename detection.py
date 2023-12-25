import cv2
import os
import keyboard
import shutil
from ultralytics import YOLO

# Clear previous prediction
if os.path.exists("runs"):
    shutil.rmtree("./runs")

# Initialize YOLO Model
model = YOLO('smartcart_yolov8.pt')
names = model.names

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Capture and save image from camera
    cv2.imwrite("./images/prediction.jpg", frame)

    if os.path.exists("./images/prediction.jpg"):
        print()

        # infer on a local image
        prediction = model("./images/prediction.jpg", conf=0.5, save=True)
        for p in prediction:
            for c in p.boxes.cls:
                print("\nPrediction:", names[int(c)])

    # Exit by pressing 'Q'
    if keyboard.is_pressed('q'):
        print("Exiting the program.")
        break

cap.release()
cv2.destroyAllWindows()