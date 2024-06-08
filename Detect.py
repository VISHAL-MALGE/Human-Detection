import cv2
import cvzone
import math
import time
from ultralytics import YOLO
import pygame

# Initialize pygame
pygame.init()

# Load the sound file ("beep.wav" with the path of beep sound file)
beep_sound = pygame.mixer.Sound("beep.wav")

# Initialize the video capture object for the laptop camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Set the resolution for the camera
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the YOLO model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Define class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella","handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed","diningtable", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone","microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "scissors","teddy bear", "hair drier", "toothbrush"]
              

prev_frame_time = 0 
new_frame_time = 0

while True:
    new_frame_time = time.time()

    # Read a frame from the webcam
    ret, img = cap.read()

    # Perform object detection
    results = model(img, stream=True)

    # Process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence and class
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Filter out only humans (class index 0)
            if cls == 0:
                # Draw bounding box
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))

                # Add class name and confidence to the image
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                # Play beep sound when human detected
                beep_sound.play()

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    # Display the resulting frame
    cv2.imshow("Image", img)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
