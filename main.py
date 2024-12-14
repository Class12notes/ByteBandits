from ultralytics import YOLO
import cv2
import cvzone
import math
#import time
import plyer
cap = cv2.VideoCapture(1)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video


model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#prev_frame_time = 0
#new_frame_time = 0
# Your existing code...

# Initialize a flag to track whether the chair was previously detected
chair_detected = False
chair_removed = False

while True:
    # new_frame_time = time.time()
    success, img = cap.read() #success is a boolean expression in this and img is frame of the video 
    results = model(img, stream=True)

    chair_removed = True

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if (currentClass == "chair" or currentClass == "person" and conf > 0.5):
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                # Set the flag to True when chair is detected
                chair_detected = True
                chair_removed = False
            else:
                # If chair was previously detected and now it's not, print "Hello, World!"
                if chair_detected and chair_removed:
                    plyer.notification.notify(
                        title="Text Message",
                        message="Alert Message"
                    )
                    chair_detected = False
                elif chair_detected:
                    print("Hello, World!")
                    chair_detected = False

    cv2.imshow("Image", img)
    cv2.waitKey(1)
