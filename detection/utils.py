import cv2
import os
from ultralytics import YOLO
import numpy as np

# function that takes the path to a video file as input
def detect_road(video_path, output_dir='media/processed_frames'):

    # Opens the video file for reading
    cap = cv2.VideoCapture(video_path)

    # Creates a background subtractor, which detects moving objects by subtracting the background.
    # Useful for detecting cars, people, etc., on roads.
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # using the car_cascade
    car_cascade = cv2.CascadeClassifier('detection/haar_cascades/cars.xml')

    # Creates an empty list to store processed frames (only foreground/motion parts).
    results = []

    # Create directory to save the prpccessed frames
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0

    # Loops while the video is still open
    while cap.isOpened():

        # read the next frame
        # ret is true if reading is successfully
        # frame contains the image (a numpy array)
        ret, frame = cap.read()

        # if no more frames
        if not ret or frame_count >= 50:
            break
        
        # Applies background subtraction to get the foreground mask (white = motion, black = background).
        fgmask = fgbg.apply(frame)

        # frame is colorful
        # most object detection methods (like Haar cascades) work on grayscale, not color
        # cv2.cvtColor() changes the color format.
        # cv2.COLOR_BGR2GRAY tells OpenCV to convert from color to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cars
        # 1.1: scale factor (how much to shrink image during scanning)
        # 1: min neighbors (higher = stricter detection)
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)

        # Draw rectangles around detected cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Stores the foreground mask (a grayscale image) in the results list.
        # results.append(fgmask)

        filename = os.path.join(output_dir, f'frame_{frame_count}.jpg')
        cv2.imwrite(filename, frame)  # Save each processed frame
        results.append(f'processed_frames/frame_{frame_count}.jpg')
        frame_count += 1

    # close the video
    cap.release()
    return results

def detect_road_yolo(video_path, output_dir='media/yolo_frames'):
    model = YOLO("yolov8n.pt")  # Or yolov8s.pt for better accuracy

    # Loads the video using OpenCV. cap is now a video reader object.
    cap = cv2.VideoCapture(video_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        # if not ret or frame_count >= 100:
        #     break

        if not ret:
            break

        # Run YOLO detection
        # model(frame) returns a list of results, one for each image (we only give 1 frame, so [0] gives the first result)
        # the detection contains: bounding boxes, confidence scores, classid(ex:car,truck,person)
        # model(frame) runs YOLO on the image: returns a list of results (even if there's only one image).
        # [0] gets the first result from the list (since we only gave one frame)
        detections = model(frame)[0]

        # Draw the detection boxes, labels and confidence scores on the frame
        # Returns a new image (annotated) with the boxes drawn on it.
        annotated = detections.plot()
        
        filename = os.path.join(output_dir, f'yolo_frame_{frame_count}.jpg')
        cv2.imwrite(filename, annotated)
        results.append(f'yolo_frames/yolo_frame_{frame_count}.jpg')
        frame_count += 1

    cap.release()
    return results

# make those frames to a video
# frames_dir:folder path where image frames are stored
# output_video_path:path to save the final video
# fps: frames per seconds (defualt 20)
def frames_to_video(frames_dir, output_video_path, fps=20):

    # get a list of all files in the directory with
    # sort the jpeg in order
    frames = sorted(os.listdir(frames_dir))

    # Filters out only .jpg image files from the folder (ignores others like .DS_Store)
    frames = [f for f in frames if f.endswith('.jpg')]

    # If the folder has no image frames, stop the function early
    if not frames:
        print("No frames found.")
        return

    # Read the first frame to get width and height
    # loads the first frame form the folder
    first_frame_path = os.path.join(frames_dir, frames[0])

    # read the first frame
    first_frame = cv2.imread(first_frame_path)

    # get the video resolution (width and height)
    # _ is the number of channels (we don't use it)
    height, width, _ = first_frame.shape

    # Defines the video codec (compression format)
    # fourcc stands for Four Character Code — it's used to define the video codec (how video is compressed and stored).
    # 'XVID' is a common codec for .avi videos
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Creates a video writer object
    # It will save frames to the file at output_video_path with the given fps and size
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Loops through all the sorted image frames
    for frame_name in frames:

        frame_path = os.path.join(frames_dir, frame_name)

        # Reads each frame from disk and writes it to the video
        frame = cv2.imread(frame_path)

        # check if each of the frames had the same height and width
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        out.write(frame)

    # Finalizes and saves the video file
    out.release()
    print("Video saved:", output_video_path)

# detect the collisions

model = YOLO("yolov8n.pt")

def compute_iou(box1, box2):

    # This finds the coordinates of the overlapping area (intersection) between the two boxes
    # x1, y1: top-left of the intersection box
    # x2, y2: bottom-right of the intersection box

    x1 = max(box1[0], box2[0])  # left side of overlap
    y1 = max(box1[1], box2[1])  # top side of overlap
    x2 = min(box1[2], box2[2])  # right side of overlap
    y2 = min(box1[3], box2[3])  # bottom side of overlap

    # refer to the picture i upload
    # If boxes don't overlap, values could be negative, so max(0, ...) is used to make area zero in that case
    inter_area = max(0, x2 - x1) * max(0, y2 - y1) 

    # minus the all area - inter-area
    union_area = (
        (box1[2] - box1[0]) * (box1[3] - box1[1])
        + (box2[2] - box2[0]) * (box2[3] - box2[1])
        - inter_area
    )
    # return IoU score = how much the boxes overlap (0 = no overlap, 1 = perfect overlap)
    return inter_area / union_area if union_area > 0 else 0

# video_path:path to input video files
# output_dir:flder where annoted frames will ba saved
def detect_collision_from_video(video_path, output_dir):

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    
    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        results = model(frame)[0]

        # filter out only vehicles detections
        # To collect only vehicle detection boxes (car, motorbike, bus, truck) from YOLOv8 output and store their bounding box coordinates
        # Create an empty list to store vehicle bounding boxes (box = detected object region)
        vehicle_boxes = []

        # zip(...)combines both
        # Each box is a tensor like this: tensor([x1, y1, x2, y2]),box = tensor([300.1, 80.5, 400.2, 160.0], device='cuda')
        # results.boxes.xyxy: bounding boxes → each is [x_min, y_min, x_max, y_max]
        # results.boxes.cls: predicted class index (like 0 = person, 2 = car, 5 = bus, etc.)
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls) in [2, 3, 5, 7]:  # car, motorbike, bus, truck

                # But OpenCV (cv2.rectangle) needs NumPy arrays, not PyTorch tensors
                # box is on GPU → .cpu() moves it to CPU memory
                # .numpy() converts the tensor to a NumPy array
                # append(...) adds it to the vehicle_boxes list
                # box.cpu().numpy() = array([300.1, 80.5, 400.2, 160.0])
                vehicle_boxes.append(box.cpu().numpy())

        # check both cars
        for i in range(len(vehicle_boxes)):
            for j in range(i + 1, len(vehicle_boxes)):

                # check all the boxes that next to each other, if the iou return the value 0.3
                if compute_iou(vehicle_boxes[i], vehicle_boxes[j]) > 0.3:

                    # first vehicle
                    x1, y1, x2, y2 = vehicle_boxes[i]
                    # Draws a red box (color (0, 0, 255)) around that vehicle
                    # the 2 is the thickness
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    # second vehicle
                    x1, y1, x2, y2 = vehicle_boxes[j]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # imageframe, the actual text to draw (wrning label), bottom left corner of the text, font type, font scale, red color bgr, thickness
                    cv2.putText(frame, "Collision", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # make the file name in sequence
        # Creates a unique name for the current frame: Example: frame_0000.jpg, frame_0001.jpg, etc
        # :04d means 4-digit zero-padded number
        frame_name = f"frame_{frame_count:04d}.jpg"

        # write the frame into the file name into the dirctory
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        frame_count += 1

    cap.release()
    return output_dir