import cv2
import os

# function that takes the path to a video file as input
def detect_road(video_path, output_dir='media/processed_frames'):

    # Opens the video file for reading
    cap = cv2.VideoCapture(video_path)

    # Creates a background subtractor, which detects moving objects by subtracting the background.
    # Useful for detecting cars, people, etc., on roads.
    fgbg = cv2.createBackgroundSubtractorMOG2()

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

        # Stores the foreground mask (a grayscale image) in the results list.
        # results.append(fgmask)

        filename = os.path.join(output_dir, f'frame_{frame_count}.jpg')
        cv2.imwrite(filename, fgmask)  # Save each processed frame
        results.append(f'processed_frames/frame_{frame_count}.jpg')
        frame_count += 1

    # close the video
    cap.release()
    return results
