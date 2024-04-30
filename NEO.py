# Import necessary libraries
import warnings
warnings.filterwarnings("ignore")


import cv2
from eye import eyeDetect
from gaze import gazeDetect
from headPose import headPoseDetect
from threading import Thread
import os

# play an alarm sound
# Ignore warnings

# Initialize video capture object with the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Get the video width and height from the video capture object
videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer object to output the video in mp4 format, 30 FPS, and the same width and height as the input video
# output = cv2.VideoWriter(
#     "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (videoWidth, videoHeight)
# )

EYE_COUNTER = 0
HEAD_COUNTER = 0
GAZE_COUNTER = 0
ALARM_ON = False

def sound_alarm(path):
    # print("Waiting")
    while ALARM_ON:
        os.system("mpg123 " + path + " > /dev/null 2>&1")

# Main loop to continuously capture video frames
while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()

    # If the frame was not successfully read, then we have reached the end of the video
    if not ret:
        break

    # Apply head pose detection to the frame
    frame,HEAD_COUNTER = headPoseDetect(frame,HEAD_COUNTER)

    # Apply eye detection to the frame
    frame, EYE_COUNTER = eyeDetect(frame, EYE_COUNTER)



    # Apply gaze detection to the frame
    frame,GAZE_COUNTER = gazeDetect(frame,GAZE_COUNTER)

    if EYE_COUNTER >= 20 or HEAD_COUNTER >= 40 or GAZE_COUNTER >= 40:
        cv2.putText(frame, "ALERT", (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        if not ALARM_ON:
            ALARM_ON = True
            alarm_thread = Thread(target=sound_alarm, args=("alarm.mp3",))
            alarm_thread.start()

    else:
        ALARM_ON = False

    # Write the frame to the output video
    # output.write(frame)

    # Display the frame in a window named "Frame"
    cv2.imshow("Frame", frame)

    # Wait for 1 ms for a key press. If 'q' is pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()

# Release the video writer and capture objects
# output.release()
cap.release()
