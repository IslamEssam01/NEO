from numpy.typing import NDArray
from scipy.spatial import distance
from imutils import face_utils
import dlib
import cv2
from cv2.typing import MatLike

# Define constants
THRESHOLD = 0.25  # Threshold for eye aspect ratio below which eye is considered closed
FRAME_CHECK = 20  # Number of consecutive frames the eye must be below the threshold for to signify a blink
CLOSED_TEXT = "*************************Closed*************************"  # Text to display when eyes are closed
OPEN_TEXT = "*************************Open*************************"  # Text to display when eyes are open


def eye_aspect_ratio(eye: NDArray) -> float:
    """
    Calculate the eye aspect ratio (EAR) given eye landmarks.

    Parameters:
    eye (NDArray): The coordinates of the eye landmarks.

    Returns:
    float: The eye aspect ratio.
    """
    # Compute the euclidean distances between the two sets of vertical eye landmarks and the horizontal eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    # The EAR is the ratio of the average of the distances of A and B to the distance C
    ear = (A + B) / (2.0 * C)
    return ear


# Initialize the face detector (HOG-based) and then create the facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("./gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat")

# Grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]


def eyeDetect(frame: MatLike, counter: int) -> tuple[MatLike, int]:
    """
    Detect eyes in the given frame and determine whether they are open or closed.

    Parameters:
    frame (MatLike): The input frame. This should be an image that contains a face with visible eyes.
    counter (int): A counter to keep track of the number of frames processed.

    Returns:
    tuple[MatLike, int]: The frame with the detected eyes drawn and their state (open or closed) indicated, and the updated counter.
    """
    # Make a copy of the frame and convert it to grayscale
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    subjects = detect(gray, 0)

    # Loop over the face detections
    for subject in subjects:
        # Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Compute the convex hull for the left and right eye, then visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
        # text = CLOSED_TEXT if ear < THRESHOLD else OPEN_TEXT
        if ear < THRESHOLD:
            counter += 1
            text = CLOSED_TEXT
        else:
            counter = 0
            text = OPEN_TEXT

        # Draw the text on the frame
        cv2.putText(img, text, (10, 325), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    # Return the frame with the drawn eyes and the state (open or closed) indicated
    return img, counter
