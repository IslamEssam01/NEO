from gaze_tracking import GazeTracking
import cv2
from cv2.typing import MatLike

# Create the GazeTracking object. This object will be used to perform gaze tracking on each frame.
gaze = GazeTracking()


def gazeDetect(frame: MatLike, counter: int) -> tuple[MatLike, int]:
    """
    Detect gaze in the given frame and annotate the frame with the gaze direction and pupil coordinates.

    This function uses the GazeTracking library to detect the user's gaze in a given frame. It annotates the frame with the
    detected gaze direction (e.g., "Looking left") and the coordinates of the left and right pupils.

    Parameters:
    frame (MatLike): The input frame. This should be an image that contains a face with visible eyes.
    counter (int): A counter to keep track of the number of frames processed.

    Returns:
    tuple[MatLike,int]: The input frame annotated with the gaze direction and pupil coordinates, and the updated counter.
    """
    # Make a copy of the frame
    img = frame.copy()

    # Refresh the gaze object with the current frame
    gaze.refresh(img)

    # Get an annotated frame from the gaze object. This frame is annotated with the detected eyes and pupils.
    img: MatLike = gaze.annotated_frame()

    # Initialize the text that will be displayed on the frame
    text = ""

    # Determine the gaze direction and set the display text accordingly
    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
        counter += 1
    elif gaze.is_left():
        text = "Looking left"
        counter += 1
    elif gaze.is_center():
        text = "Looking center"
        counter = 0

    # Draw the gaze direction text on the frame
    cv2.putText(img, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # Get the coordinates of the left and right pupils
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    # Draw the pupil coordinates on the frame
    cv2.putText(
        img,
        f"Left pupil:  {left_pupil}",
        (90, 130),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (147, 58, 31),
        1,
    )
    cv2.putText(
        img,
        f"Right pupil: {right_pupil}",
        (90, 165),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (147, 58, 31),
        1,
    )

    # Return the annotated frame
    return img, counter
