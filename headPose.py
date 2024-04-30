import numpy as np
import pickle
import mediapipe as mp
import cv2
import pandas as pd
from cv2.typing import MatLike

# Define constants for the facial landmarks we're interested in
NOSE = 1
FOREHEAD = 10
LEFT_EYE = 33
MOUTH_LEFT = 61
CHIN = 199
RIGHT_EYE = 263
MOUTH_RIGHT = 291


def load_model(model_path: str):
    """
    Load the model from the specified path.

    Parameters:
    model_path (str): The path to the model file.

    Returns:
    The loaded model, or None if an error occurred.
    """
    try:
        return pickle.load(open(model_path, "rb"))
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Load the model
model = load_model("./model.pkl")


def extract_features(img: MatLike, face_mesh: mp.solutions.face_mesh.FaceMesh) -> list:
    """
    Extract features from the image using the face mesh.

    Parameters:
    img (MatLike): The input image.
    face_mesh (mp.solutions.face_mesh.FaceMesh): The face mesh object.

    Returns:
    A list of feature values.
    """
    result = face_mesh.process(img)
    face_features = []

    # If landmarks are detected, extract the features we're interested in
    if result.multi_face_landmarks is not None:
        for face_landmarks in result.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [
                    FOREHEAD,
                    NOSE,
                    MOUTH_LEFT,
                    MOUTH_RIGHT,
                    CHIN,
                    LEFT_EYE,
                    RIGHT_EYE,
                ]:
                    face_features.append(lm.x)
                    face_features.append(lm.y)

    return face_features


def normalize(poses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the poses dataframe.

    Parameters:
    poses_df (pd.DataFrame): The dataframe containing the pose data.

    Returns:
    A new dataframe with the normalized pose data.
    """
    normalized_df = poses_df.copy()

    # Center the data around the nose and scale it
    for dim in ["x", "y"]:
        for feature in [
            "forehead_",
            "nose_",
            "mouth_left_",
            "mouth_right_",
            "left_eye_",
            "chin_",
            "right_eye_",
        ]:
            normalized_df[feature + dim] = (
                poses_df[feature + dim] - poses_df["nose_" + dim]
            )

        diff = normalized_df["mouth_right_" + dim] - normalized_df["left_eye_" + dim]
        for feature in [
            "forehead_",
            "nose_",
            "mouth_left_",
            "mouth_right_",
            "left_eye_",
            "chin_",
            "right_eye_",
        ]:
            normalized_df[feature + dim] = normalized_df[feature + dim] / diff

    return normalized_df


def draw_axes(
    img: MatLike,
    pitch: float,
    yaw: float,
    roll: float,
    tx: int,
    ty: int,
    size: int = 50,
) -> MatLike:
    """
    Draw axes on the image.

    Parameters:
    img (MatLike): The input image.
    pitch (float): The pitch angle.
    yaw (float): The yaw angle.
    roll (float): The roll angle.
    tx (int): The x-coordinate of the translation vector.
    ty (int): The y-coordinate of the translation vector.
    size (int, optional): The size of the axes. Defaults to 50.

    Returns:
    A new image with the axes drawn.
    """
    yaw = -yaw
    rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
    axes_points = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * size).astype(int)
    axes_points[0, :] = axes_points[0, :] + tx
    axes_points[1, :] = axes_points[1, :] + ty

    new_img = img.copy()
    cv2.line(
        new_img,
        tuple(axes_points[:, 3].ravel()),
        tuple(axes_points[:, 0].ravel()),
        (255, 0, 0),
        3,
    )
    cv2.line(
        new_img,
        tuple(axes_points[:, 3].ravel()),
        tuple(axes_points[:, 1].ravel()),
        (0, 255, 0),
        3,
    )
    cv2.line(
        new_img,
        tuple(axes_points[:, 3].ravel()),
        tuple(axes_points[:, 2].ravel()),
        (0, 0, 255),
        3,
    )

    return new_img


def generate_column_names() -> list:
    """
    Generate column names.

    Returns:
    A list of column names.
    """
    cols = []
    for pos in [
        "nose_",
        "forehead_",
        "left_eye_",
        "mouth_left_",
        "chin_",
        "right_eye_",
        "mouth_right_",
    ]:
        for dim in ("x", "y"):
            cols.append(pos + dim)
    return cols


# Generate the column names
cols = generate_column_names()


def headPoseDetect(frame: MatLike, counter: int) -> tuple[MatLike, int]:
    """
    Detect head pose in the given frame.

    Parameters:
    frame (MatLike): The input frame.
    counter (int): A counter to keep track of the number of frames processed.

    Returns:
    tuple[MatLike, int]: The frame with the detected head pose drawn, and the updated counter.
    """
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    img = frame.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 1)

    img_h, img_w, img_c = img.shape
    text = ""
    face_features = extract_features(img, face_mesh)
    if len(face_features):
        face_features_df = pd.DataFrame([face_features], columns=cols)
        face_features_normalized = normalize(face_features_df)
        pitch_pred, yaw_pred, roll_pred = model.predict(
            face_features_normalized
        ).ravel()
        nose_x = face_features_df["nose_x"].values * img_w
        nose_y = face_features_df["nose_y"].values * img_h
        img = draw_axes(img, pitch_pred, yaw_pred, roll_pred, nose_x, nose_y)

        if pitch_pred > 0.3:
            text = "Top"
            counter += 1
            if yaw_pred > 0.3:
                text = "Top Left"
            elif yaw_pred < -0.3:
                text = "Top Right"
        elif pitch_pred < -0.3:
            counter += 1
            text = "Bottom"
            if yaw_pred > 0.3:
                text = "Bottom Left"
            elif yaw_pred < -0.3:
                text = "Bottom Right"
        elif yaw_pred > 0.3:
            counter += 1
            text = "Left"
        elif yaw_pred < -0.3:
            counter += 1
            text = "Right"
        else:
            counter = 0
            text = "Forward"
    img = cv2.flip(img, 1)
    cv2.putText(img, text, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img, counter
