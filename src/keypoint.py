# Gesture Control for PowerPoint
# Author: AJ Saludares
# Date: 28 May 2025

import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import time
import atexit # For ensuring resources are released

# --- MediaPipe Pose Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles # For nicer default styles

pose_detector = None # Will be initialized in main

def initialize_mediapipe():
    global pose_detector
    if pose_detector is None:
        print("Initializing MediaPipe Pose detector...")
        pose_detector = mp_pose.Pose(
            static_image_mode=False,      # Process as video stream
            model_complexity=1,           # 0=lite, 1=full, 2=heavy. 1 is a good balance.
            smooth_landmarks=True,        # Reduce jitter
            enable_segmentation=False,    # We don't need segmentation for keypoints only
            smooth_segmentation=True,     # (irrelevant if enable_segmentation=False)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("MediaPipe Pose detector initialized.")

def cleanup_mediapipe():
    global pose_detector
    if pose_detector:
        print("Closing MediaPipe Pose detector...")
        pose_detector.close()
        pose_detector = None # Mark as closed
        print("MediaPipe Pose detector closed.")

# Register cleanup function to be called on script exit
atexit.register(cleanup_mediapipe)

# --- Global variable for FPS calculation ---
prev_time = 0

# --- Gradio processing function ---
def process_frame_mediapipe(frame_np):
    global prev_time
    global pose_detector

    if pose_detector is None:
        # This might happen if Gradio calls this before main has fully run
        # or if there was an issue during initialization.
        # For robustness, we can try initializing here, but it's better done once.
        initialize_mediapipe()
        if pose_detector is None: # Still None after trying to init
             print("Error: MediaPipe Pose detector not initialized.")
             return np.zeros((480, 640, 3), dtype=np.uint8) # Return black frame

    if frame_np is None:
        # Return a black image if no frame is received (e.g., camera not ready)
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # MediaPipe expects RGB. Gradio webcam image is already RGB NumPy array.
    # For drawing, make a writable copy.
    annotated_image = frame_np.copy()

    # Process the frame with MediaPipe Pose
    # To improve performance, optionally mark the image as not writeable before
    # passing it to MediaPipe if you were passing `frame_np` directly and not its copy.
    # frame_np.flags.writeable = False # Example if processing original
    results = pose_detector.process(frame_np) # Process original RGB frame_np, draw on 'annotated_image'
    # frame_np.flags.writeable = True # Example if processing original

    # Draw the pose annotations on the image.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style() # Prettier points
        )
        # You can also customize drawing specs:
        # landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
        # connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)

    # Calculate FPS
    current_time = time.time()
    fps = 0
    if prev_time > 0:
        fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS on the frame
    cv2.putText(annotated_image, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) # Red color for FPS

    return annotated_image

# --- Gradio Interface ---
interface_description = (
    "Real-time human body keypoint tracking using MediaPipe Pose. "
)

iface_mediapipe = gr.Interface(
    fn=process_frame_mediapipe,
    inputs=gr.Image(sources="webcam", type="numpy", label="Webcam Feed", streaming=True),
    outputs=gr.Image(type="numpy", label="Output with MediaPipe Keypoints"),
    title="Live Human Body Keypoint Tracking (MediaPipe)",
    description=interface_description,
    live=True # Crucial for continuous processing
)

# --- Launch the Gradio app ---
if __name__ == '__main__':
    initialize_mediapipe() # Initialize MediaPipe when the script starts
    print("Launching Gradio interface with MediaPipe...")
    try:
        iface_mediapipe.launch(share=True)
    except Exception as e:
        print(f"An error occurred during Gradio launch: {e}")
    finally:
        print("Gradio app has been closed or an error occurred.")