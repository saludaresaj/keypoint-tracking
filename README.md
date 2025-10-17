# Human Body Keypoint Tracking with MediaPipe and Gradio

This project demonstrates **real-time human body keypoint tracking** using a webcam interface.  
By combining **MediaPipe Pose** for body landmark detection and **Gradio** for an interactive browser display,  
it provides a clear, hands-on example of modern pose estimation in action.

---

## 1. Introduction

Human pose estimation, often referred to as keypoint detection, is a computer vision task that identifies  
important body joints such as the shoulders, elbows, hips, and knees from images or videos.  
It has a wide range of applications in motion analysis, fitness tracking, animation, and human-computer interaction.

This project builds a simple but effective demo that performs real-time pose tracking directly from a laptop webcam.  
Using MediaPipe's pre-trained **Pose** model, it detects and connects 33 human body landmarks.  
The system is wrapped in a **Gradio interface** so it can be easily accessed through a web browser.  
When available, GPU acceleration can be used to improve performance.

---

## 2. How It Works

The program processes each webcam frame in real time through the following steps:

1. Capture an image from the webcam feed.  
2. Use **MediaPipe Pose** to estimate the position of 33 body keypoints.  
3. Draw the detected landmarks and their skeletal connections.  
4. Display the annotated frame along with the system's frame rate (FPS).  

---

## 3. Project Files

- **`app.py`:** Main application script that runs MediaPipe Pose and launches the Gradio interface.
- **`requirements.txt`:** Python dependencies needed to run the demo.

---

## 4. Requirements

Install the required libraries with:
```bash
pip install -r requirements.txt
