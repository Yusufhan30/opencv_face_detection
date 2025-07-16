# 🧑‍💻 Face Detection with OpenCV

This is a simple Python script that uses OpenCV to perform face detection on webcam input using Haar cascade classifiers.

## 🚀 How it works
- Opens your webcam stream
- Converts each frame to grayscale
- Uses `haarcascade_frontalface_default.xml` to detect faces
- Draws rectangles around detected faces
- Press `q` to exit

## 🐍 How to run
Make sure you have OpenCV installed:

```bash
pip install opencv-python
