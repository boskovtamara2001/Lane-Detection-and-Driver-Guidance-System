# Lane-Detection-and-Driver-Guidance-System
This project is part of the Multimedia Systems in the Automotive Industry course. It is a program designed to process video footage and images to detect white lane markings on the road. The program provides visual guidance to the driver by overlaying direction lines on the video, indicating the optimal path to stay within their lane.

**Features**
**Camera Calibration**
The program includes functionality to calibrate the camera using chessboard patterns.
Camera calibration compensates for lens distortion to ensure accurate lane detection.
The calibration process computes the camera matrix and distortion coefficients based on multiple chessboard images.

**Lane Detection**
Processes images and videos to detect lane markings.
Combines techniques such as edge detection and color thresholding for robust detection of both white and yellow lane markings.

**Driver Guidance**
Overlays direction lines on the video feed to provide real-time guidance for lane following.

**Camera Calibration Process**
Input: A set of chessboard images captured by the camera.

Steps:
  1. Detect inner corners on the chessboard images using the cv2.findChessboardCorners function.
  2. Map 2D image points (chessboard corners) to corresponding 3D real-world points.
  3. Compute the camera matrix and distortion coefficients using cv2.calibrateCamera.
  4. Undistort sample images to verify the calibration process.

**Prerequisites**
Python 3.8+
OpenCV library (cv2)
NumPy library (numpy)

**How to Run**
Install Dependencies:
pip install opencv-python numpy  

**File Structure**
camera_cal/: Contains calibration images.
test_images/: Sample road images for testing.
camera_calibration.py: Script for camera calibration.
lane_detection.py: Script for detecting lane markings and providing guidance.

**Future Enhancements**
Implement real-time lane detection using a live video feed.
Add support for curved lane detection and predictive path calculations.
