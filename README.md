**Lane-Detection-and-Driver-Guidance-SystemProject**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Camera Calibration
To compute the camera matrix, the following steps are necessary:  
1. Obtain a set of chessboard images taken from different angles (e.g., stored in `/camera_cal`).  
2. Define the dimensions of the chessboard (number of inner corners along rows and columns).  
3. Create a 3D matrix representing the chessboard corners in real-world coordinates.  
4. Convert the images to grayscale to accurately detect edges.  
5. Detect edges using the function `cv2.findChessboardCorners`.  
6. Use the function `cv2.calibrateCamera` to calculate the camera matrix, distortion coefficients, rotation, and translation vectors by minimizing the re-projection error between the real-world and image points.

   ![image](https://github.com/user-attachments/assets/696698ac-26ea-41f6-abff-5ed5bbad3c65)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

TODO: Add your text here!!!

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

TODO: Add your text here!!!

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

TODO: Add your text here!!!

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

