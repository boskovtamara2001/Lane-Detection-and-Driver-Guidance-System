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
#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
To compute the camera matrix, the following steps are necessary:  
1. Obtain a set of chessboard images taken from different angles (e.g., stored in `/camera_cal`).  
2. Define the dimensions of the chessboard (number of inner corners along rows and columns).  
3. Create a 3D matrix representing the chessboard corners in real-world coordinates.  
4. Convert the images to grayscale to accurately detect edges.  
5. Detect edges using the function `cv2.findChessboardCorners`.  
6. Use the function `cv2.calibrateCamera` to calculate the camera matrix, distortion coefficients, rotation, and translation vectors by minimizing the re-projection error between the real-world and image points.

Example of a distortion corrected calibration image:
   ![image](https://github.com/user-attachments/assets/696698ac-26ea-41f6-abff-5ed5bbad3c65)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
Thresholded binary image is created through a combination of color transformations and edge detection techniques. The process begins with converting the input image to grayscale using cv2.cvtColor(), which simplifies the image by removing color information and focusing on intensity values. This grayscale image is then processed using the Canny edge detection algorithm (cv2.Canny()), which identifies areas of high intensity gradients, effectively highlighting the edges of the lane markings.

The result is a binary image, where pixel values of 1 (white) represent detected edges and pixel values of 0 (black) represent areas with no significant intensity changes. This binary image is essential for further lane detection and fitting processes in the pipeline.

Transformations are performed in lines 56-57.
Example of a Binary Image Result:
   ![image](https://github.com/user-attachments/assets/f41ba28f-7cab-4577-86de-d28603aeddac)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
Source and Destination Points: I first defined the source points (src) that correspond to the region of interest in the image (the lane markings on a straight section of the road). The destination points (dst) were chosen to map these source points into a top-down view of the road. The line_dst_offset was used to adjust the width of the transformed lane area to make it more consistent.

The source and destination points are defined as follows:
```python
src = np.float32([[440, 340], [530, 340], [850, 530], [190, 530]])
dst = np.float32([[src[3][0] + line_dst_offset, 0],
                  [src[2][0] - line_dst_offset, 0],
                  [src[2][0] - line_dst_offset, 720],
                  [src[3][0] + line_dst_offset, 720]])
```
                  
Transformation Matrix: Using the cv2.getPerspectiveTransform function, I computed the matrix that maps the source points to the destination points.
python
transform_matrix = cv2.getPerspectiveTransform(src, dst)
inv_transform_matrix = cv2.getPerspectiveTransform(dst, src)
Warping the Image: With the transformation matrix, I applied the perspective warp using cv2.warpPerspective. This transformed the image into a bird's-eye view of the lane markings, which helps in better lane detection. The transformation is performed in the process_frame function.

python
img_pt = cv2.warpPerspective(edges, transform_matrix, dsize=(edges.shape[1], edges.shape[0]), flags=cv2.INTER_LINEAR)
The perspective transform is crucial for lane detection because it makes the road appear as if viewed from above, simplifying the analysis of lane boundaries.

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

