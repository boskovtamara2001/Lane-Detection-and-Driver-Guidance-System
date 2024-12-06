import cv2
import numpy as np
import glob

# 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
# Define chessboard dimensions
chessboard_dims = (9, 6)  # Number of inner corners along rows and columns
square_size = 1.0  # Physical size of one square on the chessboard

# Create a 3D grid of points representing the chessboard corners in real-world coordinates
object_points_template = np.zeros((chessboard_dims[0] * chessboard_dims[1], 3), np.float32)
object_points_template[:, :2] = np.indices((chessboard_dims[0], chessboard_dims[1])).T.reshape(-1, 2) * square_size

# Lists to store 3D real-world points and 2D image points
object_points = []  # 3D points in real-world space
image_points = []   # 2D points in image space (detected chessboard corners)

# Load all calibration images
image_files = glob.glob('camera_cal/*.jpg')  # Modify the path as needed

for image_path in image_files:
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect chessboard corners
    found, corners = cv2.findChessboardCorners(grayscale_image, chessboard_dims, None)

    if found:
        # Append detected corners and corresponding real-world points
        image_points.append(corners)
        object_points.append(object_points_template)

# Perform camera calibration to get the camera matrix and distortion coefficients
success, camera_matrix, distortion_coeffs, rotation_vectors, translation_vectors = cv2.calibrateCamera(
    object_points, image_points, grayscale_image.shape[::-1], None, None)

# 2. Apply a distortion correction to raw images
test_image = cv2.imread('test_images/whiteCarLaneSwitch.jpg')  # Load the test image for lane detection

# Correct distortion based on the camera calibration results
corrected_image = cv2.undistort(test_image, camera_matrix, distortion_coeffs, None, camera_matrix)

# 3. Use color transforms, gradients, etc., to create a thresholded binary image
grayscale_corrected = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Apply Canny edge detection to detect edges in the image
edges = cv2.Canny(grayscale_corrected, 150, 170, apertureSize=3)  # Adjust threshold as needed

# 4. Apply a perspective transform to rectify the binary image ("bird's-eye view")
# Define the source points for the perspective transform (coordinates in the image)
src = np.float32([[440, 340], [530, 340], [850, 530], [190, 530]])

# Define the destination points (coordinates after the transformation)
line_dst_offset = 150  # Larger offset gives a wider "bird's-eye" view
dst = np.float32([[src[3][0] + line_dst_offset, 0],
                  [src[2][0] - line_dst_offset, 0],
                  [src[2][0] - line_dst_offset, edges.shape[0]],
                  [src[3][0] + line_dst_offset, edges.shape[0]]])

# Compute the perspective transform matrix
transform_matrix = cv2.getPerspectiveTransform(src, dst)

# Apply the perspective transformation to the image
img_pt = cv2.warpPerspective(edges, transform_matrix, dsize=(edges.shape[1], edges.shape[0]), flags=cv2.INTER_LINEAR)

# 5. Detect lane pixels and fit to find the lane boundary
# Create a histogram of the image to find the base of the lanes
histogram = np.sum(img_pt[img_pt.shape[0]//2:, :], axis=0)
midpoint = int(histogram.shape[0] // 2)  # Find the midpoint of the histogram
leftx_base = np.argmax(histogram[:midpoint])  # Find the base of the left lane
rightx_base = np.argmax(histogram[midpoint:]) + midpoint  # Find the base of the right lane

# Define the number of sliding windows for lane search
nwindows = 9
window_height = int(img_pt.shape[0] / nwindows)  # Height of each window
nonzero = img_pt.nonzero()  # Get the indices of all non-zero pixels
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# Set initial positions for the left and right lanes
leftx_current = leftx_base
rightx_current = rightx_base
margin = 100  # Width of the window for searching
minpix = 50   # Minimum number of pixels required to consider a lane

left_lane_inds = []
right_lane_inds = []

# Slide through the image and find the pixels corresponding to the lanes
for window in range(nwindows):
    win_y_low = img_pt.shape[0] - (window + 1) * window_height
    win_y_high = img_pt.shape[0] - window * window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    # Find the indices of the pixels that lie within the window for both lanes
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                      (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    
    # Append the indices of the detected lane pixels
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    
    # Update the search positions based on the detected pixels
    if len(good_left_inds) > minpix:
        leftx_current = int(np.mean(nonzerox[good_left_inds]))  # Recalculate the left lane position
    if len(good_right_inds) > minpix:
        rightx_current = int(np.mean(nonzerox[good_right_inds]))  # Recalculate the right lane position

# Combine the indices from all windows
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Get the x and y coordinates of the detected lane pixels
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Fit a second-degree polynomial to the detected lane pixels
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# 6. Calculate lane curvature and vehicle position
# Define conversion factors
ym_per_pix = 30 / 720  # Meters per pixel in y-dimension
xm_per_pix = 3.7 / 700  # Meters per pixel in x-dimension

# Generate y-values
ploty = np.linspace(0, img_pt.shape[0] - 1, img_pt.shape[0])

# Calculate curvature for left and right lanes
left_curverad = ((1 + (2 * left_fit[0] * np.max(ploty) * ym_per_pix + left_fit[1])**2)**1.5) / np.abs(2 * left_fit[0])
right_curverad = ((1 + (2 * right_fit[0] * np.max(ploty) * ym_per_pix + right_fit[1])**2)**1.5) / np.abs(2 * right_fit[0])

# Average the curvature
curvature = (left_curverad + right_curverad) / 2

# Calculate vehicle position with respect to the lane center
lane_center = (leftx_base + rightx_base) / 2
image_center = img_pt.shape[1] / 2
center_offset = (image_center - lane_center) * xm_per_pix

# 9. Warp the detected lane boundaries back onto the original image and annotate it
lane_image = np.zeros_like(test_image)

# Generate x-values for plotting
left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

# Create lane area
lane_pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
lane_pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
lane_pts = np.hstack((lane_pts_left, lane_pts_right))

cv2.fillPoly(lane_image, np.int_([lane_pts]), (0, 255, 0))

# Compute the inverse perspective transform matrix
inv_transform_matrix = cv2.getPerspectiveTransform(dst, src)

# Warp the lane image back to the original perspective
lane_overlay = cv2.warpPerspective(lane_image, inv_transform_matrix, (test_image.shape[1], test_image.shape[0]))

# Combine the overlay with the original image
annotated_image = cv2.addWeighted(test_image, 1, lane_overlay, 0.5, 0)

# Add curvature and position annotations
curvature_text = f"Radius of Curvature: {curvature:.2f} m"
position_text = f"Vehicle is {abs(center_offset):.2f} m {'left' if center_offset < 0 else 'right'} of center"

cv2.putText(annotated_image, curvature_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(annotated_image, position_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Display the final annotated image
cv2.imshow('Lane Detection', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
