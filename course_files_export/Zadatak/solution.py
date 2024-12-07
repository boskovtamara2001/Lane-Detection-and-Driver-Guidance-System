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

# 2. Define the source and destination points for perspective transform
# These points should roughly correspond to the lane markings on a straight section of road
src = np.float32([[440, 340], [530, 340], [850, 530], [190, 530]])
line_dst_offset = 150  # Offset to define the width of the transformed lane
dst = np.float32([[src[3][0] + line_dst_offset, 0],
                  [src[2][0] - line_dst_offset, 0],
                  [src[2][0] - line_dst_offset, 720],
                  [src[3][0] + line_dst_offset, 720]])

# Compute the perspective transform and its inverse
transform_matrix = cv2.getPerspectiveTransform(src, dst)
inv_transform_matrix = cv2.getPerspectiveTransform(dst, src)

# Function to process each video frame
def process_frame(frame):
    # 3. Apply distortion correction
    corrected_image = cv2.undistort(frame, camera_matrix, distortion_coeffs, None, camera_matrix)

    # Convert to grayscale and detect edges
    grayscale_corrected = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    edges = cv2.Canny(grayscale_corrected, 150, 270, apertureSize=3)  # Apply Canny edge detection

    # 4. Warp the image to a bird's-eye view
    img_pt = cv2.warpPerspective(edges, transform_matrix, dsize=(edges.shape[1], edges.shape[0]), flags=cv2.INTER_LINEAR)

    # 5. Detect lane pixels and fit to find the lane boundary
    # Create a histogram to find lane pixels
    histogram = np.sum(img_pt[img_pt.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])  # Base of the left lane
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint  # Base of the right lane

    # Sliding window parameters
    nwindows = 9
    window_height = int(img_pt.shape[0] / nwindows)
    nonzero = img_pt.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50

    # Arrays to store lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Define window boundaries
        win_y_low = img_pt.shape[0] - (window + 1) * window_height
        win_y_high = img_pt.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify pixels within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append detected indices
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Recenter the window if sufficient pixels are found
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the lane indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract lane pixel coordinates
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit second-degree polynomials to the lane lines
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate y-values and corresponding x-values for the lane lines
    ploty = np.linspace(0, img_pt.shape[0] - 1, img_pt.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # 6. Determine the curvature of the lane and vehicle position with respect to center
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    left_curverad = ((1 + (2 * left_fit[0] * np.max(ploty) * ym_per_pix + left_fit[1])**2)**1.5) / np.abs(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * np.max(ploty) * ym_per_pix + right_fit[1])**2)**1.5) / np.abs(2 * right_fit[0])
    curvature = (left_curverad + right_curverad) / 2
    lane_center = (leftx_base + rightx_base) / 2
    image_center = img_pt.shape[1] / 2
    center_offset = (image_center - lane_center) * xm_per_pix

    # 7. Warp the detected lane boundaries back onto the original image
    lane_image = np.zeros_like(frame)
    lane_pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    lane_pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]).astype(int)
    cv2.fillPoly(lane_image, np.int_([np.hstack((lane_pts_left, lane_pts_right))]), (0, 255, 0))
    lane_overlay = cv2.warpPerspective(lane_image, inv_transform_matrix, (frame.shape[1], frame.shape[0]))
    annotated_frame = cv2.addWeighted(frame, 1, lane_overlay, 0.5, 0)

    # Add curvature and offset annotations
    curvature_text = f"Radius of Curvature: {curvature:.2f} m"
    position_text = f"Vehicle is {abs(center_offset):.2f} m {'left' if center_offset < 0 else 'right'} of center"
    cv2.putText(annotated_frame, curvature_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(annotated_frame, position_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return annotated_frame

# 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
v_path = "test_videos/project_video02.mp4"
cap = cv2.VideoCapture(v_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)
    cv2.imshow("Lane Detection", processed_frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
