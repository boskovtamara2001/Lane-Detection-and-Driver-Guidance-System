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

# Perform camera calibration
success, camera_matrix, distortion_coeffs, rotation_vectors, translation_vectors = cv2.calibrateCamera(
    object_points, image_points, grayscale_image.shape[::-1], None, None)

# Load a test image for processing
test_image = cv2.imread('test_images/whiteCarLaneSwitch.jpg')

# 2. Apply a distortion correction to raw images
corrected_image = cv2.undistort(test_image, camera_matrix, distortion_coeffs, None, camera_matrix)

# 3. Use color transforms, gradients, etc., to create a thresholded binary image
grayscale_corrected = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(grayscale_corrected, 50, 150)  # Adjust thresholds as needed

# Display the original corrected image and the edges
# cv2.imshow("Corrected Image", corrected_image)
# cv2.imshow("Canny Edges", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 4. Apply a perspective transform to rectify binary image ("birds-eye view").
# Define the source points (4 points from the input image)
src = [
    [440, 340],  # Top-left
    [530, 340],  # Top-right
    [850, 530],  # Bottom-right
    [190, 530]   # Bottom-left
]

# Define the destination points for perspective transformation (with a wider perspective)
line_dst_offset = 150  # Larger offset for wider perspective
dst = [
    [src[3][0] + line_dst_offset, 0],                       # Top-left
    [src[2][0] - line_dst_offset, 0],                       # Top-right
    [src[2][0] - line_dst_offset, edges.shape[0]],     # Bottom-right
    [src[3][0] + line_dst_offset, edges.shape[0]]      # Bottom-left
]

# Convert points to float32 for compatibility with OpenCV functions
src = np.float32(src)
dst = np.float32(dst)

# Compute the perspective transformation matrix
transform_matrix = cv2.getPerspectiveTransform(src, dst)

# Apply the perspective transformation
img_pt = cv2.warpPerspective(edges, transform_matrix, dsize=(edges.shape[1], edges.shape[0]), flags=cv2.INTER_LINEAR)

# Display the resulting image
cv2.imshow('Birds-eye Perspective Transform', img_pt)
cv2.waitKey(0)
cv2.destroyAllWindows()