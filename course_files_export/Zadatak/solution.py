import cv2
import numpy as np
import glob

# 1. Define the chessboard dimensions and square size
chessboard_dims = (9, 6)  # Number of inner corners along rows and columns
square_size = 1.0  # Physical size of a square on the chessboard (e.g., 1 unit)

# Prepare a 3D grid of points representing the chessboard corners in real-world coordinates
object_points_template = np.zeros((chessboard_dims[0] * chessboard_dims[1], 3), np.float32)
object_points_template[:, :2] = np.indices((chessboard_dims[0], chessboard_dims[1])).T.reshape(-1, 2) * square_size

# Lists to store 3D real-world points and 2D image points
object_points = []  # 3D points in real-world space
image_points = []   # 2D points in image space (detected chessboard corners)

# 2. Load all calibration images
image_files = glob.glob('camera_cal/*.jpg')  # Modify path as needed

for image_path in image_files:
    # Read the image and convert it to grayscale
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Detect chessboard corners in the image
    found, corners = cv2.findChessboardCorners(grayscale_image, chessboard_dims, None)

    if found:
        # Add detected corners to the image points list
        image_points.append(corners)
        # Add corresponding real-world points to the object points list
        object_points.append(object_points_template)

        # Optional: visualize the detected corners on the chessboard
        # cv2.drawChessboardCorners(image, chessboard_dims, corners, found)
        # cv2.imshow('Detected Corners', image)
        # cv2.waitKey(500)

# cv2.destroyAllWindows()

# 4. Perform camera calibration to obtain the camera matrix and distortion coefficients
success, camera_matrix, distortion_coeffs, rotation_vectors, translation_vectors = cv2.calibrateCamera(
    object_points, image_points, grayscale_image.shape[::-1], None, None)

# Print the results
print("Camera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(distortion_coeffs)

# 5. Load a test image for undistortion
test_image = cv2.imread('test_images\challange00101.jpg')

# Undistort the image using the calibration results
corrected_image = cv2.undistort(test_image, camera_matrix, distortion_coeffs, None, camera_matrix)

# Optional: display the original distorted image and the corrected undistorted image
# cv2.imshow('Original Distorted Image', test_image)
# cv2.imshow('Corrected Undistorted Image', corrected_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
