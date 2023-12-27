import cv2
import numpy as np
from scipy.interpolate import CubicSpline

def create_lut(control_points):
    # Extract the x and y coordinates of the control points
    x_points, y_points = zip(*control_points)

    # Create a cubic spline passing through the control points
    spline = CubicSpline(x_points, y_points)

    # Generate the LUT by evaluating the spline for each input intensity
    lut = spline(np.arange(256))

    # Clip the LUT to the valid range and convert to uint8
    lut = np.clip(lut, 0, 255).astype('uint8')
    return lut

# Define the control points for the curve
# You will need to add more points to match the curve precisely
# The given point (114, 55) is included here
control_points = [(0, 0), (114, 55), (255, 255)]

# Create the LUT based on the curve
lut = create_lut(control_points)

# Apply the LUT to an image
def apply_lut(image, lut):
    # Apply the LUT to each channel of the image
    if len(image.shape) == 3:
        return cv2.merge([cv2.LUT(channel, lut) for channel in cv2.split(image)])
    else:
        return cv2.LUT(image, lut)

# Load your image
image = cv2.imread('00035.tif', cv2.IMREAD_COLOR)

# Apply the LUT to the image
transformed_image = apply_lut(image, lut)

# Save the transformed image
cv2.imwrite('transformed_image.jpg', transformed_image)
