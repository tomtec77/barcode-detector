import numpy as np
import argparse
import cv2

# Parse the input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# Load the image and convert to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute the Scharr gradient magnitude representation of the image in
# both the x and y direction
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

# Subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

cv2.imshow("Gradient", gradient)
cv2.waitKey(0)

# Blur and threshold the image
# Apply an average blur to the gradient image, using a 9x9 kernel.
# This helps smooth out high frequency noise in the gradient
# representation of the image
blurred = cv2.blur(gradient, (9,9))

# Threshold the blurred image - any pixel not greater than 225 is set to
# zero (black); otherwise, the pixel is set to 255 (white)
(_, thresh) = cv2.threshold(blurred, 225, 225, cv2.THRESH_BINARY)

cv2.imshow("Blur and threshold", thresh)
cv2.waitKey(0)

# After the previous step there are gaps between the vertical bars. In
# order to close the gaps and make it easier for the algorithm to detect
# the "blob"-like region of the barcode, start by constructing a
# rectangular kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))

# Apply the kernel to the thresholded image
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cv2.imshow("After morphological transformation", closed)
cv2.waitKey(0)

# Remove the small blobs in the image that aren't part of the actual
# barcode, but may interfere with the contour detection.
# Perform 4 iterations of erosions ("erode" white pixels in the image,
# thus removing small blobs) followed by 4 iterations of dilations
# ("dilate" remaining white pixels to grow the white regions back out)
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)

cv2.imshow("After erosion and dilation", closed)
cv2.waitKey(0)

# Find the contours in the thresholded image, then sort the contours by
# their area, keeping only the largest one
(_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

# Compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))

# Draw a bounding box around the detected barcode and display the image
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.imshow("Detected barcode", image)
cv2.waitKey(0)



