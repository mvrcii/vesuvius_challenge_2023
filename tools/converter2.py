import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load images
MIN_MATCH_COUNT = 10
# img1 = cv.imread('img1bw.png', cv.IMREAD_GRAYSCALE)  # queryImage
# img2 = cv.imread('img2bw.png', cv.IMREAD_GRAYSCALE)  # trainImage

img1 = cv.imread('img1bw.png', cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread('img2bw.png', cv.IMREAD_GRAYSCALE)  # trainImage

# Find feature matches
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Filter good feature matches
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

# Abort if not enough good matches
if len(good) < MIN_MATCH_COUNT:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    exit()

src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
# Find homography matrix and do perspective transform
M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

M = np.array([[1.271064661994988798e+00, 4.578582273462539903e-02, -2.503758383655157047e+02],
              [4.092334932268167774e-02, 1.120982027416608018e+00, 4.719520913204767965e+00],
              [2.693060144129175550e-04, 4.690222400693215690e-05, 1.000000000000000000e+00]])

M = np.array([[7.98553694e-01, -5.40821307e-02, -6.36690724e+01],
                           [-2.22129869e-02, 8.65956681e-01, 4.28450289e+01],
                           [-1.30516607e-04, -5.57522890e-05, 1.00000000e+00]])

print(M)
print(type(M))
img1Reg = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
cv2.imwrite("adjusted.png", img1Reg)
matchesMask = mask.ravel().tolist()
h, w = img1.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv.perspectiveTransform(pts, M)
img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(img3, 'gray'), plt.show()
