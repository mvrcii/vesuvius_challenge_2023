import cv2
import numpy as np

points1 = []
points2 = []


def select_point(event, x, y, flags, params):
    global points1, points2, counter

    if event == cv2.EVENT_LBUTTONDOWN:
        points1.append((x, y))
        cv2.circle(params[0], (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image 1", params[0])
        print(f"Point selected on Image 1: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        points2.append((x, y))
        cv2.circle(params[1], (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image 2", params[1])
        print(f"Point selected on Image 2: ({x}, {y})")


def alignImages(im1, im2):
    global points1, points2

    # Display images for point selection
    cv2.imshow("Image 1", im1)
    cv2.imshow("Image 2", im2)
    cv2.setMouseCallback("Image 1", select_point, [im1, im2])
    cv2.setMouseCallback("Image 2", select_point, [im1, im2])

    print("Select points in both images. Left click for image 1, right click for image 2. Press 'q' to proceed.")

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or len(points1) >= 4 or len(points2) >= 4:
            break

    cv2.destroyAllWindows()

    # Ensure equal number of points in both images
    min_points = min(len(points1), len(points2))
    points1 = np.float32(points1[:min_points])
    points2 = np.float32(points2[:min_points])

    # Find homography
    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


if __name__ == '__main__':
    # Read reference image
    refFilename = "img1bw.png"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "img2bw.png"
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    if imReference is None or im is None:
        print("Error loading images!")
        exit()

    print("Aligning images ...")
    # Registered image will be restored in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)
