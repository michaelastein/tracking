import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide Tkinter main window
Tk().withdraw()

# Select images
img1_path = askopenfilename(title="Select the first image")
img2_path = askopenfilename(title="Select the second image")

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Initialize ORB
orb = cv2.ORB_create(2000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches (for reference)
match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

if len(matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is not None:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Define corners (FOV outlines)
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)

        # Warp image 1 corners and content
        warped_corners1 = cv2.perspectiveTransform(corners1, H)
        warped_img1 = cv2.warpPerspective(img1, H, (w2, h2))

        # Combine corners to build a large canvas
        all_corners = np.vstack((warped_corners1, corners2))
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 50)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 50)

        width = xmax - xmin
        height = ymax - ymin
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Shift so everything fits
        shift = np.array([[[-xmin, -ymin]]], dtype=np.float32)

        # Warp both images into the larger canvas space
        trans_mat = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
        warped_img2_big = cv2.warpPerspective(img2, trans_mat, (width, height))
        warped_img1_big = cv2.warpPerspective(warped_img1, trans_mat, (width, height))

        # Blend both images (image 1 semi-transparent)
        blended = cv2.addWeighted(warped_img2_big, 1.0, warped_img1_big, 0.5, 0)

        # Draw FOV outlines
        shifted_corners2 = corners2 + shift
        shifted_warped1 = warped_corners1 + shift
        cv2.polylines(blended, [np.int32(shifted_corners2)], True, (0, 0, 255), 2)
        cv2.polylines(blended, [np.int32(shifted_warped1)], True, (0, 255, 0), 2)
        cv2.putText(blended, "Image 2", tuple(np.int32(shifted_corners2[0, 0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(blended, "Image 1", tuple(np.int32(shifted_warped1[0, 0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show results
        cv2.imshow("ORB Matches", match_img)
        cv2.imshow("FOV + Image Overlay (Green=Image1, Red=Image2)", blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Homography could not be computed.")
else:
    print("Not enough matches to compute alignment.")
