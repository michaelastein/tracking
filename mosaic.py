import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

# Hide the main Tkinter window
Tk().withdraw()

# Open file dialog to select up to 10 images
img_paths = askopenfilenames(title="Select up to 10 images", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
if len(img_paths) < 2:
    print("Select at least 2 images.")
    exit()

# Load all images
images = [cv2.imread(path) for path in img_paths]

def stitch_images(img1, img2):
    """Stitch two images together using ORB and homography."""
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        print("Not enough matches found.")
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("Homography could not be computed.")
        return None

    # Warp img1 to img2
    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + img2.shape[1]
    result = cv2.warpPerspective(img1, H, (width, height))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    return result

# Iteratively stitch all images
mosaic = images[0]
for i in range(1, min(len(images), 10)):
    mosaic = stitch_images(mosaic, images[i])
    if mosaic is None:
        print(f"Could not stitch image {i+1}. Skipping.")
        mosaic = images[i]  # fallback to next image

# Show the final mosaic
cv2.imshow("Mosaic", mosaic)
cv2.waitKey(0)
cv2.destroyAllWindows()
