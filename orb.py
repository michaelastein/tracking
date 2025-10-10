import cv2
import numpy as np
from tkinter import Tk, Canvas, Scrollbar, Frame, Label, BOTH, RIGHT, LEFT, Y, NW, Toplevel
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilenames

# --- Tk root ---
root = Tk()
root.withdraw()

# Select exactly two images
img_paths = askopenfilenames(
    title="Select two images",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
)
if len(img_paths) != 2:
    print("Please select exactly two images.")
    exit()

images = [cv2.imread(p) for p in img_paths]

# --- ORB + BFMatcher ---
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

homographies = [np.eye(3)]
match_imgs = []

for i in range(1, len(images)):
    img1 = images[i-1]
    img2 = images[i]

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw top 50 matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_imgs.append(match_img)

    if len(matches) < 10:
        homographies.append(homographies[-1])
        continue

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        H = np.eye(3)
    homographies.append(homographies[-1] @ H)

# --- Compute global canvas size ---
all_corners = []
for img, H in zip(images, homographies):
    h, w = img.shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(corners, H)
    all_corners.append(warped)
all_corners = np.vstack(all_corners)
[xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 50)
[xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 50)
width = xmax - xmin
height = ymax - ymin
shift = np.array([[[-xmin, -ymin]]], dtype=np.float32)

colors = [(0,255,0),(0,0,255)]  # Green, Red

canvas_fov = np.zeros((height, width, 3), dtype=np.uint8)

for i, (img, H) in enumerate(zip(images, homographies)):
    h, w = img.shape[:2]
    trans = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float32)
    H_shifted = trans @ H
    warped_img = cv2.warpPerspective(img, H_shifted, (width, height))
    mask = np.any(warped_img > 0, axis=2)
    canvas_fov[mask] = warped_img[mask]

    # Draw outlines
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, H) + shift
    cv2.polylines(canvas_fov, [np.int32(warped_corners)], True, colors[i % len(colors)], 2)
    cv2.putText(canvas_fov, f"Image {i+1}", tuple(np.int32(warped_corners[0,0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % len(colors)], 2)

# --- Tkinter canvas window ---
fov_window = Toplevel(root)
fov_window.title("Global FOV Visualization")
fov_canvas = Canvas(fov_window, width=1000, height=800, bg="black")
fov_canvas.pack(fill=BOTH, expand=1)

# Scale down initially
scale_factor = 0.8
canvas_fov_small = cv2.resize(canvas_fov, (int(width*scale_factor), int(height*scale_factor)))
img_rgb = cv2.cvtColor(canvas_fov_small, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img_rgb)
im_tk_fov = ImageTk.PhotoImage(im_pil)
fov_img_obj = fov_canvas.create_image(0, 0, anchor=NW, image=im_tk_fov)
fov_canvas.image = im_tk_fov
fov_canvas.orig_image = im_pil

fov_scale = 1.0
def fov_zoom(event):
    global fov_scale
    factor = 1.1 if event.delta > 0 else 0.9
    fov_scale *= factor
    new_size = (int(fov_canvas.orig_image.width * fov_scale),
                int(fov_canvas.orig_image.height * fov_scale))
    resized = fov_canvas.orig_image.resize(new_size)
    fov_canvas.image = ImageTk.PhotoImage(resized)
    fov_canvas.itemconfig(fov_img_obj, image=fov_canvas.image)

fov_canvas.bind("<MouseWheel>", fov_zoom)

# --- Scrollable ORB matches ---
matches_window = Toplevel(root)
matches_window.title("ORB Feature Matches")
matches_window.geometry("1200x900")
frame = Frame(matches_window)
frame.pack(fill=BOTH, expand=1)
canvas_scroll = Canvas(frame)
scrollbar = Scrollbar(frame, orient="vertical", command=canvas_scroll.yview)
scrollable_frame = Frame(canvas_scroll)
scrollable_frame.bind("<Configure>", lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
canvas_scroll.create_window((0,0), window=scrollable_frame, anchor=NW)
canvas_scroll.configure(yscrollcommand=scrollbar.set)
canvas_scroll.pack(side=LEFT, fill=BOTH, expand=1)
scrollbar.pack(side=RIGHT, fill=Y)

photo_refs = []
matches_zoom_scale = 0.8
for img in match_imgs:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_rgb)
    new_size = (int(im_pil.width*matches_zoom_scale), int(im_pil.height*matches_zoom_scale))
    resized = im_pil.resize(new_size)
    im_tk = ImageTk.PhotoImage(resized)
    photo_refs.append(im_tk)
    label = Label(scrollable_frame, image=im_tk)
    label.pack(pady=10)

root.deiconify()
root.mainloop()
