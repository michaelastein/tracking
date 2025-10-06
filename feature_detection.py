import cv2
import numpy as np
from tkinter import Tk, Canvas, Scrollbar, Frame, Label, BOTH, RIGHT, LEFT, Y, NW, Toplevel
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilenames

# --- Single Tk root ---
root = Tk()
root.withdraw()

# --- Select images ---
img_paths = askopenfilenames(
    title="Select up to 10 consecutive images",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
)
img_paths = list(img_paths)
if len(img_paths) < 2:
    print("Please select at least two images.")
    exit()

images = [cv2.imread(p) for p in img_paths]
num_images = len(images)

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
    # Top 50 matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_img = cv2.resize(match_img, (int(match_img.shape[1]*0.8), int(match_img.shape[0]*0.8)))
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

# --- Compute global FOV ---
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

colors = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 255, 128), (255, 128, 128), (128, 128, 255), (255, 255, 255)
]

canvas_fov = np.zeros((height, width, 3), dtype=np.uint8)

for i, (img, H) in enumerate(zip(images, homographies)):
    h, w = img.shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, H) + shift

    trans = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float32)
    H_shifted = trans @ H
    warped_img = cv2.warpPerspective(img, H_shifted, (width, height))
    mask = np.any(warped_img > 0, axis=2)
    canvas_fov[mask] = warped_img[mask]

    color = colors[i % len(colors)]
    cv2.polylines(canvas_fov, [np.int32(warped_corners)], True, color, 2)
    cv2.putText(canvas_fov, f"Image {i+1}", tuple(np.int32(warped_corners[0,0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# --- Global FOV window ---
fov_window = Toplevel(root)
fov_window.title("Global FOV Visualization")
fov_canvas = Canvas(fov_window, width=1000, height=800, bg="black")
fov_canvas.pack(fill=BOTH, expand=1)

scale_factor = 0.8
canvas_fov_small = cv2.resize(canvas_fov, (int(width*scale_factor), int(height*scale_factor)))
im_pil = Image.fromarray(cv2.cvtColor(canvas_fov_small, cv2.COLOR_BGR2RGB))
im_tk_fov = ImageTk.PhotoImage(im_pil)
fov_img_obj = fov_canvas.create_image(0,0, anchor=NW, image=im_tk_fov)

fov_scale = 1.0
fov_canvas.image = im_tk_fov
fov_canvas.orig_image = im_pil

def fov_zoom(event):
    global fov_scale
    factor = 1.1 if event.delta>0 else 0.9
    fov_scale *= factor
    new_size = (int(fov_canvas.orig_image.width*fov_scale),
                int(fov_canvas.orig_image.height*fov_scale))
    resized = fov_canvas.orig_image.resize(new_size)
    fov_canvas.image = ImageTk.PhotoImage(resized)
    fov_canvas.itemconfig(fov_img_obj, image=fov_canvas.image)
fov_canvas.bind("<MouseWheel>", fov_zoom)

pan_data = {"x":0,"y":0}
def start_pan(event):
    pan_data["x"] = event.x
    pan_data["y"] = event.y
def do_pan(event):
    dx = event.x - pan_data["x"]
    dy = event.y - pan_data["y"]
    fov_canvas.move(fov_img_obj, dx, dy)
    pan_data["x"] = event.x
    pan_data["y"] = event.y
fov_canvas.bind("<ButtonPress-1>", start_pan)
fov_canvas.bind("<B1-Motion>", do_pan)

# --- Scrollable ORB matches window ---
matches_window = Toplevel(root)
matches_window.title("ORB Feature Matches Scrollable View")
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
    im_tk = ImageTk.PhotoImage(im_pil.resize(new_size))
    photo_refs.append(im_tk)
    label = Label(scrollable_frame, image=im_tk)
    label.pack(pady=10)

def matches_mousewheel(event):
    global matches_zoom_scale
    if not scrollable_frame.winfo_exists():
        return
    if (event.state & 0x4):  # Ctrl pressed
        factor = 1.1 if event.delta>0 else 0.9
        matches_zoom_scale *= factor
        for i, (img, label) in enumerate(zip(match_imgs, scrollable_frame.winfo_children())):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img_rgb)
            new_size = (int(im_pil.width*matches_zoom_scale), int(im_pil.height*matches_zoom_scale))
            im_tk = ImageTk.PhotoImage(im_pil.resize(new_size))
            photo_refs[i] = im_tk
            label.configure(image=im_tk)
    else:
        canvas_scroll.yview_scroll(int(-1*(event.delta/120)), "units")
matches_window.bind("<MouseWheel>", matches_mousewheel)

# --- Third window: 2 rows of 5 images ---
side_window = Toplevel(root)
side_window.title("All Images 2x5 Grid")
canvas_side = Canvas(side_window, width=1500, height=800, bg="black")
canvas_side.pack(fill=BOTH, expand=1)

photo_refs_side = []
img_positions = []
img_objects = []
rows, cols = 2, 5
margin = 10
x_offset, y_offset = margin, margin
max_height_row = 0
for i, img in enumerate(images):
    h, w = img.shape[:2]
    scale = min(200/h, 200/w)
    img_resized = cv2.resize(img, (int(w*scale), int(h*scale)))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_rgb)
    im_tk = ImageTk.PhotoImage(im_pil)
    photo_refs_side.append(im_tk)
    img_obj = canvas_side.create_image(x_offset, y_offset, anchor=NW, image=im_tk)
    img_objects.append(img_obj)
    img_positions.append([x_offset, y_offset, w*scale, h*scale, scale])
    x_offset += int(w*scale) + margin
    max_height_row = max(max_height_row, int(h*scale))
    if (i+1)%cols == 0:
        x_offset = margin
        y_offset += max_height_row + margin
        max_height_row = 0

side_scale = 1.0
markers = []
point_radius = 5

def side_zoom(event):
    global side_scale
    if not (event.state & 0x4):
        return
    factor = 1.1 if event.delta>0 else 0.9
    side_scale *= factor
    canvas_side.delete("all")
    for i, img in enumerate(images):
        x0, y0, w_orig, h_orig, scale_orig = img_positions[i]
        new_w = int(images[i].shape[1]*scale_orig*side_scale)
        new_h = int(images[i].shape[0]*scale_orig*side_scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        im_tk = ImageTk.PhotoImage(im_pil)
        photo_refs_side[i] = im_tk
        new_x = int(x0*side_scale)
        new_y = int(y0*side_scale)
        img_obj = canvas_side.create_image(new_x, new_y, anchor=NW, image=im_tk)
        img_objects[i] = img_obj
        img_positions[i][0] = new_x
        img_positions[i][1] = new_y
        img_positions[i][2] = new_w
        img_positions[i][3] = new_h
    for m in markers:
        canvas_side.delete(m)
    markers.clear()

canvas_side.bind("<MouseWheel>", side_zoom)

# Click to visualize points
def click_event(event):
    for m in markers: canvas_side.delete(m)
    markers.clear()
    for idx, (x0, y0, w, h, _) in enumerate(img_positions):
        if x0 <= event.x <= x0+w and y0 <= event.y <= y0+h:
            px = (event.x - x0)/(w/images[idx].shape[1])
            py = (event.y - y0)/(h/images[idx].shape[0])
            click_point = np.array([px, py, 1]).reshape(3,1)
            for j, (xj0, yj0, wj, hj, _) in enumerate(img_positions):
                H = np.linalg.inv(homographies[j]) @ homographies[idx]
                pt = H @ click_point
                pt /= pt[2,0]
                x_img = int(pt[0,0]*(wj/images[j].shape[1]) + xj0)
                y_img = int(pt[1,0]*(hj/images[j].shape[0]) + yj0)
                if 0 <= x_img-xj0 < wj and 0 <= y_img-yj0 < hj:
                    m = canvas_side.create_oval(x_img-point_radius, y_img-point_radius,
                                                x_img+point_radius, y_img+point_radius,
                                                outline="red", width=2)
                    markers.append(m)
            break

canvas_side.bind("<Button-1>", click_event)

# --- Close root when all windows closed ---
def check_close():
    if not (fov_window.winfo_exists() or matches_window.winfo_exists() or side_window.winfo_exists()):
        root.destroy()
fov_window.protocol("WM_DELETE_WINDOW", lambda: (fov_window.destroy(), check_close()))
matches_window.protocol("WM_DELETE_WINDOW", lambda: (matches_window.destroy(), check_close()))
side_window.protocol("WM_DELETE_WINDOW", lambda: (side_window.destroy(), check_close()))

root.deiconify()
root.mainloop()
