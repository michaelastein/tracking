import cv2
import numpy as np
from tkinter import Tk, Canvas, Scrollbar, Frame, Label, BOTH, RIGHT, LEFT, Y, NW, Toplevel
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilenames
import piexif
import time

# --- Helpers ---
def rational_to_float(r):
    if isinstance(r, tuple):
        return r[0]/r[1]
    return float(r)

def gps_to_decimal(coord, ref):
    deg = rational_to_float(coord[0])
    min_ = rational_to_float(coord[1])
    sec = rational_to_float(coord[2])
    decimal = deg + min_/60 + sec/3600
    if ref in ['S','W']:
        decimal = -decimal
    return decimal

def extract_gps_from_exif(exif_dict):
    gps_ifd = exif_dict.get("GPS", {})
    lat_tag = gps_ifd.get(piexif.GPSIFD.GPSLatitude)
    lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef)
    lon_tag = gps_ifd.get(piexif.GPSIFD.GPSLongitude)
    lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef)
    alt_tag = gps_ifd.get(piexif.GPSIFD.GPSAltitude)
    alt_ref = gps_ifd.get(piexif.GPSIFD.GPSAltitudeRef, 0)
    lat = gps_to_decimal(lat_tag, lat_ref)
    lon = gps_to_decimal(lon_tag, lon_ref)
    alt = rational_to_float(alt_tag)
    if isinstance(alt_ref, (bytes, bytearray)):
        alt_ref_val = int(alt_ref[0])
    else:
        alt_ref_val = int(alt_ref)
    if alt_ref_val == 1:
        alt = -alt
    return lat, lon, alt

# --- Main ---
start_time = time.time()
root = Tk()
root.withdraw()

print("Select all images...")
img_paths = askopenfilenames(title="Select images")
if not img_paths:
    exit()

images = [cv2.imread(p) for p in img_paths]
num_images = len(images)
print(f"Number of images: {num_images}")

# --- Extract GPS ---
positions = []
for path in img_paths:
    exif_dict = piexif.load(path)
    lat, lon, alt = extract_gps_from_exif(exif_dict)
    positions.append([lat, lon, alt])
positions = np.array(positions)

# --- Find nearest neighbors (up to 15) ---
MAX_NEIGHBORS = 15
neighbors = []
for i in range(num_images):
    distances = np.linalg.norm(positions - positions[i], axis=1)
    idxs = np.argsort(distances)[1:MAX_NEIGHBORS+1]
    neighbors.append(list(idxs))

# --- ORB features ---
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- Compute homographies for neighbor pairs ---
homographies = {}
print("Computing homographies between neighbor images...")
for i, nbrs in enumerate(neighbors):
    img1 = images[i]
    kp1, des1 = orb.detectAndCompute(img1, None)
    for j in nbrs:
        if (i,j) in homographies or (j,i) in homographies:
            continue
        img2 = images[j]
        kp2, des2 = orb.detectAndCompute(img2, None)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) >= 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                homographies[(i,j)] = H
        print(f"Homography between {i} ↔ {j} computed, {len(matches)} matches")

# --- Display images in scrollable 6-column grid ---
window = Toplevel(root)
window.title("Images with Neighbor Matches")
frame = Frame(window)
frame.pack(fill=BOTH, expand=1)
canvas_scroll = Canvas(frame)
scrollbar = Scrollbar(frame, orient="vertical", command=canvas_scroll.yview)
scrollable_frame = Frame(canvas_scroll)
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
)
canvas_scroll.create_window((0,0), window=scrollable_frame, anchor=NW)
canvas_scroll.configure(yscrollcommand=scrollbar.set)
canvas_scroll.pack(side=LEFT, fill=BOTH, expand=1)
scrollbar.pack(side=RIGHT, fill=Y)

photo_refs = []
image_canvases = []

cols = 6
margin = 10
thumb_w, thumb_h = 200, 150

for idx, img in enumerate(images):
    h, w = img.shape[:2]
    scale = min(thumb_w/w, thumb_h/h)
    img_resized = cv2.resize(img, (int(w*scale), int(h*scale)))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_rgb)
    im_tk = ImageTk.PhotoImage(im_pil)
    photo_refs.append(im_tk)

    row = idx//cols
    col = idx%cols
    x = margin + col*(thumb_w+margin)
    y = margin + row*(thumb_h+margin)

    img_canvas = Canvas(scrollable_frame, width=thumb_w, height=thumb_h, bg="black")
    img_canvas.grid(row=row, column=col, padx=margin, pady=margin)
    img_canvas.create_image(0,0, anchor=NW, image=im_tk)
    image_canvases.append((img_canvas, scale))
    
# --- Click and mark points ---
markers = []

def click_event(event, idx):
    global markers
    for m in markers:
        for c, _ in image_canvases:
            c.delete(m)
    markers.clear()
    
    img_canvas, scale = image_canvases[idx]
    x = event.x/scale
    y = event.y/scale
    print(f"Clicked image {idx} at ({x:.1f},{y:.1f})")
    
    for jdx in range(num_images):
        H = homographies.get((idx,jdx))
        if H is None:
            H = homographies.get((jdx,idx))
            if H is not None:
                H = np.linalg.inv(H)
        if H is None:
            continue
        pt = np.array([x, y, 1]).reshape(3,1)
        pt_trans = H @ pt
        pt_trans /= pt_trans[2,0]
        xc, yc = pt_trans[0,0]*image_canvases[jdx][1], pt_trans[1,0]*image_canvases[jdx][1]
        m = image_canvases[jdx][0].create_oval(xc-5, yc-5, xc+5, yc+5, outline="red", width=2)
        markers.append(m)
    
    # Also mark the clicked point on the original image
    m = img_canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, outline="red", width=2)
    markers.append(m)

for idx, (img_canvas, _) in enumerate(image_canvases):
    img_canvas.bind("<Button-1>", lambda e, i=idx: click_event(e, i))

# --- Close ---
def on_close():
    root.destroy()
window.protocol("WM_DELETE_WINDOW", on_close)

root.deiconify()
root.mainloop()
print(f"Total computation time: {time.time() - start_time:.2f} seconds")
