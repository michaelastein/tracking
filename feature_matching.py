import cv2
import numpy as np
import piexif
from tkinter import Tk, Canvas, Frame, Scrollbar, Label, LEFT, RIGHT, Y, NW
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilenames
import math
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

start_time = time.time()

# ----------------- Progress helper -----------------
progress_lock = threading.Lock()
def print_progress(current, total, stage_name, last_print=[-1]):
    percent = int((current / total) * 100)
    with progress_lock:
        if percent // 5 != last_print[0] or current == total:
            print(f"[{stage_name}] Progress: {percent}% ({current}/{total})")
            last_print[0] = percent // 5

# ----------------- Helper functions -----------------
def rational_to_float(r):
    if isinstance(r, (tuple, list)):
        return r[0] / r[1]
    return float(r)

def gps_to_decimal(coord, ref):
    deg = rational_to_float(coord[0])
    minute = rational_to_float(coord[1])
    sec = rational_to_float(coord[2])
    val = deg + minute / 60.0 + sec / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode(errors='ignore')
    if ref in ['S', 's', 'W', 'w']:
        val = -val
    return val

def extract_gps_from_exif(exif_dict):
    gps_ifd = exif_dict.get("GPS", {})
    lat_tag = gps_ifd.get(piexif.GPSIFD.GPSLatitude)
    lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef)
    lon_tag = gps_ifd.get(piexif.GPSIFD.GPSLongitude)
    lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef)
    alt_tag = gps_ifd.get(piexif.GPSIFD.GPSAltitude)
    alt_ref = gps_ifd.get(piexif.GPSIFD.GPSAltitudeRef, 0)
    if not (lat_tag and lat_ref and lon_tag and lon_ref and alt_tag is not None):
        raise ValueError("Missing GPS fields in EXIF.")
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

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(np.sqrt(a))

def preprocess_for_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    return gray

# ----------------- GUI File Selection -----------------
root = Tk()
root.withdraw()
img_paths = askopenfilenames(
    title="Select images (up to 1000)",
    filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
)
if len(img_paths) < 1:
    print("No images selected.")
    exit()

else:
    print(f"Selected {len(img_paths)} images.")

# ----------------- Load images and GPS -----------------
images = []
gps_positions = []
orig_sizes = []
for path in img_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: could not read {path}")
        continue
    images.append(img)
    h, w = img.shape[:2]
    orig_sizes.append((w, h))
    try:
        exif_dict = piexif.load(path)
        gps = extract_gps_from_exif(exif_dict)
        gps_positions.append(gps)
    except Exception as e:
        print(f"Warning: no GPS in {path}, {e}")
        gps_positions.append((None, None, None))
print(f"Loaded {len(images)} images with GPS info.")

# ----------------- Parameters -----------------
threshold_meters = 100.0
ratio_test = 0.7
ransac_thresh = 4.0
min_inliers = 20
dist_consistency_thresh = 40.0
max_workers = 8

# ----------------- Detector/descriptor -----------------
try:
    detector = cv2.SIFT_create()
    descriptor_type = 'float'
    print("Using SIFT.")
except Exception:
    try:
        detector = cv2.AKAZE_create()
        descriptor_type = 'binary'
        print("Using AKAZE.")
    except Exception:
        detector = cv2.ORB_create(nfeatures=3000)
        descriptor_type = 'binary'
        print("Using ORB (fallback).")

# ----------------- Compute features -----------------
kp_list = [None] * len(images)
des_list = [None] * len(images)

def compute_features_for_index(idx):
    img = images[idx]
    gray = preprocess_for_features(img)
    kp, des = detector.detectAndCompute(gray, None)
    kp_list[idx] = kp
    des_list[idx] = None if des is None else des.copy()

with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(images)))) as ex:
    futures = [ex.submit(compute_features_for_index, i) for i in range(len(images))]
    done_count = 0
    for f in as_completed(futures):
        done_count += 1
        print_progress(done_count, len(futures), "Feature extraction")

# ----------------- GPS neighbor prefiltering -----------------
neighbors = []
for i, (lat_i, lon_i, _) in enumerate(gps_positions):
    if lat_i is None: continue
    for j, (lat_j, lon_j, _) in enumerate(gps_positions):
        if i >= j or lat_j is None: continue
        if haversine(lat_i, lon_i, lat_j, lon_j) <= threshold_meters:
            neighbors.append((i, j))
print(f"Found {len(neighbors)} likely neighbor pairs based on GPS.")
if len(neighbors) == 0:
    print("No GPS neighbors found — falling back to all pairs (this may be slow).")
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            neighbors.append((i, j))
    print(f"Total pairs to attempt: {len(neighbors)}")

# ----------------- FLANN matcher -----------------
if descriptor_type == 'float':
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
else:
    index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

def ensure_flann_dtype(des):
    if des is None:
        return None
    if descriptor_type == 'float' and des.dtype != np.float32:
        return des.astype(np.float32)
    if descriptor_type != 'float' and des.dtype != np.uint8:
        return des.astype(np.uint8)
    return des

for i in range(len(des_list)):
    if des_list[i] is not None:
        des_list[i] = ensure_flann_dtype(des_list[i])

# ----------------- Matching + homography -----------------
match_cache = {}
H_dict = {}
H_inliers = {}

def match_and_filter_pairs(i, j):
    key = (i, j)
    if key in match_cache:
        return match_cache[key]

    des_i = des_list[i]
    des_j = des_list[j]
    kp_i = kp_list[i]
    kp_j = kp_list[j]
    if des_i is None or des_j is None or kp_i is None or kp_j is None:
        match_cache[key] = []
        return []

    des_i_q = ensure_flann_dtype(des_i)
    des_j_q = ensure_flann_dtype(des_j)
    try:
        knn_j_i = matcher.knnMatch(des_j_q, des_i_q, k=2)
        knn_i_j = matcher.knnMatch(des_i_q, des_j_q, k=2)
    except cv2.error:
        bf = cv2.BFMatcher(cv2.NORM_L2 if descriptor_type == 'float' else cv2.NORM_HAMMING, crossCheck=False)
        knn_j_i = bf.knnMatch(des_j_q, des_i_q, k=2)
        knn_i_j = bf.knnMatch(des_i_q, des_j_q, k=2)

    def filter_good(knn):
        good = []
        for m_n in knn:
            if len(m_n) == 2 and m_n[0].distance < ratio_test * m_n[1].distance:
                good.append(m_n[0])
        return good

    good_j_i = filter_good(knn_j_i)
    good_i_j = filter_good(knn_i_j)

    best_j_to_i = {m.queryIdx: m.trainIdx for m in good_j_i}
    best_i_to_j = {m.queryIdx: m.trainIdx for m in good_i_j}
    mutual = [(q, t) for q, t in best_j_to_i.items() if t in best_i_to_j and best_i_to_j[t] == q]

    if not mutual:
        match_cache[key] = []
        return []

    pts_j = np.array([kp_j[q].pt for q, _ in mutual])
    pts_i = np.array([kp_i[t].pt for _, t in mutual])
    vecs = pts_i - pts_j
    mean_vec = np.mean(vecs, axis=0)
    dists = np.linalg.norm(vecs - mean_vec, axis=1)
    keep_mask = dists <= dist_consistency_thresh
    filtered = [mutual[idx] for idx, k in enumerate(keep_mask) if k]
    match_cache[key] = filtered
    return filtered

def compute_homography_for_pair(pair):
    i, j = pair
    matches = match_and_filter_pairs(i, j)
    kp_i = kp_list[i]
    kp_j = kp_list[j]
    if len(matches) < min_inliers:
        return None
    src_pts = np.float32([kp_j[q].pt for q, _ in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp_i[t].pt for _, t in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None or int(np.sum(mask)) < min_inliers:
        return None
    in_src = src_pts[mask.ravel()==1].reshape(-1,2)
    in_dst = dst_pts[mask.ravel()==1].reshape(-1,2)
    return (i, j, H, in_src.copy(), in_dst.copy())

pairs_to_process = neighbors.copy()
print(f"Computing homographies for {len(pairs_to_process)} pairs (parallel)...")
done_pairs = 0
with ThreadPoolExecutor(max_workers=min(max_workers, len(pairs_to_process))) as ex:
    futures = {ex.submit(compute_homography_for_pair, p): p for p in pairs_to_process}
    for fut in as_completed(futures):
        res = fut.result()
        done_pairs += 1
        print_progress(done_pairs, len(pairs_to_process), "Homography")
        if res is None:
            continue
        i, j, H, in_src, in_dst = res
        H_dict[(j,i)] = H
        H_inliers[(j,i)] = in_dst.copy()
        try:
            H_inv = np.linalg.inv(H)
            H_dict[(i,j)] = H_inv
            H_inliers[(i,j)] = in_src.copy()
        except np.linalg.LinAlgError:
            pass

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60
print(f"Feature extraction + matching + homography setup runtime: {elapsed_minutes:.2f} minutes or {end_time - start_time:.2f} seconds for {len(img_paths)} images.")
print(f"Computed {len(H_dict)//2} good homography pairs.")

# ----------------- Graph helpers -----------------
adj = {}
for (a,b) in H_dict.keys():
    adj.setdefault(a, set()).add(b)
    adj.setdefault(b, set()).add(a)

def node_connected_component(start):
    if start not in adj: return {start}
    seen = {start}
    queue = [start]
    while queue:
        u = queue.pop(0)
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return seen

def shortest_path(u, v):
    if u==v: return [u]
    if u not in adj: return None
    from collections import deque
    q = deque([u])
    parent = {u: None}
    while q:
        cur = q.popleft()
        for nb in adj.get(cur, ()):
            if nb not in parent:
                parent[nb] = cur
                if nb == v:
                    path = [v]
                    p = v
                    while parent[p] is not None:
                        p = parent[p]
                        path.append(p)
                    path.reverse()
                    return path
                q.append(nb)
    return None

# ----------------- GUI Thumbnail Grid -----------------
root.deiconify()
root.title("Images Grid with Clickable Points")
canvas_side = Canvas(root, width=1400, height=800, bg="black")
scrollbar = Scrollbar(root, orient="vertical", command=canvas_side.yview)
canvas_side.configure(yscrollcommand=scrollbar.set)
canvas_side.pack(side=LEFT, fill="both", expand=1)
scrollbar.pack(side=RIGHT, fill=Y)
frame = Frame(canvas_side, bg="black")
canvas_side.create_window((0,0), window=frame, anchor=NW)

photo_refs_side = []
img_positions = []
markers = []
cols = 6
img_size = 200

for i, img in enumerate(images):
    h, w = img.shape[:2]
    scale = min(img_size / h, img_size / w)
    disp_w, disp_h = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (disp_w, disp_h))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_rgb)
    im_tk = ImageTk.PhotoImage(im_pil)
    photo_refs_side.append(im_tk)

    # Frame für Bild + Name
    img_frame = Frame(frame, bd=1, relief="solid", bg="black")
    img_frame.grid(row=i // cols, column=i % cols, padx=5, pady=5)

    lbl = Label(img_frame, image=im_tk)
    lbl.pack()
    img_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
    name_lbl = Label(img_frame, text=img_name, fg="white", bg="black", font=("Arial", 9))
    name_lbl.pack(fill="x")

    img_positions.append((lbl, scale, disp_w, disp_h))
    lbl.bind("<Button-1>", lambda e, idx=i: click_event(e, idx))

canvas_side.update_idletasks()
canvas_side.config(scrollregion=canvas_side.bbox("all"))

# ----------------- Click Event -----------------
def click_event(event, idx):
    global markers
    for m in markers:
        try: m.destroy()
        except: pass
    markers.clear()
    lbl, scale, disp_w, disp_h = img_positions[idx]
    x_click = event.x / scale
    y_click = event.y / scale
    pt = np.array([x_click, y_click, 1.0]).reshape(3,1)

    comp = node_connected_component(idx)
    m0 = Label(lbl.master, bg="red")
    m0.place(in_=lbl, x=int(x_click*scale), y=int(y_click*scale), anchor=NW, width=6, height=6)
    markers.append(m0)

    for other in comp:
        if other == idx: continue
        path = shortest_path(idx, other)
        if path is None: continue
        cur_pt = pt.copy()
        ok = True
        for k in range(len(path)-1):
            a = path[k]
            b = path[k+1]
            H = H_dict.get((a,b))
            if H is None:
                ok = False
                break
            try:
                cur_pt = H @ cur_pt
                if abs(cur_pt[2,0]) < 1e-8:
                    ok = False
                    break
                cur_pt = cur_pt / cur_pt[2,0]
            except Exception:
                ok = False
                break
        if not ok: continue
        lbl_c, scale_c, disp_w_c, disp_h_c = img_positions[other]
        x_disp, y_disp = cur_pt[0,0]*scale_c, cur_pt[1,0]*scale_c
        if 0 <= x_disp <= disp_w_c and 0 <= y_disp <= disp_h_c:
            m = Label(lbl_c.master, bg="yellow")
            m.place(in_=lbl_c, x=int(x_disp), y=int(y_disp), anchor=NW, width=6, height=6)
            markers.append(m)

root.mainloop()
