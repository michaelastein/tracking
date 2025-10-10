import cv2
import numpy as np
import piexif
from tkinter import Tk, Canvas, Frame, Scrollbar, Label, LEFT, RIGHT, Y, NW
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilenames
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

start_time = time.time()

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
    return 2 * R * math.asin(math.sqrt(a))

def convex_hull_2d(points):
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return np.array([])
    pts_unique = np.unique(np.round(pts, 6), axis=0)
    if pts_unique.shape[0] < 3:
        return np.array([])
    pts_sorted = pts_unique[np.lexsort((pts_unique[:,1], pts_unique[:,0]))]
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts_sorted:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper=[]
    for p in reversed(pts_sorted):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1])
    if hull.shape[0] < 3:
        return np.array([])
    return hull

def point_in_convex_polygon(point, polygon):
    if polygon is None or polygon.size == 0:
        return False
    x, y = float(point[0]), float(point[1])
    prev_sign = None
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i+1)%len(polygon)]
        cp = (b[0]-a[0])*(y-a[1]) - (b[1]-a[1])*(x-a[0])
        if abs(cp) < 1e-9:
            continue
        sign = cp > 0
        if prev_sign is None:
            prev_sign = sign
        elif prev_sign != sign:
            return False
    return True

def point_in_hull(point, hull_points):
    pts = np.asarray(hull_points, dtype=np.float64)
    if pts.shape[0] < 3:
        return False
    pts_unique = np.unique(np.round(pts, 6), axis=0)
    if pts_unique.shape[0] < 3:
        return False
    hull = convex_hull_2d(pts_unique)
    if hull.size == 0:
        min_xy = np.min(pts_unique, axis=0)
        max_xy = np.max(pts_unique, axis=0)
        x, y = float(point[0]), float(point[1])
        eps = 1e-6
        return (min_xy[0]-eps <= x <= max_xy[0]+eps) and (min_xy[1]-eps <= y <= max_xy[1]+eps)
    return point_in_convex_polygon(point, hull)

# ----------------- Image preprocessing -----------------
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
if len(img_paths) > 1000:
    print(f"Selected {len(img_paths)} images, using first 1000.")
    img_paths = img_paths[:1000]
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
threshold_meters = 100.0     # GPS neighbor prefilter
ratio_test = 0.7             # Lowe's ratio test (lower -> stricter)
ransac_thresh = 4.0          # RANSAC reprojection threshold (px)
min_inliers = 20             # minimum inliers to accept homography
dist_consistency_thresh = 40.0  # spatial consistency filter (px)
max_workers = 8              # parallel workers

# ----------------- Detector/descriptor -----------------
detector = None
descriptor_type = None
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
    print(f"Features for image {idx}: {0 if kp is None else len(kp)} keypoints")

with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(images)))) as ex:
    futures = [ex.submit(compute_features_for_index, i) for i in range(len(images))]
    for f in as_completed(futures):
        pass

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

# ----------------- Matching utilities -----------------
match_cache = {}  # key: (i,j) ordered — returns list of (qIdx_in_j, tIdx_in_i)

def match_and_filter_pairs(i, j):
    """
    Return list of tuples (qidx_in_j, tidx_in_i) mapping descriptors from j -> i
    """
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

    good_j_i = []
    for m_n in knn_j_i:
        if len(m_n) != 2: continue
        m, n = m_n
        if m.distance < ratio_test * n.distance:
            good_j_i.append(m)

    good_i_j = []
    for m_n in knn_i_j:
        if len(m_n) != 2: continue
        m, n = m_n
        if m.distance < ratio_test * n.distance:
            good_i_j.append(m)

    best_j_to_i = {m.queryIdx: m.trainIdx for m in good_j_i}
    best_i_to_j = {m.queryIdx: m.trainIdx for m in good_i_j}

    mutual = []
    for qidx_j, tidx_i in best_j_to_i.items():
        if tidx_i in best_i_to_j and best_i_to_j[tidx_i] == qidx_j:
            mutual.append((qidx_j, tidx_i))

    if len(mutual) == 0:
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

# ----------------- Rotation-aware homography computation -----------------
H_dict = {}
H_inliers = {}

def compute_homography_for_pair(pair):
    i, j = pair
    # attempt normal matching first
    matches = match_and_filter_pairs(i, j)
    kp_i = kp_list[i]
    kp_j = kp_list[j]

    # helper to build H from matches given kp_j (source), kp_i (dest)
    def build_h_from_pairs(pairs, kp_src, kp_dst):
        if len(pairs) < 8:
            return None
        src_pts = np.float32([kp_src[q].pt for q, _ in pairs]).reshape(-1,1,2)
        dst_pts = np.float32([kp_dst[t].pt for _, t in pairs]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if H is None or mask is None:
            return None
        if int(np.sum(mask)) < min_inliers:
            return None
        in_src = src_pts[mask.ravel()==1].reshape(-1,2)
        in_dst = dst_pts[mask.ravel()==1].reshape(-1,2)
        return (H, in_src.copy(), in_dst.copy())

    res = build_h_from_pairs(matches, kp_j, kp_i)
    if res is not None:
        H, in_src, in_dst = res
        return (i, j, H, in_src, in_dst)

    # If normal matching failed, try rotated 180° on image j (source rotated)
    # rotate image j by 180 and compute features/descriptors on the rotated image
    try:
        img_j_rot = cv2.rotate(images[j], cv2.ROTATE_180)
        gray_j_rot = preprocess_for_features(img_j_rot)
        kp_j_rot, des_j_rot = detector.detectAndCompute(gray_j_rot, None)
        if des_j_rot is None or des_list[i] is None or kp_j_rot is None:
            return None
        des_i_q = ensure_flann_dtype(des_list[i])
        des_j_rot_q = ensure_flann_dtype(des_j_rot)

        # knn both directions between j_rot and i
        try:
            knn_jrot_i = matcher.knnMatch(des_j_rot_q, des_i_q, k=2)
            knn_i_jrot = matcher.knnMatch(des_i_q, des_j_rot_q, k=2)
        except cv2.error:
            bf = cv2.BFMatcher(cv2.NORM_L2 if descriptor_type == 'float' else cv2.NORM_HAMMING, crossCheck=False)
            knn_jrot_i = bf.knnMatch(des_j_rot_q, des_i_q, k=2)
            knn_i_jrot = bf.knnMatch(des_i_q, des_j_rot_q, k=2)

        good_jrot_i = []
        for m_n in knn_jrot_i:
            if len(m_n) != 2: continue
            m, n = m_n
            if m.distance < ratio_test * n.distance:
                good_jrot_i.append(m)
        good_i_jrot = []
        for m_n in knn_i_jrot:
            if len(m_n) != 2: continue
            m, n = m_n
            if m.distance < ratio_test * n.distance:
                good_i_jrot.append(m)

        best_jrot_to_i = {m.queryIdx: m.trainIdx for m in good_jrot_i}
        best_i_to_jrot = {m.queryIdx: m.trainIdx for m in good_i_jrot}

        mutual_rot = []
        for qj, ti in best_jrot_to_i.items():
            if ti in best_i_to_jrot and best_i_to_jrot[ti] == qj:
                mutual_rot.append((qj, ti))

        if len(mutual_rot) == 0:
            return None

        # spatial consistency
        pts_jrot = np.array([kp_j_rot[q].pt for q, _ in mutual_rot])
        pts_i = np.array([kp_i[t].pt for _, t in mutual_rot])
        vecs = pts_i - pts_jrot
        mean_vec = np.mean(vecs, axis=0)
        dists = np.linalg.norm(vecs - mean_vec, axis=1)
        keep_mask = dists <= dist_consistency_thresh
        filtered_rot = [mutual_rot[idx] for idx, k in enumerate(keep_mask) if k]

        res_rot = build_h_from_pairs(filtered_rot, kp_j_rot, kp_i)
        if res_rot is None:
            return None

        H_rot, in_src_rot, in_dst_rot = res_rot
        # transform to account for 180° rotation of j: original_j -> rotated coords matrix
        w_rot, h_rot = img_j_rot.shape[1], img_j_rot.shape[0]
        rot_180 = np.array([[-1, 0, w_rot],
                            [0, -1, h_rot],
                            [0,  0,    1]], dtype=np.float64)
        H_total = H_rot @ rot_180  # maps original j coords -> i coords
        return (i, j, H_total, in_src_rot, in_dst_rot)
    except Exception as e:
        # rotated attempt failed
        return None

# compute homographies in parallel
pairs_to_process = neighbors.copy()
print(f"Computing rotation-aware homographies for {len(pairs_to_process)} pairs (parallel)...")
with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(pairs_to_process)))) as ex:
    futures = {ex.submit(compute_homography_for_pair, p): p for p in pairs_to_process}
    for fut in as_completed(futures):
        res = fut.result()
        if res is None:
            continue
        i, j, H, in_src, in_dst = res
        # store j->i mapping as before (H maps coordinates in j to coordinates in i)
        H_dict[(j, i)] = H
        H_inliers[(j, i)] = in_dst.copy()
        try:
            H_inv = np.linalg.inv(H)
            H_dict[(i, j)] = H_inv
            H_inliers[(i, j)] = in_src.copy()
        except np.linalg.LinAlgError:
            pass

end_time = time.time()
print(f"Feature extraction + matching + homography setup runtime: {end_time - start_time:.2f} seconds")
print(f"Computed {len(H_dict)//2} good homography pairs (bidirectional stored when invertible).")

# ----------------- Build adjacency & graph helpers (no external lib) -----------------
adj = {}
for (a,b) in H_dict.keys():
    adj.setdefault(a, set()).add(b)
    adj.setdefault(b, set()).add(a)

def node_connected_component(start):
    # BFS
    if start not in adj:
        return {start}
    seen = set([start])
    queue = [start]
    while queue:
        u = queue.pop(0)
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return seen

def shortest_path(u, v):
    if u == v:
        return [u]
    if u not in adj:
        return None
    from collections import deque
    q = deque()
    q.append(u)
    parent = {u: None}
    while q:
        cur = q.popleft()
        for nb in adj.get(cur, ()):
            if nb not in parent:
                parent[nb] = cur
                if nb == v:
                    # build path
                    path = [v]
                    p = v
                    while parent[p] is not None:
                        p = parent[p]
                        path.append(p)
                    path.reverse()
                    return path
                q.append(nb)
    return None

print(f"Graph built with {len(adj)} nodes and approx. {sum(len(v) for v in adj.values())//2} edges.")
# ----------------- GUI Thumbnail Grid -----------------
root.deiconify()
root.title("Images Grid with Clickable Points")
canvas_side = Canvas(root, width=1400, height=800, bg="black")
scrollbar = Scrollbar(root, orient="vertical", command=canvas_side.yview)
canvas_side.configure(yscrollcommand=scrollbar.set)
canvas_side.pack(side=LEFT, fill="both", expand=1)
scrollbar.pack(side=RIGHT, fill=Y)
frame = Frame(canvas_side)
canvas_side.create_window((0,0), window=frame, anchor=NW)

photo_refs_side = []
img_positions = []
markers = []
cols = 6
img_size = 200

for i, img in enumerate(images):
    h, w = img.shape[:2]
    scale = min(img_size / h, img_size / w)
    disp_w, disp_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (disp_w, disp_h))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_rgb)
    im_tk = ImageTk.PhotoImage(im_pil)
    photo_refs_side.append(im_tk)
    lbl = Label(frame, image=im_tk, bd=2, relief="solid")
    lbl.grid(row=i // cols, column=i % cols, padx=5, pady=5)
    img_positions.append((lbl, scale, disp_w, disp_h))

# ----------------- Click Event (multi-hop propagation using shortest paths) -----------------
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

    # connected component images
    comp = node_connected_component(idx)
    # Always show on original
    m0 = Label(frame, bg="red")
    m0.place(in_=lbl, x=int(x_click*scale), y=int(y_click*scale), anchor=NW, width=6, height=6)
    markers.append(m0)

    for other in comp:
        if other == idx:
            continue
        path = shortest_path(idx, other)
        if path is None:
            continue
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
        if not ok:
            continue
        lbl_c, scale_c, disp_w_c, disp_h_c = img_positions[other]
        x_disp, y_disp = cur_pt[0,0]*scale_c, cur_pt[1,0]*scale_c
        if 0 <= x_disp <= disp_w_c and 0 <= y_disp <= disp_h_c:
            m = Label(frame, bg="yellow")
            m.place(in_=lbl_c, x=int(x_disp), y=int(y_disp), anchor=NW, width=6, height=6)
            markers.append(m)

for i, (lbl, _, _, _) in enumerate(img_positions):
    lbl.bind("<Button-1>", lambda e, idx=i: click_event(e, idx))

canvas_side.update_idletasks()
canvas_side.config(scrollregion=canvas_side.bbox("all"))
root.mainloop()
