import cv2
import numpy as np
import piexif
from tkinter import Tk, Canvas, Frame, Scrollbar, Label, LEFT, RIGHT, Y, NW
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilenames
import math
import time
import threading
from queue import Queue

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
    if ref in ['S','s','W','w']:
        val = -val
    return val

def extract_gps_from_exif(exif_dict):
    gps_ifd = exif_dict.get("GPS",{})
    lat_tag = gps_ifd.get(piexif.GPSIFD.GPSLatitude)
    lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef)
    lon_tag = gps_ifd.get(piexif.GPSIFD.GPSLongitude)
    lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef)
    alt_tag = gps_ifd.get(piexif.GPSIFD.GPSAltitude)
    alt_ref = gps_ifd.get(piexif.GPSIFD.GPSAltitudeRef,0)
    if not (lat_tag and lat_ref and lon_tag and lon_ref and alt_tag is not None):
        raise ValueError("Missing GPS fields in EXIF.")
    lat = gps_to_decimal(lat_tag, lat_ref)
    lon = gps_to_decimal(lon_tag, lon_ref)
    alt = rational_to_float(alt_tag)
    if isinstance(alt_ref,(bytes,bytearray)):
        alt_ref_val = int(alt_ref[0])
    else:
        alt_ref_val = int(alt_ref)
    if alt_ref_val==1:
        alt=-alt
    return lat, lon, alt

def haversine(lat1, lon1, lat2, lon2):
    R=6371000
    phi1, phi2=math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a=math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

def convex_hull_2d(points):
    pts = np.asarray(points,dtype=np.float64)
    if pts.shape[0]<3:
        return np.array([])
    pts_unique = np.unique(np.round(pts,6),axis=0)
    if pts_unique.shape[0]<3:
        return np.array([])
    pts_sorted = pts_unique[np.lexsort((pts_unique[:,1], pts_unique[:,0]))]
    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts_sorted:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:
            lower.pop()
        lower.append(tuple(p))
    upper=[]
    for p in reversed(pts_sorted):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:
            upper.pop()
        upper.append(tuple(p))
    hull=np.array(lower[:-1]+upper[:-1])
    if hull.shape[0]<3:
        return np.array([])
    return hull

def point_in_convex_polygon(point, polygon):
    if polygon is None or polygon.size==0:
        return False
    x,y = float(point[0]),float(point[1])
    prev_sign=None
    for i in range(len(polygon)):
        a=polygon[i]
        b=polygon[(i+1)%len(polygon)]
        cp=(b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])
        if abs(cp)<1e-9:
            continue
        sign = cp>0
        if prev_sign is None:
            prev_sign=sign
        elif prev_sign!=sign:
            return False
    return True

def point_in_hull(point,hull_points):
    pts=np.asarray(hull_points,dtype=np.float64)
    if pts.shape[0]<3:
        return False
    pts_unique=np.unique(np.round(pts,6),axis=0)
    if pts_unique.shape[0]<3:
        return False
    hull=convex_hull_2d(pts_unique)
    if hull.size==0:
        min_xy = np.min(pts_unique,axis=0)
        max_xy = np.max(pts_unique,axis=0)
        x,y=float(point[0]),float(point[1])
        eps=1e-6
        return (min_xy[0]-eps<=x<=max_xy[0]+eps) and (min_xy[1]-eps<=y<=max_xy[1]+eps)
    return point_in_convex_polygon(point,hull)

# ----------------- GUI File Selection -----------------
root=Tk()
root.withdraw()
img_paths=askopenfilenames(title="Select images (up to 1000)",
                            filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
if len(img_paths)<1:
    print("No images selected.")
    exit()
if len(img_paths)>1000:
    print(f"Selected {len(img_paths)} images, using first 1000.")
    img_paths=img_paths[:1000]
else:
    print(f"Selected {len(img_paths)} images.")

# ----------------- Load images and GPS -----------------
images=[]
gps_positions=[]
orig_sizes=[]
for path in img_paths:
    img=cv2.imread(path)
    if img is None:
        print(f"Warning: could not read {path}")
        continue
    images.append(img)
    h,w=img.shape[:2]
    orig_sizes.append((w,h))
    try:
        exif_dict=piexif.load(path)
        gps=extract_gps_from_exif(exif_dict)
        gps_positions.append(gps)
    except Exception as e:
        print(f"Warning: no GPS in {path}, {e}")
        gps_positions.append((None,None,None))
print(f"Loaded {len(images)} images with GPS info.")

# ----------------- GPS prefiltering -----------------
threshold_meters=100.0
neighbors=[]
for i,(lat_i,lon_i,_) in enumerate(gps_positions):
    if lat_i is None: continue
    for j,(lat_j,lon_j,_) in enumerate(gps_positions):
        if i>=j or lat_j is None: continue
        if haversine(lat_i,lon_i,lat_j,lon_j)<=threshold_meters:
            neighbors.append((i,j))
print(f"Found {len(neighbors)} likely neighbor pairs based on GPS.")

# ----------------- Lazy ORB with Multithreading -----------------
orb=cv2.ORB_create(5000)
kp_list=[None]*len(images)
des_list=[None]*len(images)
needed_idxs=set()
for i,j in neighbors:
    needed_idxs.add(i)
    needed_idxs.add(j)

def orb_worker(q):
    while True:
        try:
            idx=q.get_nowait()
        except:
            break
        img=images[idx]
        kp,des=orb.detectAndCompute(img,None)
        kp_list[idx]=kp
        des_list[idx]=des
        print(f"ORB computed for image {idx} ({0 if kp is None else len(kp)} keypoints)")
        q.task_done()

queue=Queue()
for idx in needed_idxs:
    queue.put(idx)
threads=[]
for _ in range(min(8,len(needed_idxs))):
    t=threading.Thread(target=orb_worker,args=(queue,))
    t.start()
    threads.append(t)
queue.join()
print("ORB feature extraction done.")

# ----------------- Matching + Homography -----------------
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)
H_dict={}
H_inliers={}
min_inliers=15
ratio_thresh=0.75
ransac_thresh=5.0
angle_thresh=60.0
dist_thresh=50.0

def filter_matches_by_angle_distance(kp1,kp2,matches,angle_thresh,dist_thresh):
    if len(matches)<2: return matches
    pts1=np.array([kp1[m.trainIdx].pt for m in matches])
    pts2=np.array([kp2[m.queryIdx].pt for m in matches])
    vecs=pts2-pts1
    angles=np.degrees(np.arctan2(vecs[:,1],vecs[:,0]))
    med_angle=np.median(angles)
    angles_diff=np.abs(angles-med_angle)
    pts_dist=np.linalg.norm(vecs-vecs.mean(axis=0),axis=1)
    filtered=[m for m,ad,dd in zip(matches,angles_diff,pts_dist) if ad<=angle_thresh and dd<=dist_thresh]
    return filtered

print("Computing homographies between GPS-filtered neighbors...")
for i,j in neighbors:
    des_i,des_j=des_list[i],des_list[j]
    kp_i,kp_j=kp_list[i],kp_list[j]
    if des_i is None or des_j is None or kp_i is None or kp_j is None: continue
    try:
        knn_j_i=bf.knnMatch(des_j,des_i,k=2)
        good=[]
        for m_n in knn_j_i:
            if len(m_n)!=2: continue
            m,n=m_n
            if m.distance<ratio_thresh*n.distance:
                good.append(m)
        good=filter_matches_by_angle_distance(kp_i,kp_j,good,angle_thresh,dist_thresh)
        if len(good)<10: continue
        src_pts=np.float32([kp_j[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts=np.float32([kp_i[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,ransac_thresh)
        if H is not None and mask is not None and np.sum(mask)>=min_inliers:
            inlier_src=src_pts[mask.ravel()==1].reshape(-1,2)
            inlier_dst=dst_pts[mask.ravel()==1].reshape(-1,2)
            H_dict[(j,i)]=H
            H_inliers[(j,i)]=inlier_dst.copy()
            try:
                H_inv=np.linalg.inv(H)
                H_dict[(i,j)]=H_inv
                H_inliers[(i,j)]=inlier_src.copy()
            except np.linalg.LinAlgError:
                print(f"Cannot invert homography {i}-{j}")
    except cv2.error as e:
        print(f"OpenCV error {i}-{j}: {e}")
    print(f"Processed pair {i}<->{j}")
print("Homography computation done.")

# ----------------- GUI Thumbnail Grid -----------------
root.deiconify()
root.title("Images Grid (click to show correspondences)")
canvas_side=Canvas(root,width=1400,height=800,bg="black")
scrollbar=Scrollbar(root,orient="vertical",command=canvas_side.yview)
canvas_side.configure(yscrollcommand=scrollbar.set)
canvas_side.pack(side=LEFT,fill="both",expand=1)
scrollbar.pack(side=RIGHT,fill=Y)
frame=Frame(canvas_side)
canvas_side.create_window((0,0),window=frame,anchor=NW)
photo_refs_side=[]
img_positions=[]
markers=[]
cols=6
img_size=200

for i,img in enumerate(images):
    h,w=img.shape[:2]
    scale=min(img_size/h,img_size/w)
    disp_w,disp_h=int(w*scale),int(h*scale)
    img_resized=cv2.resize(img,(disp_w,disp_h))
    img_rgb=cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
    im_pil=Image.fromarray(img_rgb)
    im_tk=ImageTk.PhotoImage(im_pil)
    photo_refs_side.append(im_tk)
    lbl=Label(frame,image=im_tk,bd=2,relief="solid")
    lbl.grid(row=i//cols,column=i%cols,padx=5,pady=5)
    img_positions.append((lbl,scale,disp_w,disp_h))
print("Thumbnails ready.")

# ----------------- Click Event -----------------
def click_event(event,idx):
    global markers
    for m in markers:
        try: m.destroy()
        except: pass
    markers.clear()
    lbl,scale,disp_w,disp_h=img_positions[idx]
    x_click=event.x/scale
    y_click=event.y/scale
    pt=np.array([x_click,y_click,1.0]).reshape(3,1)
    print(f"Clicked image {idx}: ({x_click:.1f},{y_click:.1f})")
    visited=set()
    queue=[(idx,pt)]
    while queue:
        cur_idx,cur_pt=queue.pop(0)
        if cur_idx in visited: continue
        visited.add(cur_idx)
        lbl_c,scale_c,disp_w_c,disp_h_c=img_positions[cur_idx]
        x_disp,y_disp=cur_pt[0,0]*scale_c,cur_pt[1,0]*scale_c
        if 0<=x_disp<=disp_w_c and 0<=y_disp<=disp_h_c:
            m=Label(frame,bg="red")
            m.place(in_=lbl_c,x=int(x_disp),y=int(y_disp),anchor=NW,width=6,height=6)
            markers.append(m)
            for (i,j),H in list(H_dict.items()):
                if i==cur_idx and j not in visited:
                    try:
                        new_pt=H@cur_pt
                        if abs(new_pt[2,0])<1e-8: continue
                        new_pt=new_pt/new_pt[2,0]
                        if (i,j) in H_inliers:
                            hull_pts=H_inliers[(i,j)]
                            if point_in_hull(new_pt[:2,0],hull_pts):
                                queue.append((j,new_pt))
                        else:
                            queue.append((j,new_pt))
                    except Exception as e:
                        print(f"Cannot apply homography {i}->{j}: {e}")

for i,(lbl,scale,_,_) in enumerate(img_positions):
    lbl.bind("<Button-1>",lambda e,idx=i: click_event(e,idx))

frame.update_idletasks()
canvas_side.config(scrollregion=canvas_side.bbox("all"))
print("Grid ready. Click any image to see corresponding points.")
root.mainloop()
