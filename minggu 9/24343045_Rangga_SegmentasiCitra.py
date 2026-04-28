import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd   

# =========================================================
# LOAD IMAGES 
# =========================================================
def load_image(path):
    img = cv2.imread(path, 0)
    if img is None:
        print(f"[ERROR] Gambar tidak ditemukan: {path}")
        exit()
    return img

images = {
    "Bimodal": load_image("bimodal.jpg"),
    "Uneven": load_image("uneven.jpg"),
    "Overlapping": load_image("koin.jpg")
}

# =========================================================
# GROUND TRUTH 
# =========================================================
def load_gt(name, fallback_img):
    path = f"gt_{name.lower()}.png"
    if os.path.exists(path):
        print(f"[INFO] Load GT: {path}")
        return load_image(path)
    else:
           return cv2.threshold(fallback_img, 100, 255, cv2.THRESH_BINARY)[1]

gts = {}
for k, img in images.items():
    gts[k] = load_gt(k, img)


# =========================================================
# THRESHOLDING
# =========================================================
def thresholding(img):
    return {
        "Global": cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1],
        "Otsu": cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
        "Adaptive Mean": cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2),
        "Adaptive Gaussian": cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    }


# =========================================================
# EDGE DETECTION 
# =========================================================
def edge(img):
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    magnitude = np.sqrt(sx**2 + sy**2)

    sobel = cv2.normalize(
        magnitude, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    orientation = np.arctan2(sy, sx) * 180 / np.pi

    prewitt = cv2.filter2D(img, -1, np.array([[1,0,-1],[1,0,-1],[1,0,-1]]))

    return {
        "Sobel": sobel,
        "Prewitt": prewitt,
        "Canny(50,150)": cv2.Canny(img,50,150),
        "Canny(100,200)": cv2.Canny(img,100,200),
        "Sobel Orientation": orientation   
    }


# =========================================================
# REGION METHODS
# =========================================================
def auto_seed(img):
    
    y, x = np.unravel_index(np.argmax(img), img.shape)
    return (y, x)

def region_growing(img, seed, th=25):
    h,w = img.shape
    res = np.zeros_like(img)
    visited = np.zeros_like(img,bool)
    stack=[seed]

    while stack:
        x,y = stack.pop()
        if x<0 or y<0 or x>=h or y>=w or visited[x,y]:
            continue

        visited[x,y]=True

        if abs(int(img[x,y])-int(img[seed]))<th:
            res[x,y]=255
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((x+dx,y+dy))

    return res


def watershed_seg(img):
    _,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    dist = cv2.distanceTransform(th,cv2.DIST_L2,5)
    _,fg = cv2.threshold(dist,0.5*dist.max(),255,0)
    fg = fg.astype(np.uint8)

    unknown = cv2.subtract(th,fg)

    _,markers = cv2.connectedComponents(fg)
    markers += 1
    markers[unknown==255] = 0

    imgc = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(imgc,markers)

    res = np.zeros_like(img)
    res[markers>1] = 255
    return res


def connected_components(img):
    _,bin = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    num_labels, labels = cv2.connectedComponents(bin)
    print(f"[INFO] Jumlah objek terdeteksi: {num_labels-1}")  # ✅ TAMBAHAN
    return bin


# =========================================================
# METRICS
# =========================================================
def metrics(p,g):
    p=(p>0); g=(g>0)

    tp=np.sum((p==1)&(g==1))
    fp=np.sum((p==1)&(g==0))
    fn=np.sum((p==0)&(g==1))
    tn=np.sum((p==0)&(g==0))

    acc=(tp+tn)/(tp+tn+fp+fn+1e-6)
    prec=tp/(tp+fp+1e-6)
    rec=tp/(tp+fn+1e-6)
    dice=2*tp/(2*tp+fp+fn+1e-6)
    iou=tp/(tp+fp+fn+1e-6)

    return acc,prec,rec,dice,iou


# =========================================================
# NOISE TEST
# =========================================================
def add_noise(img):
    noise=np.random.normal(0,20,img.shape)
    return np.clip(img+noise,0,255).astype(np.uint8)


# =========================================================
# MAIN PIPELINE
# =========================================================
for name, img in images.items():

    print("\n==============================")
    print("IMAGE:", name)
    print("==============================")

    gt = gts[name]
    results = {}
    times = {}  

    # Threshold
    for k,v in thresholding(img).items():
        start = time.time()
        results[k] = v
        times[k] = time.time() - start

    # Edge
    for k,v in edge(img).items():
        start = time.time()
        results[k] = v
        times[k] = time.time() - start

    # Region
    seed = auto_seed(img)  
    start = time.time()
    results["Region Growing"] = region_growing(img,seed)
    times["Region Growing"] = time.time() - start

    start = time.time()
    results["Watershed"] = watershed_seg(img)
    times["Watershed"] = time.time() - start

    start = time.time()
    results["Connected Component"] = connected_components(img)
    times["Connected Component"] = time.time() - start

    best=None
    best_score=0

    print(f"{'Method':<25} {'Dice':<10} {'IoU':<10} {'Time(s)'}")
    print("-"*60)

    table = []  

    for k,v in results.items():

        if "Orientation" in k:
            continue

        acc,pre,rec,dice,iou = metrics(v,gt)

        print(f"{k:<25} {dice:.3f}     {iou:.3f}     {times.get(k,0):.4f}")

        table.append([k,acc,pre,rec,dice,iou,times.get(k,0)])

        if dice > best_score:
            best_score = dice
            best = k

    df = pd.DataFrame(table, columns=["Method","Accuracy","Precision","Recall","Dice","IoU","Time"])
    print("\nTABEL HASIL:")
    print(df)

    print("\nBEST METHOD:", best)

    # Robustness (noise)
    noisy = add_noise(img)
    noisy_res = thresholding(noisy)["Otsu"]
    _,_,_,dice_n,_ = metrics(noisy_res,gt)
    print("Otsu (Noise Test):", round(dice_n,3))

    # Overlay contour
    overlay = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    contours,_ = cv2.findContours(results[best],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay,contours,-1,(0,255,0),1)

    # Visualisasi
    plt.figure(figsize=(12,6))

    plt.subplot(2,3,1); plt.imshow(img,cmap='gray'); plt.title("Original"); plt.axis('off')
    plt.subplot(2,3,2); plt.imshow(gt,cmap='gray'); plt.title("Ground Truth"); plt.axis('off')
    plt.subplot(2,3,3); plt.imshow(results["Otsu"],cmap='gray'); plt.title("Otsu"); plt.axis('off')

    plt.subplot(2,3,4); plt.imshow(results["Canny(50,150)"],cmap='gray'); plt.title("Canny"); plt.axis('off')
    plt.subplot(2,3,5); plt.imshow(results["Watershed"],cmap='gray'); plt.title("Watershed"); plt.axis('off')
    plt.subplot(2,3,6); plt.imshow(overlay); plt.title(f"Overlay ({best})"); plt.axis('off')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(noisy, cmap='gray')
    plt.title("Noisy Image")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(noisy_res, cmap='gray')
    plt.title("Otsu (Noisy)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()