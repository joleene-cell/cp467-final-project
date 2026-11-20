from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Paths 
IMG_ROOT   = Path("../images/frames")
SPLITS_DIR = Path("../splits")
RESIZE_WIDTH = 800       

# CORNER ORDER: top-left, top-right, bottom-right, bottom-left
# Corners are (x, y) float32 arrays of shape (4, 2).
# Note: the dataset CV stores them as TL, BL, BR, TR, so gt_corners() reorders them

# Load one of the split files as a table
def load_split(name):
    assert name in ("validation", "test"), "use 'validation' or 'test'"
    return pd.read_csv(SPLITS_DIR / f"{name}.csv")

# Read one frame and shirnk it so every method gets the same size image
def load_frame(row, width=RESIZE_WIDTH):
    img = cv2.imread(str(IMG_ROOT / row["image_path"]))
    # check if file missing
    if img is None:
        raise FileNotFoundError(f"Could not read {IMG_ROOT / row['image_path']}")
    # makes sure never to enlarge a small image
    scale = min(1.0, width / img.shape[1])
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img, scale


# Ground truth corners as a 4x2 array in TL, TR, BR, BL order
def gt_corners(row, scale=1.0):
    pts = np.array([[row["tl_x"], row["tl_y"]],
                    [row["tr_x"], row["tr_y"]],
                    [row["br_x"], row["br_y"]],
                    [row["bl_x"], row["bl_y"]]], dtype=np.float32)
    # multiply by scale so the orders match the resized image
    return pts * scale

# Sort any 4 points into TL, TR, BR, BL
def order_corners(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)                 # x + y
    d = np.diff(pts, axis=1).ravel()    # y - x
    return np.array([pts[np.argmin(s)],    # top-left: smallest x + y
                     pts[np.argmin(d)],    # top-right: smallest y - x
                     pts[np.argmax(s)],    # bottom-right: largest x + y
                     pts[np.argmax(d)]],   # bottom-left: largest y - x
                    dtype=np.float32)

# Draw the 4 corners 
def draw_corners(img, corners, color=(0, 255, 0), thickness=3):
    out = img.copy()
    cv2.polylines(out, [np.asarray(corners).astype(np.int32)], True, color, thickness)
    return out

# Annotation check. Draw the ground truth on 15 random test frames
if __name__ == "__main__":
    out_dir = Path("../output_images/annotation_check")
    out_dir.mkdir(parents=True, exist_ok=True)
    test = load_split("test").sample(15, random_state=0)
    for i, row in test.iterrows():
        img, scale = load_frame(row)
        cv2.imwrite(str(out_dir / f"{i}.jpg"), draw_corners(img, gt_corners(row, scale)))
    print("Saved 15 images to", out_dir)