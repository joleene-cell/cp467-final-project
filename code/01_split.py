from pathlib import Path
import numpy as np
import pandas as pd

# Settings
EVERY_NTH = 40   # keep every 40th frame per video
VAL_FRAC  = 0.30 # 30% of the videos go to validation
SEED      = 42

# Load the table
csv_path = "../dataset/frames/metadata.csv"
df = pd.read_csv(csv_path)
print("CSV:", csv_path)
print("Total frames:", len(df))

# Preprocessing: define each video and drop invalid rows
df["video_id"] = df["bg_name"] + "_" + df["model_name"]
corner_cols = ["tl_x","tl_y","bl_x","bl_y","br_x","br_y","tr_x","tr_y"]
df = df.dropna(subset=corner_cols)
print("Videos:", df["video_id"].nunique())
print("Frames per video (min/median/max):",
      df.groupby("video_id").size().agg(["min","median","max"]).tolist())

# Keep every 40th frame
df = df.sort_values(["video_id", "frame_index"])
df["_i"] = df.groupby("video_id").cumcount()
df = df[df["_i"] % EVERY_NTH == 0].drop(columns="_i")

# Split the videos into validation and test
rng = np.random.default_rng(SEED)
videos = df[["video_id", "modeltype_name"]].drop_duplicates()
val_videos = []
for _, g in videos.groupby("modeltype_name"):
    ids = g["video_id"].to_numpy().copy()
    rng.shuffle(ids)
    val_videos += list(ids[:max(1, round(len(ids) * VAL_FRAC))])

val_df  = df[df["video_id"].isin(val_videos)]
test_df = df[~df["video_id"].isin(val_videos)]

# Leakage check. Ensure no frames from the same video may appear in both splits
assert set(val_df["video_id"]).isdisjoint(set(test_df["video_id"]))

# Save 
Path("../splits").mkdir(exist_ok=True)
val_df.to_csv("../splits/validation.csv", index=False)
test_df.to_csv("../splits/test.csv", index=False)

# Stats
print("\nValidation:", len(val_df), "frames,", val_df["video_id"].nunique(), "videos")
print("Test:      ", len(test_df), "frames,", test_df["video_id"].nunique(), "videos")
print("\nVideos per document type:")
print(pd.concat({"validation": val_df.groupby("modeltype_name")["video_id"].nunique(),
                 "test":       test_df.groupby("modeltype_name")["video_id"].nunique()}, axis=1))