import pandas as pd, shutil
from pathlib import Path

# Copies the frames listed in split files

src = Path("../dataset/frames") # full dataset
dst = Path("../images/frames") # subset folder

val = pd.read_csv("../splits/validation.csv")
test = pd.read_csv("../splits/test.csv")
both = pd.concat([val, test]) # list of frames to copy

#  
for p in both["image_path"]:
    out = dst / p
    out.parent.mkdir(parents=True, exist_ok=True) 
    shutil.copy(src / p, out) # extract image

both.to_csv(dst / "metadata_subset.csv", index=False)
print(len(both), "frames copied") # total rows copied