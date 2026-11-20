# Document Corner Detection: Contour vs. Hough

Group project comparing two classical methods for finding the four corners of
a document in smartphone frames.

## Folders

- `code/`: scripts
- `splits/`: `validation.csv` (221 frames) and `test.csv` (480 frames)
- `images/frames/`: 701 frames from SmartDoc 2015 Challenge 1
- `output_images/`: annotation check and detection results

## How to run

Run all scripts from inside `code/`, since paths use `../`.

```
pip install -r requirements.txt
cd code
python3 utils.py
```

## Data

SmartDoc 2015 Challenge 1 [1], from the author's release [2]. The full dataset
has 24,889 frames from 150 videos. We kept every 40th frame (701 frames) and
split by video, so no video is in both sets. Corners are annotated in every
frame. The full dataset is not included, and it is only needed to rerun
`split.py` and `make_subset.py`.

## Rules for the team

- Load data only through `utils.py`.
- Corner order is TL, TR, BR, BL.
- `detect_corners(img)` returns 4 corners, or `None` if it fails.
- Tune on `validation`. Use `test` once, at the end.

## License

Frames are from [2] under the MIT License (see `images/frames/LICENSE-smartdoc15-ch1.txt`).

## References

[1] J.-C. Burie et al., "ICDAR2015 competition on smartphone document capture
and OCR (SmartDoc)," ICDAR, 2015, doi: 10.1109/ICDAR.2015.7333943.

[2] J. Chazalon, "smartdoc15-ch1-dataset," GitHub, v2.0.0.
https://github.com/jchazalon/smartdoc15-ch1-dataset
