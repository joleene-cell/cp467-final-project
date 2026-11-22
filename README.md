# Document Corner Detection: Contour vs. Hough

Group project comparing two classical methods for finding the four corners of
a document in smartphone frames.

## Folders

- `code/`: scripts
- `splits/`: `validation.csv` (221 frames) and `test.csv` (480 frames)
- `images/frames/`: 701 frames from SmartDoc 2015 Challenge 1
- `output_images/`: annotation check and detection results
- `results/`: experiment results, tables, figures, and analysis

## How to run

Run all scripts from inside `code/`, since paths use `../`.

```
pip install -r requirements.txt
cd code
python3 utils.py
python3 contour.py
python3 hough.py
python3 evaluation.py
python3 experiments.py
```

## Data

SmartDoc 2015 Challenge 1 [1], from the author's release [2]. The full dataset
has 24,889 frames from 150 videos. We kept every 40th frame (701 frames) and
split by video, so no video is in both sets. Corners are annotated in every
frame. The full dataset is not included, and it is only needed to rerun
`01_split.py` and `01_extract_subset.py`.

## Contour Method

The contour-based method is implemented in `code/contour.py`. It detects document boundaries using the following steps:

- Convert the image to grayscale and apply Gaussian blur to reduce noise.
- Use Canny edge detection to identify strong edges.
- Apply morphological closing to connect small gaps in the detected edges.
- Find and sort contours by area.
- Ignore contours smaller than 5% of the image area.
- Use polygon approximation to identify four-sided document candidates.
- Return the detected corners in TL, TR, BR, BL order.
  The method was tuned using only the validation split. Qualitative outputs are saved as edge images and result images, where red represents the ground-truth boundary and green represents the predicted document boundary.

## Hough Method

The Hough-based method is implemented in `code/hough.py`. It detects document boundaries using the following steps:

- Convert the image to grayscale and apply Gaussian blur to reduce noise.
- Use Canny edge detection to identify strong edges.
- Apply morphological closing to connect small gaps in detected edges.
- Use the probabilistic Hough line transform to detect straight line segments.
- Filter and merge line segments that belong to the same document edge.
- Group lines by similar orientations and identify approximately perpendicular pairs of document edges.
- Calculate intersections between the selected lines to estimate the four document corners.
- Score candidate quadrilaterals based on their geometry, edge support, and coverage to select the most suitable document boundary.
- Return the detected corners in TL, TR, BR, BL order.
  The method was tuned using only the validation split. Qualitative outputs are saved as edge images and result images, where red represents the ground-truth boundary and green represents the predicted document boundary.

## Evaluation

The evaluation is implemented in `code/evaluation.py` and is used to measure the performance of the document corner detection methods. The evaluation includes:

- Intersection over Union (IoU) between the predicted and ground-truth document boundaries.
- Corner error based on the distance between predicted and ground-truth corners.
- Runtime measurement for each detection method.
  These metrics are used to compare the performance of the contour-based and Hough-based approaches

## Experiments

The experiment workflow is implemented in `code/experiments.py`. It runs both the contour and Hough detection methods on a selected dataset split and evaluates their results.
The script:

- Runs both detection methods on each frame.
- Records detection success, IoU, corner error, and runtime.
- Generates comparison tables and figures.
- Produces breakdowns and failure analysis for the results.
- Saves experiment outputs in the results/ directory.
  Experiments are first run on the validation split for development and tuning. The test split is reserved for the final evaluation.

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
