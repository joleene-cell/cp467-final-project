from pathlib import Path
import cv2
import numpy as np

from utils import load_frame, load_split, gt_corners, order_corners, draw_corners


# -----------------------------------------------------------------------------
# Hough parameters
# -----------------------------------------------------------------------------
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 55
MIN_LINE_LENGTH_RATIO = 0.10
MAX_LINE_GAP = 30

# Keep only useful Hough segments.
MAX_LINES = 60
MAX_ANCHORS = 8
MAX_FAMILY_LINES = 30
MAX_LINE_PAIRS = 12

# Geometry constraints.
MIN_PAIR_SEPARATION_RATIO = 0.08
MIN_AREA_RATIO = 0.035
MAX_AREA_RATIO = 0.70
MIN_SIDE_RATIO = 0.07
MIN_ANGLE = 40
MAX_ANGLE = 140

# A document edge should have continuous edge support.
MIN_EDGE_SUPPORT = 0.25
MIN_SEGMENT_COVERAGE = 0.18


# -----------------------------------------------------------------------------
# Line helpers
# -----------------------------------------------------------------------------
def line_from_segment(x1, y1, x2, y2):
    a = float(y1 - y2)
    b = float(x2 - x1)
    norm = np.hypot(a, b)
    if norm < 1e-6:
        return None

    a /= norm
    b /= norm
    c = a * x1 + b * y1

    if a < 0 or (abs(a) < 1e-8 and b < 0):
        a, b, c = -a, -b, -c

    return {
        "a": a,
        "b": b,
        "c": c,
        "angle": float(np.arctan2(y2 - y1, x2 - x1) % np.pi),
        "length": float(np.hypot(x2 - x1, y2 - y1)),
        "segment": np.array([[x1, y1], [x2, y2]], dtype=np.float32),
    }


def angle_difference(a, b):
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


def intersection(l1, l2):
    det = l1["a"] * l2["b"] - l2["a"] * l1["b"]
    if abs(det) < 1e-6:
        return None

    x = (l1["c"] * l2["b"] - l2["c"] * l1["b"]) / det
    y = (l1["a"] * l2["c"] - l2["a"] * l1["c"]) / det
    return np.array([x, y], dtype=np.float32)


def line_distance(l1, l2):
    return abs(l1["c"] - l2["c"])


# -----------------------------------------------------------------------------
# Hough line preparation
# -----------------------------------------------------------------------------
def touches_border(line, shape, border=5):
    h, w = shape[:2]
    p = line["segment"]
    x1, y1 = p[0]
    x2, y2 = p[1]

    n = max(20, int(line["length"] / 5))
    t = np.linspace(0, 1, n)
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    on_border = (
        (x <= border) | (x >= w - 1 - border) |
        (y <= border) | (y >= h - 1 - border)
    )
    return np.mean(on_border) > 0.65


def merge_collinear(lines, angle_tol=3, distance_tol=10):
    """Merge broken Hough pieces belonging to the same physical edge."""
    groups = []

    for line in sorted(lines, key=lambda x: x["length"], reverse=True):
        placed = False
        for group in groups:
            ref = group[0]
            if angle_difference(line["angle"], ref["angle"]) > np.deg2rad(angle_tol):
                continue
            if abs(line["c"] - ref["c"]) > distance_tol:
                continue
            group.append(line)
            placed = True
            break

        if not placed:
            groups.append([line])

    merged = []
    for group in groups:
        ref = group[0]
        direction = np.array(
            [np.cos(ref["angle"]), np.sin(ref["angle"])], dtype=np.float32
        )
        points = np.vstack([x["segment"] for x in group])
        proj = points @ direction
        p1 = points[np.argmin(proj)]
        p2 = points[np.argmax(proj)]

        result = line_from_segment(p1[0], p1[1], p2[0], p2[1])
        if result is not None:
            merged.append(result)

    return merged


def get_line_families(lines):
    """Create many possible orientation pairs instead of trusting one dominant angle."""
    families = []
    lines = sorted(lines, key=lambda x: x["length"], reverse=True)

    for anchor in lines[:MAX_ANCHORS]:
        a = anchor["angle"]
        family_a = []
        family_b = []

        for line in lines:
            d = angle_difference(line["angle"], a)
            if d <= np.deg2rad(12):
                family_a.append(line)
            elif np.deg2rad(55) <= d <= np.deg2rad(125):
                family_b.append(line)

        if len(family_a) >= 2 and len(family_b) >= 2:
            families.append((family_a[:MAX_FAMILY_LINES], family_b[:MAX_FAMILY_LINES]))

    return families


def line_pairs(family, shape):
    h, w = shape[:2]
    min_sep = MIN_PAIR_SEPARATION_RATIO * min(h, w)
    pairs = []

    for i in range(len(family)):
        for j in range(i + 1, len(family)):
            l1, l2 = family[i], family[j]
            sep = line_distance(l1, l2)
            if sep < min_sep:
                continue

            # Prefer long, well-separated lines.
            score = sep * np.sqrt(l1["length"] * l2["length"])
            pairs.append((score, l1, l2))

    pairs.sort(key=lambda x: x[0], reverse=True)
    return [(a, b) for _, a, b in pairs[:MAX_LINE_PAIRS]]


# -----------------------------------------------------------------------------
# Edge/segment support
# -----------------------------------------------------------------------------
def segment_coverage(line, p1, p2):
    seg = line["segment"]
    s0, s1 = seg
    v = s1 - s0
    denom = float(np.dot(v, v))
    if denom < 1e-6:
        return 0.0

    t1 = float(np.dot(p1 - s0, v) / denom)
    t2 = float(np.dot(p2 - s0, v) / denom)
    lo, hi = sorted((t1, t2))
    overlap = max(0.0, min(hi, 1.0) - max(lo, 0.0))

    side_length = np.linalg.norm(p2 - p1)
    if side_length < 1e-6:
        return 0.0

    return float(overlap * line["length"] / side_length)


def edge_support(edge_distance, p1, p2, max_distance=2.5):
    """Fast, vectorized edge support using a precomputed distance transform."""
    length = np.linalg.norm(p2 - p1)
    if length < 10:
        return 0.0

    # 40 samples are enough to determine whether a long straight boundary
    # is actually supported by the Canny edge image. This replaces the old
    # Python loop over every pixel and is much faster.
    n = min(80, max(30, int(length / 8)))
    points = np.linspace(p1, p2, n)
    xs = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, edge_distance.shape[1] - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, edge_distance.shape[0] - 1)

    distances = edge_distance[ys, xs]
    return float(np.mean(distances <= max_distance))


# -----------------------------------------------------------------------------
# Candidate scoring
# -----------------------------------------------------------------------------
def score_quad(corners, boundary_lines, edge_distance):
    h, w = edge_distance.shape[:2]

    # Keep intersections safely inside the image.
    margin_x = 0.01 * w
    margin_y = 0.01 * h
    if np.any(corners[:, 0] < margin_x) or np.any(corners[:, 0] > w - margin_x):
        return None
    if np.any(corners[:, 1] < margin_y) or np.any(corners[:, 1] > h - margin_y):
        return None

    contour = corners.reshape(-1, 1, 2).astype(np.float32)
    if not cv2.isContourConvex(contour):
        return None

    area = abs(cv2.contourArea(contour))
    area_ratio = area / (w * h)
    if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
        return None

    sides = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(sides, axis=1)
    if np.min(lengths) < MIN_SIDE_RATIO * min(h, w):
        return None

    # Reject extremely distorted quadrilaterals.
    for i in range(4):
        v1 = corners[i - 1] - corners[i]
        v2 = corners[(i + 1) % 4] - corners[i]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-6:
            return None
        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1, 1)))
        if angle < MIN_ANGLE or angle > MAX_ANGLE:
            return None

    coverages = []
    supports = []
    for i, line in enumerate(boundary_lines):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        coverage = segment_coverage(line, p1, p2)
        support = edge_support(edge_distance, p1, p2)

        if coverage < MIN_SEGMENT_COVERAGE and support < MIN_EDGE_SUPPORT:
            return None

        coverages.append(coverage)
        supports.append(support)

    # ------------------------------------------------------------------
    # IMPORTANT: prefer the OUTER document boundary.
    #
    # A common failure is:
    #   left/right document borders + an internal horizontal table line.
    #
    # That creates a perfectly valid smaller rectangle, so area and line
    # separation need to be strong parts of the score.
    # ------------------------------------------------------------------
    opposite_sep_1 = line_distance(boundary_lines[0], boundary_lines[2])
    opposite_sep_2 = line_distance(boundary_lines[1], boundary_lines[3])

    # Normalize separation by image size. Full document sides should usually
    # be considerably farther apart than an internal table/text line.
    sep1 = min(1.0, opposite_sep_1 / (0.35 * min(h, w)))
    sep2 = min(1.0, opposite_sep_2 / (0.35 * min(h, w)))
    separation_score = 0.5 * (sep1 + sep2)

    # Area is deliberately weighted heavily. This is the main protection
    # against selecting a smaller rectangle inside the document.
    area_score = min(1.0, area_ratio / 0.12)

    # Long proposed sides are another useful signal.
    length_score = min(1.0, np.mean(lengths) / (0.35 * max(h, w)))

    # Document-like aspect ratio. The actual project images contain mostly
    # portrait documents, but keep this soft so perspective is allowed.
    width_est = 0.5 * (lengths[0] + lengths[2])
    height_est = 0.5 * (lengths[1] + lengths[3])
    aspect = width_est / max(height_est, 1e-6)
    aspect_score = np.exp(-0.5 * ((aspect - 0.78) / 0.25) ** 2)

    support_score = np.mean(supports)
    coverage_score = np.mean(coverages)

    return (
        2.8 * area_score
        + 2.0 * separation_score
        + 1.2 * support_score
        + 0.8 * coverage_score
        + 0.5 * length_score
        + 0.4 * aspect_score
    )


# -----------------------------------------------------------------------------
# Main detector
# -----------------------------------------------------------------------------
def detect_hough(img):
    """Detect document corners using Hough line segments."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # Small closing fills tiny breaks in the document boundary.
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    h, w = img.shape[:2]
    min_len = int(MIN_LINE_LENGTH_RATIO * min(h, w))

    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=min_len,
        maxLineGap=MAX_LINE_GAP,
    )

    if raw is None:
        return None, edges

    lines = []
    for x1, y1, x2, y2 in raw[:, 0]:
        line = line_from_segment(x1, y1, x2, y2)
        if line is not None:
            if not touches_border(line, img.shape):
                lines.append(line)

    if len(lines) < 4:
        return None, edges

    # Merge broken pieces from the same physical edge.
    lines = merge_collinear(lines)
    lines.sort(key=lambda x: x["length"], reverse=True)
    lines = lines[:MAX_LINES]

    # Distance transform lets us score candidate boundaries without
    # repeatedly scanning pixels in Python.
    edge_distance = cv2.distanceTransform(
        cv2.bitwise_not(edges),
        cv2.DIST_L2,
        3,
    )

    families = get_line_families(lines)
    if not families:
        return None, edges

    best_score = -np.inf
    best = None

    for family_a, family_b in families:
        pairs_a = line_pairs(family_a, img.shape)
        pairs_b = line_pairs(family_b, img.shape)

        for a1, a2 in pairs_a:
            for b1, b2 in pairs_b:
                angle = np.degrees(angle_difference(a1["angle"], b1["angle"]))
                if angle < 55 or angle > 125:
                    continue

                # Intersections of the two line pairs.
                p_tl = intersection(a1, b1)
                p_tr = intersection(a1, b2)
                p_br = intersection(a2, b2)
                p_bl = intersection(a2, b1)

                if any(p is None for p in (p_tl, p_tr, p_br, p_bl)):
                    continue

                corners = np.array([p_tl, p_tr, p_br, p_bl], dtype=np.float32)

                # Lines corresponding to the sides in the same order as corners.
                boundary_lines = [a1, b2, a2, b1]
                score = score_quad(corners, boundary_lines, edge_distance)
                if score is None:
                    continue

                if score > best_score:
                    best_score = score
                    best = corners

    if best is None:
        return None, edges

    return order_corners(best), edges


# -----------------------------------------------------------------------------
# Validation test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    out = Path("../output_images/hough")
    out.mkdir(parents=True, exist_ok=True)

    validation = load_split("validation")
    sample = validation.sample(min(50, len(validation)), random_state=0)

    success = 0
    fail = 0

    for i, row in sample.iterrows():
        img, scale = load_frame(row)
        gt = gt_corners(row, scale)

        predicted, edges = detect_hough(img)

        if predicted is None:
            fail += 1
            print(f"{i}: no Hough quadrilateral detected")
            output = draw_corners(img, gt, color=(0, 0, 255))
        else:
            success += 1
            print(f"{i}: Hough quadrilateral detected")
            output = draw_corners(img, gt, color=(0, 0, 255))
            output = draw_corners(output, predicted, color=(0, 255, 0))

        cv2.imwrite(str(out / f"{i}_result.jpg"), output)
        cv2.imwrite(str(out / f"{i}_edges.jpg"), edges)

    print()
    print("Hough line test finished")
    print("Successful:", success)
    print("Failed:", fail)
