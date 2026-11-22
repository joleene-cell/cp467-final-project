"""
Runs each method, scores it with evaluation.py, and makes the tables,
figures and failure analysis for the Results section.

    python3 experiments.py --split validation
    python3 experiments.py --split test             # final run, do this once
    python3 experiments.py --split test --reuse     # redo figures, no rerun

Output: ../results/<split>/per_frame.csv and ../results/<split>/analysis/
"""
import argparse
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ast

from contour import detect_contour
from hough import detect_hough
from utils import load_split, load_frame, gt_corners


def load_evaluation_functions():
    # evaluation.py runs a loop at the top level, so load only its imports and functions
    tree = ast.parse((Path(__file__).parent / "evaluation.py").read_text())
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    ns = {}
    exec(compile(ast.Module(keep, type_ignores=[]), "evaluation.py", "exec"), ns)
    return ns["IoU_calculation"], ns["calculate_corner_difference"], ns["record_runtime"]


IoU_calculation, calculate_corner_difference, record_runtime = load_evaluation_functions()

IOU_SUCCESS = 0.9
BOOT_REPS = 2000
RED, GREEN = (0, 0, 255), (0, 255, 0)
PALETTE = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd"]


# ---------- running the methods ----------
def get_methods():
    # both detectors return (corners, edges); keep the corners
    return {"contour": lambda img: detect_contour(img)[0],
            "hough": lambda img: detect_hough(img)[0]}


def run_methods(split, methods):
    df = load_split(split)
    rows = []
    for idx, row in df.iterrows():
        img, scale = load_frame(row)
        gt = gt_corners(row, scale)
        for name, fn in methods.items():
            try:
                pred, secs = record_runtime(fn, img)
            except Exception as e:
                print(f"{name} failed on frame {idx}: {e}")
                pred, secs = None, np.nan
            if pred is not None:
                pred = np.asarray(pred, dtype=np.float32).reshape(4, 2)
            flat = np.full(8, np.nan) if pred is None else pred.reshape(-1)
            r = {"method": name, "row_index": idx, "frame": row["image_path"],
                 "video": row["video_id"], "background": row["bg_name"],
                 "doctype": row["modeltype_name"], "detected": pred is not None,
                 "iou": IoU_calculation(pred, gt, img.shape),
                 "mean_err": calculate_corner_difference(pred, gt),
                 "runtime_s": secs}
            r.update({f"p{i // 2}{'xy'[i % 2]}": float(flat[i]) for i in range(8)})
            rows.append(r)
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(df)} frames")
    return pd.DataFrame(rows)


# ---------- helpers ----------
def md_table(df, index=True):
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(out)


def latex_table(df, caption, label):
    d = df.reset_index()
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")
    lines = [r"\begin{table}[h]", r"\centering",
             rf"\begin{{tabular}}{{{'l' + 'c' * (d.shape[1] - 1)}}}", r"\hline",
             " & ".join(esc(c) for c in d.columns) + r" \\", r"\hline"]
    for _, r in d.iterrows():
        lines.append(" & ".join(esc(v) for v in r.tolist()) + r" \\")
    lines += [r"\hline", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def boot_ci(values, videos, reps=None, seed=0):
    # mean and 95% CI, resampling whole videos
    reps = reps or BOOT_REPS
    values = np.asarray(values, dtype=float)
    ok = ~np.isnan(values)
    if ok.sum() == 0:
        return np.nan, np.nan, np.nan
    g = pd.DataFrame({"v": np.asarray(videos)[ok], "x": values[ok]}).groupby("v")["x"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), (reps, len(s)))
    means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return s.sum() / c.sum(), *np.percentile(means, [2.5, 97.5])


def fmt_ci(m, lo, hi, d=3):
    return "n/a" if np.isnan(m) else f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def colors(methods):
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}


# ---------- frame features and tags ----------
def frame_features(img, gt):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(gt).astype(np.int32)], 1)
    k = np.ones((13, 13), np.uint8)
    inner = (mask == 1) & (cv2.erode(mask, k) == 0)
    outer = (cv2.dilate(mask, k) == 1) & (mask == 0)
    contrast = abs(float(gray[inner].mean()) - float(gray[outer].mean())) if inner.any() and outer.any() else np.nan
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    side = lambda a, b: float(np.linalg.norm(gt[a] - gt[b]))
    top, right, bottom, left = side(0, 1), side(1, 2), side(3, 2), side(0, 3)
    diag = float(np.hypot(h, w))
    return dict(contrast=contrast,
                sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                bg_edge=float((edges[mask == 0] > 0).mean()),
                skew=max(top / bottom, bottom / top, left / right, right / left),
                area_frac=float(mask.sum() / (h * w)),
                margin=min(gt[:, 0].min(), gt[:, 1].min(), w - gt[:, 0].max(), h - gt[:, 1].max()) / diag)


def get_features(split, res, out):
    path = out / "frame_features.csv"
    if path.exists():
        feats = pd.read_csv(path, index_col="row_index")
        if set(res["row_index"]) <= set(feats.index):
            return feats
    df = load_split(split)
    rows = {}
    for i in sorted(res["row_index"].unique()):
        img, scale = load_frame(df.loc[i])
        rows[i] = frame_features(img, gt_corners(df.loc[i], scale))
    feats = pd.DataFrame.from_dict(rows, orient="index")
    feats.index.name = "row_index"
    feats.to_csv(path)
    return feats


# tag -> (feature, which end counts). Thresholds are quartiles of the split.
TAGS = {"low_contrast": ("contrast", "low"), "blurry": ("sharpness", "low"),
        "cluttered_bg": ("bg_edge", "high"), "strong_tilt": ("skew", "high"),
        "small_doc": ("area_frac", "low"), "near_border": ("margin", "abs")}


def tag_frames(feats):
    t = pd.DataFrame(index=feats.index)
    for tag, (f, d) in TAGS.items():
        if d == "low":
            t[tag] = feats[f] <= feats[f].quantile(0.25)
        elif d == "high":
            t[tag] = feats[f] >= feats[f].quantile(0.75)
        else:
            t[tag] = feats[f] < 0.02
    return t


# ---------- tables ----------
def main_table(res, methods, thresholds=(0.5, 0.7, 0.8, 0.9, 0.95)):
    rows, sweep = {}, {}
    for m in methods:
        g = res[res.method == m]
        det = g[g.detected]
        rows[m] = {
            "frames": len(g),
            "detection_rate": fmt_ci(*boot_ci(g.detected.astype(float), g.video)),
            f"correct_rate (IoU>={IOU_SUCCESS})": fmt_ci(*boot_ci(g.correct.astype(float), g.video)),
            "mean_IoU (misses=0)": fmt_ci(*boot_ci(g.iou, g.video)),
            "mean_IoU (detected only)": fmt_ci(*boot_ci(det.iou, det.video)),
            "corner_err_px (detected)": fmt_ci(*boot_ci(det.mean_err, det.video), d=2),
            "median_corner_err_px": "n/a" if det.empty else f"{det.mean_err.median():.2f}",
            "runtime_ms (mean)": f"{g.runtime_s.mean() * 1000:.1f}"}
        sweep[m] = {f"IoU>={t}": round(float((g.iou >= t).mean()), 4) for t in thresholds}
    return pd.DataFrame(rows).T, pd.DataFrame(sweep).T


def breakdown(res, methods, by):
    out = {}
    for m in methods:
        grp = res[res.method == m].groupby(by)
        out[(m, "n")] = grp.size()
        out[(m, "correct")] = grp["correct"].mean().round(3)
        out[(m, "detected")] = grp["detected"].mean().round(3)
        out[(m, "mean_IoU")] = grp["iou"].mean().round(3)
    t = pd.DataFrame(out)
    t.columns = [f"{m}: {k}" for m, k in t.columns]
    return t


def paired_comparison(res, a, b):
    wi = res.pivot(index="row_index", columns="method", values="iou")
    wc = res.pivot(index="row_index", columns="method", values="correct")
    vid = res.drop_duplicates("row_index").set_index("row_index")["video"].loc[wi.index].to_numpy()
    only_a = int((wc[a] & ~wc[b]).sum())
    only_b = int((~wc[a] & wc[b]).sum())
    d = boot_ci((wi[a] - wi[b]).to_numpy(), vid)
    cd = boot_ci((wc[a].astype(float) - wc[b].astype(float)).to_numpy(), vid)
    p = mcnemar_exact(only_a, only_b)
    row = {"both_correct": int((wc[a] & wc[b]).sum()),
           f"only_{a}_correct": only_a, f"only_{b}_correct": only_b,
           "neither_correct": int((~wc[a] & ~wc[b]).sum()),
           f"correct_rate_diff ({a}-{b})": fmt_ci(*cd),
           f"mean_IoU_diff ({a}-{b})": fmt_ci(*d),
           "mcnemar_p (approx.)": f"{p:.4f}"}
    return pd.DataFrame({f"{a} vs {b}": row}).T, dict(only_a=only_a, only_b=only_b, cd=cd)


def failure_analysis(res, methods, tags):
    summary, per_frame = [], []
    for m in methods:
        g = res[res.method == m].set_index("row_index")
        failed = ~g["correct"]
        ft = tags.loc[g.index]
        for t in TAGS:
            has = ft[t]
            summary.append({"method": m, "tag": t,
                            "failed_with_tag": int((failed & has).sum()),
                            "share_of_failures": round(float((failed & has).sum() / max(failed.sum(), 1)), 3),
                            "fail_rate_with_tag": round(float(failed[has].mean()), 3) if has.any() else np.nan,
                            "fail_rate_without": round(float(failed[~has].mean()), 3) if (~has).any() else np.nan})
        for i in g.index[failed]:
            per_frame.append({"method": m, "row_index": i, "frame": g.loc[i, "frame"],
                              "background": g.loc[i, "background"], "doctype": g.loc[i, "doctype"],
                              "failure_type": "no_detection" if not g.loc[i, "detected"] else "wrong_quad",
                              "iou": round(float(g.loc[i, "iou"]), 3),
                              "tags": ";".join(t for t in TAGS if tags.loc[i, t]) or "none"})
    return pd.DataFrame(summary), pd.DataFrame(per_frame)


# ---------- figures ----------
def grouped_bars(ax, methods, col, groups, values_fn):
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        ax.bar(np.arange(len(groups)) + i * w, values_fn(m), w, label=m, color=col[m])
    ax.set_xticks(np.arange(len(groups)) + w * (len(methods) - 1) / 2)


def fig_overall(res, methods, col, out):
    fig, ax = plt.subplots(figsize=(5 + len(methods), 4))
    names = ["Detection rate", f"Correct (IoU≥{IOU_SUCCESS})", "Mean IoU"]
    stats = {m: [boot_ci(s, res[res.method == m].video) for s in
                 (res[res.method == m].detected.astype(float), res[res.method == m].correct.astype(float),
                  res[res.method == m].iou)] for m in methods}
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        mean = np.array([v[0] for v in stats[m]])
        err = np.array([[v[0] - v[1] for v in stats[m]], [v[2] - v[0] for v in stats[m]]])
        ax.bar(np.arange(3) + i * w, mean, w, yerr=err, capsize=3, label=m, color=col[m])
    ax.set_xticks(np.arange(3) + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Overall performance (95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_overall.png", dpi=200)
    plt.close(fig)


def fig_threshold(res, methods, col, out):
    ts = np.linspace(0.5, 0.99, 50)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m in methods:
        iou = res[res.method == m].iou.to_numpy()
        ax.plot(ts, [(iou >= t).mean() for t in ts], label=m, color=col[m])
    ax.axvline(IOU_SUCCESS, ls="--", c="gray", lw=1)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Fraction of frames with IoU ≥ threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_threshold_curve.png", dpi=200)
    plt.close(fig)


def fig_breakdown(res, methods, col, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (by, title) in zip(axes, [("background", "By background"), ("doctype", "By document type")]):
        cats = sorted(res[by].unique())
        grouped_bars(ax, methods, col, cats,
                     lambda m: res[res.method == m].groupby(by)["correct"].mean().reindex(cats).values)
        ax.set_xticklabels([c.replace("background", "bg ") for c in cats], rotation=30, ha="right")
        ax.set_title(title)
    axes[0].set_ylabel(f"Correct rate (IoU ≥ {IOU_SUCCESS})")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "fig_breakdown.png", dpi=200)
    plt.close(fig)


def fig_error(res, methods, col, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m in methods:
        axes[0].hist(res[(res.method == m) & res.detected].iou, bins=np.linspace(0, 1, 41),
                     histtype="step", lw=2, label=m, color=col[m])
    axes[0].set_xlabel("IoU (detected frames only)")
    axes[0].set_ylabel("Frames")
    axes[0].legend()
    axes[1].boxplot([res[(res.method == m) & res.detected].mean_err.to_numpy() for m in methods],
                    tick_labels=methods)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Mean corner error (px, log scale)")
    fig.tight_layout()
    fig.savefig(out / "fig_error_dist.png", dpi=200)
    plt.close(fig)


def fig_runtime(res, methods, col, out):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    means = [res[res.method == m].runtime_s.mean() * 1000 for m in methods]
    ax.bar(methods, means, color=[col[m] for m in methods])
    for i, v in enumerate(means):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    ax.set_ylabel("Mean runtime per frame (ms)")
    fig.tight_layout()
    fig.savefig(out / "fig_runtime.png", dpi=200)
    plt.close(fig)


def fig_failure_causes(res, methods, feats, tags, col, out):
    feat_names = [("contrast", "Border contrast"), ("sharpness", "Sharpness"),
                  ("bg_edge", "Background edge density"), ("skew", "Perspective tilt"),
                  ("area_frac", "Document area fraction")]
    labels = ["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    flat = axes.ravel()
    for ax, (f, title) in zip(flat, feat_names):
        q = pd.qcut(feats[f].rank(method="first"), 4, labels=labels)

        def fail_rate(m):
            g = res[res.method == m].set_index("row_index")
            return (~g.correct).groupby(q.loc[g.index], observed=False).mean().values
        grouped_bars(ax, methods, col, labels, fail_rate)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Failure rate")
    names = list(TAGS)

    def tag_share(m):
        g = res[res.method == m].set_index("row_index")
        failed = g.index[~g.correct]
        return [tags.loc[failed, t].mean() if len(failed) else 0 for t in names]
    grouped_bars(flat[5], methods, col, names, tag_share)
    flat[5].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    flat[5].set_title("Share of failed frames with each tag")
    flat[5].set_ylim(0, 1.05)
    flat[0].legend()
    fig.tight_layout()
    fig.savefig(out / "fig_failure_causes.png", dpi=200)
    plt.close(fig)


# ---------- qualitative examples ----------
def overlay(df, r):
    row = df.loc[r["row_index"]]
    img, scale = load_frame(row)
    out = img.copy()
    cv2.polylines(out, [np.round(gt_corners(row, scale)).astype(np.int32)], True, RED, 3)
    if r["detected"]:
        pred = np.array([[r[f"p{i}x"], r[f"p{i}y"]] for i in range(4)])
        cv2.polylines(out, [np.round(pred).astype(np.int32)], True, GREEN, 3)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def pick_diverse(indices, backgrounds, n, rng):
    # random frames, from different backgrounds where possible
    idx = list(rng.permutation(list(indices)))
    chosen, seen = [], set()
    for i in idx:
        if backgrounds[i] not in seen:
            chosen.append(i)
            seen.add(backgrounds[i])
        if len(chosen) == n:
            return chosen
    for i in idx:
        if i not in chosen:
            chosen.append(i)
        if len(chosen) == n:
            break
    return chosen


def qualitative(res, df, methods, out, per_cat=2):
    rng = np.random.default_rng(0)
    bgs = res.drop_duplicates("row_index").set_index("row_index")["background"]
    by = {m: res[res.method == m].set_index("row_index") for m in methods}
    if len(methods) >= 2:
        a, b = methods[:2]
        ca, cb = by[a]["correct"], by[b]["correct"]
        cats = [("Both correct", ca & cb), (f"Only {a} correct", ca & ~cb),
                (f"Only {b} correct", ~ca & cb), ("Both fail", ~ca & ~cb)]
        cols_m = [a, b]
    else:
        g = by[methods[0]]
        cats = [("Correct", g["correct"]), ("Wrong quadrilateral", g["detected"] & ~g["correct"]),
                ("No detection", ~g["detected"])]
        cols_m = methods
    cats = [(n, s) for n, s in cats if s.any()]
    ncols = per_cat * len(cols_m)
    fig, axes = plt.subplots(len(cats), ncols, figsize=(3.6 * ncols, 2.4 * len(cats)), squeeze=False)
    for r, (name, sel) in enumerate(cats):
        chosen = pick_diverse(sel.index[sel], bgs, per_cat, rng)
        for e in range(per_cat):
            for k, m in enumerate(cols_m):
                ax = axes[r, e * len(cols_m) + k]
                ax.axis("off")
                if e < len(chosen):
                    rr = by[m].loc[chosen[e]].copy()
                    rr["row_index"] = chosen[e]
                    ax.imshow(overlay(df, rr))
                    ax.set_title(f"{m}: IoU {rr['iou']:.2f}" if rr["detected"] else f"{m}: no detection", fontsize=8)
                if e == 0 and k == 0:
                    ax.text(-0.04, 0.5, name, transform=ax.transAxes, rotation=90,
                            va="center", ha="right", fontsize=9, fontweight="bold")
    fig.suptitle("Red = ground truth, green = prediction", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "qualitative.png", dpi=200)
    plt.close(fig)


def failure_gallery(res, df, methods, tags, out, n=8):
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(len(methods), n, figsize=(2.6 * n, 2.4 * len(methods)), squeeze=False)
    for r, m in enumerate(methods):
        g = res[res.method == m].set_index("row_index")
        failed = list(g.index[~g.correct])
        chosen = list(rng.permutation(failed)[:n]) if failed else []
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(chosen):
                rr = g.loc[chosen[c]].copy()
                rr["row_index"] = chosen[c]
                ax.imshow(overlay(df, rr))
                t = [k.replace("_", " ") for k in TAGS if tags.loc[chosen[c], k]]
                head = "no detection" if not rr["detected"] else f"IoU {rr['iou']:.2f}"
                ax.set_title(head + "\n" + (", ".join(t[:3]) or "no flag"), fontsize=7)
            if c == 0:
                ax.text(-0.05, 0.5, m, transform=ax.transAxes, rotation=90, va="center", ha="right",
                        fontsize=10, fontweight="bold")
    fig.suptitle("Random failure cases (red = ground truth, green = prediction)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "failure_gallery.png", dpi=200)
    plt.close(fig)


# ---------- text for the report ----------
def report_numbers(split, res, methods, main, byb, byd, paired_info, fsum, ftype):
    L = [f"# Report numbers ({split} split)", "",
         f"Frames: {res.row_index.nunique()}, videos: {res.video.nunique()}. "
         f"Correct = IoU >= {IOU_SUCCESS}. CIs: 95%, bootstrap over videos.", "",
         "## Main table", md_table(main), "", "## Draft sentences"]
    for m in methods:
        g = res[res.method == m]
        det = g[g.detected]
        c = boot_ci(g.correct.astype(float), g.video)
        L.append(f"- {m} detected a document in {g.detected.mean() * 100:.1f}% of frames and localized it correctly "
                 f"(IoU >= {IOU_SUCCESS}) in {c[0] * 100:.1f}% (95% CI {c[1] * 100:.1f}-{c[2] * 100:.1f}%). "
                 f"On detected frames, mean IoU was {det.iou.mean():.3f} and mean corner error {det.mean_err.mean():.2f} px. "
                 f"Mean runtime was {g.runtime_s.mean() * 1000:.1f} ms per frame.")
        bg = g.groupby("background")["correct"].mean()
        dt = g.groupby("doctype")["correct"].mean()
        L.append(f"- {m}: best background {bg.idxmax()} ({bg.max() * 100:.0f}% correct), worst {bg.idxmin()} "
                 f"({bg.min() * 100:.0f}%); best document type {dt.idxmax()} ({dt.max() * 100:.0f}%), worst {dt.idxmin()} ({dt.min() * 100:.0f}%).")
        f = fsum[fsum.method == m].sort_values("share_of_failures", ascending=False).head(3)
        ft = ftype[ftype.method == m].failure_type.value_counts()
        L.append(f"- {m} failures: {int(ft.get('no_detection', 0))} no-detection, {int(ft.get('wrong_quad', 0))} wrong-quadrilateral. "
                 f"Most common tags: " + ", ".join(f"{r.tag.replace('_', ' ')} ({r.share_of_failures * 100:.0f}% of failures)" for r in f.itertuples()) + ".")
    if paired_info:
        a, b, info = paired_info
        lo, hi = info["cd"][1], info["cd"][2]
        verdict = (f"{a} was better" if lo > 0 else f"{b} was better" if hi < 0
                   else "the difference is not clearly distinguishable from zero")
        L.append(f"- The methods differ on {info['only_a'] + info['only_b']} frames ({a} only: {info['only_a']}, "
                 f"{b} only: {info['only_b']}). Correct rate difference ({a} minus {b}): {info['cd'][0] * 100:+.1f} points "
                 f"(95% CI {lo * 100:+.1f} to {hi * 100:+.1f}), so {verdict}.")
    L += ["", "## By background", md_table(byb), "", "## By document type", md_table(byd)]
    return "\n".join(L)


# ---------- main ----------
def main():
    global BOOT_REPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--reuse", action="store_true", help="use saved per_frame.csv, do not rerun methods")
    ap.add_argument("--reps", type=int, default=BOOT_REPS, help="bootstrap repetitions")
    args = ap.parse_args()
    BOOT_REPS = args.reps

    base = Path("../results") / args.split
    out = base / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    csv = base / "per_frame.csv"

    if args.reuse and csv.exists():
        res = pd.read_csv(csv)
    else:
        if args.split == "test":
            print("WARNING: test split. Parameters should be frozen; run this once.")
        res = run_methods(args.split, get_methods())
        res.to_csv(csv, index=False)
    res["correct"] = res["iou"] >= IOU_SUCCESS

    methods = list(res.method.unique())
    col = colors(methods)
    df = load_split(args.split)
    print(f"{args.split}: {methods}, {res.row_index.nunique()} frames")

    main_t, sweep = main_table(res, methods)
    main_t.to_csv(out / "main_table.csv")
    (out / "main_table.md").write_text(md_table(main_t))
    (out / "main_table.tex").write_text(latex_table(main_t[main_t.columns[:5]], f"Results on the {args.split} split.", "tab:main"))
    sweep.to_csv(out / "threshold_sweep.csv")
    byb, byd = breakdown(res, methods, "background"), breakdown(res, methods, "doctype")
    for name, t in [("by_background", byb), ("by_doctype", byd)]:
        t.to_csv(out / f"{name}.csv")
        (out / f"{name}.md").write_text(md_table(t))

    paired_info = None
    if len(methods) >= 2:
        a, b = methods[:2]
        pt, info = paired_comparison(res, a, b)
        pt.to_csv(out / "paired_comparison.csv")
        (out / "paired_comparison.md").write_text(md_table(pt))
        paired_info = (a, b, info)

    feats = get_features(args.split, res, out)
    tags = tag_frames(feats)
    fsum, ftype = failure_analysis(res, methods, tags)
    fsum.to_csv(out / "failure_summary.csv", index=False)
    (out / "failure_summary.md").write_text(md_table(fsum, index=False))
    ftype.to_csv(out / "failure_tags.csv", index=False)

    fig_overall(res, methods, col, out)
    fig_threshold(res, methods, col, out)
    fig_breakdown(res, methods, col, out)
    fig_error(res, methods, col, out)
    fig_runtime(res, methods, col, out)
    fig_failure_causes(res, methods, feats, tags, col, out)
    qualitative(res, df, methods, out)
    failure_gallery(res, df, methods, tags, out)

    (out / "report_numbers.md").write_text(
        report_numbers(args.split, res, methods, main_t, byb, byd, paired_info, fsum, ftype))
    print(main_t.T.to_string())
    if paired_info:
        print("\n" + pt.T.to_string())
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
