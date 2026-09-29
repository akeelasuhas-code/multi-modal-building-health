"""evaluate.py -- measure the crack segmentation against ground truth.

  python evaluate.py --synthetic            regression benchmark on generated concrete (proves the plumbing)
  python evaluate.py --images IMG_DIR --masks MASK_DIR [--size 256] [--limit 200] [--out results.csv]
                                            REAL evaluation on a labelled dataset (OmniCrack30k test subsets,
                                            CRACK500, DeepCrack, ...). This is the number to put in a paper.

The synthetic benchmark can only show that the code is correct (widths, lengths, no false alarms on clean
texture). It says nothing about accuracy on real photographs.

Published clIoU values (OmniCrack30k) use images resized to 256 x 256 and tolerance tau = 4 px, which is why
--size defaults to 256. Use --size 0 to evaluate at native resolution.
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

import bhas_core as core


def legacy_baseline(bgr, min_area=50):
    """The ORIGINAL method from the first prototype (adaptive threshold + contour area filter)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, np.uint8)
    defect = 0.0
    for c in cs:
        a = cv2.contourArea(c)
        if a > min_area:
            defect += a
            cv2.drawContours(mask, [c], -1, 1, -1)
    S = min(round(defect / gray.size * 100 * 8.5, 1), 100.0)
    return mask.astype(bool), S, cv2.countNonZero(th) / gray.size * 100


# ------------------------------------------------------------------ synthetic
def synthetic_benchmark(n=12, verbose=True):
    from synth import make_sample
    out = {}
    # 1. crack-free false alarms
    old_S, new_S, old_raw, new_area = [], [], [], []
    for seed in range(100, 100 + n):
        b, *_ = make_sample(seed, n_cracks=0)
        _, S, raw = legacy_baseline(b)
        vr = core.analyze_image(b)
        old_S.append(S); old_raw.append(raw)
        new_S.append(vr.severity["S"]); new_area.append(vr.measurement.crack_area_frac * 100)
    out["cleanwall"] = dict(old_S_mean=np.mean(old_S), old_raw_area_mean=np.mean(old_raw),
                            new_S_max=max(new_S), new_area_max=max(new_area))
    # 2. cracked: clIoU and width error vs the same measurement on the GT mask
    rows = []
    for wpx in (2, 4, 8, 12):
        old_c, new_c, werr, lratio = [], [], [], []
        for seed in range(10, 10 + n // 2):
            b, cg, _, _ = make_sample(seed, n_cracks=1, widths=(wpx,))
            lm, _, _ = legacy_baseline(b)
            vr = core.analyze_image(b)
            gm, _, _ = core.measure_cracks(cg, np.zeros_like(cg), None, core.VisionParams())
            old_c.append(core.cl_iou(lm, cg)); new_c.append(core.cl_iou(vr.crack, cg))
            werr.append(vr.measurement.width_px_p95 - gm.width_px_p95)
            lratio.append(vr.measurement.length_px / gm.length_px)
        rows.append(dict(width=wpx, old_clIoU=np.mean(old_c), new_clIoU=np.mean(new_c),
                         width_err_px=np.mean(werr), length_ratio=np.mean(lratio)))
    out["cracked"] = rows
    # 3. faint cracks
    fa = []
    for dk in (20, 25, 30, 45):
        ok = 0
        for seed in range(20, 26):
            b, cg, _, _ = make_sample(seed, n_cracks=1, widths=(4,), darkness=dk)
            ok += core.cl_iou(core.analyze_image(b).crack, cg) > 0.5
        fa.append((dk, ok, 6))
    out["faint"] = fa
    # 4. hard shadow
    sh = []
    for seed in range(40, 44):
        b, *_ = make_sample(seed, n_cracks=0)
        b = b.astype(np.float32); b[:, 200:300] *= 0.65
        sh.append(core.analyze_image(np.clip(b, 0, 255).astype(np.uint8)).severity["S"])
    out["shadow_S_max"] = max(sh)
    if verbose:
        c = out["cleanwall"]
        print("CRACK-FREE CONCRETE (should read ~0):")
        print(f"  old method: mean S = {c['old_S_mean']:.1f}, raw 'defect area' = {c['old_raw_area_mean']:.1f}%")
        print(f"  new method: max  S = {c['new_S_max']:.1f}, max crack area = {c['new_area_max']:.3f}%")
        print("CRACKED (clIoU, tolerance 4 px; width error vs same measurement on ground-truth mask):")
        for r in rows:
            print(f"  {r['width']:2d}px: old {r['old_clIoU']:.2f} -> new {r['new_clIoU']:.2f} | width err {r['width_err_px']:+.1f}px | length ratio {r['length_ratio']:.2f}")
        print("FAINT CRACKS (4 px wide) detected:", ", ".join(f"{d} gray levels: {o}/{t}" for d, o, t in fa))
        print(f"HARD SHADOW ON CLEAN WALL: max S = {out['shadow_S_max']:.1f}")
    return out


# ------------------------------------------------------------------ real datasets
def _find_mask(mask_dir, stem):
    for suf in ("", "_mask", "_gt", "_label", "_lab", "-mask", "_seg"):
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif"):
            p = os.path.join(mask_dir, stem + suf + ext)
            if os.path.exists(p):
                return p
    return None


def evaluate_folder(img_dir, mask_dir, size=256, limit=None, out_csv=None, tau=4, compare_legacy=True):
    import csv
    paths = sorted(p for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp") for p in glob.glob(os.path.join(img_dir, e)))
    rows = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        mp = _find_mask(mask_dir, stem)
        if mp is None:
            continue
        im, mk = cv2.imread(p, cv2.IMREAD_COLOR), cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if im is None or mk is None:
            continue
        if size:
            im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
            mk = cv2.resize(mk, (size, size), interpolation=cv2.INTER_NEAREST)
        vr = core.analyze_image(im)
        gt = cv2.resize((mk > 127).astype(np.uint8), (vr.crack.shape[1], vr.crack.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        ps = core.pixel_scores(vr.crack, gt)
        row = dict(image=stem, clIoU=core.cl_iou(vr.crack, gt, tau), clDice=core.cl_dice(vr.crack, gt),
                   iou=ps["iou"], f1=ps["f1"], gt_has_crack=int(gt.any()))
        if compare_legacy:
            lm, _, _ = legacy_baseline(cv2.resize(im, (vr.crack.shape[1], vr.crack.shape[0])))
            row["legacy_clIoU"] = core.cl_iou(lm, gt, tau)
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    if not rows:
        raise SystemExit("No image/mask pairs found (masks must share the image file stem).")
    keys = [k for k in rows[0] if k not in ("image", "gt_has_crack")]
    print(f"\n{len(rows)} image/mask pairs, size={size or 'native'}, tau={tau}px")
    pos = [r for r in rows if r["gt_has_crack"]]
    for k in keys:
        s = f"  {k:14s} mean over all = {np.mean([r[k] for r in rows]):.3f}"
        if pos:
            s += f" | over images that contain cracks = {np.mean([r[k] for r in pos]):.3f}"
        print(s)
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print("per-image results written to", out_csv)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--images"); ap.add_argument("--masks")
    ap.add_argument("--size", type=int, default=256); ap.add_argument("--limit", type=int)
    ap.add_argument("--tau", type=int, default=4); ap.add_argument("--out")
    a = ap.parse_args()
    if a.synthetic:
        synthetic_benchmark()
    elif a.images and a.masks:
        evaluate_folder(a.images, a.masks, a.size, a.limit, a.out, a.tau)
    else:
        ap.print_help()
