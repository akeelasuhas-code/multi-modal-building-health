"""Synthetic concrete images with exact ground truth.

This is a *plumbing and regression* benchmark: it proves the maths is implemented
correctly (widths, lengths, false alarms). It does NOT replace evaluation on real
datasets (OmniCrack30k, CRACK500, DeepCrack) -- use evaluate.py --images/--masks for that.
"""
import cv2
import numpy as np


def concrete_texture(h=512, w=512, rng=None, gradient=True, pores=True):
    rng = rng or np.random.default_rng()
    base = np.full((h, w), 170.0, np.float32)
    for sigma, amp in ((1.0, 5.0), (3.0, 4.5), (8.0, 4.0), (20.0, 3.0)):
        n = rng.normal(0, 1, (h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        n /= n.std() + 1e-6
        base += amp * n
    if pores:
        for _ in range(rng.integers(40, 90)):
            cx, cy = rng.integers(0, w), rng.integers(0, h)
            r = int(rng.integers(1, 4))
            cv2.circle(base, (int(cx), int(cy)), r, float(170 - rng.uniform(15, 35)), -1)
    if gradient:
        gx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        gy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        ang = rng.uniform(0, 2 * np.pi)
        base += 18.0 * (np.cos(ang) * gx + np.sin(ang) * gy)
    return base


def add_crack(img, gt, rng, width_px, length_px=350, darkness=70.0):
    h, w = img.shape
    x, y = float(rng.integers(w // 6, 5 * w // 6)), float(rng.integers(h // 6, 5 * h // 6))
    theta = rng.uniform(0, np.pi)
    pts = [(x, y)]
    step = 6
    for _ in range(int(length_px / step)):
        theta += rng.normal(0, 0.18)
        x += step * np.cos(theta)
        y += step * np.sin(theta)
        pts.append((x, y))
    layer = np.zeros((h, w), np.uint8)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(layer, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 255, int(width_px), cv2.LINE_8)
    gt |= layer > 0
    soft = cv2.GaussianBlur(layer.astype(np.float32) / 255.0, (0, 0), 0.7)
    img -= darkness * soft
    return width_px


def add_blob(img, gt_blob, rng, radius=40, darkness=38.0):
    h, w = img.shape
    cx, cy = int(rng.integers(radius * 2, w - radius * 2)), int(rng.integers(radius * 2, h - radius * 2))
    layer = np.zeros((h, w), np.uint8)
    cv2.ellipse(layer, (cx, cy), (radius, int(radius * rng.uniform(0.6, 1.0))),
                float(rng.uniform(0, 180)), 0, 360, 255, -1)
    gt_blob |= layer > 0
    soft = cv2.GaussianBlur(layer.astype(np.float32) / 255.0, (0, 0), 2.5)
    img -= darkness * soft


def make_sample(seed, n_cracks=1, widths=(4,), blob=False, size=512, darkness=70.0, gradient=True, _tries=0):
    """Returns (bgr uint8, crack_gt bool, blob_gt bool, list_of_true_widths_px).
    Cracks that run off the image and end up as short stubs are regenerated (they are not realistic cracks)."""
    rng = np.random.default_rng(seed + 1000 * _tries)
    img = concrete_texture(size, size, rng, gradient=gradient)
    crack_gt = np.zeros((size, size), bool)
    blob_gt = np.zeros((size, size), bool)
    used = []
    for i in range(n_cracks):
        wpx = widths[i % len(widths)]
        used.append(add_crack(img, crack_gt, rng, wpx, darkness=darkness))
    if blob:
        add_blob(img, blob_gt, rng)
    if n_cracks and _tries < 20:
        from skimage.morphology import skeletonize
        if skeletonize(crack_gt).sum() < 0.55 * 350 * n_cracks:
            return make_sample(seed, n_cracks, widths, blob, size, darkness, gradient, _tries + 1)
    img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return bgr, crack_gt, blob_gt, used
