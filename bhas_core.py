"""bhas_core.py -- computation engine for the Building Health Auditing System.

Everything numeric lives here (no Streamlit, no Earth Engine) so it can be unit-tested.

Pipeline
  1. segment_defects   : illumination flattening + black top-hat + hysteresis + shape filter
  2. measure_cracks    : skeleton, distance-transform width, weighted length, density
  3. compute_severity  : S = 100 * sum_i a_i * min(1, f_i / f_crit_i)
  4. fit_insar         : d(t) = c + v t + a sin(2 pi t) + b cos(2 pi t); v_vert = v_LOS / cos(theta)
  5. estimate_construction_year : step-change detector on annual NDBI/NDVI
  6. fuse              : weighted risk over the modalities that are actually available
  7. cl_iou / cl_dice  : evaluation metrics for thin structures

PROVISIONAL constants (marked below) are engineering defaults, not fitted values.
Calibrate them against expert grades before publishing any accuracy claim.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label
from skimage.morphology import skeletonize

SEVERITY_ORDER = ["STRUCTURALLY SOUND", "MONITORING RECOMMENDED", "CRITICAL ACTION REQUIRED"]


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
@dataclass
class VisionParams:
    max_side: int = 1600            # long side is capped here for speed (scale is tracked)
    contrast_min: float = 14.0      # absolute floor (gray levels): a crack must be at least this dark
    k_noise: float = 5.0            # ...and exceed k * robust texture noise of the top-hat response (tuned on synthetic texture; re-tune on real photos)
    hyst_low_ratio: float = 0.6     # low hysteresis threshold = ratio * high
    se_frac: float = 0.03           # top-hat structuring element as fraction of long side
    min_len_frac: float = 0.06      # min skeleton length (fraction of long side) to count as a crack
    min_elongation: float = 8.0     # skeleton length / mean width; compact blobs fail this
    blob_contrast: float = 22.0     # gray levels below flattened background for a dark anomaly
    blob_min_area_frac: float = 0.004
    prune_frac: float = 0.012       # spur pruning length as fraction of long side


@dataclass
class SeverityParams:
    # PROVISIONAL: w_crit is a commonly cited serviceability crack width; rho_crit and a_crit
    # have no code basis. Verify with your faculty / IS 456 / ACI 224R and calibrate.
    w_crit_mm: float = 0.3
    rho_crit_m_per_m2: float = 1.0
    a_crit: float = 0.10
    px_area_crit: float = 0.01      # uncalibrated fallback: crack-pixel fraction treated as critical
    weights: tuple = (0.5, 0.25, 0.25)   # crack width, crack density, dark-anomaly area
    sat_mult: float = 2.0           # a component reaches its maximum at sat_mult x its critical value
    sound_max: float = 10.0
    monitor_max: float = 35.0


@dataclass
class FusionParams:
    # PROVISIONAL, not fitted. Use fit_logistic() once you have expert labels.
    w_vision: float = 0.60
    w_age: float = 0.15
    w_insar: float = 0.25
    age_ref_years: float = 50.0
    v_crit_mm_yr: float = 10.0
    sound_max: float = 0.10
    monitor_max: float = 0.35


# ----------------------------------------------------------------------------
# Physical scale
# ----------------------------------------------------------------------------
def gsd_from_camera(distance_mm: float, focal_mm: float, sensor_width_mm: float, image_width_px: float) -> float:
    """GSD (mm/px) = D * sensor_width / (f * image_width_px). Assumes the camera is square-on to the wall."""
    return distance_mm * sensor_width_mm / (focal_mm * image_width_px)


def gsd_from_reference(ref_length_mm: float, ref_length_px: float) -> float:
    return ref_length_mm / ref_length_px


# ----------------------------------------------------------------------------
# Segmentation
# ----------------------------------------------------------------------------
def _odd(n: float, lo: int = 3) -> int:
    n = max(lo, int(round(n)))
    return n if n % 2 == 1 else n + 1


def load_and_scale(bgr: np.ndarray, max_side: int):
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return bgr, s


@dataclass
class _Response:
    gray: np.ndarray
    flat: np.ndarray
    tophat: np.ndarray
    th_med: float
    th_sigma: float


def _mad_sigma(a: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def compute_response(bgr: np.ndarray, p: VisionParams) -> _Response:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    long_side = max(h, w)

    # illumination flattening (removes gradients and soft shadows)
    # (background is smooth, so it is computed at quarter resolution and upsampled: same result, ~10x faster)
    q = 4 if long_side >= 800 else 1
    small = cv2.resize(gray, (max(2, w // q), max(2, h // q)), interpolation=cv2.INTER_AREA) if q > 1 else gray
    bg_s = cv2.GaussianBlur(small, (0, 0), 0.15 * min(h, w) / q)
    bg = cv2.resize(bg_s, (w, h), interpolation=cv2.INTER_LINEAR) if q > 1 else bg_s
    flat = gray - bg
    flat -= np.median(flat)

    # multi-scale black top-hat: closing removes dark structures narrower than the SE; the difference keeps them.
    # Two scales so both hairlines and wide cracks respond (a crack near the SE size would be attenuated).
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    tophat = None
    for frac in (p.se_frac, 2.6 * p.se_frac):
        se = _odd(frac * long_side, 9)
        pad = se
        gp = cv2.copyMakeBorder(g, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (se, se))   # rectangular SE: separable, fast
        th = (cv2.morphologyEx(gp, cv2.MORPH_CLOSE, kernel) - gp)[pad:-pad, pad:-pad]
        tophat = th if tophat is None else np.maximum(tophat, th)
    return _Response(gray, flat, tophat, float(np.median(tophat)), _mad_sigma(tophat))


def _remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components smaller than min_size (independent of scikit-image version)."""
    lab = label(mask, connectivity=2)
    if lab.max() == 0:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.int32)
    return ndi.convolve(skel.astype(np.int32), k, mode="constant") - skel.astype(np.int32)


def prune_spurs(skel: np.ndarray, min_len: int) -> np.ndarray:
    """Remove side branches shorter than min_len that end at a junction. Isolated segments are kept."""
    skel = skel.copy()
    H, W = skel.shape
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for _ in range(2):
        nb = _neighbour_count(skel)
        endpoints = np.argwhere(skel & (nb == 1))
        removed_any = False
        for y, x in endpoints:
            if not skel[y, x]:
                continue
            path, visited, cur, hit_junction = [(y, x)], {(y, x)}, (y, x), False
            while len(path) <= min_len:
                cand = []
                for dy, dx in nbrs:
                    ny, nx = cur[0] + dy, cur[1] + dx
                    if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] and (ny, nx) not in visited:
                        cand.append((ny, nx))
                if not cand:
                    break
                if len(cand) > 1 or nb[cand[0]] >= 3:
                    hit_junction = True
                    break
                cur = cand[0]
                visited.add(cur)
                path.append(cur)
            if hit_junction and len(path) <= min_len:
                for py, px_ in path:
                    skel[py, px_] = False
                removed_any = True
        if not removed_any:
            break
    return skeletonize(skel)


def skeleton_length(skel: np.ndarray) -> float:
    """Arc length in pixels: orthogonal links weigh 1, diagonal links sqrt(2).
    A diagonal link is counted only when it is not already bridged orthogonally (no double counting)."""
    s = skel.astype(bool)
    H, W = s.shape
    total = 0.0
    orth = (s[:, :-1] & s[:, 1:]).sum() + (s[:-1, :] & s[1:, :]).sum()
    total += float(orth)
    # diagonal down-right: (y,x)-(y+1,x+1); bridged if (y,x+1) or (y+1,x) set
    d1 = s[:-1, :-1] & s[1:, 1:] & ~(s[:-1, 1:] | s[1:, :-1])
    # diagonal down-left: (y,x+1)-(y+1,x); bridged if (y,x) or (y+1,x+1) set
    d2 = s[:-1, 1:] & s[1:, :-1] & ~(s[:-1, :-1] | s[1:, 1:])
    total += math.sqrt(2.0) * float(d1.sum() + d2.sum())
    return total


def _half_max_refine(crack: np.ndarray, tophat: np.ndarray) -> np.ndarray:
    """Keep a pixel only if its top-hat response is >= 50% of the ridge peak at its nearest skeleton point
    (full-width-at-half-maximum criterion). Removes the blur/hysteresis halo that would otherwise inflate
    widths. Tolerance-band width error is therefore ~1 px instead of ~3 px."""
    skel = skeletonize(crack)
    if not skel.any():
        return crack
    peak_img = ndi.maximum_filter(tophat, size=3)
    _, (iy, ix) = ndi.distance_transform_edt(~skel, return_indices=True)
    peak = peak_img[iy, ix]
    refined = crack & (tophat >= 0.5 * peak)
    return refined if refined.any() else crack


def _extract(resp: _Response, p: VisionParams, scale: float):
    """Threshold + shape filtering. scale multiplies the contrast thresholds (used for the uncertainty band)."""
    h, w = resp.gray.shape
    long_side = max(h, w)
    t_hi = max(p.contrast_min, resp.th_med + p.k_noise * resp.th_sigma) * scale
    t_lo = p.hyst_low_ratio * t_hi
    cand = apply_hysteresis_threshold(resp.tophat, t_lo, t_hi)
    cand = _remove_small(cand, max(12, int(0.00004 * h * w)))

    skel_all = skeletonize(cand)
    lab = label(cand, connectivity=2)
    n = lab.max()
    crack = np.zeros_like(cand)
    if n > 0:
        area = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64)
        sk_len = np.bincount(lab[skel_all].ravel(), minlength=n + 1).astype(np.float64)
        min_len = p.min_len_frac * long_side
        keep = np.zeros(n + 1, bool)
        for i in range(1, n + 1):
            if sk_len[i] >= min_len and sk_len[i] > 0 and (sk_len[i] / max(1.0, area[i] / sk_len[i])) >= p.min_elongation:
                keep[i] = True
        crack = keep[lab]

    if crack.any():
        crack = _half_max_refine(crack, resp.tophat)

    # dark anomalies (possible spalling / moisture stains), excluding crack pixels
    flat_s = cv2.GaussianBlur(resp.flat, (0, 0), 2.0)
    dark = flat_s < -(p.blob_contrast * scale)
    dark &= ~cv2.dilate(crack.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)
    dark = _remove_small(dark, max(30, int(p.blob_min_area_frac * h * w)))

    # band-like dark regions that span (almost) the whole image are almost always shadows, not defects
    lab_d = label(dark, connectivity=2)
    shadow = np.zeros_like(dark)
    for i, sl in enumerate(ndi.find_objects(lab_d), start=1):
        if sl is None:
            continue
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if hh >= 0.9 * h or ww >= 0.9 * w:
            shadow |= lab_d == i
    dark &= ~shadow
    return crack, dark, float(shadow.sum() / (h * w))


# ----------------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------------
@dataclass
class CrackMeasurement:
    calibrated: bool
    gsd_mm_per_px: Optional[float]
    n_pixels: int
    crack_area_frac: float
    blob_area_frac: float
    length_px: float
    width_px_max: float
    width_px_p95: float
    width_px_mean: float
    length_m: Optional[float]
    width_mm_max: Optional[float]
    width_mm_p95: Optional[float]
    width_mm_mean: Optional[float]
    density_m_per_m2: Optional[float]
    density_px_per_px2: float
    image_area_m2: Optional[float] = None
    blob_area_m2: Optional[float] = None


def measure_cracks(crack: np.ndarray, blob: np.ndarray, gsd_mm_per_px: Optional[float], p: VisionParams):
    h, w = crack.shape
    long_side = max(h, w)
    if crack.any():
        skel0 = skeletonize(crack)
        skel = prune_spurs(skel0, max(4, int(p.prune_frac * long_side)))
        if not skel.any():
            skel = skel0
        dt = ndi.distance_transform_edt(crack)
        widths = 2.0 * dt[skel] - 1.0          # exact for odd-thickness lines
        widths = np.clip(widths, 1.0, None)
        L = skeleton_length(skel)
        w_max, w95, wmean = float(widths.max()), float(np.percentile(widths, 95)), float(widths.mean())
    else:
        skel, dt = np.zeros_like(crack), np.zeros(crack.shape, np.float32)
        L, w_max, w95, wmean = 0.0, 0.0, 0.0, 0.0
    A = float(h * w)
    m = CrackMeasurement(
        calibrated=gsd_mm_per_px is not None, gsd_mm_per_px=gsd_mm_per_px,
        n_pixels=int(crack.sum()), crack_area_frac=float(crack.sum() / A), blob_area_frac=float(blob.sum() / A),
        length_px=L, width_px_max=w_max, width_px_p95=w95, width_px_mean=wmean,
        length_m=None, width_mm_max=None, width_mm_p95=None, width_mm_mean=None,
        density_m_per_m2=None, density_px_per_px2=L / A,
    )
    if gsd_mm_per_px:
        g = gsd_mm_per_px
        m.length_m = L * g / 1000.0
        m.width_mm_max, m.width_mm_p95, m.width_mm_mean = w_max * g, w95 * g, wmean * g
        m.density_m_per_m2 = 1000.0 * L / (A * g)     # (L*g/1000 m) / (A*g^2/1e6 m^2)
        m.image_area_m2 = A * g * g / 1e6
        m.blob_area_m2 = float(blob.sum()) * g * g / 1e6
    return m, skel, dt


# ----------------------------------------------------------------------------
# Severity
# ----------------------------------------------------------------------------
def status_from_score(score: float, lo: float, hi: float) -> str:
    return SEVERITY_ORDER[0] if score <= lo else SEVERITY_ORDER[1] if score <= hi else SEVERITY_ORDER[2]


def compute_severity(m: CrackMeasurement, sp: SeverityParams) -> dict:
    """S = 100 * sum_i a_i * min(1, f_i / (m * f_crit_i)),  m = sat_mult = 2
    (a component is half-way to its maximum AT the critical value and saturates at 2x, so a crack far beyond the
    limit still scores worse than one just over it). Weights are renormalised to the components available."""
    a1, a2, a3 = sp.weights
    if m.calibrated:
        f = [m.width_mm_p95 / sp.w_crit_mm, m.density_m_per_m2 / sp.rho_crit_m_per_m2, m.blob_area_frac / sp.a_crit]
        w = [a1, a2, a3]
        basis = "calibrated (mm)"
    else:
        f = [m.crack_area_frac / sp.px_area_crit, m.blob_area_frac / sp.a_crit]
        w = [a1 + a2, a3]
        basis = "UNCALIBRATED (pixel-relative; provide a scale for mm-based severity)"
    w = np.array(w) / sum(w)
    f = np.array(f) / sp.sat_mult
    S = 100.0 * float(np.sum(w * np.minimum(1.0, f)))
    return {"S": S, "status": status_from_score(S, sp.sound_max, sp.monitor_max), "basis": basis,
            "components": [float(min(1.0, x)) for x in f]}


# ----------------------------------------------------------------------------
# Visual outputs
# ----------------------------------------------------------------------------
def render_outputs(bgr, crack, blob, skel, dt, wcrit_px=None):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    overlay[blob] = (0.55 * overlay[blob] + 0.45 * np.array([255, 165, 0])).astype(np.uint8)
    overlay[crack] = (0.45 * overlay[crack] + 0.55 * np.array([255, 30, 30])).astype(np.uint8)
    width_map = rgb.copy()
    width_map = (0.55 * width_map).astype(np.uint8)
    if skel.any():
        wpx = np.where(skel, 2 * dt - 1, 0)
        # calibrated: colour against the crack-width limit (red = at/above 1.5x limit); else normalise to the widest
        ref = 1.5 * wcrit_px if wcrit_px else max(1.0, float(wpx.max()))
        norm = np.clip(wpx / ref, 0, 1)
        col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        col = cv2.cvtColor(col, cv2.COLOR_BGR2RGB)
        thick = cv2.dilate(skel.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        colD = cv2.dilate(col, np.ones((3, 3), np.uint8))
        width_map[thick] = colD[thick]
    return overlay, width_map


# ----------------------------------------------------------------------------
# Full vision analysis
# ----------------------------------------------------------------------------
@dataclass
class VisionResult:
    measurement: CrackMeasurement
    severity: dict
    S_low: float
    S_high: float
    scale_factor: float
    shadow_frac: float
    crack: np.ndarray = field(repr=False, default=None)
    blob: np.ndarray = field(repr=False, default=None)
    skeleton: np.ndarray = field(repr=False, default=None)
    overlay: np.ndarray = field(repr=False, default=None)
    width_map: np.ndarray = field(repr=False, default=None)
    original_rgb: np.ndarray = field(repr=False, default=None)
    width_profile_px: np.ndarray = field(repr=False, default=None)


def analyze_image(bgr_original: np.ndarray, gsd_mm_per_px_original: Optional[float] = None,
                  vp: VisionParams = VisionParams(), sp: SeverityParams = SeverityParams(),
                  band_scales=(1.25, 1.0, 0.8)) -> VisionResult:
    bgr, s = load_and_scale(bgr_original, vp.max_side)
    gsd = None if gsd_mm_per_px_original is None else gsd_mm_per_px_original / s   # mm per *processed* px
    resp = compute_response(bgr, vp)

    Ss = []
    central = None
    for sc in band_scales:
        crack, blob, shadow_frac = _extract(resp, vp, sc)
        m, skel, dt = measure_cracks(crack, blob, gsd, vp)
        sev = compute_severity(m, sp)
        Ss.append(sev["S"])
        if sc == 1.0:
            central = (crack, blob, m, skel, dt, sev, shadow_frac)
    crack, blob, m, skel, dt, sev, shadow_frac = central
    wcrit_px = (sp.w_crit_mm / gsd) if gsd else None
    overlay, width_map = render_outputs(bgr, crack, blob, skel, dt, wcrit_px)
    prof = (2 * dt[skel] - 1) if skel.any() else np.array([])
    return VisionResult(m, sev, min(Ss), max(Ss), s, shadow_frac, crack, blob, skel, overlay, width_map,
                        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), prof)


# ----------------------------------------------------------------------------
# InSAR
# ----------------------------------------------------------------------------
def fit_insar(t_years, d_mm, incidence_deg: Optional[float] = None, seasonal: bool = True) -> dict:
    """OLS fit of d(t) = c + v t [+ a sin(2 pi t) + b cos(2 pi t)]; t in decimal years, d in mm (LOS).
    v_vert = v_LOS / cos(theta) assumes purely vertical motion. Negative = away from satellite (subsidence);
    check the sign convention of the tool that produced your data. SE assumes independent residuals
    and is therefore optimistic for real InSAR series."""
    t = np.asarray(t_years, float)
    d = np.asarray(d_mm, float)
    ok = np.isfinite(t) & np.isfinite(d)
    t, d = t[ok], d[ok]
    n = len(t)
    if n < 4 or np.ptp(t) <= 0:
        raise ValueError("Need at least 4 valid, distinct epochs.")
    use_seas = seasonal and np.ptp(t) >= 1.5 and n >= 8
    cols = [np.ones(n), t - t.mean()]
    if use_seas:
        cols += [np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, d, rcond=None)
    resid = d - X @ beta
    dof = max(1, n - X.shape[1])
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    v, se = float(beta[1]), float(math.sqrt(cov[1, 1]))
    ss_tot = float(((d - d.mean()) ** 2).sum())
    out = {"v_los_mm_yr": v, "v_los_se": se, "n_epochs": int(n), "seasonal_used": bool(use_seas),
           "seasonal_amp_mm": float(math.hypot(beta[2], beta[3])) if use_seas else 0.0,
           "r2": 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan"),
           "rmse_mm": math.sqrt(float(resid @ resid) / n), "v_vert_mm_yr": None, "v_vert_se": None,
           "fit_t": t, "fit_d": d, "fit_curve": X @ beta}
    if incidence_deg:
        c = math.cos(math.radians(incidence_deg))
        out["v_vert_mm_yr"], out["v_vert_se"] = v / c, se / c
    return out


def parse_insar_csv(df) -> tuple:
    """Accepts a DataFrame with a date column and a displacement (mm) column."""
    import pandas as pd
    cols = {c.lower().strip(): c for c in df.columns}
    dcol = next((cols[k] for k in cols if "date" in k), df.columns[0])
    vcol = next((cols[k] for k in cols if "disp" in k or "mm" in k), df.columns[1])
    dt = pd.to_datetime(df[dcol], errors="coerce")
    t = dt.dt.year + (dt.dt.dayofyear - 1) / 365.25
    return t.to_numpy(float), pd.to_numeric(df[vcol], errors="coerce").to_numpy(float)


# ----------------------------------------------------------------------------
# Building age: step-change detector on annual composites
# ----------------------------------------------------------------------------
def _wmean(a, lo, hi, min_valid):
    seg = a[lo:hi]
    seg = seg[np.isfinite(seg)]
    return float(seg.mean()) if len(seg) >= min_valid else float("nan")


def estimate_construction_year(years, ndvi, ndbi, k: int = 3, min_valid: int = 2, tau: float = 0.15,
                               z_min: float = 3.0, current_year: Optional[int] = None) -> dict:
    """For each candidate year s:
         Delta(s) = [mean NDBI(s..s+k-1) - mean NDBI(s-k..s-1)] - [mean NDVI(s..s+k-1) - mean NDVI(s-k..s-1)]
       year_hat = argmax_s Delta(s), accepted only if Delta_max >= tau and its robust z-score >= z_min.
       'year' is the first composite year showing the built-up signal."""
    import datetime
    cy = current_year or datetime.date.today().year
    years = np.asarray(years, int)
    ndvi = np.asarray(ndvi, float)
    ndbi = np.asarray(ndbi, float)
    order = np.argsort(years)
    years, ndvi, ndbi = years[order], ndvi[order], ndbi[order]
    n = len(years)
    deltas = np.full(n, np.nan)
    for i in range(k, n - k + 1):
        b = _wmean(ndbi, i - k, i, min_valid), _wmean(ndbi, i, i + k, min_valid)
        v = _wmean(ndvi, i - k, i, min_valid), _wmean(ndvi, i, i + k, min_valid)
        if all(np.isfinite([*b, *v])):
            deltas[i] = (b[1] - b[0]) - (v[1] - v[0])
    valid = np.isfinite(deltas)
    res = {"status": "inconclusive", "year": None, "year_range": None, "age": None, "age_range": None,
           "delta_max": None, "z": None, "years": years, "delta": deltas, "ndvi": ndvi, "ndbi": ndbi,
           "message": ""}
    if valid.sum() < 5:
        res["message"] = "Too few valid annual composites (cloud cover or short record)."
        return res
    i_star = int(np.nanargmax(deltas))
    d_max = float(deltas[i_star])
    sigma = max(_mad_sigma(deltas[valid]), 0.02)
    z = (d_max - float(np.nanmedian(deltas))) / sigma
    res.update(delta_max=d_max, z=float(z))
    if d_max >= tau and z >= z_min:
        near = np.where(valid & (deltas >= 0.9 * d_max))[0]
        y = int(years[i_star])
        res.update(status="detected", year=y, year_range=(int(years[near.min()]), int(years[near.max()])),
                   age=cy - y, age_range=(cy - int(years[near.max()]), cy - int(years[near.min()])),
                   message="Built-up transition detected in the annual record.")
    else:
        res["message"] = ("No clear construction signal in the record. The building probably predates it, was built "
                          "in the last few years, or is too small for 30 m pixels. Age is NOT estimated.")
    return res


# ----------------------------------------------------------------------------
# Fusion
# ----------------------------------------------------------------------------
def fuse(S: float, vision_status: str, age_years: Optional[float], v_vert_mm_yr: Optional[float],
         fp: FusionParams = FusionParams()) -> dict:
    """R = sum_i w_i c_i / sum_i w_i over the modalities actually available (no mock inputs).
       c_vision = S/100, c_age = min(1, age/age_ref), c_insar = min(1, |v_vert|/v_crit).
       Final status = the more severe of the vision-only status and the fused status."""
    comps, weights = {"vision": S / 100.0}, {"vision": fp.w_vision}
    if age_years is not None:
        comps["age"], weights["age"] = min(1.0, max(0.0, age_years / fp.age_ref_years)), fp.w_age
    if v_vert_mm_yr is not None:
        comps["insar"], weights["insar"] = min(1.0, abs(v_vert_mm_yr) / fp.v_crit_mm_yr), fp.w_insar
    R = sum(weights[k] * comps[k] for k in comps) / sum(weights.values())
    fused_status = status_from_score(R, fp.sound_max, fp.monitor_max)
    final = max(vision_status, fused_status, key=SEVERITY_ORDER.index)
    return {"R": float(R), "components": comps, "weights": weights, "fused_status": fused_status,
            "final_status": final, "modalities": list(comps)}


def fit_logistic(X, y, l2: float = 1e-2, iters: int = 5000, lr: float = 0.5):
    """Tiny numpy logistic regression for fitting fusion weights on expert labels (0/1). Returns (beta, mu, sd)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ beta))
        g = Z.T @ (p - y) / len(y) + l2 * np.r_[0, beta[1:]]
        beta -= lr * g
    return beta, mu, sd


def auroc(scores, labels) -> float:
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    cmp = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return float(cmp)


# ----------------------------------------------------------------------------
# Evaluation metrics for thin structures (OmniCrack30k definition of clIoU)
# ----------------------------------------------------------------------------
def _dilate(mask: np.ndarray, tau: int) -> np.ndarray:
    if tau <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tau + 1, 2 * tau + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def cl_iou(pred: np.ndarray, gt: np.ndarray, tau: int = 4) -> float:
    """clIoU_tau = |TP| / (|TP| + |FP| + |FN|) on skeletons with tolerance radius tau.
       TP = S_T & dilate(S_P);  FP = S_P & ~dilate(S_T);  FN = S_T & ~dilate(S_P)."""
    Sp, St = skeletonize(pred.astype(bool)), skeletonize(gt.astype(bool))
    if not Sp.any() and not St.any():
        return 1.0
    tp = int((St & _dilate(Sp, tau)).sum())
    fp = int((Sp & ~_dilate(St, tau)).sum())
    fn = int((St & ~_dilate(Sp, tau)).sum())
    d = tp + fp + fn
    return tp / d if d else 1.0


def cl_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """clDice = 2 Tprec Tsens / (Tprec + Tsens); Tprec = |S_P & M_T|/|S_P|, Tsens = |S_T & M_P|/|S_T|."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    Sp, St = skeletonize(pred), skeletonize(gt)
    if not Sp.any() and not St.any():
        return 1.0
    if not Sp.any() or not St.any():
        return 0.0
    tp_ = (Sp & gt).sum() / Sp.sum()
    ts_ = (St & pred).sum() / St.sum()
    return float(2 * tp_ * ts_ / (tp_ + ts_)) if (tp_ + ts_) > 0 else 0.0


def pixel_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return {"iou": iou, "f1": f1}


def result_to_record(vr: VisionResult) -> dict:
    d = asdict(vr.measurement)
    d.update(S=vr.severity["S"], S_low=vr.S_low, S_high=vr.S_high, shadow_frac=vr.shadow_frac, vision_status=vr.severity["status"],
             severity_basis=vr.severity["basis"])
    return d
