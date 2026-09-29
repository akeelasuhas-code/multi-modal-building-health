"""bhas_pipeline.py -- one function that runs a full audit and returns everything the UI/report needs.

Modalities that are not supplied are simply omitted (never mocked); fusion renormalises over what exists.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Callable, Optional

import bhas_core as core


@dataclass
class AuditInputs:
    name: str = ""
    is_public: bool = False
    lat: float = 17.4065
    lon: float = 78.4772
    radius_m: float = 45.0
    gsd_mm_px: Optional[float] = None          # mm per pixel of the ORIGINAL image; None = uncalibrated
    want_age: bool = False
    insar_series: Optional[tuple] = None       # (t_years, d_mm)
    insar_velocity: Optional[float] = None     # direct LOS velocity mm/yr (alternative to a series)
    insar_incidence_deg: Optional[float] = None
    age_k: int = 3
    age_tau: float = 0.15
    age_z_min: float = 3.0


def insar_from_velocity(v_los: float, incidence_deg: Optional[float]) -> dict:
    out = {"v_los_mm_yr": float(v_los), "v_los_se": None, "n_epochs": None, "seasonal_used": False,
           "seasonal_amp_mm": None, "r2": None, "rmse_mm": None, "v_vert_mm_yr": None, "v_vert_se": None,
           "fit_t": None, "fit_d": None, "fit_curve": None}
    if incidence_deg:
        out["v_vert_mm_yr"] = v_los / math.cos(math.radians(incidence_deg))
    return out


def run_audit(bgr, inp: AuditInputs, vp: core.VisionParams = None, sp: core.SeverityParams = None,
              fp: core.FusionParams = None, age_fetcher: Optional[Callable] = None,
              now: Optional[dt.datetime] = None) -> dict:
    vp, sp, fp = vp or core.VisionParams(), sp or core.SeverityParams(), fp or core.FusionParams()
    now = now or dt.datetime.now()
    warnings = []

    # 1. vision
    vr = core.analyze_image(bgr, inp.gsd_mm_px, vp, sp)
    m = vr.measurement
    if not m.calibrated:
        warnings.append("No image scale supplied: severity is pixel-relative only and crack widths are in pixels. "
                        "Provide mm/pixel (or camera geometry, or a reference object) for millimetre results.")
    else:
        min_w = 3.0 * m.gsd_mm_per_px
        if min_w > sp.w_crit_mm:
            warnings.append(f"Resolution too coarse: the smallest reliably measurable crack is about 3 pixels = "
                            f"{min_w:.2f} mm, which is larger than the {sp.w_crit_mm:g} mm limit. Photograph closer "
                            f"(target <= {sp.w_crit_mm / 3:.2f} mm/pixel) before trusting a 'sound' result.")
    if m.crack_area_frac > 0.05 or m.blob_area_frac > 0.25:
        warnings.insert(0, f"Unusually large defect coverage (cracks {m.crack_area_frac * 100:.1f}% and dark anomalies "
                           f"{m.blob_area_frac * 100:.0f}% of the image; real cracked walls rarely exceed about 5% crack area). "
                           "This usually means the photo contains objects, edges, vegetation, people or hard shadows rather "
                           "than only a wall surface. Retake a close, square-on photo of the surface before relying on this result.")
    if vr.shadow_frac > 0.02:
        warnings.append(f"A band-like dark region covering {vr.shadow_frac * 100:.0f}% of the image was ignored as a probable shadow.")
    if vr.S_high - vr.S_low > 20:
        warnings.append(f"Severity is sensitive to the detection threshold (range {vr.S_low:.0f} to {vr.S_high:.0f}); "
                        "retake the photo with even lighting or review the overlay manually.")
    warnings.append("Severity and fusion constants are provisional engineering defaults, not values fitted to expert grades.")

    # 2. building age
    age, age_error = None, None
    if inp.want_age:
        if age_fetcher is None:
            age_error = "Earth Engine is not connected; building age skipped."
        else:
            try:
                df = age_fetcher(inp.lat, inp.lon, inp.radius_m)
                age = core.estimate_construction_year(df["year"].to_numpy(), df["ndvi"].to_numpy(),
                                                      df["ndbi"].to_numpy(), k=inp.age_k, tau=inp.age_tau,
                                                      z_min=inp.age_z_min, current_year=now.year)
                age["table"] = df
                if inp.radius_m < 30:
                    warnings.append("Building-age buffer is smaller than one 30 m Landsat pixel; expect mixed-pixel error.")
            except Exception as e:  # noqa: BLE001
                age_error = f"Building age failed: {e}"
        if age_error:
            warnings.append(age_error)
        elif age and age["status"] != "detected":
            warnings.append("Building age: " + age["message"])

    # 3. InSAR (only from data the user supplied)
    insar = None
    try:
        if inp.insar_series is not None:
            insar = core.fit_insar(inp.insar_series[0], inp.insar_series[1], inp.insar_incidence_deg)
            warnings.append("InSAR standard error assumes independent residuals and is optimistic for real time series.")
        elif inp.insar_velocity is not None:
            insar = insar_from_velocity(inp.insar_velocity, inp.insar_incidence_deg)
        if insar is not None and insar["v_vert_mm_yr"] is None:
            warnings.append("InSAR supplied without an incidence angle: vertical velocity unavailable, so InSAR is not used in fusion.")
    except Exception as e:  # noqa: BLE001
        insar = None
        warnings.append(f"InSAR could not be fitted: {e}")

    # 4. fusion over what actually exists
    age_years = age["age"] if (age and age["status"] == "detected") else None
    v_vert = insar["v_vert_mm_yr"] if insar else None
    fusion = core.fuse(vr.severity["S"], vr.severity["status"], age_years, v_vert, fp)

    record = core.result_to_record(vr)
    record.update(name=inp.name, is_public=int(inp.is_public), lat=inp.lat, lon=inp.lon,
                  age_years=age_years, construction_year=age["year"] if age_years is not None else None,
                  v_vert_mm_yr=v_vert, risk_R=fusion["R"], final_status=fusion["final_status"],
                  calibrated=int(m.calibrated), width_mm_p95=m.width_mm_p95, length_m=m.length_m,
                  density_m_per_m2=m.density_m_per_m2)
    return {"vision": vr, "age": age, "age_error": age_error, "insar": insar, "fusion": fusion,
            "warnings": warnings, "inputs": inp, "record": record, "created_at": now,
            "params": {"vision": vp, "severity": sp, "fusion": fp}}
