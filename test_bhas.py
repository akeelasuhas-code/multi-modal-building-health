"""Run with:  pytest -q test_bhas.py"""
import datetime as dt
import math

import cv2
import numpy as np
import pandas as pd
import pytest

import bhas_core as core
import bhas_db as db
import bhas_pipeline as pipe
import bhas_report as report
from synth import make_sample


# ---------------------------------------------------------------- metrics
def test_cl_iou_identity_and_disjoint():
    m = np.zeros((100, 100), bool); m[50, 10:90] = True
    assert core.cl_iou(m, m) == 1.0
    other = np.zeros_like(m); other[10, 10:90] = True
    assert core.cl_iou(m, other) == 0.0
    assert core.cl_iou(np.zeros_like(m), np.zeros_like(m)) == 1.0


def test_cl_iou_tolerance():
    a = np.zeros((100, 100), bool); a[50, 10:90] = True
    b = np.zeros_like(a); b[53, 10:90] = True          # 3 px offset
    assert core.cl_iou(a, b, tau=4) > 0.95
    assert core.cl_iou(a, b, tau=1) < 0.1


def test_cl_dice_bounds():
    a = np.zeros((80, 80), bool); a[40, 5:75] = True
    assert core.cl_dice(a, a) == pytest.approx(1.0)
    assert core.cl_dice(a, np.zeros_like(a)) == 0.0


# ---------------------------------------------------------------- measurement geometry
def test_skeleton_length_horizontal_and_diagonal():
    s = np.zeros((60, 60), bool); s[30, 5:55] = True                    # 49 links
    assert core.skeleton_length(s) == pytest.approx(49.0)
    d = np.zeros((60, 60), bool)
    for i in range(40):
        d[5 + i, 5 + i] = True                                          # 39 diagonal links
    assert core.skeleton_length(d) == pytest.approx(39 * math.sqrt(2), rel=1e-6)


def test_prune_spurs_removes_short_side_branch_keeps_main_line():
    s = np.zeros((60, 60), bool); s[30, 5:55] = True
    s[25:30, 30] = True                                                 # 5-px spur
    p = core.prune_spurs(s, min_len=8)
    assert not p[26, 30]
    assert p[30, 10:50].all()


def test_gsd_formulas():
    assert core.gsd_from_reference(210.0, 840.0) == pytest.approx(0.25)
    assert core.gsd_from_camera(1000, 5.6, 7.0, 4000) == pytest.approx(1000 * 7.0 / (5.6 * 4000))


def test_width_and_length_recovery_against_ground_truth():
    vp = core.VisionParams()
    for wpx in (4, 8):
        bgr, cg, _, _ = make_sample(11, n_cracks=1, widths=(wpx,))
        vr = core.analyze_image(bgr)
        gt, _, _ = core.measure_cracks(cg, np.zeros_like(cg), None, vp)
        assert abs(vr.measurement.width_px_p95 - gt.width_px_p95) <= 1.0
        assert vr.measurement.length_px / gt.length_px == pytest.approx(1.0, abs=0.06)
        assert core.cl_iou(vr.crack, cg) > 0.9


def test_mm_measurement_is_scale_invariant():
    bgr, _, _, _ = make_sample(11, n_cracks=1, widths=(8,))
    a = core.analyze_image(bgr, 0.5).measurement
    big = cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    b = core.analyze_image(big, 0.25).measurement
    assert b.width_mm_p95 == pytest.approx(a.width_mm_p95, rel=0.12)
    assert b.length_m == pytest.approx(a.length_m, rel=0.12)


# ---------------------------------------------------------------- false alarms (the original bug)
def test_no_false_alarm_on_clean_textured_concrete():
    for seed in range(200, 206):
        bgr, *_ = make_sample(seed, n_cracks=0)
        vr = core.analyze_image(bgr)
        assert vr.measurement.crack_area_frac == 0.0
        assert vr.severity["status"] == "STRUCTURALLY SOUND"


def test_hard_shadow_is_not_a_defect():
    bgr, *_ = make_sample(41, n_cracks=0)
    b = bgr.astype(np.float32); b[:, 200:300] *= 0.65
    vr = core.analyze_image(np.clip(b, 0, 255).astype(np.uint8))
    assert vr.severity["S"] == 0.0 and vr.shadow_frac > 0.1


def test_severity_increases_with_crack_width():
    S = []
    for w in (2, 6, 14):
        bgr, *_ = make_sample(11, n_cracks=1, widths=(w,))
        S.append(core.analyze_image(bgr, 0.1).severity["S"])
    assert S[0] < S[1] <= S[2]          # keeps rising past the limit instead of saturating at it


def test_uncalibrated_is_flagged():
    bgr, *_ = make_sample(11, n_cracks=1, widths=(6,))
    vr = core.analyze_image(bgr, None)
    assert "UNCALIBRATED" in vr.severity["basis"] and vr.measurement.width_mm_p95 is None


# ---------------------------------------------------------------- construction year
def _series(build, rng, years, noise=0.04):
    ndbi = np.where(years >= build, 0.02, -0.17) if build else np.full(len(years), -0.17)
    ndvi = np.where(years >= build, 0.16, 0.30) if build else np.full(len(years), 0.30)
    return ndvi + rng.normal(0, noise, len(years)), ndbi + rng.normal(0, noise, len(years))


def test_construction_year_detected_within_one_year():
    rng, years = np.random.default_rng(0), np.arange(1985, 2026)
    hits = 0
    for b in (1996, 2001, 2006, 2011, 2016):
        nv, nb = _series(b, rng, years)
        r = core.estimate_construction_year(years, nv, nb, current_year=2026)
        hits += r["status"] == "detected" and abs(r["year"] - b) <= 1
    assert hits >= 4


def test_flat_series_gives_no_age():
    years = np.arange(1985, 2026)
    r = core.estimate_construction_year(years, np.full(len(years), 0.3), np.full(len(years), -0.17))
    assert r["status"] == "inconclusive" and r["age"] is None


def test_too_few_composites_gives_no_age():
    years = np.arange(1985, 2026)
    nan = np.full(len(years), np.nan)
    assert core.estimate_construction_year(years, nan, nan)["status"] == "inconclusive"


# ---------------------------------------------------------------- InSAR
def test_insar_recovers_velocity_and_vertical_projection():
    rng = np.random.default_rng(2)
    t = np.sort(2021 + rng.random(45) * 3)
    d = -6.0 * (t - t.mean()) + 4 * np.sin(2 * np.pi * t + 0.7) + rng.normal(0, 1.5, 45)
    f = core.fit_insar(t, d, 39.0)
    assert f["v_los_mm_yr"] == pytest.approx(-6.0, abs=4 * f["v_los_se"] + 0.3)
    assert f["v_vert_mm_yr"] == pytest.approx(f["v_los_mm_yr"] / math.cos(math.radians(39)))
    assert f["seasonal_amp_mm"] == pytest.approx(4.0, abs=1.5)


def test_insar_rejects_too_little_data():
    with pytest.raises(ValueError):
        core.fit_insar([2021, 2021.1], [0, 1])


def test_parse_insar_csv():
    df = pd.DataFrame({"date": ["2022-01-01", "2023-01-01"], "disp_mm": [0.0, -5.0]})
    t, d = core.parse_insar_csv(df)
    assert t[1] - t[0] == pytest.approx(1.0, abs=0.01) and d[1] == -5.0


# ---------------------------------------------------------------- fusion
def test_fusion_uses_only_available_modalities():
    f = core.fuse(40.0, "CRITICAL ACTION REQUIRED", None, None)
    assert f["modalities"] == ["vision"] and f["R"] == pytest.approx(0.40)


def test_fusion_final_status_is_more_severe_of_two():
    f = core.fuse(5.0, "STRUCTURALLY SOUND", None, -30.0)          # clean wall but fast subsidence
    assert f["final_status"] != "STRUCTURALLY SOUND"
    g = core.fuse(80.0, "CRITICAL ACTION REQUIRED", 0, 0.0)
    assert g["final_status"] == "CRITICAL ACTION REQUIRED"


def test_logistic_fit_and_auroc():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2)); y = (1.5 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.7, 300) > 0).astype(int)
    b, mu, sd = core.fit_logistic(X, y)
    p = 1 / (1 + np.exp(-(np.column_stack([np.ones(300), (X - mu) / sd]) @ b)))
    assert core.auroc(p, y) > 0.9
    assert core.auroc([0.1, 0.2, 0.3], [0, 0, 0]) != core.auroc([0.1, 0.2, 0.3], [0, 0, 0])   # nan when one class


# ---------------------------------------------------------------- pipeline: nothing mocked
def test_pipeline_never_invents_satellite_data():
    bgr, *_ = make_sample(11, n_cracks=1, widths=(5,))
    r = pipe.run_audit(bgr, pipe.AuditInputs(gsd_mm_px=0.2))
    assert r["insar"] is None and r["age"] is None
    assert r["fusion"]["modalities"] == ["vision"]
    assert r["record"]["age_years"] is None and r["record"]["v_vert_mm_yr"] is None


def test_pipeline_age_failure_is_reported_not_hidden():
    bgr, *_ = make_sample(11, n_cracks=1, widths=(5,))
    def boom(*a): raise RuntimeError("ee down")
    r = pipe.run_audit(bgr, pipe.AuditInputs(want_age=True), age_fetcher=boom)
    assert r["age"] is None and "ee down" in r["age_error"] and any("ee down" in w for w in r["warnings"])


def test_pipeline_coarse_resolution_warning():
    bgr, *_ = make_sample(11, n_cracks=1, widths=(5,))
    r = pipe.run_audit(bgr, pipe.AuditInputs(gsd_mm_px=0.5))
    assert any("Resolution too coarse" in w for w in r["warnings"])


# ---------------------------------------------------------------- escalation
def _crit(name="School", public=1, status="CRITICAL ACTION REQUIRED"):
    return dict(name=name, is_public=public, lat=17.4, lon=78.4, S=60.0, S_low=55.0, S_high=62.0,
                vision_status=status, final_status=status, calibrated=1)


def test_escalation_ladder_timing_and_clock_reset(tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    t0 = dt.datetime(2026, 1, 1)
    db.add_audit(conn, _crit(), now=t0)
    assert db.run_escalation(conn, now=t0 + dt.timedelta(days=10)) == []
    n = db.run_escalation(conn, now=t0 + dt.timedelta(days=15))
    assert len(n) == 1 and n[0]["to"] == db.LADDER[1]
    assert db.run_escalation(conn, now=t0 + dt.timedelta(days=16)) == []          # clock reset
    n = db.run_escalation(conn, now=t0 + dt.timedelta(days=31))
    assert n[0]["to"] == db.LADDER[2]
    for k in range(3):
        n = db.run_escalation(conn, now=t0 + dt.timedelta(days=50 + 20 * k))
    assert n[0]["top_of_ladder"] and "highest authority" in n[0]["message"]
    assert len(db.list_escalations(conn)) >= 4


def test_escalation_ignores_private_acknowledged_and_non_critical(tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    t0 = dt.datetime(2026, 1, 1)
    db.add_audit(conn, _crit("private", public=0), now=t0)
    db.add_audit(conn, _crit("minor", status="MONITORING RECOMMENDED"), now=t0)
    a = db.add_audit(conn, _crit("acked"), now=t0)
    db.set_state(conn, a, "ACKNOWLEDGED")
    assert db.run_escalation(conn, now=t0 + dt.timedelta(days=100)) == []


def test_bad_workflow_state_rejected(tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    a = db.add_audit(conn, _crit())
    with pytest.raises(ValueError):
        db.set_state(conn, a, "DONE")


# ---------------------------------------------------------------- report
def test_pdf_builds_for_calibrated_and_uncalibrated():
    bgr, *_ = make_sample(11, n_cracks=1, widths=(5,), blob=True)
    for gsd in (0.1, None):
        r = pipe.run_audit(bgr, pipe.AuditInputs(name="Test", gsd_mm_px=gsd))
        pdf = report.build_pdf(r)
        assert pdf[:4] == b"%PDF" and len(pdf) > 10_000


def test_severity_formula_by_hand():
    sp = core.SeverityParams()
    m = core.CrackMeasurement(True, 0.1, 100, 0.0, 0.0, 100.0, 5.0, 4.0, 3.0, 0.01, 0.6, 0.6, 0.5, 1.0, 0.001,
                              0.1, 0.0)
    m.width_mm_p95, m.density_m_per_m2, m.blob_area_frac = 0.3, 1.0, 0.10       # everything exactly AT its critical value
    s = core.compute_severity(m, sp)
    assert s["S"] == pytest.approx(50.0)                                        # half of maximum by construction
    m.width_mm_p95 = 0.05
    assert core.compute_severity(m, sp)["S"] < s["S"]


def test_non_surface_photo_triggers_plausibility_warning():
    bgr, *_ = make_sample(11, n_cracks=9, widths=(12,))          # absurd amount of "cracking"
    r = pipe.run_audit(bgr, pipe.AuditInputs(gsd_mm_px=0.1))
    assert r["vision"].measurement.crack_area_frac > 0.05
    assert r["warnings"][0].startswith("Unusually large defect coverage")
