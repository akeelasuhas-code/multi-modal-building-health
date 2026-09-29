"""Building Health Auditing System - Streamlit app.

All computation lives in bhas_core / bhas_pipeline; this file is UI only.
Run locally:  streamlit run app.py
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import bhas_core as core
import bhas_db as db
import bhas_gee as gee
import bhas_pipeline as pipe
import bhas_report as report

st.set_page_config(page_title="Building Health Auditor", layout="wide")

GEE_PROJECT = "suhas-proj1"


# ----------------------------------------------------------------------------- helpers
@st.cache_resource
def ee_status():
    try:
        secrets = st.secrets
        _ = "gee_service_account" in secrets
    except Exception:  # no secrets.toml
        secrets = None
    return gee.init_ee(project=GEE_PROJECT, secrets=secrets)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def age_table(lat, lon, radius):
    return gee.fetch_annual_indices(lat, lon, radius)


def get_conn():
    # A fresh connection per call: Streamlit runs sessions on different threads and SQLite
    # connections must not be shared across them. Opening one is cheap.
    return db.connect()


def status_box(text):
    (st.error if text == core.SEVERITY_ORDER[2] else st.warning if text == core.SEVERITY_ORDER[1] else st.success)(
        f"### {text}")


def decode(upload):
    arr = np.frombuffer(upload.getvalue(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ----------------------------------------------------------------------------- sidebar
ee_ok, ee_msg = ee_status()
with st.sidebar:
    st.header("System")
    (st.success if ee_ok else st.warning)(f"Earth Engine: {ee_msg[:140]}")
    st.caption("Log storage: local SQLite. On Streamlit Community Cloud it resets when the app restarts; "
               "use the CSV export for anything you must keep.")
    st.header("Settings")
    k_noise = st.slider("Crack sensitivity (noise multiplier k)", 3.5, 7.0, 5.0, 0.5,
                        help="Lower = finds fainter cracks but risks false alarms on textured concrete. 5 was chosen on "
                             "synthetic concrete; re-tune on real photos.")
    w_crit = st.number_input("Critical crack width (mm)", 0.05, 2.0, 0.30, 0.05,
                             help="Provisional. Confirm the limit for your exposure class in IS 456 / ACI 224R.")
    with st.expander("Advanced: provisional constants"):
        rho_crit = st.number_input("Critical crack density (m per m2)", 0.1, 10.0, 1.0, 0.1)
        a_crit = st.number_input("Critical dark-anomaly fraction", 0.01, 1.0, 0.10, 0.01)
        w1 = st.slider("Weight: crack width", 0.0, 1.0, 0.50, 0.05)
        w2 = st.slider("Weight: crack density", 0.0, 1.0, 0.25, 0.05)
        w3 = st.slider("Weight: dark anomaly area", 0.0, 1.0, 0.25, 0.05)
        sound_max = st.number_input("Sound up to S =", 1.0, 50.0, 10.0, 1.0)
        monitor_max = st.number_input("Monitoring up to S =", 5.0, 90.0, 35.0, 1.0)
        st.markdown("**Fusion**")
        fw_v = st.slider("Fusion weight: vision", 0.0, 1.0, 0.60, 0.05)
        fw_a = st.slider("Fusion weight: building age", 0.0, 1.0, 0.15, 0.05)
        fw_i = st.slider("Fusion weight: InSAR", 0.0, 1.0, 0.25, 0.05)
        v_crit = st.number_input("Critical vertical velocity (mm/yr)", 1.0, 100.0, 10.0, 1.0)
        st.markdown("**Building-age detector**")
        age_tau = st.number_input("Minimum step size", 0.05, 0.5, 0.15, 0.01)
        age_z = st.number_input("Minimum z-score", 1.5, 6.0, 3.0, 0.5)

vp = core.VisionParams(k_noise=k_noise)
wsum = max(1e-9, w1 + w2 + w3)
sp = core.SeverityParams(w_crit_mm=w_crit, rho_crit_m_per_m2=rho_crit, a_crit=a_crit,
                         weights=(w1 / wsum, w2 / wsum, w3 / wsum), sound_max=sound_max, monitor_max=monitor_max)
fp = core.FusionParams(w_vision=fw_v, w_age=fw_a, w_insar=fw_i, v_crit_mm_yr=v_crit)

st.title("Building Health Auditing System")
tab_audit, tab_log, tab_method = st.tabs(["Audit", "Log and escalation", "Method"])

# ----------------------------------------------------------------------------- audit tab
with tab_audit:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("1. Building")
        name = st.text_input("Name / identifier", "")
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", value=17.4065, format="%.5f")
        lon = c2.number_input("Longitude", value=78.4772, format="%.5f")
        is_public = st.checkbox("Public facility (school, hospital, government office): enables escalation")

        st.subheader("2. Photo")
        up = st.file_uploader("Facade or concrete surface photo", type=["jpg", "jpeg", "png"])
        img_w = None
        if up is not None:
            bgr_preview = decode(up)
            if bgr_preview is None:
                st.error("Could not read that image.")
                up = None
            else:
                img_w = bgr_preview.shape[1]
                st.image(cv2.cvtColor(bgr_preview, cv2.COLOR_BGR2RGB), caption=f"{bgr_preview.shape[1]} x {bgr_preview.shape[0]} px",
                         width="stretch")

        st.subheader("3. Scale")
        scale_mode = st.radio("How do you know the size of a pixel?",
                              ["Not calibrated", "I know mm per pixel", "Camera geometry", "Reference object in frame"],
                              help="Without a scale, widths are in pixels and severity is pixel-relative only.")
        gsd = None
        if scale_mode == "I know mm per pixel":
            gsd = st.number_input("mm per pixel (original photo)", 0.005, 20.0, 0.25, 0.005, format="%.3f")
        elif scale_mode == "Camera geometry":
            a, b, c = st.columns(3)
            dist_m = a.number_input("Distance to wall (m)", 0.1, 50.0, 1.0, 0.1)
            focal = b.number_input("Focal length (mm)", 1.0, 200.0, 5.6, 0.1,
                                   help="Real focal length, not the 35 mm equivalent.")
            sens_w = c.number_input("Sensor width (mm)", 1.0, 40.0, 7.0, 0.1)
            if img_w:
                gsd = core.gsd_from_camera(dist_m * 1000, focal, sens_w, img_w)
                st.caption(f"= {gsd:.3f} mm/pixel (assumes the camera faces the wall square-on)")
        elif scale_mode == "Reference object in frame":
            a, b = st.columns(2)
            ref_mm = a.number_input("Real length of object (mm)", 1.0, 5000.0, 210.0, 1.0, help="e.g. 210 mm = short side of A4")
            ref_px = b.number_input("Its length in the original photo (px)", 1.0, 20000.0, 800.0, 1.0)
            gsd = core.gsd_from_reference(ref_mm, ref_px)
            st.caption(f"= {gsd:.3f} mm/pixel")

        st.subheader("4. Satellite (optional)")
        want_age = st.checkbox("Estimate construction year from Landsat", value=False, disabled=not ee_ok,
                               help=None if ee_ok else "Earth Engine is not connected on this server.")
        radius = st.slider("Building buffer (m)", 15, 150, 45, 5, disabled=not want_age)
        insar_mode = st.radio("InSAR ground motion", ["None", "Upload time series (CSV)", "Enter velocity"], horizontal=True)
        insar_series, insar_v, inc = None, None, None
        if insar_mode != "None":
            inc = st.number_input("Incidence angle (deg)", 20.0, 55.0, 39.0, 0.5,
                                  help="Sentinel-1 is typically 30 to 46 degrees; use the value for your track.")
        if insar_mode == "Upload time series (CSV)":
            f = st.file_uploader("CSV with a date column and a displacement column in mm (LOS)", type=["csv"])
            if f is not None:
                try:
                    insar_series = core.parse_insar_csv(pd.read_csv(f))
                    st.caption(f"{int(np.isfinite(insar_series[1]).sum())} epochs read.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not read CSV: {e}")
        elif insar_mode == "Enter velocity":
            insar_v = st.number_input("LOS velocity (mm/yr)", -100.0, 100.0, 0.0, 0.1)

        go = st.button("Run audit", type="primary", disabled=up is None)

    if go and up is not None:
        inputs = pipe.AuditInputs(name=name, is_public=is_public, lat=lat, lon=lon, radius_m=float(radius), gsd_mm_px=gsd,
                                  want_age=want_age, insar_series=insar_series, insar_velocity=insar_v,
                                  insar_incidence_deg=inc, age_tau=age_tau, age_z_min=age_z)
        with st.spinner("Analysing image" + (" and querying Landsat" if want_age else "") + "..."):
            res = pipe.run_audit(decode(up), inputs, vp, sp, fp, age_fetcher=age_table if (want_age and ee_ok) else None)
        st.session_state["result"] = res
        st.session_state["saved_id"] = None
        st.session_state["pdf"] = None

    with right:
        res = st.session_state.get("result")
        if res is None:
            st.info("Upload a photo, set the scale if you can, and press Run audit. Results stay on screen until the next run.")
        else:
            vr, m, fu = res["vision"], res["vision"].measurement, res["fusion"]
            status_box(fu["final_status"])
            a, b, c, d = st.columns(4)
            a.metric("Severity S", f"{vr.severity['S']:.0f}", help=f"Plausible range {vr.S_low:.0f} to {vr.S_high:.0f}")
            a.caption(f"range {vr.S_low:.0f} to {vr.S_high:.0f}")
            b.metric("Crack width (p95)", f"{m.width_mm_p95:.2f} mm" if m.calibrated else f"{m.width_px_p95:.1f} px")
            c.metric("Crack length", f"{m.length_m:.2f} m" if m.calibrated else f"{m.length_px:.0f} px")
            d.metric("Fused risk R", f"{fu['R']:.2f}", help="Uses " + ", ".join(fu["modalities"]))
            st.caption(f"Severity basis: {vr.severity['basis']}. Fusion used: {', '.join(fu['modalities'])}.")

            t1, t2, t3 = st.tabs(["Detected defects", "Crack width map", "Original"])
            t1.image(vr.overlay, caption="Red: cracks. Orange: dark anomalies (possible spalling or moisture).", width="stretch")
            t2.image(vr.width_map, caption=("Colour = width relative to the critical crack width (red at 1.5x or more)."
                                            if m.calibrated else "Colour = width relative to the widest crack in this image."),
                     width="stretch")
            t3.image(vr.original_rgb, width="stretch")

            if m.calibrated:
                st.caption(f"Density {m.density_m_per_m2:.2f} m/m2 | dark anomaly area {m.blob_area_m2 * 1e4:.1f} cm2 "
                           f"({m.blob_area_frac * 100:.1f}%) | imaged surface {m.image_area_m2:.3f} m2")
            if len(vr.width_profile_px):
                with st.expander("Width distribution along the crack"):
                    g = m.gsd_mm_per_px if m.calibrated else 1.0
                    hist, edges = np.histogram(vr.width_profile_px * g, bins=12)
                    st.bar_chart(pd.DataFrame({"skeleton points": hist},
                                              index=[f"{e:.2f}" for e in edges[:-1]]), x_label="width " + ("(mm)" if m.calibrated else "(px)"))

            if res["age"] is not None:
                ag = res["age"]
                st.markdown("**Construction year (Landsat step detector)**")
                if ag["status"] == "detected":
                    conf = "high" if ag["z"] >= 4 else "moderate"
                    st.success(f"Built about {ag['year']} (range {ag['year_range'][0]} to {ag['year_range'][1]}), "
                               f"age about {ag['age']} years. Step {ag['delta_max']:.2f}, z = {ag['z']:.1f} ({conf} confidence).")
                else:
                    st.warning(ag["message"])
                tbl = pd.DataFrame({"NDBI": ag["ndbi"], "NDVI": ag["ndvi"], "step score": ag["delta"]}, index=ag["years"])
                st.line_chart(tbl)
            if res["insar"] is not None:
                ins = res["insar"]
                st.markdown("**InSAR ground motion**")
                txt = f"LOS velocity {ins['v_los_mm_yr']:.2f} mm/yr"
                if ins.get("v_los_se"):
                    txt += f" (+/- {ins['v_los_se']:.2f})"
                if ins["v_vert_mm_yr"] is not None:
                    txt += f"; vertical {ins['v_vert_mm_yr']:.2f} mm/yr (negative = subsidence, if your tool uses that sign)"
                st.info(txt)
                if ins["fit_t"] is not None:
                    st.line_chart(pd.DataFrame({"displacement (mm)": ins["fit_d"], "fit": ins["fit_curve"]}, index=ins["fit_t"]))

            with st.expander("Fusion breakdown"):
                st.dataframe(pd.DataFrame({"component (0-1)": fu["components"], "weight": fu["weights"]}))
                st.caption(f"Vision-only status: {vr.severity['status']}. Fused status: {fu['fused_status']}. "
                           "The reported status is the more severe of the two.")
            with st.expander(f"Warnings and assumptions ({len(res['warnings'])})", expanded=True):
                for w in res["warnings"]:
                    st.write("- " + w)

            s1, s2 = st.columns(2)
            if st.session_state.get("saved_id") is None:
                if s1.button("Save to audit log"):
                    st.session_state["saved_id"] = db.add_audit(get_conn(), res["record"])
                    st.rerun()
            else:
                s1.success(f"Saved as audit #{st.session_state['saved_id']}")
            if st.session_state.get("pdf") is None:
                st.session_state["pdf"] = report.build_pdf(res)
            s2.download_button("Download repair report (PDF)", st.session_state["pdf"],
                               file_name=f"audit_{(res['inputs'].name or 'building').replace(' ', '_')}_{res['created_at']:%Y%m%d}.pdf",
                               mime="application/pdf")

# ----------------------------------------------------------------------------- log tab
with tab_log:
    conn = get_conn()
    df = db.list_audits(conn)
    st.subheader("Audit log")
    if df.empty:
        st.info("No audits saved yet.")
    else:
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Export log (CSV)", df.to_csv(index=False).encode(), "audit_log.csv", "text/csv")
        st.markdown("**Update an audit**")
        c1, c2, c3, c4 = st.columns([1, 1.4, 2, 1])
        aid = c1.selectbox("Audit #", df["id"].tolist())
        state = c2.selectbox("Workflow state", list(db.STATES))
        note = c3.text_input("Note (optional)")
        if c4.button("Update"):
            db.set_state(conn, int(aid), state, note or None)
            st.rerun()

    st.subheader("Escalation")
    st.caption("Public facilities only. A CRITICAL audit with no acknowledgement moves one level up the authority ladder "
               "each time the waiting period passes. Notices are generated and logged here; they are not emailed.")
    days = st.number_input("Waiting period per level (days)", 1.0, 365.0, 14.0, 1.0)
    if st.button("Run escalation check"):
        notices = db.run_escalation(conn, delta_days=days)
        if not notices:
            st.success("Nothing is overdue.")
        for n in notices:
            st.warning(f"Audit #{n['audit_id']} escalated to {n['to']}" + (" (top of ladder)" if n["top_of_ladder"] else ""))
            st.code(n["message"])
    el = db.list_escalations(conn)
    if not el.empty:
        st.dataframe(el, width="stretch", hide_index=True)
    st.caption("Ladder: " + " > ".join(db.LADDER))

# ----------------------------------------------------------------------------- method tab
with tab_method:
    st.subheader("What is computed")
    st.markdown("**1. Crack extraction.** Illumination flattening, multi-scale black top-hat, hysteresis threshold at "
                "k times the robust texture noise, then shape filtering (skeleton length and elongation) so pores and "
                "compact blobs are not counted as cracks. A half-maximum criterion removes the blur halo from the width.")
    st.latex(r"w(x)=\big(2\,\mathrm{DT}(x)-1\big)\cdot \mathrm{GSD},\qquad \mathrm{GSD}=\frac{D\,s_w}{f\,W_{px}}\ \ \text{or}\ \ \frac{L_{ref}}{L_{ref,px}}")
    st.markdown("Length is the skeleton arc length (diagonal links weigh sqrt 2) after spur pruning; density is length per unit area.")
    st.markdown("**2. Severity.**")
    st.latex(r"S=100\sum_i a_i\,\min\!\Big(1,\frac{f_i}{2\,f_{crit,i}}\Big),\quad f=\Big(w_{95},\ \rho,\ A_{anomaly}/A\Big)")
    st.markdown("A component is half its maximum at its critical value and saturates at twice that. "
                "The range shown comes from re-running with detection thresholds x1.25 and x0.8.")
    st.markdown("**3. Construction year.** Annual dry-season Landsat 5/7/8/9 composites of NDBI and NDVI over the buffer.")
    st.latex(r"\Delta(s)=\big[\overline{NDBI}_{s..s+k-1}-\overline{NDBI}_{s-k..s-1}\big]-\big[\overline{NDVI}_{s..s+k-1}-\overline{NDVI}_{s-k..s-1}\big]")
    st.markdown("The year is argmax of the step score, accepted only if the step and its robust z-score pass the thresholds; "
                "otherwise no age is reported.")
    st.markdown("**4. InSAR.** Ordinary least squares with a seasonal term, then projection to vertical.")
    st.latex(r"d(t)=c+vt+a\sin 2\pi t+b\cos 2\pi t,\qquad v_{vert}\approx v_{LOS}/\cos\theta")
    st.markdown("**5. Fusion.** Only the modalities you actually supplied are used.")
    st.latex(r"R=\frac{\sum_i w_i c_i}{\sum_i w_i},\quad c_{vision}=\tfrac{S}{100},\ c_{age}=\min(1,\tfrac{age}{50}),\ c_{insar}=\min(1,\tfrac{|v_{vert}|}{v_{crit}})")
    st.subheader("Known limitations")
    st.markdown(
        "- Classical image processing, not a trained network. Fine for screening; evaluate on OmniCrack30k or your own labelled photos "
        "before claiming accuracy (see evaluate.py).\n"
        "- Widths need a correct scale and a square-on camera. Cracks under about 3 pixels wide are over-estimated.\n"
        "- Critical width, density and area limits, weights and thresholds are provisional defaults until calibrated against expert grades.\n"
        "- Dark anomalies are not typed (spall vs stain vs shadow). Band-like shadows are ignored.\n"
        "- Landsat pixels are 30 m: small buildings mix with their surroundings. Age is skipped when the signal is unclear.\n"
        "- InSAR is only as good as the data you provide; the fit standard error is optimistic.")
