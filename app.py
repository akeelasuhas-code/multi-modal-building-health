"""Building Health Check - Streamlit app (UI only; all maths is in bhas_core / bhas_pipeline)."""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import bhas_core as core
import bhas_db as db
import bhas_gee as gee
import bhas_pipeline as pipe
import bhas_report as rep

st.set_page_config(page_title="Building Health Check", page_icon="🏢", layout="wide",
                   initial_sidebar_state="collapsed")
GEE_PROJECT = "suhas-proj1"

REF_OBJECTS = {"A4 sheet, short side (210 mm)": 210.0, "A4 sheet, long side (297 mm)": 297.0,
               "ID or bank card, long side (85.6 mm)": 85.6, "Something else (I'll type its size)": None}
SCALE_MODES = ["📏 Object of known size in the photo", "🔢 I know the mm per pixel",
               "📷 Camera distance and lens", "⏭️ Skip (no measurements in mm)"]
STATE_LABEL = {"PENDING_ACTION": "⏳ Waiting for action", "ACKNOWLEDGED": "👍 Acknowledged",
               "RESOLVED": "✅ Repaired / closed", "ESCALATED": "📣 Escalated"}


# ----------------------------------------------------------------------------- helpers
@st.cache_resource
def ee_status():
    try:
        secrets = st.secrets
        _ = "gee_service_account" in secrets
    except Exception:
        secrets = None
    return gee.init_ee(project=GEE_PROJECT, secrets=secrets)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def age_table(lat, lon, radius):
    return gee.fetch_annual_indices(lat, lon, radius)


def get_conn():
    # fresh connection each time: Streamlit sessions run on different threads
    return db.connect()


def decode(upload):
    return cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)


def ruler_image(rgb):
    """Photo with a pixel grid so the user can read off the length of a reference object."""
    img = rgb.copy()
    h, w = img.shape[:2]
    target = max(w, h) / 10
    step = min([10, 20, 25, 50, 100, 200, 250, 500, 1000], key=lambda s: abs(s - target))
    grid = img.copy()
    for x in range(0, w, step):
        cv2.line(grid, (x, 0), (x, h - 1), (255, 255, 0), max(1, w // 800))
    for y in range(0, h, step):
        cv2.line(grid, (0, y), (w - 1, y), (255, 255, 0), max(1, w // 800))
    img = cv2.addWeighted(grid, 0.45, img, 0.55, 0)
    fs, th = max(0.45, max(w, h) / 1400), max(1, max(w, h) // 700)
    for x in range(0, w, step * 2):
        for col, t in (((0, 0, 0), th + 2), ((255, 255, 255), th)):
            cv2.putText(img, str(x), (x + 3, int(22 * fs + 6)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, t, cv2.LINE_AA)
    for y in range(step * 2, h, step * 2):
        for col, t in (((0, 0, 0), th + 2), ((255, 255, 255), th)):
            cv2.putText(img, str(y), (4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, col, t, cv2.LINE_AA)
    return img, step


def verdict_card(status):
    V = rep.VERDICT[status]
    st.markdown(
        f"""<div style="border-left:10px solid {V['color']};background:rgba(127,127,127,0.09);
        padding:14px 18px;border-radius:8px;margin:4px 0 10px 0">
        <div style="font-size:1.55rem;font-weight:700;line-height:1.25">{V['icon']} {V['title']}</div>
        <div style="margin-top:6px;font-size:1.02rem">{V['meaning']}</div></div>""",
        unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sidebar (engineers only)
ee_ok, ee_msg = ee_status()
with st.sidebar:
    st.header("⚙️ Engineer settings")
    st.caption("Most users never need to touch these.")
    k_noise = st.slider("Crack sensitivity (lower = finds fainter cracks, more false alarms)", 3.5, 7.0, 5.0, 0.5)
    w_crit = st.number_input("Reference crack width limit (mm)", 0.05, 2.0, 0.30, 0.05,
                             help="Provisional. Check the limit for your exposure class (IS 456 / ACI 224R).")
    with st.expander("Score weights and bands (provisional)"):
        rho_crit = st.number_input("Reference crack density (m per m2)", 0.1, 10.0, 1.0, 0.1)
        a_crit = st.number_input("Reference damaged-area fraction", 0.01, 1.0, 0.10, 0.01)
        w1 = st.slider("Weight: crack width", 0.0, 1.0, 0.50, 0.05)
        w2 = st.slider("Weight: amount of cracking", 0.0, 1.0, 0.25, 0.05)
        w3 = st.slider("Weight: damaged patches", 0.0, 1.0, 0.25, 0.05)
        sound_max = st.number_input("'No significant damage' up to score", 1.0, 50.0, 10.0, 1.0)
        monitor_max = st.number_input("'Needs monitoring' up to score", 5.0, 90.0, 35.0, 1.0)
    with st.expander("Satellite fusion and age detector (provisional)"):
        fw_v = st.slider("Weight: photo", 0.0, 1.0, 0.60, 0.05)
        fw_a = st.slider("Weight: building age", 0.0, 1.0, 0.15, 0.05)
        fw_i = st.slider("Weight: ground movement", 0.0, 1.0, 0.25, 0.05)
        v_crit = st.number_input("Reference vertical velocity (mm/yr)", 1.0, 100.0, 10.0, 1.0)
        age_tau = st.number_input("Age detector: minimum step", 0.05, 0.5, 0.15, 0.01)
        age_z = st.number_input("Age detector: minimum z-score", 1.5, 6.0, 3.0, 0.5)
    with st.expander("System status"):
        st.write(("✅ " if ee_ok else "⚪ ") + ("Satellite data connected." if ee_ok else
                 "Satellite data (Google Earth Engine) is not set up on this server, so the building-age option is off."))
        st.caption("Saved inspections are stored on this server and are wiped when the app restarts. "
                   "Use 'Download all (CSV)' to keep a copy.")

vp = core.VisionParams(k_noise=k_noise)
wsum = max(1e-9, w1 + w2 + w3)
sp = core.SeverityParams(w_crit_mm=w_crit, rho_crit_m_per_m2=rho_crit, a_crit=a_crit,
                         weights=(w1 / wsum, w2 / wsum, w3 / wsum), sound_max=sound_max, monitor_max=monitor_max)
fp = core.FusionParams(w_vision=fw_v, w_age=fw_a, w_insar=fw_i, v_crit_mm_yr=v_crit)

# ----------------------------------------------------------------------------- header
st.title("🏢 Building Health Check")
st.write("Upload a close-up photo of a wall or concrete surface. You get a damage verdict, measured cracks "
         "and a repair report you can download.")
h1, h2, h3 = st.columns(3)
h1.info("**1. Upload a photo**  \nOne wall area, taken straight on.")
h2.info("**2. Add a size reference**  \nAn A4 sheet or ID card in the photo gives sizes in mm.")
h3.info("**3. Get the verdict**  \nScore, crack list and a PDF report.")

tab_new, tab_saved, tab_how = st.tabs(["🔍 New inspection", "📁 Saved inspections", "ℹ️ How it works"])

# ============================================================================= NEW INSPECTION
with tab_new:
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        # ---------- step 1
        with st.container(border=True):
            st.subheader("Step 1 · Photo")
            up = st.file_uploader("Choose a photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
            with st.expander("📸 How to take a good photo"):
                st.markdown("- Hold the phone **parallel to the wall**, not at an angle.\n"
                            "- Stand **30 to 60 cm** away so cracks are clearly visible.\n"
                            "- Use **even light**. Avoid hard shadows and flash glare.\n"
                            "- Only the wall: no people, plants, windows or sky in the frame.\n"
                            "- Tape an **A4 sheet** or hold an **ID card** flat against the wall for scale.")
            bgr_preview = None
            if up is not None:
                bgr_preview = decode(up)
                if bgr_preview is None:
                    st.error("That file could not be read as an image.")
                    up = None

        # ---------- step 2
        with st.container(border=True):
            st.subheader("Step 2 · Size reference")
            st.caption("Without this, cracks can be found but not measured in millimetres.")
            mode = st.radio("How can we tell how big things are?", SCALE_MODES, index=0, label_visibility="collapsed")
            gsd = None
            if mode == SCALE_MODES[0]:
                obj = st.selectbox("What is in the photo?", list(REF_OBJECTS))
                ref_mm = REF_OBJECTS[obj]
                if ref_mm is None:
                    ref_mm = st.number_input("Its real length (mm)", 1.0, 5000.0, 100.0, 1.0)
                if bgr_preview is not None:
                    ruled, step = ruler_image(cv2.cvtColor(bgr_preview, cv2.COLOR_BGR2RGB))
                    st.image(ruled, width="stretch",
                             caption=f"Grid lines every {step} pixels. Count how many pixels long the object is.")
                ref_px = st.number_input("How many pixels long is it in the photo?", min_value=1.0, max_value=20000.0,
                                         value=None, step=1.0, placeholder="e.g. 640, read it off the grid")
                if ref_px:
                    gsd = core.gsd_from_reference(ref_mm, ref_px)
                    st.success(f"Scale set: 1 pixel = {gsd:.3f} mm")
                else:
                    st.caption("Enter the pixel length to switch on millimetre measurements.")
            elif mode == SCALE_MODES[1]:
                gsd = st.number_input("Millimetres per pixel (original photo)", 0.005, 20.0, 0.25, 0.005, format="%.3f")
            elif mode == SCALE_MODES[2]:
                a, b, c = st.columns(3)
                dist_m = a.number_input("Distance to wall (m)", 0.1, 50.0, 0.5, 0.1)
                focal = b.number_input("Lens focal length (mm)", 1.0, 200.0, 5.6, 0.1, help="Real focal length, not 35 mm equivalent.")
                sens_w = c.number_input("Sensor width (mm)", 1.0, 40.0, 7.0, 0.1)
                if bgr_preview is not None:
                    gsd = core.gsd_from_camera(dist_m * 1000, focal, sens_w, bgr_preview.shape[1])
                    st.caption(f"1 pixel = {gsd:.3f} mm (assumes the camera faces the wall straight on)")
            elif bgr_preview is not None:
                st.image(cv2.cvtColor(bgr_preview, cv2.COLOR_BGR2RGB), width="stretch")

        # ---------- step 3
        with st.container(border=True):
            st.subheader("Step 3 · Details (optional)")
            c1, c2 = st.columns(2)
            name = c1.text_input("Building name", placeholder="e.g. Block B, east wall")
            inspector = c2.text_input("Inspected by", placeholder="Your name")
            is_public = st.checkbox("This is a public building (school, hospital, government office)",
                                    help="Public buildings with urgent results can be escalated to higher authorities "
                                         "if nobody acts on them. See 'Saved inspections'.")
            with st.expander("🛰️ Satellite data (advanced)"):
                st.caption("Optional extra evidence. The photo result works without it.")
                c1, c2 = st.columns(2)
                lat = c1.number_input("Latitude", value=17.4065, format="%.5f")
                lon = c2.number_input("Longitude", value=78.4772, format="%.5f")
                want_age = st.checkbox("Estimate when the building was built (Landsat, 1985 onwards)",
                                       disabled=not ee_ok,
                                       help=None if ee_ok else "Needs Google Earth Engine, which is not set up on this server.")
                radius = st.slider("Area around the building to check (m)", 15, 150, 45, 5, disabled=not want_age)
                insar_mode = st.radio("Ground movement data (InSAR)", ["None", "Upload a time series (CSV)", "Type a velocity"],
                                      horizontal=True)
                insar_series, insar_v, inc = None, None, None
                if insar_mode != "None":
                    inc = st.number_input("Satellite incidence angle (degrees)", 20.0, 55.0, 39.0, 0.5)
                if insar_mode == "Upload a time series (CSV)":
                    f = st.file_uploader("CSV with a date column and a displacement column (mm)", type=["csv"])
                    if f is not None:
                        try:
                            insar_series = core.parse_insar_csv(pd.read_csv(f))
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Could not read the CSV: {e}")
                elif insar_mode == "Type a velocity":
                    insar_v = st.number_input("Line-of-sight velocity (mm per year)", -100.0, 100.0, 0.0, 0.1)

        go = st.button("🔍 Analyse photo", type="primary", disabled=up is None, width="stretch")
        if up is None:
            st.caption("Upload a photo in Step 1 to enable the button.")

    if go and up is not None:
        inputs = pipe.AuditInputs(name=name, is_public=is_public, lat=lat, lon=lon, radius_m=float(radius), gsd_mm_px=gsd,
                                  want_age=want_age, insar_series=insar_series, insar_velocity=insar_v,
                                  insar_incidence_deg=inc, age_tau=age_tau, age_z_min=age_z)
        with st.spinner("Analysing the photo" + (" and checking satellite records" if want_age else "") + "..."):
            res = pipe.run_audit(decode(up), inputs, vp, sp, fp, age_fetcher=age_table if (want_age and ee_ok) else None)
        st.session_state.update(result=res, saved_id=None, pdf=None, inspector=inspector)

    # ---------- results
    with right:
        res = st.session_state.get("result")
        if res is None:
            with st.container(border=True):
                st.subheader("Your results will appear here")
                st.write("After you press **Analyse photo** you will see:")
                st.markdown("- a plain-language verdict: 🟢 no significant damage, 🟠 needs monitoring, or 🔴 needs urgent inspection\n"
                            "- the cracks marked and numbered on your photo\n"
                            "- crack widths and lengths in millimetres (if you added a size reference)\n"
                            "- a downloadable PDF report with next steps")
        else:
            vr, m, fu = res["vision"], res["vision"].measurement, res["fusion"]
            items = rep.crack_inventory(vr)
            verdict_card(fu["final_status"])

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Damage score", f"{vr.severity['S']:.0f} / 100",
                      help=f"Likely range {vr.S_low:.0f} to {vr.S_high:.0f}. 0 = no visible damage, 100 = severe.")
            if not items:
                k2.metric("Widest cracks", "None")
                k3.metric("Total crack length", "0 cm")
            elif m.calibrated:
                k2.metric("Widest cracks", f"{m.width_mm_p95:.2f} mm", help=f"Typical width of the widest cracks. Reference limit {sp.w_crit_mm:g} mm.")
                k3.metric("Total crack length", f"{m.length_m * 100:.0f} cm")
            else:
                k2.metric("Widest cracks", "needs scale", help="Add a size reference in Step 2 to measure in mm.")
                k3.metric("Cracked area", f"{m.crack_area_frac * 100:.1f}%")
            k4.metric("Cracks found", f"{len(items)}")
            if m.calibrated:
                st.caption(f"Reference crack-width limit: {sp.w_crit_mm:g} mm. Damaged patches: "
                           f"{m.blob_area_m2 * 1e4:.0f} cm² ({m.blob_area_frac * 100:.1f}% of the photo).")

            c1, c2 = st.columns(2)
            c1.image(vr.original_rgb, caption="Your photo", width="stretch")
            c2.image(rep.numbered_overlay(vr, items), caption="Red = cracks (numbered), orange = damaged patches", width="stretch")

            st.markdown("**What to do next**")
            for i, s in enumerate(rep.VERDICT[fu["final_status"]]["steps"], start=1):
                st.write(f"{i}. {s}")

            if st.session_state.get("pdf") is None:
                st.session_state["pdf"] = rep.build_pdf(res, st.session_state.get("inspector", ""))
            a, b = st.columns(2)
            a.download_button("📄 Download full report (PDF)", st.session_state["pdf"], type="primary", width="stretch",
                              file_name=f"{rep.report_id(res)}_{(res['inputs'].name or 'building').replace(' ', '_')}.pdf",
                              mime="application/pdf")
            if st.session_state.get("saved_id") is None:
                if b.button("💾 Save this inspection", width="stretch"):
                    st.session_state["saved_id"] = db.add_audit(get_conn(), res["record"])
                    st.rerun()
            else:
                b.success(f"Saved as inspection #{st.session_state['saved_id']}")

            with st.expander("📊 Why this score?", expanded=True):
                for bd in rep.score_breakdown(res):
                    st.write(f"**{bd['factor']}**: {bd['points']:.0f} of {bd['max_points']:.0f} points")
                    st.progress(min(1.0, bd["level"]))
                    st.caption(bd["explain"])

            important = [w for w in res["warnings"] if not w.startswith("Severity and fusion constants")]
            if important:
                with st.expander(f"⚠️ Things to check ({len(important)})", expanded=True):
                    for w in important:
                        st.write("- " + w)

            with st.expander(f"📋 Crack-by-crack list ({len(items)})"):
                if not items:
                    st.write("No cracks were detected.")
                elif m.calibrated:
                    st.dataframe(pd.DataFrame([{"Crack": i["id"], "Length (cm)": round(i["length_mm"] / 10, 1),
                                                "Typical width (mm)": round(i["w95_mm"], 2), "Max width (mm)": round(i["wmax_mm"], 2),
                                                "Direction": i["orientation"], "Width class": i["cls"]} for i in items]),
                                 hide_index=True, width="stretch")
                    cs = rep.class_summary(items)
                    st.dataframe(pd.DataFrame([{"Width class": c["class"], "Cracks": c["count"], "Total length (m)": round(c["length_m"], 2),
                                                "Typical repair (confirm with engineer)": c["repair"]} for c in cs]),
                                 hide_index=True, width="stretch")
                else:
                    st.dataframe(pd.DataFrame([{"Crack": i["id"], "Length (px)": round(i["length_px"]),
                                                "Typical width (px)": round(i["w95_px"], 1), "Direction": i["orientation"]}
                                               for i in items]), hide_index=True, width="stretch")
                    st.caption("Add a size reference to get millimetres and repair classes.")

            if res["age"] is not None or res["insar"] is not None:
                with st.expander("🛰️ Satellite results"):
                    ag = res["age"]
                    if ag is not None:
                        if ag["status"] == "detected":
                            st.success(f"Built around **{ag['year']}** (range {ag['year_range'][0]} to {ag['year_range'][1]}), "
                                       f"so roughly **{ag['age']} years old**.")
                        else:
                            st.warning(ag["message"])
                        st.line_chart(pd.DataFrame({"Built-up index (NDBI)": ag["ndbi"], "Vegetation index (NDVI)": ag["ndvi"]},
                                                   index=ag["years"]))
                    ins = res["insar"]
                    if ins is not None:
                        txt = f"Ground movement along the satellite's line of sight: {ins['v_los_mm_yr']:.1f} mm per year"
                        if ins["v_vert_mm_yr"] is not None:
                            txt += f"; vertical: {ins['v_vert_mm_yr']:.1f} mm per year (negative usually means sinking)."
                        st.info(txt)
                        if ins["fit_t"] is not None:
                            st.line_chart(pd.DataFrame({"Measured (mm)": ins["fit_d"], "Fitted trend (mm)": ins["fit_curve"]},
                                                       index=ins["fit_t"]))

            with st.expander("🔧 Details for engineers"):
                st.image(vr.width_map, width="stretch",
                         caption="Crack centrelines coloured by width" + (" relative to the limit (red = 1.5x or more)." if m.calibrated else "."))
                if len(vr.width_profile_px):
                    g = m.gsd_mm_per_px if m.calibrated else 1.0
                    hist, edges = np.histogram(vr.width_profile_px * g, bins=12)
                    st.bar_chart(pd.DataFrame({"points along cracks": hist}, index=[f"{e:.2f}" for e in edges[:-1]]),
                                 x_label="width " + ("(mm)" if m.calibrated else "(px)"))
                st.write(f"Severity basis: {vr.severity['basis']}. Photo-only verdict: "
                         f"{rep.VERDICT[vr.severity['status']]['title']}. Combined risk R = {fu['R']:.2f} from "
                         f"{', '.join(fu['modalities'])}.")
                st.dataframe(pd.DataFrame({"component (0-1)": fu["components"], "weight": fu["weights"]}))
                st.caption("All reference values, weights and score bands are provisional and not yet calibrated against engineers' grades.")

# ============================================================================= SAVED INSPECTIONS
with tab_saved:
    conn = get_conn()
    df = db.list_audits(conn)
    st.subheader("Saved inspections")
    if df.empty:
        st.info("Nothing saved yet. After analysing a photo, press **Save this inspection**.")
    else:
        view = pd.DataFrame({
            "#": df["id"], "Date": df["created_at"].str[:16], "Building": df["name"].fillna("").replace("", "Not named"),
            "Result": df["final_status"].map(lambda s: f"{rep.VERDICT[s]['icon']} {rep.VERDICT[s]['title']}" if s in rep.VERDICT else s),
            "Score": df["S"].round(0), "Widest cracks (mm)": df["width_mm_p95"].round(2),
            "Public": df["is_public"].map({1: "Yes", 0: "No"}), "Follow-up": df["workflow_state"].map(STATE_LABEL)})
        st.dataframe(view, hide_index=True, width="stretch")
        st.download_button("⬇️ Download all (CSV)", df.to_csv(index=False).encode(), "inspections.csv", "text/csv")

        with st.container(border=True):
            st.markdown("**Update follow-up status**")
            c1, c2, c3, c4 = st.columns([0.8, 1.4, 2, 0.8], vertical_alignment="bottom")
            aid = c1.selectbox("Inspection #", df["id"].tolist())
            new_state = c2.selectbox("New status", list(STATE_LABEL), format_func=lambda s: STATE_LABEL[s])
            note = c3.text_input("Note (optional)", placeholder="e.g. engineer visit booked for 12 Oct")
            if c4.button("Update"):
                db.set_state(conn, int(aid), new_state, note or None)
                st.rerun()

    st.subheader("Escalation for public buildings")
    st.write("If a public building gets a 🔴 urgent result and nobody marks it as acknowledged, it moves one level up "
             "this chain each time the waiting period passes:")
    st.write(" → ".join(db.LADDER))
    st.caption("This creates escalation notices here. It does not send emails.")
    c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
    days = c1.number_input("Waiting period per level (days)", 1.0, 365.0, 14.0, 1.0)
    if c2.button("Check for overdue inspections"):
        notices = db.run_escalation(conn, delta_days=days)
        if not notices:
            st.success("Nothing is overdue.")
        for n in notices:
            st.warning(f"Inspection #{n['audit_id']} escalated to {n['to']}" + (" (top of the chain)" if n["top_of_ladder"] else ""))
            st.code(n["message"], language=None)
    el = db.list_escalations(conn)
    if not el.empty:
        with st.expander(f"Escalation history ({len(el)})"):
            st.dataframe(el, hide_index=True, width="stretch")

# ============================================================================= HOW IT WORKS
with tab_how:
    st.subheader("How it works")
    st.markdown(
        "**1. Finding cracks.** The app evens out the lighting, enhances thin dark lines, and keeps only long, thin shapes. "
        "Pores, dirt spots and texture are ignored. Dark areas wider than any crack are reported separately as damaged "
        "patches (possible spalling or damp). Long shadows across the whole photo are ignored.\n\n"
        "**2. Measuring them.** Each crack's width is measured all along its centre line. A size reference "
        "(A4 sheet, ID card) converts pixels to millimetres.\n\n"
        "**3. Scoring.** Three things add up to the damage score out of 100: how wide the cracks are, how much "
        "cracking there is per square metre, and how much of the surface has damaged patches.\n\n"
        "**4. Satellite data (optional).** Old Landsat images can show roughly when the building appeared, and InSAR "
        "data can show if the ground is moving. They are only used if you provide them.")
    st.markdown("**What the verdicts mean**")
    st.table(pd.DataFrame({"Verdict": [f"{v['icon']} {v['title']}" for v in rep.VERDICT.values()],
                           "Score": [f"0 to {sp.sound_max:g}", f"{sp.sound_max:g} to {sp.monitor_max:g}", f"above {sp.monitor_max:g}"],
                           "Suggested action": [v["steps"][0] for v in rep.VERDICT.values()]}))
    st.markdown("**Limits to keep in mind**")
    st.markdown("- It only sees the photographed surface, not inside the wall or the cause of the damage.\n"
                "- Results need a straight-on photo and a correct size reference. Cracks under about 3 pixels wide read too wide.\n"
                "- The score's reference values are provisional and not yet calibrated against engineers' judgements.\n"
                "- This is a screening tool. An urgent result means 'get an engineer', not a final diagnosis.")
    with st.expander("For engineers: formulas"):
        st.latex(r"w(x)=\big(2\,\mathrm{DT}(x)-1\big)\cdot \mathrm{GSD},\qquad \mathrm{GSD}=\frac{L_{ref}}{L_{ref,px}}\ \ \text{or}\ \ \frac{D\,s_w}{f\,W_{px}}")
        st.latex(r"S=100\sum_i a_i\,\min\!\Big(1,\frac{f_i}{2\,f_{crit,i}}\Big),\quad f=\Big(w_{95},\ \rho,\ A_{patch}/A\Big)")
        st.latex(r"\Delta(s)=\big[\overline{NDBI}_{s..s+k-1}-\overline{NDBI}_{s-k..s-1}\big]-\big[\overline{NDVI}_{s..s+k-1}-\overline{NDVI}_{s-k..s-1}\big]")
        st.latex(r"d(t)=c+vt+a\sin 2\pi t+b\cos 2\pi t,\quad v_{vert}\approx v_{LOS}/\cos\theta,\quad R=\frac{\sum_i w_i c_i}{\sum_i w_i}")
