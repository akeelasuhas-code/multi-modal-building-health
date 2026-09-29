"""bhas_report.py -- two-audience PDF: page 1 plain language, page 2 technical + contractor quantities."""
from __future__ import annotations

import io

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

STATUS_COLOR = {"STRUCTURALLY SOUND": colors.HexColor("#2e7d32"),
                "MONITORING RECOMMENDED": colors.HexColor("#ef6c00"),
                "CRITICAL ACTION REQUIRED": colors.HexColor("#c62828")}

PLAIN = {
    "STRUCTURALLY SOUND": ("No significant surface damage was found in the photographed area.",
                           "No repair is needed now. Re-inspect at your normal interval or after heavy rain or nearby construction."),
    "MONITORING RECOMMENDED": ("Some cracking or surface damage was found. It is not urgent on its own but can worsen if ignored.",
                               "Have a civil engineer look at it within the next few weeks, seal visible cracks, and re-photograph in 1 to 3 months to see if it is growing."),
    "CRITICAL ACTION REQUIRED": ("Significant damage was found. Wide cracks, extensive cracking or large damaged areas can affect safety and durability.",
                                 "Arrange a site inspection by a qualified structural engineer promptly. Consider restricting access to the affected area until it has been checked."),
}


def _png(rgb: np.ndarray, max_w: int = 900) -> io.BytesIO:
    h, w = rgb.shape[:2]
    if w > max_w:
        rgb = cv2.resize(rgb, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return io.BytesIO(buf.tobytes())


def _img(rgb, width_mm):
    h, w = rgb.shape[:2]
    return Image(_png(rgb), width=width_mm * mm, height=width_mm * mm * h / w)


def _fmt(v, nd=2, unit=""):
    return "n/a" if v is None else f"{v:.{nd}f}{unit}"


def build_pdf(result: dict) -> bytes:
    vr, fusion, age, insar = result["vision"], result["fusion"], result["age"], result["insar"]
    inp, m, sp = result["inputs"], result["vision"].measurement, result["params"]["severity"]
    status = fusion["final_status"]
    S = vr.severity["S"]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title="Structural Health Audit", author="BHAS")
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=19, leading=23, alignment=0, spaceAfter=4)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=8, spaceAfter=3)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=13.5)
    SM = ParagraphStyle("SM", parent=B, fontSize=8.3, leading=10.5, textColor=colors.HexColor("#444444"))
    story = []

    # ---------------- page 1: layman ----------------
    story += [Paragraph("Structural Health Audit", H1),
              Paragraph(f"{inp.name or 'Unnamed building'} &nbsp;|&nbsp; {result['created_at']:%d %b %Y} &nbsp;|&nbsp; "
                        f"lat {inp.lat:.5f}, lon {inp.lon:.5f}", SM), Spacer(1, 5 * mm)]
    banner = Table([[status]], colWidths=[178 * mm], rowHeights=[13 * mm])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), STATUS_COLOR[status]),
                                ("TEXTCOLOR", (0, 0), (-1, -1), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 15),
                                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [banner, Spacer(1, 4 * mm)]
    what, todo = PLAIN[status]
    story += [Paragraph("What this means", H2), Paragraph(what, B),
              Paragraph("What to do", H2), Paragraph(todo, B), Paragraph("Damage score", H2)]
    # simple bar gauge
    g = Table([["", ""]], colWidths=[max(1.0, 178 * mm * S / 100), max(1.0, 178 * mm * (1 - S / 100))], rowHeights=[6 * mm])
    g.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), STATUS_COLOR[vr.severity["status"]]),
                           ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#e0e0e0"))]))
    story += [g, Paragraph(f"{S:.0f} out of 100 (plausible range {vr.S_low:.0f} to {vr.S_high:.0f}, depending on detection "
                           f"sensitivity). 0 = no visible damage, 100 = severe.", SM), Spacer(1, 3 * mm),
              _img(vr.overlay, 120),
              Paragraph("Red = cracks found. Orange = dark or damaged patches. Only the photographed area is assessed.", SM),
              Spacer(1, 3 * mm),
              Paragraph("This is an automated screening aid, not a certified structural assessment. It cannot see inside the "
                        "structure and only reflects the photo supplied.", SM),
              PageBreak()]

    # ---------------- page 2: technical ----------------
    story += [Paragraph("Technical summary for engineers and contractors", H1)]
    cal = "calibrated" if m.calibrated else "UNCALIBRATED (pixel units)"
    rows = [["Measurement", "Value"],
            ["Scale", f"{cal}" + (f", {m.gsd_mm_per_px:.3f} mm/px (processed image)" if m.calibrated else "")],
            ["Crack width, 95th percentile", _fmt(m.width_mm_p95, 2, " mm") if m.calibrated else _fmt(m.width_px_p95, 1, " px")],
            ["Crack width, maximum", _fmt(m.width_mm_max, 2, " mm") if m.calibrated else _fmt(m.width_px_max, 1, " px")],
            ["Total crack length", _fmt(m.length_m, 2, " m") if m.calibrated else _fmt(m.length_px, 0, " px")],
            ["Crack density", _fmt(m.density_m_per_m2, 2, " m/m2") if m.calibrated else _fmt(m.density_px_per_px2, 4, " px/px2")],
            ["Dark anomaly area (possible spalling/moisture)",
             (f"{m.blob_area_m2 * 1e4:.1f} cm2 ({m.blob_area_frac * 100:.1f}% of image)" if m.calibrated
              else f"{m.blob_area_frac * 100:.1f}% of image")],
            ["Imaged surface area", _fmt(m.image_area_m2, 2, " m2") if m.calibrated else "n/a"],
            ["Severity S (basis)", f"{S:.1f}  [{vr.S_low:.1f} to {vr.S_high:.1f}]  -  {vr.severity['basis']}"],
            ["Severity components (0-1)", ", ".join(f"{c:.2f}" for c in vr.severity["components"])]]
    if age and age["status"] == "detected":
        rows.append(["Construction year (Landsat step detector)",
                     f"{age['year']} (range {age['year_range'][0]}-{age['year_range'][1]}); age {age['age']} y; "
                     f"step {age['delta_max']:.2f}, z = {age['z']:.1f}"])
    else:
        rows.append(["Construction year", "not estimated" + (f" ({age['message']})" if age else "")])
    if insar:
        rows.append(["InSAR vertical velocity",
                     _fmt(insar["v_vert_mm_yr"], 2, " mm/yr") + (f" +/- {insar['v_vert_se']:.2f}" if insar.get("v_vert_se") else "")
                     + f"  (LOS {insar['v_los_mm_yr']:.2f} mm/yr)"])
    else:
        rows.append(["InSAR", "no data supplied - not used"])
    rows.append(["Fused risk R", f"{fusion['R']:.2f} from " + ", ".join(fusion["modalities"]) +
                 f"  -> {fusion['fused_status']}"])
    # wrap long cells
    for r in rows[1:]:
        r[0], r[1] = Paragraph(str(r[0]), SM), Paragraph(str(r[1]), SM)
    t = Table(rows, colWidths=[62 * mm, 116 * mm], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474f")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, 0), 9),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0bec5")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7f8")])]))
    story += [t, Spacer(1, 4 * mm)]
    ims = Table([[_img(vr.overlay, 84), _img(vr.width_map, 84)]], colWidths=[89 * mm, 89 * mm])
    story += [ims, Paragraph("Left: detected cracks (red) and dark anomalies (orange). Right: crack centrelines coloured by local width "
                             + ("(blue = thin, red = 1.5x the crack-width limit or wider)." if m.calibrated else "(blue = thin, red = widest in this image)."), SM), Spacer(1, 3 * mm)]

    lim = ["S = 100 x sum(a_i x min(1, f_i / (2 x f_crit_i))) with f = crack width p95 (critical " + f"{sp.w_crit_mm:g} mm), crack density "
           f"(critical {sp.rho_crit_m_per_m2:g} m/m2), dark-anomaly fraction (critical {sp.a_crit:g}); weights "
           f"{tuple(round(x, 2) for x in sp.weights)}. A component is half its maximum at its critical value.",
           "Widths come from a distance transform on the crack skeleton; accuracy needs the camera square-on to the wall "
           "and a correct scale. Cracks thinner than about 3 pixels are over-estimated.",
           "Crack-critical width and the other reference values above are PROVISIONAL defaults pending calibration against "
           "expert grades; treat the numeric severity as a triage ranking, not a code-compliance check.",
           "Dark anomalies are not classified by type; a shadow, stain or spall can look alike. Band-like shadows are ignored.",
           "Satellite age is a 30 m step detector on NDBI/NDVI; it is not estimated when the signal is unclear. "
           "InSAR is used only when data are supplied."]
    story += [Paragraph("Method and limitations", H2)] + [Paragraph("- " + s, SM) for s in lim]
    doc.build(story)
    return buf.getvalue()
