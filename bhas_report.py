"""bhas_report.py -- detailed inspection report (PDF) + helpers shared with the app.

Page 1: summary for building owners (verdict, what it means, next steps, photos).
Page 2: detailed findings (measurements, crack-by-crack inventory, width classes, repair options).
Page 3: supporting data, method, limitations and sign-off.
"""
from __future__ import annotations

import io
import math

import cv2
import numpy as np
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)
from scipy import ndimage as ndi
from skimage.measure import label

import bhas_core as core

# ----------------------------------------------------------------------------- shared wording
VERDICT = {
    "STRUCTURALLY SOUND": {
        "icon": "🟢", "title": "No significant damage found", "color": "#2e7d32",
        "meaning": "The photographed area shows no cracks or damaged patches large enough to matter.",
        "steps": ["No repair needed now.",
                  "Re-inspect in about 12 months, or sooner after heavy rain or nearby construction.",
                  "Keep this report to compare against the next inspection."]},
    "MONITORING RECOMMENDED": {
        "icon": "🟠", "title": "Needs monitoring", "color": "#ef6c00",
        "meaning": "Some cracking or surface damage was found. It is not urgent on its own, but it can grow if ignored.",
        "steps": ["Ask a civil engineer to look at it (suggested within 4 to 6 weeks).",
                  "Seal visible cracks to keep water out.",
                  "Photograph the same spot again in 1 to 3 months, with the same size reference, to see if it is growing."]},
    "CRITICAL ACTION REQUIRED": {
        "icon": "🔴", "title": "Needs urgent inspection", "color": "#c62828",
        "meaning": "Significant damage was found: wide cracks, a lot of cracking, or large damaged patches.",
        "steps": ["Get a qualified structural engineer to inspect the site (suggested within 7 days).",
                  "Consider restricting access to the affected area until it has been checked.",
                  "Do not just paint over the cracks; the cause needs to be found first."]},
}

# report width classes (mm, by typical width of each crack). Indicative, not a code classification.
WIDTH_CLASSES = [(0.1, "Hairline"), (0.3, "Fine"), (1.0, "Medium"), (float("inf"), "Wide")]
REPAIR_OPTIONS = {
    "Hairline": "Usually cosmetic. Coat or seal during routine maintenance and monitor.",
    "Fine": "Seal to keep water out (flexible sealant or low-viscosity resin) and monitor for growth.",
    "Medium": "Engineer should find the cause first. Resin injection or rout-and-seal are common repairs.",
    "Wide": "Structural engineer assessment before any repair; may indicate movement or overloading.",
}


def width_class(w_mm: float) -> str:
    for lim, name in WIDTH_CLASSES:
        if w_mm < lim:
            return name
    return "Wide"


def report_id(res) -> str:
    return f"BHAS-{res['created_at']:%Y%m%d-%H%M%S}"


# ----------------------------------------------------------------------------- crack inventory
def crack_inventory(vr) -> list:
    """One entry per connected crack: length, typical (p95) and max width, orientation, width class."""
    g = vr.measurement.gsd_mm_per_px
    lab = label(vr.crack, connectivity=2)
    if lab.max() == 0:
        return []
    dt = ndi.distance_transform_edt(vr.crack)
    skel = vr.skeleton
    items = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        comp = lab[sl] == i
        sk = skel[sl] & comp
        if sk.sum() < 3:
            continue
        w = np.clip(2 * dt[sl][sk] - 1, 1, None)
        L = core.skeleton_length(sk)
        ys, xs = np.nonzero(sk)
        ev, evec = np.linalg.eigh(np.cov(np.vstack([xs, ys]).astype(float)))
        vx, vy = evec[:, -1]
        ang = (math.degrees(math.atan2(-vy, vx)) + 180.0) % 180.0
        orient = ("Horizontal" if (ang < 22.5 or ang >= 157.5) else "Vertical" if 67.5 <= ang < 112.5 else "Diagonal")
        it = {"length_px": float(L), "w95_px": float(np.percentile(w, 95)), "wmax_px": float(w.max()),
              "orientation": orient, "cx": float(sl[1].start + xs.mean()), "cy": float(sl[0].start + ys.mean())}
        if g:
            it.update(length_mm=L * g, w95_mm=it["w95_px"] * g, wmax_mm=it["wmax_px"] * g,
                      cls=width_class(it["w95_px"] * g))
        items.append(it)
    items.sort(key=lambda d: -d["length_px"])
    for k, it in enumerate(items, start=1):
        it["id"] = f"C{k}"
    return items


def numbered_overlay(vr, items, max_labels: int = 15) -> np.ndarray:
    img = vr.overlay.copy()
    h, w = img.shape[:2]
    fs = max(0.4, min(w, h) / 900.0)
    th = max(1, int(round(fs * 2)))
    for it in items[:max_labels]:
        txt = it["id"]
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        x = int(min(max(2, it["cx"] - tw / 2), w - tw - 4))
        y = int(min(max(tht + 4, it["cy"] - 6), h - 4))
        cv2.rectangle(img, (x - 3, y - tht - 4), (x + tw + 3, y + 4), (255, 255, 255), -1)
        cv2.rectangle(img, (x - 3, y - tht - 4), (x + tw + 3, y + 4), (30, 30, 30), 1)
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th, cv2.LINE_AA)
    return img


def class_summary(items) -> list:
    out = []
    for _, name in WIDTH_CLASSES:
        sel = [i for i in items if i.get("cls") == name]
        if sel:
            out.append({"class": name, "count": len(sel), "length_m": sum(i["length_mm"] for i in sel) / 1000.0,
                        "repair": REPAIR_OPTIONS[name]})
    return out


def score_breakdown(res) -> list:
    """How many of the S points each factor contributes (they add up to S)."""
    vr, sp = res["vision"], res["params"]["severity"]
    comps = vr.severity["components"]
    a1, a2, a3 = sp.weights
    INFO = {"width": ("Crack width", "how wide the cracks are compared with the safe limit", a1),
            "density": ("Amount of cracking", "total crack length per square metre of wall", a2),
            "patch": ("Damaged patches", "dark patches such as spalling or damp, as a share of the photo", a3),
            "area": ("Cracked area", "share of the photo covered by cracks (no scale given)", a1 + a2)}
    keys = vr.severity.get("keys") or (["width", "density", "patch"] if vr.measurement.calibrated else ["area", "patch"])
    w = np.array([INFO[k][2] for k in keys]); w = w / w.sum()
    return [{"factor": INFO[k][0], "explain": INFO[k][1], "points": 100.0 * wi * ci, "max_points": 100.0 * wi, "level": ci}
            for k, wi, ci in zip(keys, w, comps)]


# ----------------------------------------------------------------------------- PDF building blocks
def _png(rgb: np.ndarray, max_w: int = 1000) -> io.BytesIO:
    h, w = rgb.shape[:2]
    if w > max_w:
        rgb = cv2.resize(rgb, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return io.BytesIO(buf.tobytes())


def _img(rgb, width_mm, max_h_mm=None):
    h, w = rgb.shape[:2]
    W, H = width_mm, width_mm * h / w
    if max_h_mm and H > max_h_mm:
        W, H = W * max_h_mm / H, max_h_mm
    return Image(_png(rgb), width=W * mm, height=H * mm)


def _f(v, nd=2, unit=""):
    return "n/a" if v is None else f"{v:.{nd}f}{unit}"


def _table(rows, widths, header=True, font=8.4):
    t = Table(rows, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), font), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c5cdd3")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474f")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")])]
    t.setStyle(TableStyle(st))
    return t


def _hist_chart(vr, width_mm=85, height_mm=48):
    m = vr.measurement
    prof = vr.width_profile_px
    d = Drawing(width_mm * mm, height_mm * mm)
    if prof is None or len(prof) == 0:
        d.add(String(10, height_mm * mm / 2, "No cracks detected", fontSize=8))
        return d
    g = m.gsd_mm_per_px or 1.0
    vals = prof * g
    counts, edges = np.histogram(vals, bins=10)
    bc = VerticalBarChart()
    bc.x, bc.y, bc.width, bc.height = 28, 22, width_mm * mm - 38, height_mm * mm - 34
    bc.data = [list(counts.astype(int))]
    bc.categoryAxis.categoryNames = [f"{e:.2f}" if m.calibrated else f"{e:.0f}" for e in edges[:-1]]
    bc.categoryAxis.labels.fontSize = 6
    bc.categoryAxis.labels.angle = 45
    bc.categoryAxis.labels.dy = -6
    bc.valueAxis.labels.fontSize = 6
    bc.valueAxis.valueMin = 0
    bc.bars[0].fillColor = colors.HexColor("#546e7a")
    d.add(bc)
    d.add(String(28, height_mm * mm - 9, "Crack width along the crack centrelines (" + ("mm" if m.calibrated else "px") + ")",
                 fontSize=7))
    return d


def _age_chart(age, width_mm=170, height_mm=45):
    d = Drawing(width_mm * mm, height_mm * mm)
    yrs, ndbi, ndvi = age["years"], age["ndbi"], age["ndvi"]
    ok = np.isfinite(ndbi) & np.isfinite(ndvi)
    if ok.sum() < 2:
        return d
    lp = LinePlot()
    lp.x, lp.y, lp.width, lp.height = 30, 18, width_mm * mm - 45, height_mm * mm - 30
    lp.data = [list(zip(yrs[ok].tolist(), ndbi[ok].tolist())), list(zip(yrs[ok].tolist(), ndvi[ok].tolist()))]
    lp.lines[0].strokeColor = colors.HexColor("#c62828")
    lp.lines[1].strokeColor = colors.HexColor("#2e7d32")
    lp.xValueAxis.labels.fontSize = lp.yValueAxis.labels.fontSize = 6
    d.add(lp)
    d.add(String(30, height_mm * mm - 8, "Built-up index NDBI (red) and vegetation index NDVI (green), one value per year",
                 fontSize=7))
    return d


def _footer(rid):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(16 * mm, 9 * mm, f"{rid}  |  Automated screening report, not a certified structural assessment")
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()
    return draw


# ----------------------------------------------------------------------------- the report
def build_pdf(res: dict, inspector: str = "") -> bytes:
    vr, fu, age, ins = res["vision"], res["fusion"], res["age"], res["insar"]
    inp, m, sp = res["inputs"], vr.measurement, res["params"]["severity"]
    status = fu["final_status"]
    V = VERDICT[status]
    S = vr.severity["S"]
    rid = report_id(res)
    items = crack_inventory(vr)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=13 * mm,
                            bottomMargin=16 * mm, title=f"Building inspection report {rid}", author="BHAS")
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=18, leading=22, alignment=0, spaceAfter=2)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=7, spaceAfter=3,
                        textColor=colors.HexColor("#263238"))
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13)
    SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10.4, textColor=colors.HexColor("#444444"))
    P = lambda s: Paragraph(s, SM)  # noqa: E731
    story = []

    # ===================== PAGE 1: summary =====================
    story += [Paragraph("Building Inspection Report", H1),
              Paragraph(f"Report {rid}", SM), Spacer(1, 3 * mm)]
    scale_txt = (f"Calibrated ({m.gsd_mm_per_px:.3f} mm per pixel)" if m.calibrated
                 else "Not calibrated: sizes in pixels only")
    info = [[P("<b>Building</b>"), P(inp.name or "Not named"), P("<b>Inspection date</b>"), P(f"{res['created_at']:%d %b %Y, %H:%M}")],
            [P("<b>Location</b>"), P(f"{inp.lat:.5f}, {inp.lon:.5f}"), P("<b>Inspector</b>"), P(inspector or "Not given")],
            [P("<b>Facility type</b>"), P("Public" if inp.is_public else "Private"), P("<b>Measurement scale</b>"), P(scale_txt)]]
    story += [_table(info, [28, 61, 32, 57], header=False, font=8), Spacer(1, 4 * mm)]

    banner = Table([[f"{V['title'].upper()}"]], colWidths=[178 * mm], rowHeights=[12 * mm])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(V["color"])),
                                ("TEXTCOLOR", (0, 0), (-1, -1), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 15),
                                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [banner, Spacer(1, 2 * mm), Paragraph(V["meaning"], B)]

    # score bar + key facts
    gauge = Table([["", ""]], colWidths=[max(1.0, 178 * mm * S / 100), max(1.0, 178 * mm * (1 - S / 100))],
                  rowHeights=[5 * mm])
    gauge.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), colors.HexColor(VERDICT[vr.severity['status']]['color'])),
                               ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#e3e6e8"))]))
    story += [Paragraph("Damage score", H2), gauge,
              Paragraph(f"<b>{S:.0f} / 100</b> (likely range {vr.S_low:.0f} to {vr.S_high:.0f}). "
                        "0 means no visible damage; 100 means severe. Score bands: 0 to "
                        f"{sp.sound_max:g} no significant damage, up to {sp.monitor_max:g} needs monitoring, above that urgent.", SM)]
    facts = [["What we measured", "Result"]]
    if m.calibrated:
        facts += [["Typical width of the widest cracks", f"{m.width_mm_p95:.2f} mm  (reference limit {sp.w_crit_mm:g} mm)"],
                  ["Total length of cracks in the photo", f"{m.length_m:.2f} m"],
                  ["Number of separate cracks", f"{len(items)}"],
                  ["Damaged patches (spalling or damp)", f"{m.blob_area_m2 * 1e4:.0f} cm2 ({m.blob_area_frac * 100:.1f}% of the photo)"]]
    else:
        facts += [["Number of separate cracks", f"{len(items)}"],
                  ["Share of photo covered by cracks", f"{m.crack_area_frac * 100:.2f}%"],
                  ["Damaged patches (spalling or damp)", f"{m.blob_area_frac * 100:.1f}% of the photo"],
                  ["Crack widths in mm", "Not available: no size reference was given"]]
    story += [Spacer(1, 2 * mm), _table(facts, [80, 98])]

    photos = Table([[_img(vr.original_rgb, 86, 70), _img(numbered_overlay(vr, items), 86, 70)],
                    [P("Photo as supplied"), P("Detected cracks (red, numbered) and damaged patches (orange)")]],
                   colWidths=[89 * mm, 89 * mm])
    photos.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [Spacer(1, 3 * mm), photos]
    story += [KeepTogether([Paragraph("What to do next", H2)] +
                           [Paragraph(f"{k}.&nbsp;&nbsp;{s}", B) for k, s in enumerate(V["steps"], start=1)])]
    story.append(PageBreak())

    # ===================== PAGE 2: detailed findings =====================
    story += [Paragraph("Detailed findings", H1), Paragraph("For engineers and repair contractors.", SM)]
    bd = score_breakdown(res)
    rows = [["Factor", "What it measures", "Points"]]
    rows += [[b["factor"], P(b["explain"]), f"{b['points']:.1f} of {b['max_points']:.0f}"] for b in bd]
    rows += [["Total", "", f"{S:.1f}"]]
    t = _table(rows, [40, 108, 30])
    t.setStyle(TableStyle([("FONTNAME", (0, len(rows) - 1), (-1, len(rows) - 1), "Helvetica-Bold")]))
    story += [Paragraph("Why this score", H2), t]

    story += [Paragraph("Crack-by-crack inventory", H2)]
    if not items:
        story.append(Paragraph("No cracks were detected in the photographed area.", B))
    else:
        if m.calibrated:
            rows = [["ID", "Length", "Typical width", "Max width", "Direction", "Width class"]]
            rows += [[i["id"], f"{i['length_mm'] / 10:.1f} cm", f"{i['w95_mm']:.2f} mm", f"{i['wmax_mm']:.2f} mm",
                      i["orientation"], i["cls"]] for i in items[:15]]
            widths = [14, 26, 30, 28, 34, 46]
        else:
            rows = [["ID", "Length (px)", "Typical width (px)", "Max width (px)", "Direction"]]
            rows += [[i["id"], f"{i['length_px']:.0f}", f"{i['w95_px']:.1f}", f"{i['wmax_px']:.1f}", i["orientation"]]
                     for i in items[:15]]
            widths = [16, 36, 44, 40, 42]
        story.append(_table(rows, widths))
        if len(items) > 15:
            story.append(Paragraph(f"{len(items) - 15} shorter crack segments are not listed; they are included in all totals.", SM))
        story.append(Paragraph("IDs match the numbers on the photo. Typical width is the 95th percentile along the crack, "
                               "which ignores single-pixel spikes; max width is the single widest point.", SM))

    cs = class_summary(items)
    if cs:
        rows = [["Width class", "Cracks", "Total length", "Typical repair options (to confirm with an engineer)"]]
        rows += [[c["class"], str(c["count"]), f"{c['length_m']:.2f} m", P(c["repair"])] for c in cs]
        story += [Paragraph("Repair quantities by width class", H2), _table(rows, [26, 16, 26, 110]),
                  Paragraph("Classes used here: hairline under 0.1 mm, fine 0.1 to 0.3 mm, medium 0.3 to 1 mm, wide 1 mm "
                            "and over (by typical width). They are indicative report categories, not a design-code classification.", SM)]
    elif not m.calibrated and items:
        story += [Paragraph("Repair quantities by width class", H2),
                  Paragraph("Not available: widths in millimetres need a size reference in the photo.", B)]

    story += [Spacer(1, 3 * mm),
              Table([[_hist_chart(vr), _img(vr.width_map, 84, 62)],
                     ["", P("Crack centrelines coloured by width" + (" relative to the limit (red = 1.5x the limit or wider)."
                                                                     if m.calibrated else " (red = widest in this photo)."))]],
                    colWidths=[90 * mm, 88 * mm])]
    story.append(PageBreak())

    # ===================== PAGE 3: supporting data and method =====================
    story += [Paragraph("Supporting data, method and limitations", H1)]
    story += [Paragraph("Satellite: estimated construction year", H2)]
    if age is None:
        story.append(Paragraph("Not assessed for this inspection.", B))
    else:
        if age["status"] == "detected":
            story.append(Paragraph(f"Built-up signal first appears in <b>{age['year']}</b> (range {age['year_range'][0]} "
                                   f"to {age['year_range'][1]}), so the building is roughly <b>{age['age']} years</b> old. "
                                   f"Step size {age['delta_max']:.2f}, confidence z = {age['z']:.1f}.", B))
        else:
            story.append(Paragraph(age["message"], B))
        story.append(_age_chart(age))
    story += [Paragraph("Satellite: ground movement (InSAR)", H2)]
    if ins is None:
        story.append(Paragraph("No InSAR data was supplied, so ground movement was not used.", B))
    else:
        txt = f"Line-of-sight velocity {ins['v_los_mm_yr']:.2f} mm/yr"
        if ins.get("v_los_se"):
            txt += f" (+/- {ins['v_los_se']:.2f})"
        if ins["v_vert_mm_yr"] is not None:
            txt += f"; vertical velocity {ins['v_vert_mm_yr']:.2f} mm/yr (negative usually means sinking)."
        story.append(Paragraph(txt, B))
    story += [Paragraph("How the overall verdict was reached", H2),
              Paragraph(f"Photo-only verdict: {VERDICT[vr.severity['status']]['title']}. Combined risk R = {fu['R']:.2f} "
                        f"using {', '.join(fu['modalities'])}, giving: {VERDICT[fu['fused_status']]['title']}. "
                        "The report shows the more serious of the two.", B)]
    if res["warnings"]:
        story += [Paragraph("Notes and warnings for this inspection", H2)] + [P("- " + w) for w in res["warnings"]]
    method = [
        "Cracks are found with image processing: lighting is evened out, thin dark lines are enhanced, and only long, thin "
        "shapes are kept so that pores and stains are not counted as cracks.",
        "Width is measured at every point along each crack's centreline and converted to millimetres using the size reference.",
        f"Damage score S = 100 x sum of (weight x min(1, measured / (2 x reference value))), with reference values "
        f"{sp.w_crit_mm:g} mm crack width, {sp.rho_crit_m_per_m2:g} m of crack per m2, and {sp.a_crit * 100:g}% damaged area.",
        "The reference values, weights and score bands are provisional defaults that have not yet been calibrated against "
        "engineers' assessments. Use the score to rank and prioritise, not as a code-compliance check.",
        "Only the photographed surface is assessed. The method cannot see inside the structure, behind finishes, or the cause "
        "of the damage. Accuracy needs the camera square-on to the wall and a correct size reference.",
    ]
    story += [Paragraph("Method and limitations", H2)] + [P("- " + s) for s in method]
    sign = Table([[P("<b>Prepared by</b> (automated system)"), P("<b>Reviewed by</b> (engineer)")],
                  [P(f"{inspector or '________________'}<br/>Date: {res['created_at']:%d %b %Y}"),
                   P("Name: ______________________<br/><br/>Signature: _________________&nbsp;&nbsp;Date: __________")]],
                 colWidths=[89 * mm, 89 * mm], rowHeights=[7 * mm, 20 * mm])
    sign.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#90a4ae")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [Spacer(1, 4 * mm), KeepTogether([Paragraph("Sign-off", H2), sign])]

    doc.build(story, onFirstPage=_footer(rid), onLaterPages=_footer(rid))
    return buf.getvalue()
