"""bhas_db.py -- persistent audit log + time-bound escalation (SQLite, standard library only).

Escalation rule (public facilities only):
    an audit whose final status is CRITICAL and whose workflow state is PENDING_ACTION or ESCALATED
    escalates one level up the authority ladder when  now - clock_start > delta_days,
    where clock_start = last escalation time, or creation time if never escalated.

NOTE: this generates escalation notices and logs them. It does NOT send email. On Streamlit Community
Cloud the local filesystem is ephemeral (it resets when the app restarts or is redeployed), so use the
CSV export, or move to Supabase/Postgres, for records that must survive.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3

import pandas as pd

DB_PATH = os.environ.get("BHAS_DB", "bhas_audits.db")
CRITICAL = "CRITICAL ACTION REQUIRED"
STATES = ("PENDING_ACTION", "ACKNOWLEDGED", "RESOLVED", "ESCALATED")
LADDER = ["Facility Engineer", "Executive Engineer", "Superintending Engineer",
          "Municipal Commissioner / District Collector"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    name TEXT, is_public INTEGER DEFAULT 0, lat REAL, lon REAL,
    S REAL, S_low REAL, S_high REAL, vision_status TEXT,
    width_mm_p95 REAL, length_m REAL, density_m_per_m2 REAL, calibrated INTEGER,
    age_years INTEGER, construction_year INTEGER,
    v_vert_mm_yr REAL, risk_R REAL, final_status TEXT,
    workflow_state TEXT DEFAULT 'PENDING_ACTION',
    escalation_level INTEGER DEFAULT 0,
    last_escalated_at TEXT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS escalation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_id INTEGER, at TEXT, from_level INTEGER, to_level INTEGER, message TEXT
);
"""


def connect(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")


def _parse(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def add_audit(conn, rec: dict, now: dt.datetime | None = None) -> int:
    now = now or dt.datetime.now()
    cols = ["name", "is_public", "lat", "lon", "S", "S_low", "S_high", "vision_status", "width_mm_p95",
            "length_m", "density_m_per_m2", "calibrated", "age_years", "construction_year", "v_vert_mm_yr",
            "risk_R", "final_status", "notes"]
    vals = [rec.get(c) for c in cols]
    cur = conn.execute(
        f"INSERT INTO audits (created_at, {','.join(cols)}) VALUES (?, {','.join('?' * len(cols))})",
        [_iso(now), *vals])
    conn.commit()
    return int(cur.lastrowid)


def list_audits(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM audits ORDER BY id DESC", conn)


def list_escalations(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM escalation_log ORDER BY id DESC", conn)


def set_state(conn, audit_id: int, state: str, note: str | None = None) -> None:
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}")
    conn.execute("UPDATE audits SET workflow_state=?, notes=COALESCE(?, notes) WHERE id=?",
                 (state, note, audit_id))
    conn.commit()


def run_escalation(conn, now: dt.datetime | None = None, delta_days: float = 14.0) -> list[dict]:
    """Escalate overdue critical public audits by one level each. Returns the notices generated."""
    now = now or dt.datetime.now()
    rows = conn.execute(
        "SELECT * FROM audits WHERE is_public=1 AND final_status=? AND workflow_state IN ('PENDING_ACTION','ESCALATED')",
        (CRITICAL,)).fetchall()
    notices = []
    for r in rows:
        start = _parse(r["last_escalated_at"] or r["created_at"])
        waited = (now - start).total_seconds() / 86400.0
        if waited <= delta_days:
            continue
        lvl = r["escalation_level"]
        new_lvl = min(lvl + 1, len(LADDER) - 1)
        top = new_lvl == lvl
        to_whom = LADDER[new_lvl]
        days_total = (now - _parse(r["created_at"])).total_seconds() / 86400.0
        msg = (f"ESCALATION NOTICE\n"
               f"To: {to_whom}\n"
               f"Subject: Unresolved CRITICAL structural audit #{r['id']} - {r['name'] or 'unnamed building'}\n\n"
               f"Audit #{r['id']} ({r['name'] or 'unnamed'}; lat {r['lat']}, lon {r['lon']}) was recorded "
               f"{_parse(r['created_at']).date()} with status {r['final_status']} "
               f"(severity S = {r['S']:.1f}, range {r['S_low']:.1f}-{r['S_high']:.1f}). "
               f"It has had no acknowledged action for {days_total:.0f} days "
               f"(threshold {delta_days:g} days per level). "
               + ("This is already the highest authority in the ladder; no further escalation is possible. "
                  if top else f"Escalated from {LADDER[lvl]} to {to_whom}. ")
               + "Please acknowledge the audit or assign repair action.")
        conn.execute("UPDATE audits SET workflow_state='ESCALATED', escalation_level=?, last_escalated_at=? WHERE id=?",
                     (new_lvl, _iso(now), r["id"]))
        conn.execute("INSERT INTO escalation_log (audit_id, at, from_level, to_level, message) VALUES (?,?,?,?,?)",
                     (r["id"], _iso(now), lvl, new_lvl, msg))
        notices.append({"audit_id": r["id"], "to": to_whom, "top_of_ladder": top, "message": msg})
    conn.commit()
    return notices
