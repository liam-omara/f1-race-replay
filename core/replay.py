"""Turns a loaded FastF1 session into replay-ready arrays, and answers
"what does the race look like at time t?".

No Streamlit or Plotly in here, so it is easy to test.

Design note: the timing tower is built from lap timing data (line-crossing
times), not from X/Y positions. Progress is interpolated linearly in time
between crossings, which is exact at every crossing, monotonic, and has none
of the wrap-around glitches you get when projecting positions onto the track
near the start/finish line. The gap between two cars is the time between them
reaching the same point of the race, the same idea as real timing loops.
The dots on the map still use real X/Y, so mid-lap the map and the tower can
differ very slightly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.track import get_track_outline, rotate_xy

STATUS_STYLE = {
    "1": ("TRACK CLEAR", "#22C55E"),
    "2": ("YELLOW FLAG", "#FACC15"),
    "3": ("YELLOW FLAG", "#FACC15"),
    "4": ("SAFETY CAR", "#F97316"),
    "5": ("RED FLAG", "#EF4444"),
    "6": ("VIRTUAL SAFETY CAR", "#F97316"),
    "7": ("VSC ENDING", "#FACC15"),
}

DRS_OPEN = (10, 12, 14)


def _empty():
    return np.array([], dtype=float)


@dataclass
class DriverData:
    abbr: str
    number: str
    name: str
    team: str
    color: str
    grid: float
    finished: bool
    cross_t: np.ndarray            # session seconds at each line crossing (first = race start)
    cross_lap: np.ndarray          # laps completed at each crossing (first = 0)
    out_time: float = float("inf")  # when a non-finisher stops for good
    lap_end_t: np.ndarray = field(default_factory=_empty)
    lap_compound: list = field(default_factory=list)
    lap_tyre: np.ndarray = field(default_factory=_empty)
    pit_in: np.ndarray = field(default_factory=_empty)
    pit_out: np.ndarray = field(default_factory=_empty)
    car_t: np.ndarray = field(default_factory=_empty)
    car_speed: np.ndarray = field(default_factory=_empty)
    car_throttle: np.ndarray = field(default_factory=_empty)
    car_brake: np.ndarray = field(default_factory=_empty)
    car_gear: np.ndarray = field(default_factory=_empty)
    car_drs: np.ndarray = field(default_factory=_empty)
    car_rpm: np.ndarray = field(default_factory=_empty)


@dataclass
class Standing:
    pos: int
    idx: int            # index into Replay.drivers
    abbr: str
    color: str
    gap: str
    interval: str
    compound: str
    tyre_age: int | None
    in_pit: bool
    out: bool
    finished: bool      # has taken the chequered flag
    progress: float     # laps completed (fractional)


@dataclass
class Replay:
    title: str
    t0: float
    dt: float
    times: np.ndarray
    total_laps: int
    drivers: list
    x: np.ndarray       # (n_drivers, n_times), rotated to match the outline
    y: np.ndarray
    outline: np.ndarray
    corners: pd.DataFrame | None
    rotation: float
    status_t: np.ndarray = field(default_factory=_empty)
    status_code: list = field(default_factory=list)


# ───────────────────────────── building ─────────────────────────────

def _sorted_unique(t, *arrays):
    t, idx = np.unique(t, return_index=True)
    return (t, *[a[idx] for a in arrays])


def _race_start(session, laps):
    try:
        ss = session.session_status
        started = ss.loc[ss["Status"] == "Started", "Time"]
        if len(started):
            return float(started.iloc[0].total_seconds())
    except Exception:
        pass
    try:
        s = laps.loc[laps["LapNumber"] == 1, "LapStartTime"].dropna()
        if len(s):
            return float(s.min().total_seconds())
    except Exception:
        pass
    first = laps.loc[laps["LapNumber"] == 1, "Time"].dropna()
    return float(first.min().total_seconds()) - 100.0


def _is_finished(status, n_laps, total_laps):
    status = "" if status is None or (isinstance(status, float) and np.isnan(status)) else str(status)
    if status.strip() == "":
        return n_laps >= 0.9 * total_laps
    s = status.lower()
    return s == "finished" or s.startswith("+") or "lap" in s


def _build_driver(session, laps, r, t0, total_laps):
    number = str(r["DriverNumber"])
    abbr = str(r["Abbreviation"])
    if number not in session.pos_data or number not in session.car_data:
        return None
    d_all = laps[laps["Driver"] == abbr].sort_values("LapNumber")
    if d_all.empty:
        return None

    finished = _is_finished(r.get("Status"), d_all["LapNumber"].max(), total_laps)

    # Line crossings. A retiree's last, unfinished lap is not a crossing.
    d_cross = d_all
    if not finished and pd.isna(d_cross.iloc[-1]["LapTime"]):
        d_cross = d_cross.iloc[:-1]
    t_end = d_cross["Time"].dt.total_seconds().to_numpy(float)
    laps_done = d_cross["LapNumber"].to_numpy(float)
    ok = np.isfinite(t_end)
    cross_t = np.concatenate([[t0], t_end[ok]])
    cross_lap = np.concatenate([[0.0], laps_done[ok]])
    running_max = np.maximum.accumulate(cross_t)
    keep = np.ones(len(cross_t), dtype=bool)
    keep[1:] = cross_t[1:] > running_max[:-1] + 1e-3
    cross_t, cross_lap = cross_t[keep], cross_lap[keep]

    # Tyre info per lap
    lap_end = d_all["Time"].dt.total_seconds().bfill().fillna(np.inf).to_numpy(float)
    lap_end = np.maximum.accumulate(lap_end)
    compound = d_all["Compound"].fillna("UNKNOWN").astype(str).str.upper().tolist()
    tyre = pd.to_numeric(d_all["TyreLife"], errors="coerce").to_numpy(float)

    # Pit stops as (in, out) pairs
    pit_in = np.sort(d_all["PitInTime"].dropna().dt.total_seconds().to_numpy(float))
    pit_out_all = np.sort(d_all["PitOutTime"].dropna().dt.total_seconds().to_numpy(float))
    pit_out = np.array(
        [pit_out_all[pit_out_all > a].min() if (pit_out_all > a).any() else np.inf for a in pit_in],
        dtype=float,
    )

    # Car telemetry
    car = session.car_data[number]
    ct = car["SessionTime"].dt.total_seconds().to_numpy(float)
    arrays = [
        car["Speed"].to_numpy(float),
        car["Throttle"].to_numpy(float),
        car["Brake"].to_numpy(float) * 100.0,
        car["nGear"].to_numpy(float),
        car["DRS"].to_numpy(float),
        car["RPM"].to_numpy(float),
    ]
    good = np.isfinite(ct)
    ct, speed, throttle, brake, gear, drs, rpm = _sorted_unique(ct[good], *[a[good] for a in arrays])

    if finished:
        out_time = float("inf")
    else:
        moving = np.where(speed > 10)[0]
        out_time = float(ct[moving[-1]]) if len(moving) else float(cross_t[-1])

    color = r.get("TeamColor")
    color = "#" + color if isinstance(color, str) and len(color) == 6 else "#9CA3AF"
    grid = pd.to_numeric(r.get("GridPosition"), errors="coerce")
    grid = 21.0 if pd.isna(grid) or grid <= 0 else float(grid)

    return DriverData(
        abbr=abbr,
        number=number,
        name=str(r.get("FullName") or abbr),
        team=str(r.get("TeamName") or ""),
        color=color,
        grid=grid,
        finished=bool(finished),
        cross_t=cross_t,
        cross_lap=cross_lap,
        out_time=out_time,
        lap_end_t=lap_end,
        lap_compound=compound,
        lap_tyre=tyre,
        pit_in=pit_in,
        pit_out=pit_out,
        car_t=ct,
        car_speed=speed,
        car_throttle=throttle,
        car_brake=brake,
        car_gear=gear,
        car_drs=drs,
        car_rpm=rpm,
    )


def build_replay(session, dt=1.0):
    """Resample every driver onto one shared timeline and collect timing data."""
    laps = session.laps
    total_laps = int(laps["LapNumber"].max())
    t0 = _race_start(session, laps)
    outline, corners, rotation = get_track_outline(session)

    drivers = []
    for _, r in session.results.iterrows():
        d = _build_driver(session, laps, r, t0, total_laps)
        if d is not None:
            drivers.append(d)
    if not drivers:
        raise ValueError("No usable driver data found for this race.")

    t_end = max(float(d.cross_t[-1]) for d in drivers)
    n = int((t_end - t0) // dt) + 1
    times = t0 + np.arange(n) * dt

    x = np.full((len(drivers), n), np.nan)
    y = np.full((len(drivers), n), np.nan)
    for k, d in enumerate(drivers):
        pos = session.pos_data[d.number]
        pt = pos["SessionTime"].dt.total_seconds().to_numpy(float)
        px = pos["X"].to_numpy(float)
        py = pos["Y"].to_numpy(float)
        ok = np.isfinite(pt) & np.isfinite(px) & np.isfinite(py)
        pt, px, py = _sorted_unique(pt[ok], px[ok], py[ok])
        if len(pt) < 2:
            continue
        x[k], y[k] = rotate_xy(np.interp(times, pt, px), np.interp(times, pt, py), rotation)

    try:
        ts = session.track_status
        status_t = ts["Time"].dt.total_seconds().to_numpy(float)
        status_code = ts["Status"].astype(str).tolist()
    except Exception:
        status_t, status_code = _empty(), []

    try:
        title = f"{session.event['EventName']} {session.event['EventDate'].year}"
    except Exception:
        title = "Race"

    return Replay(
        title=title, t0=t0, dt=dt, times=times, total_laps=total_laps, drivers=drivers,
        x=x, y=y, outline=outline, corners=corners, rotation=rotation,
        status_t=status_t, status_code=status_code,
    )


# ───────────────────────── querying at time t ─────────────────────────

def fmt_clock(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def track_status_at(rp, t):
    if len(rp.status_t) == 0:
        return "1"
    i = int(np.searchsorted(rp.status_t, t, side="right")) - 1
    return rp.status_code[i] if i >= 0 else "1"


def _tyre_at(d, t):
    if len(d.lap_end_t) == 0:
        return "UNKNOWN", None
    i = min(int(np.searchsorted(d.lap_end_t, t, side="right")), len(d.lap_end_t) - 1)
    age = d.lap_tyre[i] if i < len(d.lap_tyre) else np.nan
    return d.lap_compound[i], (None if np.isnan(age) else int(age))


def _in_pit(d, t):
    return bool(np.any((d.pit_in <= t) & (t < d.pit_out)))


def _gap_text(a, b, t):
    """Gap of driver b (info dict) behind driver a (info dict), in seconds or laps."""
    laps_down = int(a["p"] - b["p"] + 1e-9)
    if laps_down >= 1:
        return f"+{laps_down} LAP" + ("S" if laps_down > 1 else "")
    da, db = a["d"], b["d"]
    tb = min(t, float(db.cross_t[-1]))
    t_a_at_b_progress = float(np.interp(b["p"], da.cross_lap, da.cross_t))
    return f"+{max(tb - t_a_at_b_progress, 0.0):.1f}"


def standings(rp, t):
    """Timing tower at session time t (seconds)."""
    info = []
    for i, d in enumerate(rp.drivers):
        p = float(np.interp(t, d.cross_t, d.cross_lap))
        out = t >= d.out_time
        # Small tie-breaks: grid order right at the start, finishing order after the flag.
        score = p - d.grid * 1e-4 * max(0.0, 1.0 - p) - float(d.cross_t[-1]) * 1e-9
        info.append({"i": i, "d": d, "p": p, "out": out, "score": score})

    running = sorted([x for x in info if not x["out"]], key=lambda x: -x["score"])
    retired = sorted([x for x in info if x["out"]], key=lambda x: -x["score"])

    rows = []
    leader = prev = None
    for pos, x in enumerate(running + retired, start=1):
        d = x["d"]
        if x["out"]:
            gap = interval = "OUT"
        elif leader is None:
            gap, interval = "Leader", "–"
            leader = prev = x
        else:
            gap = _gap_text(leader, x, t)
            interval = _gap_text(prev, x, t)
            prev = x
        compound, age = _tyre_at(d, t)
        rows.append(Standing(
            pos=pos, idx=x["i"], abbr=d.abbr, color=d.color, gap=gap, interval=interval,
            compound=compound, tyre_age=age, in_pit=_in_pit(d, t), out=x["out"],
            finished=d.finished and t >= float(d.cross_t[-1]), progress=x["p"],
        ))
    return rows


def race_lap(rp, rows):
    """Lap the leader is currently on."""
    running = [r for r in rows if not r.out]
    lead = running[0] if running else rows[0]
    return int(min(rp.total_laps, int(lead.progress) + 1))


def telemetry_window(d, t, window):
    """Last `window` seconds of car telemetry up to time t, x-axis relative to now."""
    if len(d.car_t) == 0:
        return None
    i0 = int(np.searchsorted(d.car_t, t - window, side="left"))
    i1 = int(np.searchsorted(d.car_t, t, side="right"))
    if i1 <= i0:
        return None
    sl = slice(i0, i1)
    return {
        "t": d.car_t[sl] - t,
        "speed": d.car_speed[sl],
        "throttle": d.car_throttle[sl],
        "brake": d.car_brake[sl],
        "gear": d.car_gear[sl],
        "drs_open": int(d.car_drs[i1 - 1]) in DRS_OPEN,
        "now": {
            "speed": float(d.car_speed[i1 - 1]),
            "throttle": float(d.car_throttle[i1 - 1]),
            "brake": float(d.car_brake[i1 - 1]),
            "gear": int(d.car_gear[i1 - 1]),
            "rpm": float(d.car_rpm[i1 - 1]),
        },
    }