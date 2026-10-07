"""Small HTML builders for the timing tower and the status banner."""
from core.replay import STATUS_STYLE, fmt_clock

COMPOUNDS = {
    "SOFT": ("S", "#EF4444"),
    "MEDIUM": ("M", "#FACC15"),
    "HARD": ("H", "#F9FAFB"),
    "INTERMEDIATE": ("I", "#22C55E"),
    "WET": ("W", "#3B82F6"),
}

_COLS = "1.9rem 3rem 1fr 1fr 3.6rem 2.4rem"
_ROW = f"display:grid;grid-template-columns:{_COLS};align-items:center;gap:.4rem;padding:.2rem .5rem;"


def tower_html(rows, selected_abbr):
    parts = [
        '<div style="background:#111827;color:#F9FAFB;border-radius:.75rem;padding:.6rem;'
        'font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.8rem;">',
        f'<div style="{_ROW}opacity:.55;font-size:.66rem;text-transform:uppercase;">'
        '<span>Pos</span><span>Drv</span><span style="text-align:right;">Gap</span>'
        '<span style="text-align:right;">Int</span><span style="text-align:center;">Tyre</span><span></span></div>',
    ]
    for r in rows:
        letter, tcolor = COMPOUNDS.get(r.compound, ("?", "#9CA3AF"))
        age = "–" if r.tyre_age is None else r.tyre_age
        if r.out:
            tag = '<span style="color:#EF4444;font-weight:700;">OUT</span>'
        elif r.in_pit:
            tag = '<span style="color:#F97316;font-weight:700;">PIT</span>'
        elif r.finished:
            tag = "🏁"
        else:
            tag = ""
        bg = "background:#1F2937;" if r.abbr == selected_abbr else ""
        parts.append(
            f'<div style="{_ROW}{bg}border-left:4px solid {r.color};">'
            f'<span style="opacity:.7;">{r.pos}</span>'
            f'<span style="font-weight:700;">{r.abbr}</span>'
            f'<span style="text-align:right;">{r.gap}</span>'
            f'<span style="text-align:right;opacity:.8;">{r.interval}</span>'
            f'<span style="text-align:center;"><b style="color:{tcolor};">{letter}</b> {age}</span>'
            f'{tag}</div>'
        )
    parts.append("</div>")
    return "".join(parts)


def banner_html(lap, total_laps, race_seconds, status_code):
    label, color = STATUS_STYLE.get(status_code, STATUS_STYLE["1"])
    return (
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'background:#111827;color:#F9FAFB;border-radius:.75rem;padding:.6rem 1rem;margin-bottom:.5rem;">'
        f'<span style="font-size:1.1rem;font-weight:700;">LAP {lap} / {total_laps}</span>'
        f'<span style="font-family:ui-monospace,Menlo,Consolas,monospace;opacity:.8;">{fmt_clock(race_seconds)}</span>'
        f'<span style="background:{color};color:#111827;font-weight:700;border-radius:.4rem;'
        f'padding:.15rem .6rem;font-size:.8rem;">{label}</span></div>'
    )