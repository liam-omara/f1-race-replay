"""Builds a self-contained HTML player for a Replay.

All animation runs in the browser (requestAnimationFrame, canvas), so Streamlit
renders the page once and never reruns while the race plays. That is what makes
playback smooth: no server round-trip and no chart redraw per frame.

The race is shipped to the browser as compact binary arrays (base64):
  pos   int16  [driver][time][x,y]       car positions on the replay grid (0.5 s)
  sa/sb        timing tower per second   (order, gap kinds, tyre, flags / gaps)
  lap          leader lap per second
  tspd/tthr/tflg  telemetry on the replay grid for every driver
"""
from __future__ import annotations

import base64
import json

import numpy as np

from core.replay import STATUS_STYLE, race_lap, standings

PLAYER_HEIGHT = 1290

COMPOUND_CODE = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
MISSING = -32768
DRS_OPEN = (10, 12, 14)


def _b64(arr, dtype):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")


def _kind_value(txt):
    """Gap text from standings() -> (kind, value). kind: 0 seconds, 1 leader, 2 laps down, 3 out."""
    if txt in ("Leader", "–"):
        return 1, 0
    if txt == "OUT":
        return 3, 0
    if "LAP" in txt:
        return 2, min(int(txt[1:].split()[0]), 65535)
    return 0, min(int(round(float(txt[1:]) * 10)), 65535)


def _nearest(times, ct, values):
    j = np.clip(np.searchsorted(ct, times), 1, len(ct) - 1)
    pick = np.where(np.abs(times - ct[j - 1]) <= np.abs(times - ct[j]), j - 1, j)
    return values[pick]


def build_payload(rp):
    times, dt = rp.times, rp.dt
    n_t, n_d = len(times), len(rp.drivers)

    # ── positions, quantised to int16 relative to the track's bounding box ──
    ox, oy = rp.outline[:, 0], rp.outline[:, 1]
    cx, cy = (ox.min() + ox.max()) / 2, (oy.min() + oy.max()) / 2
    half = max(np.ptp(ox), np.ptp(oy)) / 2 * 1.15 + 500
    qscale = 30000.0 / half
    pos = np.full((n_d, n_t, 2), MISSING, dtype="<i2")
    for k in range(n_d):
        ok = np.isfinite(rp.x[k]) & np.isfinite(rp.y[k])
        pos[k, ok, 0] = np.clip(np.round((rp.x[k][ok] - cx) * qscale), -32000, 32000)
        pos[k, ok, 1] = np.clip(np.round((rp.y[k][ok] - cy) * qscale), -32000, 32000)

    # ── timing tower, one snapshot per second ──
    step = max(1, int(round(1.0 / dt)))
    s_idx = list(range(0, n_t, step))
    n_s = len(s_idx)
    sa = np.zeros((n_s, n_d, 5), dtype=np.uint8)
    sb = np.zeros((n_s, n_d, 2), dtype=np.uint16)
    lap = np.zeros(n_s, dtype=np.uint8)
    for s, i in enumerate(s_idx):
        rows = standings(rp, float(times[i]))
        lap[s] = race_lap(rp, rows)
        for rank, r in enumerate(rows):
            gk, gv = _kind_value(r.gap)
            ik, iv = _kind_value(r.interval)
            comp = COMPOUND_CODE.get(r.compound, 5)
            age = 255 if r.tyre_age is None else min(r.tyre_age, 254)
            flags = (1 if r.in_pit else 0) | (2 if r.finished else 0) | (4 if r.out else 0)
            sa[s, rank] = (r.idx, gk | (ik << 2), comp, age, flags)
            sb[s, rank] = (gv, iv)

    # ── telemetry on the same grid as positions ──
    tspd = np.zeros((n_d, n_t), dtype="<u2")
    tthr = np.zeros((n_d, n_t), dtype=np.uint8)
    tflg = np.zeros((n_d, n_t), dtype=np.uint8)
    for k, d in enumerate(rp.drivers):
        if len(d.car_t) < 2:
            continue
        tspd[k] = np.clip(np.round(np.interp(times, d.car_t, d.car_speed, left=0, right=0)), 0, 65535)
        tthr[k] = np.clip(np.round(np.interp(times, d.car_t, d.car_throttle, left=0, right=0)), 0, 100)
        gear = np.clip(np.round(_nearest(times, d.car_t, d.car_gear)), 0, 15).astype(np.uint8)
        brake = (_nearest(times, d.car_t, d.car_brake) > 50).astype(np.uint8)
        drs = np.isin(_nearest(times, d.car_t, d.car_drs), DRS_OPEN).astype(np.uint8)
        tflg[k] = gear | (brake << 4) | (drs << 5)

    final = standings(rp, float(times[-1]))
    corners = []
    if rp.corners is not None and len(rp.corners):
        corners = [
            {"x": round(float(r.X), 1), "y": round(float(r.Y), 1), "l": str(r.Label)}
            for r in rp.corners.itertuples()
        ]

    return {
        "title": rp.title,
        "dt": float(dt),
        "nT": n_t,
        "nD": n_d,
        "nS": n_s,
        "stepSec": step * float(dt),
        "t0": float(rp.t0),
        "totalLaps": int(rp.total_laps),
        "follow": int(final[0].idx),
        "drivers": [
            {"abbr": d.abbr, "name": d.name, "team": d.team, "color": d.color, "number": d.number}
            for d in rp.drivers
        ],
        "cx": float(cx), "cy": float(cy), "qscale": float(qscale),
        "outline": [int(round(v)) for v in rp.outline.reshape(-1)],
        "corners": corners,
        "statusT": [float(v) for v in rp.status_t],
        "statusC": [str(v) for v in rp.status_code],
        "statusStyle": {k: list(v) for k, v in STATUS_STYLE.items()},
        "pos": _b64(pos, "<i2"),
        "sa": _b64(sa, np.uint8),
        "sb": _b64(sb, "<u2"),
        "lap": _b64(lap, np.uint8),
        "tspd": _b64(tspd, "<u2"),
        "tthr": _b64(tthr, np.uint8),
        "tflg": _b64(tflg, np.uint8),
    }


def build_html(rp):
    payload = build_payload(rp)
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.replace("__TITLE__", payload["title"]).replace("__PAYLOAD__", blob)


TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; background: #0B1220; color: #F9FAFB;
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  #app { padding: 10px 12px 12px; }
  .bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
  .bar label { font-size: .7rem; text-transform: uppercase; opacity: .6; display: flex; flex-direction: column; gap: 2px; }
  button, select { background: #1F2937; color: #F9FAFB; border: 1px solid #374151; border-radius: .5rem;
    padding: .4rem .6rem; font-size: .85rem; font-family: inherit; }
  button { cursor: pointer; font-weight: 700; min-width: 5.5rem; background: #DC2626; border-color: #DC2626; }
  button:hover { filter: brightness(1.1); }
  #slider { flex: 1; min-width: 200px; accent-color: #DC2626; }
  #timeTxt { font-family: ui-monospace, Menlo, Consolas, monospace; opacity: .8; min-width: 4.5rem; text-align: right; }
  .banner { display: flex; justify-content: space-between; align-items: center; background: #111827;
    border-radius: .75rem; padding: .55rem 1rem; margin-bottom: 8px; height: 44px; }
  #lapTxt { font-size: 1.05rem; font-weight: 700; }
  #clockTxt { font-family: ui-monospace, Menlo, Consolas, monospace; opacity: .8; }
  #statusTxt { color: #111827; font-weight: 700; border-radius: .4rem; padding: .15rem .6rem; font-size: .8rem; }
  .grid { display: grid; grid-template-columns: minmax(0, 2.2fr) minmax(300px, 1fr); gap: 10px; }
  #mapWrap { background: #0B1220; border-radius: .75rem; overflow: hidden; height: 600px; }
  canvas { display: block; width: 100%; }
  #tower { background: #111827; border-radius: .75rem; padding: .5rem; height: 600px; overflow-y: auto;
    font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .8rem; }
  .row { display: grid; grid-template-columns: 1.7rem 3rem 1fr 1fr 3.6rem 2.4rem; align-items: center; gap: .4rem;
    padding: 0 .5rem; height: 26px; border-left: 4px solid #6B7280; cursor: pointer; }
  .row.head { opacity: .55; font-size: .66rem; text-transform: uppercase; cursor: default; border-left-color: transparent; }
  .row.sel { background: #1F2937; }
  .row:not(.head):hover { background: #19222f; }
  .r { text-align: right; } .c { text-align: center; }
  .telhead { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin: 12px 0 4px; }
  .telhead h3 { margin: 0; font-size: 1.05rem; }
  .metrics { display: flex; gap: 22px; }
  .metrics div { display: flex; flex-direction: column; }
  .metrics span { font-size: .66rem; text-transform: uppercase; opacity: .55; }
  .metrics b { font-size: 1.15rem; font-family: ui-monospace, Menlo, Consolas, monospace; }
  #telCanvas { background: #0B1220; border-radius: .75rem; }
  .hint { font-size: .72rem; opacity: .5; margin-top: 6px; }
</style>
</head>
<body>
<div id="app">
  <div class="bar">
    <button id="playBtn">&#9654; Play</button>
    <label>Speed<select id="speedSel"></select></label>
    <label>Follow driver<select id="driverSel"></select></label>
    <label>Trace<select id="winSel"></select></label>
    <input id="slider" type="range" min="0" max="1" step="1" value="0">
    <span id="timeTxt">0:00:00</span>
  </div>
  <div class="banner"><span id="lapTxt"></span><span id="clockTxt"></span><span id="statusTxt"></span></div>
  <div class="grid">
    <div id="mapWrap"><canvas id="mapCanvas"></canvas></div>
    <div id="tower"></div>
  </div>
  <div class="telhead">
    <h3 id="telTitle">Telemetry</h3>
    <div class="metrics">
      <div><span>Speed</span><b id="mSpeed">-</b></div>
      <div><span>Gear</span><b id="mGear">-</b></div>
      <div><span>Throttle</span><b id="mThr">-</b></div>
      <div><span>Brake</span><b id="mBrk">-</b></div>
      <div><span>DRS</span><b id="mDrs">-</b></div>
    </div>
  </div>
  <canvas id="telCanvas"></canvas>
  <div class="hint">Space: play/pause &middot; &larr; &rarr;: &plusmn;10 s (Shift: &plusmn;60 s) &middot; click a car or a tower row to follow that driver</div>
</div>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
(function () {
"use strict";
var $ = function (id) { return document.getElementById(id); };
var P = JSON.parse($("payload").textContent);

function dec(s, T) {
  var b = atob(s), n = b.length, u = new Uint8Array(n);
  for (var i = 0; i < n; i++) u[i] = b.charCodeAt(i);
  return T === Uint8Array ? u : new T(u.buffer);
}
var POS = dec(P.pos, Int16Array), SA = dec(P.sa, Uint8Array), SB = dec(P.sb, Uint16Array);
var LAP = dec(P.lap, Uint8Array), TSPD = dec(P.tspd, Uint16Array), TTHR = dec(P.tthr, Uint8Array), TFLG = dec(P.tflg, Uint8Array);

var nT = P.nT, nD = P.nD, nS = P.nS, dt = P.dt, step = P.stepSec, dur = (nT - 1) * dt;
var MISSING = -32768, MAP_H = 600, TEL_H = 440;
var COMP = [["S", "#EF4444"], ["M", "#FACC15"], ["H", "#F9FAFB"], ["I", "#22C55E"], ["W", "#3B82F6"], ["?", "#9CA3AF"]];

// ── state ──
var t = 0, playing = false, speed = 10, win = 60, sel = P.follow, lastNow = 0, curS = -1;
var dirty = true, dragging = false, curStatus = "1", outNow = new Uint8Array(nD);
var carPx = [];

// ── controls ──
[1, 2, 5, 10, 20, 50].forEach(function (v) {
  var o = document.createElement("option"); o.value = v; o.textContent = v + "×"; if (v === speed) o.selected = true; $("speedSel").appendChild(o);
});
[30, 60, 120].forEach(function (v) {
  var o = document.createElement("option"); o.value = v; o.textContent = v + "s"; if (v === win) o.selected = true; $("winSel").appendChild(o);
});
P.drivers.forEach(function (d, i) {
  var o = document.createElement("option"); o.value = i; o.textContent = d.abbr + " – " + d.name; $("driverSel").appendChild(o);
});
$("driverSel").value = sel;
$("slider").max = Math.floor(dur);

function setPlaying(p) {
  playing = p; lastNow = performance.now(); dirty = true;
  $("playBtn").innerHTML = p ? "&#9208; Pause" : "&#9654; Play";
}
function setSel(i) { sel = i; $("driverSel").value = i; curS = -1; dirty = true; }
function seek(v) { t = Math.max(0, Math.min(dur, v)); if (t >= dur && playing) setPlaying(false); dirty = true; }

$("playBtn").onclick = function () { if (!playing && t >= dur) t = 0; setPlaying(!playing); };
$("speedSel").onchange = function () { speed = +this.value; };
$("winSel").onchange = function () { win = +this.value; dirty = true; };
$("driverSel").onchange = function () { setSel(+this.value); };
$("slider").oninput = function () { seek(+this.value); };
$("slider").onpointerdown = function () { dragging = true; };
window.addEventListener("pointerup", function () { dragging = false; });
document.addEventListener("keydown", function (e) {
  if (e.target && e.target.tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); $("playBtn").click(); }
  else if (e.code === "ArrowRight") seek(t + (e.shiftKey ? 60 : 10));
  else if (e.code === "ArrowLeft") seek(t - (e.shiftKey ? 60 : 10));
});
$("tower").addEventListener("click", function (e) {
  var r = e.target.closest(".row[data-i]"); if (r) setSel(+r.getAttribute("data-i"));
});

// ── helpers ──
function pad2(n) { return (n < 10 ? "0" : "") + n; }
function clock(sec) { sec = Math.max(0, Math.floor(sec)); return Math.floor(sec / 3600) + ":" + pad2(Math.floor(sec % 3600 / 60)) + ":" + pad2(sec % 60); }
function statusAt(tt) {
  var a = P.statusT, lo = 0, hi = a.length - 1, r = -1;
  while (lo <= hi) { var m = (lo + hi) >> 1; if (a[m] <= tt) { r = m; lo = m + 1; } else hi = m - 1; }
  return r < 0 ? "1" : P.statusC[r];
}
function gapText(kind, v, isGap) {
  if (kind === 1) return isGap ? "Leader" : "–";
  if (kind === 3) return "OUT";
  if (kind === 2) return "+" + v + " LAP" + (v > 1 ? "S" : "");
  return "+" + (v / 10).toFixed(1);
}

// ── map geometry ──
var O = P.outline, minX = 1e18, maxX = -1e18, minY = 1e18, maxY = -1e18;
for (var k = 0; k < O.length; k += 2) {
  if (O[k] < minX) minX = O[k]; if (O[k] > maxX) maxX = O[k];
  if (O[k + 1] < minY) minY = O[k + 1]; if (O[k + 1] > maxY) maxY = O[k + 1];
}
var mapCanvas = $("mapCanvas"), mctx, staticC = document.createElement("canvas"), sctx, mapW = 800, sc = 1, offX = 0, offY = 0;
function toX(X) { return offX + (X - minX) * sc; }
function toY(Y) { return MAP_H - (offY + (Y - minY) * sc); }

function setupCanvas(c, w, h) {
  var dpr = window.devicePixelRatio || 1;
  c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
  if (c.style) { c.style.height = h + "px"; }
  var ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return ctx;
}
function layout() {
  mapW = Math.max(300, $("mapWrap").clientWidth);
  mctx = setupCanvas(mapCanvas, mapW, MAP_H);
  sctx = setupCanvas(staticC, mapW, MAP_H);
  var pad = 40, rx = Math.max(1, maxX - minX), ry = Math.max(1, maxY - minY);
  sc = Math.min((mapW - 2 * pad) / rx, (MAP_H - 2 * pad) / ry);
  offX = (mapW - rx * sc) / 2; offY = (MAP_H - ry * sc) / 2;
  drawStatic();
  var tw = $("telCanvas").parentNode.clientWidth - 24;
  telCtx = setupCanvas($("telCanvas"), Math.max(300, tw), TEL_H); telW = Math.max(300, tw);
  dirty = true;
}
var telCtx, telW = 800;

function drawStatic() {
  var ctx = sctx, style = P.statusStyle[curStatus];
  var line = (curStatus === "1" || !style) ? "#4B5563" : style[1];
  ctx.clearRect(0, 0, mapW, MAP_H);
  ctx.lineJoin = "round"; ctx.lineCap = "round";
  ctx.beginPath();
  for (var k = 0; k < O.length; k += 2) { var x = toX(O[k]), y = toY(O[k + 1]); if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
  ctx.strokeStyle = "#1F2937"; ctx.lineWidth = 16; ctx.stroke();
  ctx.strokeStyle = line; ctx.lineWidth = 9; ctx.stroke();
  ctx.font = "10px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillStyle = "#9CA3AF";
  P.corners.forEach(function (c) { ctx.fillText(c.l, toX(c.x), toY(c.y)); });
  if (O.length >= 4) {
    var x0 = toX(O[0]), y0 = toY(O[1]), x1 = toX(O[2]), y1 = toY(O[3]);
    var dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy) || 1, nx = -dy / L * 10, ny = dx / L * 10;
    ctx.beginPath(); ctx.moveTo(x0 - nx, y0 - ny); ctx.lineTo(x0 + nx, y0 + ny);
    ctx.strokeStyle = "#F9FAFB"; ctx.lineWidth = 3; ctx.stroke();
  }
}

function carXY(d, tt) {
  var f = Math.min(nT - 1, tt / dt), i = Math.floor(f), j = Math.min(nT - 1, i + 1), a = f - i, b = d * nT * 2;
  var x0 = POS[b + 2 * i], y0 = POS[b + 2 * i + 1], x1 = POS[b + 2 * j], y1 = POS[b + 2 * j + 1];
  var m0 = x0 === MISSING, m1 = x1 === MISSING;
  if (m0 && m1) return null;
  if (m0) { x0 = x1; y0 = y1; } else if (m1) { x1 = x0; y1 = y0; }
  return [P.cx + (x0 + (x1 - x0) * a) / P.qscale, P.cy + (y0 + (y1 - y0) * a) / P.qscale];
}

function drawMap() {
  var ctx = mctx;
  ctx.clearRect(0, 0, mapW, MAP_H);
  ctx.drawImage(staticC, 0, 0, mapW, MAP_H);
  ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.font = "bold 10px sans-serif";
  var order = [], i;
  for (i = 0; i < nD; i++) if (i !== sel) order.push(i);
  order.push(sel);
  carPx = [];
  order.forEach(function (d) {
    var p = carXY(d, t); if (!p) return;
    var x = toX(p[0]), y = toY(p[1]), s = d === sel, r = s ? 9 : 7;
    carPx.push([d, x, y]);
    ctx.globalAlpha = outNow[d] ? 0.35 : 1;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832);
    ctx.fillStyle = P.drivers[d].color; ctx.fill();
    ctx.lineWidth = s ? 3 : 1.5; ctx.strokeStyle = s ? "#FFFFFF" : "#0B1220"; ctx.stroke();
    ctx.fillStyle = "#F9FAFB"; ctx.fillText(P.drivers[d].abbr, x, y - r - 2);
    ctx.globalAlpha = 1;
  });
}
mapCanvas.addEventListener("click", function (e) {
  var rc = mapCanvas.getBoundingClientRect(), x = e.clientX - rc.left, y = e.clientY - rc.top, best = -1, bd = 18;
  carPx.forEach(function (c) { var dd = Math.hypot(c[1] - x, c[2] - y); if (dd < bd) { bd = dd; best = c[0]; } });
  if (best >= 0) setSel(best);
});

// ── timing tower (updates once per second of race time) ──
function updateTower(s) {
  var h = '<div class="row head"><span>Pos</span><span>Drv</span><span class="r">Gap</span><span class="r">Int</span><span class="c">Tyre</span><span></span></div>';
  for (var r = 0; r < nD; r++) {
    var o = (s * nD + r) * 5, idx = SA[o], kinds = SA[o + 1], comp = COMP[Math.min(SA[o + 2], 5)], age = SA[o + 3], fl = SA[o + 4];
    var gv = SB[(s * nD + r) * 2], iv = SB[(s * nD + r) * 2 + 1], d = P.drivers[idx];
    outNow[idx] = (fl & 4) ? 1 : 0;
    var tag = (fl & 4) ? '<span style="color:#EF4444;font-weight:700">OUT</span>' : (fl & 1) ? '<span style="color:#F97316;font-weight:700">PIT</span>' : (fl & 2) ? "🏁" : "";
    h += '<div class="row' + (idx === sel ? " sel" : "") + '" data-i="' + idx + '" style="border-left-color:' + d.color + '">' +
      '<span style="opacity:.7">' + (r + 1) + '</span><span style="font-weight:700">' + d.abbr + '</span>' +
      '<span class="r">' + gapText(kinds & 3, gv, true) + '</span><span class="r" style="opacity:.8">' + gapText(kinds >> 2, iv, false) + '</span>' +
      '<span class="c"><b style="color:' + comp[1] + '">' + comp[0] + '</b> ' + (age === 255 ? "–" : age) + '</span>' + tag + '</div>';
  }
  $("tower").innerHTML = h;
}

// ── telemetry traces ──
function panel(ctx, pts, x0, x1, top, h, ymin, ymax, color, fill, stepped, ticks) {
  var L = x0, W = x1 - x0;
  ctx.strokeStyle = "#1F2937"; ctx.lineWidth = 1; ctx.fillStyle = "#6B7280"; ctx.font = "10px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  ticks.forEach(function (v) {
    var y = top + h - (v - ymin) / (ymax - ymin) * h;
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + W, y); ctx.stroke(); ctx.fillText(v, L - 6, y);
  });
  if (!pts.length) return;
  ctx.save(); ctx.beginPath(); ctx.rect(L, top - 2, W, h + 4); ctx.clip();
  ctx.beginPath();
  var first = true, py = 0;
  pts.forEach(function (p) {
    var y = top + h - (p[1] - ymin) / (ymax - ymin) * h;
    if (first) { ctx.moveTo(p[0], y); first = false; }
    else { if (stepped) ctx.lineTo(p[0], py); ctx.lineTo(p[0], y); }
    py = y;
  });
  if (fill) {
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.lineTo(pts[pts.length - 1][0], top + h); ctx.lineTo(pts[0][0], top + h); ctx.closePath();
    ctx.globalAlpha = 0.25; ctx.fillStyle = color; ctx.fill(); ctx.globalAlpha = 1;
  } else { ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke(); }
  ctx.restore();
}
function drawTel() {
  var ctx = telCtx, W = telW, H = TEL_H, L = 52, R = 10, B = 24, gap = 10, usable = H - B - 3 * gap;
  ctx.clearRect(0, 0, W, H);
  var base = sel * nT, iEnd = Math.min(nT - 1, Math.floor(t / dt)), iStart = Math.max(0, Math.ceil((t - win) / dt));
  function X(tt) { return L + (tt - (t - win)) / win * (W - L - R); }
  var spd = [], thr = [], brk = [], gear = [], i;
  for (i = iStart; i <= iEnd; i++) {
    var x = X(i * dt), f = TFLG[base + i];
    spd.push([x, TSPD[base + i]]); thr.push([x, TTHR[base + i]]); brk.push([x, (f >> 4) & 1]); gear.push([x, f & 15]);
  }
  if (iEnd < nT - 1 && iEnd >= iStart) {                       // extend to "now" so the trace doesn't lag the clock
    var a = (t - iEnd * dt) / dt, xe = X(t);
    spd.push([xe, TSPD[base + iEnd] + (TSPD[base + iEnd + 1] - TSPD[base + iEnd]) * a]);
    thr.push([xe, TTHR[base + iEnd] + (TTHR[base + iEnd + 1] - TTHR[base + iEnd]) * a]);
    brk.push([xe, brk[brk.length - 1][1]]); gear.push([xe, gear[gear.length - 1][1]]);
  }
  var hs = usable * 0.4, ho = usable * 0.2, y = 3;
  panel(ctx, spd, L, W - R, y, hs, 0, 360, P.drivers[sel].color, false, false, [0, 100, 200, 300]); y += hs + gap;
  panel(ctx, thr, L, W - R, y, ho, 0, 100, "#22C55E", true, false, [0, 50, 100]); y += ho + gap;
  panel(ctx, brk, L, W - R, y, ho, 0, 1, "#EF4444", true, true, [0, 1]); y += ho + gap;
  panel(ctx, gear, L, W - R, y, ho, 0, 8, "#E5E7EB", false, true, [2, 4, 6, 8]);
  ctx.fillStyle = "#9CA3AF"; ctx.font = "10px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
  ctx.fillText("km/h", 4, 12); ctx.fillText("Throttle", 4, 3 + hs + gap + 8); ctx.fillText("Brake", 4, 3 + hs + ho + 2 * gap + 8); ctx.fillText("Gear", 4, 3 + hs + 2 * ho + 3 * gap + 8);
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (var q = 0; q <= 4; q++) { var s = -win + q * win / 4; ctx.fillText(q === 4 ? "now" : s + "s", X(t + s), H - B + 6); }
  // metrics
  var f2 = TFLG[base + iEnd], sp = TSPD[base + iEnd];
  setText("mSpeed", Math.round(sp) + " km/h"); setText("mGear", f2 & 15 || "N"); setText("mThr", TTHR[base + iEnd] + "%");
  setText("mBrk", (f2 >> 4) & 1 ? "ON" : "off"); setText("mDrs", (f2 >> 5) & 1 ? "OPEN" : "closed");
}
var textCache = {};
function setText(id, v) { v = String(v); if (textCache[id] !== v) { textCache[id] = v; $(id).textContent = v; } }

// ── banner + render loop ──
function render() {
  var s = Math.min(nS - 1, Math.floor(t / step));
  if (s !== curS) { curS = s; updateTower(s); }
  var code = statusAt(P.t0 + t);
  if (code !== curStatus) { curStatus = code; drawStatic(); }
  var st = P.statusStyle[code] || P.statusStyle["1"];
  setText("lapTxt", "LAP " + LAP[s] + " / " + P.totalLaps);
  setText("clockTxt", clock(t));
  setText("timeTxt", clock(t));
  setText("telTitle", "Telemetry – " + P.drivers[sel].abbr + " (" + P.drivers[sel].team + ")");
  var b = $("statusTxt"); if (b.textContent !== st[0]) { b.textContent = st[0]; b.style.background = st[1]; }
  if (!dragging) $("slider").value = Math.floor(t);
  drawMap(); drawTel();
}
function frame(now) {
  if (playing) {
    var d = Math.min(0.25, (now - lastNow) / 1000);
    t += d * speed;
    if (t >= dur) { t = dur; setPlaying(false); }
    dirty = true;
  }
  lastNow = now;
  if (dirty) { dirty = false; render(); }
  requestAnimationFrame(frame);
}

window.addEventListener("resize", layout);
if (window.ResizeObserver) new ResizeObserver(function () { layout(); }).observe($("mapWrap"));
window.__replay = { seek: seek, setSel: setSel, setPlaying: setPlaying, state: function () { return { t: t, playing: playing, sel: sel, dur: dur }; } };
layout();
requestAnimationFrame(frame);
})();
</script>
</body>
</html>
"""