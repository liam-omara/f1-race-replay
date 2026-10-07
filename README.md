# F1 Race Replay

Replay any Formula 1 race from the last few seasons in your browser. Watch all 20 cars move around the circuit at 60 fps, follow a live timing tower, and inspect any driver's telemetry at any moment. Built with [FastF1](https://github.com/theOehrly/Fast-F1) and Streamlit, with the player itself written in plain JavaScript on a canvas.


## Features

- **Track map:** circuit outline with corner numbers and every car in team colours, in the standard orientation
- **Smooth playback:** the player runs in the browser (canvas + `requestAnimationFrame`), so there are no Streamlit reruns and no flicker. Play/pause, scrub to any second, 1× to 50× speed, keyboard shortcuts
- **Live timing tower:** position, gap to leader, interval to the car ahead, tyre compound and age, PIT and OUT markers, chequered flag
- **Race status:** lap counter, race clock, and a banner for yellow flags, safety car, VSC and red flags (the track outline changes colour too)
- **Driver telemetry:** follow any driver (dropdown, click a car, or click a tower row) and see speed, gear, throttle, brake and DRS, plus traces for the last 30, 60 or 120 seconds
- **Season and race dropdowns** that load the real calendar from FastF1

## Getting started

```bash
git clone https://github.com/liam-omara/f1-race-replay.git
cd f1-race-replay
pip install -r requirements.txt
streamlit run app.py
```

Then open the address Streamlit prints (usually `http://localhost:8501`), pick a year and a race, and press Play.

The first time you open a race, FastF1 downloads its position and car data, which can take a few minutes. It is cached in a local `cache/` folder after that. Building the replay takes a little longer and is cached by Streamlit until you restart it.

## Controls

| Control | What it does |
|---|---|
| Play / Pause (or Space) | Start and stop the replay |
| Speed | 1× to 50× real time |
| Scrubber | Jump to any point in the race |
| ← / → (Shift for ±60 s) | Skip 10 seconds back or forward |
| Follow driver | Choose whose telemetry to show |
| Click a car or a tower row | Follow that driver |
| Trace | How much recent history the telemetry graphs show |

## Project structure

```
app.py               Streamlit page: year and race pickers, embeds the player
core/
  loader.py          FastF1 session loading and race calendar
  track.py           Track outline, corner labels, rotation
  replay.py          Builds the replay arrays; standings, gaps, tyres, status at time t
  player.py          Packs the race into compact binary arrays and the HTML/JS player
tests/               pytest suite using a synthetic race (no downloads needed)
```

`core/replay.py` has no Streamlit in it, so the race logic can be tested on its own.

## How it works

1. **Shared timeline.** Every driver's position samples (each on its own timestamps) are interpolated onto one 0.5-second grid, then rotated by the circuit's rotation angle so they line up with the track outline.
2. **Timing tower from lap timing, not X/Y.** Progress is interpolated linearly between each driver's line-crossing times. This is exact at every crossing, always increasing, and avoids the wrap-around glitches you get from projecting positions onto the track near the start/finish line. The gap between two cars is the time between them reaching the same point of the race, the same idea as real timing loops. A car a full lap or more behind shows as `+1 LAP`.
3. **Ordering.** Cars are sorted by progress, with grid order breaking ties at the start and finishing order after the flag. Retired cars drop to the bottom and show `OUT`.
4. **Playback in the browser.** Positions, a per-second timing-tower snapshot and telemetry are packed into base64 `int16`/`uint8` arrays and embedded in one self-contained HTML page. JavaScript interpolates car positions every frame and advances time by real elapsed time × speed, so Streamlit never reruns while the race plays. Streamlit's earlier rerun-per-tick approach caused a visible flash on every update, which is why the player lives in the browser.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests build a small synthetic race (a leader, a close follower, a lapped car and a retirement) and check ordering, gaps, lapped cars, retirements, pit status, track status, telemetry windows, the binary payload, and a full run of the Streamlit app.

## Limitations

- Race sessions only, and only seasons where FastF1 has position data.
- Mid-lap, the timing tower and the dots on the map can differ slightly, because one is interpolated from lap timing and the other comes from X/Y position.
- Lap 1 order is approximate (all cars start from the same time, ordered by grid position at the very start).
- Gaps during safety cars, red flags and pit stops follow the lap timing and can look odd.
- The timing tower updates once per second of race time.
- The first build of a race takes a while (resampling plus one standings snapshot per second). It is cached afterwards.

## Ideas for next steps

- Colour the track by speed or gear for the followed driver
- Pit stop timeline and tyre strategy strip under the scrubber
- Gap-to-car-ahead chart over the race
- Qualifying and sprint sessions
- Compare two drivers' telemetry overlaid

## Data

Race data comes from FastF1, which pulls from official F1 timing sources. This project is unofficial and not affiliated with Formula 1.