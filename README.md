# F1 Race Replay

A race replay and telemetry system for Formula 1, built on FastF1 and Streamlit.
Watch every car move around the circuit with a live timing tower and per-driver telemetry.

> Work in progress.

## Roadmap
- [x] Track outline and corner numbers
- [ ] Resample all drivers onto a shared timeline
- [ ] All cars on the map at a chosen timestamp
- [ ] Time scrubber and play/pause
- [ ] Live leaderboard with gaps and intervals
- [ ] Driver telemetry panel
- [ ] Track status (yellow flags, safety car)

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```