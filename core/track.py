import numpy as np
import pandas as pd


def rotate_xy(x, y, angle_deg):
    """Rotate coordinate arrays by the circuit's rotation angle (degrees)."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return x * c - y * s, x * s + y * c


def get_track_outline(session):
    """Track outline (closed loop) and corner labels, rotated to the standard orientation.

    Returns (outline Nx2 array, corners DataFrame[X, Y, Label] or None, rotation in degrees).
    Coordinates are FastF1 position units (1/10 metre).
    """
    lap = session.laps.pick_fastest()
    pos = lap.get_pos_data()
    x = pos["X"].to_numpy(float)
    y = pos["Y"].to_numpy(float)

    try:
        info = session.get_circuit_info()
        rotation = float(info.rotation)
        corner_df = info.corners.copy()
    except Exception:
        rotation, corner_df = 0.0, None

    ox, oy = rotate_xy(x, y, rotation)
    outline = np.column_stack([ox, oy])
    outline = np.vstack([outline, outline[:1]])  # close the lap so there's no gap at the line

    corners = None
    if corner_df is not None and not corner_df.empty:
        cx, cy = rotate_xy(corner_df["X"].to_numpy(float), corner_df["Y"].to_numpy(float), rotation)
        letters = corner_df["Letter"].fillna("").astype(str) if "Letter" in corner_df else ""
        labels = corner_df["Number"].astype(int).astype(str) + letters
        corners = pd.DataFrame({"X": cx, "Y": cy, "Label": labels.to_numpy()})

    return outline, corners, rotation