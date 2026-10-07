import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.replay import STATUS_STYLE

BG = "#0B1220"


def track_figure(rp, idx, rows, selected_abbr, status_code="1"):
    """Track map with every car at time index idx."""
    fig = go.Figure()

    line_color = "#4B5563" if status_code == "1" else STATUS_STYLE.get(status_code, ("", "#4B5563"))[1]
    fig.add_trace(go.Scatter(
        x=rp.outline[:, 0], y=rp.outline[:, 1], mode="lines",
        line=dict(color=line_color, width=10), hoverinfo="skip",
    ))

    if rp.corners is not None and len(rp.corners):
        fig.add_trace(go.Scatter(
            x=rp.corners["X"], y=rp.corners["Y"], mode="text", text=rp.corners["Label"],
            textfont=dict(color="#9CA3AF", size=10), hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=[rp.outline[0, 0]], y=[rp.outline[0, 1]], mode="markers",
        marker=dict(symbol="line-ns", size=16, color="#F9FAFB", line=dict(width=3, color="#F9FAFB")),
        hovertext=["Start / finish"], hoverinfo="text",
    ))

    by_idx = {r.idx: r for r in rows}
    xs, ys, colors, sizes, opac, edge, edge_w, labels, hover = [], [], [], [], [], [], [], [], []
    for i, d in enumerate(rp.drivers):
        r = by_idx[i]
        selected = d.abbr == selected_abbr
        xs.append(rp.x[i, idx])
        ys.append(rp.y[i, idx])
        colors.append(d.color)
        sizes.append(20 if selected else 14)
        opac.append(0.35 if r.out else 1.0)
        edge.append("#FFFFFF" if selected else "#0B1220")
        edge_w.append(3 if selected else 1)
        labels.append(d.abbr)
        hover.append(f"P{r.pos} {d.name}<br>{d.team}")

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=labels, textposition="top center",
        textfont=dict(color="#F9FAFB", size=10),
        marker=dict(size=sizes, color=colors, opacity=opac, line=dict(color=edge, width=edge_w)),
        hovertext=hover, hoverinfo="text",
    ))

    pad = 0.08 * max(np.ptp(rp.outline[:, 0]), np.ptp(rp.outline[:, 1]))
    fig.update_layout(
        height=640, showlegend=False, paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=0, r=0, t=10, b=0), uirevision="map",
        xaxis=dict(visible=False, range=[rp.outline[:, 0].min() - pad, rp.outline[:, 0].max() + pad]),
        yaxis=dict(visible=False, range=[rp.outline[:, 1].min() - pad, rp.outline[:, 1].max() + pad],
                   scaleanchor="x", scaleratio=1),
    )
    return fig


def telemetry_figure(win, color, window):
    """Speed / throttle / brake / gear traces for the last `window` seconds."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        row_heights=[0.4, 0.2, 0.2, 0.2],
    )
    t = win["t"]
    fig.add_trace(go.Scatter(x=t, y=win["speed"], mode="lines", line=dict(color=color, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=win["throttle"], mode="lines", fill="tozeroy",
                             line=dict(color="#22C55E", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=win["brake"], mode="lines", fill="tozeroy",
                             line=dict(color="#EF4444", width=1.5, shape="hv")), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=win["gear"], mode="lines",
                             line=dict(color="#E5E7EB", width=1.5, shape="hv")), row=4, col=1)

    fig.update_yaxes(title_text="km/h", row=1, col=1)
    fig.update_yaxes(title_text="Throttle %", range=[0, 105], row=2, col=1)
    fig.update_yaxes(title_text="Brake", range=[0, 105], row=3, col=1)
    fig.update_yaxes(title_text="Gear", range=[0, 9], dtick=1, row=4, col=1)
    fig.update_xaxes(range=[-window, 0], title_text="seconds before now", row=4, col=1)
    fig.update_layout(
        height=460, showlegend=False, paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="#E5E7EB"), margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_xaxes(gridcolor="#1F2937")
    fig.update_yaxes(gridcolor="#1F2937")
    return fig