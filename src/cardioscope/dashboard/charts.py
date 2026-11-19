"""Consistent interactive charts using aggregate research rows."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from .evidence import Record, site_run

CYAN = "#38BDF8"
TEAL = "#2DD4BF"


def style(fig: Any, height: int = 360) -> Any:
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color="#F1F5F9", size=13),
        margin=dict(l=14, r=14, t=24, b=30),
        legend=dict(orientation="h", y=-0.18),
        hoverlabel=dict(bgcolor="#1E2D45", font_color="#F1F5F9"),
    )
    fig.update_xaxes(gridcolor="#25334A", zeroline=False)
    fig.update_yaxes(gridcolor="#25334A", zeroline=False)
    return fig


def bars(labels: list[str], values: list[float | None], ytitle: str, unit: bool = False) -> Any:
    fig = go.Figure(
        go.Bar(
            x=labels, y=values, marker_color=CYAN, hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"
        )
    )
    fig.update_yaxes(title=ytitle, range=[0, 1] if unit else None)
    return style(fig)


def hospital_chart(sites: list[Record], family: str, tier: str, selected: str | None = None) -> Any:
    rows = [site_run(s, family, tier) for s in sites]
    values = [r["metrics"]["balanced_accuracy"] if r else None for r in rows]
    intervals = [r["intervals"]["balanced_accuracy"]["bounds"] if r else None for r in rows]
    fig = go.Figure(
        go.Bar(
            x=[s["hospital"] for s in sites],
            y=values,
            marker_color=[CYAN if selected in (None, s["hospital"]) else "#477088" for s in sites],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[
                    hi - v if v is not None and bound else 0
                    for v, bound in zip(values, intervals, strict=True)
                    for hi in [bound[1] if bound else 0]
                ],
                arrayminus=[
                    v - bound[0] if v is not None and bound else 0
                    for v, bound in zip(values, intervals, strict=True)
                ],
                color="#94A3B8",
                thickness=1.5,
            ),
            customdata=[[s["test"], s["positive"], s["negative"]] for s in sites],
            hovertemplate="<b>%{x}</b><br>Balanced accuracy %{y:.3f}<br>%{customdata[0]} records"
            "<br>%{customdata[1]} positive · %{customdata[2]} negative<extra></extra>",
        )
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="#64748B")
    fig.update_yaxes(range=[0, 1.04], title="Balanced accuracy", tickformat=".0%")
    return style(fig, 370)


def confusion(m: Record) -> Any:
    fig = go.Figure(
        go.Heatmap(
            z=[[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
            x=["Predicted 0", "Predicted 1"],
            y=["Recorded 0", "Recorded 1"],
            texttemplate="%{z}",
            textfont=dict(size=24),
            showscale=False,
            colorscale=[[0, "#18243A"], [1, "#167CA5"]],
            hovertemplate="%{y} · %{x}<br>%{z} records<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return style(fig, 300)


def calibration(rows: list[Record]) -> Any:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Agreement",
            line=dict(color="#64748B", dash="dot"),
        )
    )
    for r in rows:
        bins = [b for b in r["diagnostics"]["bins"] if b["n"]]
        fig.add_trace(
            go.Scatter(
                x=[b["mean_p"] for b in bins],
                y=[b["observed"] for b in bins],
                customdata=[b["n"] for b in bins],
                mode="lines+markers",
                marker=dict(size=9),
                name=r["mode"].capitalize(),
                line=dict(color=CYAN if r["mode"] == "raw" else TEAL),
                hovertemplate=(
                    "Mean score %{x:.3f}<br>Observed %{y:.3f}<br>%{customdata} "
                    "records<extra></extra>"
                ),
            )
        )
    fig.update_xaxes(title="Mean model score", range=[0, 1])
    fig.update_yaxes(title="Recorded positive fraction", range=[0, 1])
    return style(fig, 420)
