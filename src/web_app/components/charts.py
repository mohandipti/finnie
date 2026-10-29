"""Shared chart styling: a validated categorical palette and Plotly layout defaults.

Series colors follow a fixed slot order (never cycled) and have separate light and dark
steps. The palette passes colorblind-separation checks in both modes; three light-mode
colors are below 3:1 contrast on the surface, so charts using them carry visible labels
or a table view.
"""

import plotly.graph_objects as go
import streamlit as st

SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
}


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


def series_colors() -> list[str]:
    return SERIES[theme_mode()]


def style(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    """Recessive grid and axes, no chart junk; Streamlit's theme supplies surface and text colors."""
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 40, "b": 8},
        showlegend=legend,
        # Legend sits below the plot so it never collides with the title.
        legend={"orientation": "h", "yanchor": "top", "y": -0.12, "x": 0},
        hoverlabel={"namelength": -1},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1, zeroline=False)
    return fig
