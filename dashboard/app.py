"""
============================================================================
 DASHBOARD/APP.PY — Planning Portal Feedback Dashboard (Streamlit)
============================================================================
Launch with:
    streamlit run dashboard/app.py

Reads ONLY from SQLite (data/metrics.db) via store.py — never touches raw
survey files or the SurveyMonkey API directly. Run run_pipeline.py (or the
scheduled GitHub Action) first to populate data.

Page structure:
    Tab 1 — Feedback ratings  : ratings/NSAT/volume side of things, split
                                 into its own Month by month (KPI cards,
                                 ratings, this-month sentiment split,
                                 styled to match the earlier GitHub Pages
                                 design) and Aggregate (Overview overlay
                                 chart + Sentiment split trend) sub-tabs.
    Tab 2 — Feedback comments : everything derived from free-text comments
                                 (tag groups, features, errors, browse
                                 reviews), split the same way into its own
                                 Month by month / Aggregate sub-tabs.

Edit THIS file if you want to:
  - Change a page's layout or charts (Section 4 — PAGE FUNCTIONS)
  - Add a brand new page (Section 4 to write it, Section 6 to register it)
  - Change chart colours globally — edit dashboard_config.py instead
============================================================================
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard_config import PALETTE, SERIES_COLORS, SENTIMENT_COLORS
from store import (
    db_exists,
    load_available_months,
    load_latest_run_date,
    load_monthly_summary,
    load_tag_group_trends,
    load_feature_trends,
    load_error_trends,
    load_review_detail,
    filter_by_months,
)

matplotlib.use("Agg")

# Emoji used in place of the raw positive/negative/neutral text labels
# (Browse Reviews) — matches the SurveyMonkey-style face icons.
SENTIMENT_EMOJI = {
    "positive": "🙂",
    "neutral": "😐",
    "negative": "🙁",
}


# ============================================================================
# SECTION 1 — PAGE CONFIG + CSS
# ============================================================================
st.set_page_config(
    page_title="Planning Portal Feedback",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_css():
    """Card + pill styling to match the earlier GitHub Pages design."""
    st.markdown(
        f"""
        <style>
        .kpi-card {{
            background: {PALETTE["paper"]};
            border: 1px solid {PALETTE["grid"]};
            border-left: 5px solid {PALETTE["neutral"]};
            border-radius: 10px;
            padding: 22px 24px 22px 28px;
            height: 100%;
        }}
        .kpi-label {{
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: {PALETTE["muted"]};
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 2.6rem;
            font-weight: 700;
            color: {PALETTE["ink"]};
            line-height: 1.1;
        }}
        .kpi-delta-up {{
            font-size: 0.85rem;
            color: {PALETTE["positive"]};
            margin-top: 4px;
        }}
        .kpi-delta-down {{
            font-size: 0.85rem;
            color: {PALETTE["negative"]};
            margin-top: 4px;
        }}
        .kpi-delta-flat {{
            font-size: 0.85rem;
            color: {PALETTE["muted"]};
            margin-top: 4px;
        }}
        div[data-testid="stButton"] > button {{
            border-radius: 999px;
            border: 1px solid {PALETTE["grid"]};
            padding: 2px 16px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _kpi_card(label: str, value: str, delta: float | None = None,
              delta_suffix: str = "", higher_is_better: bool = True,
              sublabel: str | None = None):
    delta_html = ""
    if delta is not None:
        if abs(delta) < 1e-9:
            cls, arrow = "kpi-delta-flat", "→"
        elif (delta > 0) == higher_is_better:
            cls, arrow = "kpi-delta-up", "↑"
        else:
            cls, arrow = "kpi-delta-down", "↓"
        delta_html = f'<div class="{cls}">{arrow} {abs(delta):.1f}{delta_suffix} vs prior month</div>'
    elif sublabel:
        # Same visual slot as the delta line, just neutral-colored —
        # keeps the card's height consistent whether it shows a trend
        # delta or a plain sublabel like "40% of respondents".
        delta_html = f'<div class="kpi-delta-flat">{sublabel}</div>'

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# SECTION 2 — CHART HELPERS (shared across pages)
# ============================================================================

def _line_trend(df: pd.DataFrame, x: str, y: str, title: str,
                 ylabel: str = "", color: str = None) -> plt.Figure:
    color = color or PALETTE["primary"]
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    ax.plot(df[x], df[y], marker="o", color=color, linewidth=2)
    ax.set_title(title, fontsize=12, loc="left", pad=10, color=PALETTE["ink"])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0, colors=PALETTE["ink"])
    ax.tick_params(axis="y", colors=PALETTE["ink"])
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    fig.tight_layout()
    return fig


def _bar_trend(df: pd.DataFrame, x: str, y: str, title: str,
               ylabel: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    ax.bar(df[x], df[y], color=PALETTE["primary"], width=0.5)
    ax.set_title(title, fontsize=12, loc="left", pad=10, color=PALETTE["ink"])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0, colors=PALETTE["ink"])
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    fig.tight_layout()
    return fig


def _sentiment_stacked_bar(df: pd.DataFrame, group_col: str, title: str,
                            pct: bool = True) -> plt.Figure:
    """Stacked bar of positive/negative/neutral counts (or %) per group_col value."""
    table = df.groupby(group_col)[["positive_count", "negative_count", "neutral_count"]].sum()
    table = table.rename(columns={
        "positive_count": "positive", "negative_count": "negative", "neutral_count": "neutral"
    })
    if pct:
        totals = table.sum(axis=1)
        table = table.div(totals, axis=0).fillna(0) * 100
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]

    n_rows = len(table)
    fig_h = max(4, n_rows * 0.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])

    categories = table.index.tolist()
    wrapped_categories = [textwrap.fill(str(c), width=18) for c in categories]
    y = np.arange(len(categories))
    lefts = np.zeros(len(categories))

    for col in ["negative", "neutral", "positive"]:
        if col not in table.columns:
            continue
        vals = table[col].values.astype(float)
        color = SENTIMENT_COLORS[col]
        bars = ax.barh(y, vals, left=lefts, label=col.title(), color=color, height=0.6)
        for rect, v in zip(bars, vals):
            if v > (2 if pct else 0.5):
                label = f"{v:.0f}%" if pct else str(int(v))
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + rect.get_height() / 2,
                        label, ha="center", va="center", fontsize=8, color=PALETTE["paper"])
        lefts += vals

    ax.set_yticks(y)
    ax.set_yticklabels(wrapped_categories, fontsize=9)
    ax.set_xlabel("% of reviews" if pct else "Reviews")
    ax.set_title(title, fontsize=12, loc="left", pad=12, color=PALETTE["ink"])
    ax.legend(frameon=True, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xlim(0, lefts.max() * 1.05 if lefts.max() > 0 else 1)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig


def _single_sentiment_bar(pos_pct: float, neg_pct: float, neu_pct: float,
                           title: str) -> plt.Figure:
    """One horizontal 100%-stacked bar — matches the 'Overall comment sentiment' card."""
    fig, ax = plt.subplots(figsize=(10, 2.2))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    segments = [("negative", neg_pct), ("neutral", neu_pct), ("positive", pos_pct)]
    y_pos = -0.35
    left = 0
    for name, val in segments:
        ax.barh([y_pos], [val], left=left, color=SENTIMENT_COLORS[name], height=1)
        if val > 3:
            ax.text(left + val / 2, y_pos, f"{val:.1f}%", ha="center", va="center",
                     fontsize=10, color=PALETTE["paper"], fontweight="bold")
        left += val
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, loc="left", pad=8, color=PALETTE["ink"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def _with_alpha(hex_color: str, alpha: float) -> str:
    """
    Blend a hex color with the white paper background at the given
    alpha, returning a solid hex color. Used to get lighter ★4/★2
    shades from the existing positive/negative palette entries without
    needing new colors in dashboard_config.py, and without matplotlib's
    bar alpha also fading gridlines/labels behind the bar.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r2 = round(r * alpha + 255 * (1 - alpha))
    g2 = round(g * alpha + 255 * (1 - alpha))
    b2 = round(b * alpha + 255 * (1 - alpha))
    return f"#{r2:02x}{g2:02x}{b2:02x}"


def _rating_distribution_bar(counts: dict, title: str) -> plt.Figure:
    """
    Horizontal ★5..★1 bar chart, green-to-red, matching the earlier
    GitHub Pages design. counts: {5: n, 4: n, 3: n, 2: n, 1: n}.
    """
    stars = [5, 4, 3, 2, 1]
    values = [counts.get(s) or 0 for s in stars]
    colors = [
        PALETTE["positive"],
        _with_alpha(PALETTE["positive"], 0.55),
        PALETTE["neutral"],
        _with_alpha(PALETTE["negative"], 0.55),
        PALETTE["negative"],
    ]
    labels = [f"★{s}" for s in stars]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    bars = ax.barh(labels, values, color=colors, height=0.6)
    max_val = max(values) if max(values) > 0 else 1
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{int(v)}", va="center", fontsize=9, color=PALETTE["ink"])
    ax.invert_yaxis()  # ★5 on top, matching the earlier design
    if title:
        ax.set_title(title, fontsize=12, loc="left", pad=10, color=PALETTE["ink"])
    ax.set_xlabel("Responses")
    ax.set_xlim(0, max_val * 1.15)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    fig.tight_layout()
    return fig


def _diverging_sentiment_bar(df: pd.DataFrame, group_col: str, score_col: str,
                              title: str) -> plt.Figure:
    """Horizontal bar from -1 (all negative) to +1 (all positive) per category."""
    table = df.groupby(group_col)[score_col].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, len(table) * 0.4)))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    colors = [SENTIMENT_COLORS["positive"] if v >= 0 else SENTIMENT_COLORS["negative"]
              for v in table.values]
    ax.barh(table.index, table.values, color=colors, height=0.6)
    ax.axvline(0, color=PALETTE["ink"], linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Sentiment score (-1 = all negative, +1 = all positive)")
    ax.set_title(title, fontsize=12, loc="left", pad=10, color=PALETTE["ink"])
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    fig.tight_layout()
    return fig


RATING_SCALE = (1, 5)


def _overlay_trend_chart(trend: pd.DataFrame) -> plt.Figure:
    """
    Single combined chart replacing the old separate volume / avg rating /
    NSAT graphs on the Overview page (change #2). Styled to match Nida's
    reference `plot_monthly_combined` (generate_dashboard.py) as closely as
    possible while pulling colors from dashboard_config.py instead of
    hardcoded hex:
      - grey bars     : total responses, on a genuinely-scaled but hidden
                        count axis (so it isn't squashed flat by rating/NSAT)
      - purple dashed : comments (total_with_feedback), same hidden count axis
      - blue line     : average rating, left axis, fixed 1-5 scale
      - green dashed  : NSAT %, right axis, its own natural scale

    Unlike the first pass, this uses a REAL third axis for the counts
    (twinx, hidden) rather than manually normalising into the rating
    axis's range — that's what was causing the label collisions
    (NSAT % text landing on top of bar-count text, legend overlapping
    the title). Fixed-point label offsets (textcoords="offset points")
    keep annotations a constant pixel distance from their marker
    regardless of axis scale, which is what actually prevents overlap.
    """
    import matplotlib.ticker as mticker

    color_avg = PALETTE["primary"]
    color_nsat = PALETTE["positive"]
    color_comment = PALETTE["accent"]
    color_resp = _with_alpha(PALETTE["muted"], 0.6)

    # Nice "Nov 2025" x-labels when the month column parses as YYYY-MM;
    # falls back to the raw string (e.g. "2025-11") otherwise.
    parsed = pd.to_datetime(trend["month"], format="%Y-%m", errors="coerce")
    month_labels = parsed.dt.strftime("%b %Y") if parsed.notna().all() else trend["month"]

    x = list(range(len(trend)))

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax1.set_facecolor(PALETTE["paper"])

    # ── Background: response bars + comment line on a hidden count axis ──
    ax_count = ax1.twinx()
    for spine in ax_count.spines.values():
        spine.set_visible(False)
    ax_count.set_yticks([])

    responses = trend["total_respondents"].astype(float)
    comments = trend["total_with_feedback"].astype(float) if "total_with_feedback" in trend else responses * 0
    count_max = max(responses.max(), comments.max())
    count_max = count_max if count_max > 0 else 1
    ax_count.set_ylim(0, count_max * 2.0)  # keeps bars/comments in the lower half

    ax_count.bar(x, responses, width=0.6, color=color_resp, zorder=1,
                 label="Total responses", edgecolor=PALETTE["muted"], linewidth=0.8)
    for xi, v in zip(x, responses):
        ax_count.annotate(f"{int(v):,}", (xi, v), textcoords="offset points",
                           xytext=(0, 4), ha="center", fontsize=8, color=PALETTE["ink"])

    ax_count.plot(x, comments, marker="^", linewidth=2, markersize=6, linestyle="--",
                  color=color_comment, zorder=3, label="Comments")
    for xi, v in zip(x, comments):
        if pd.notna(v):
            ax_count.annotate(f"{int(v):,}", (xi, v), textcoords="offset points",
                               xytext=(0, 8), ha="center", fontsize=8, color=color_comment)

    # ── Average rating (left axis, fixed 1-5 scale) ──────────────────────
    ax1.plot(x, trend["avg_rating"], marker="o", linewidth=2.5, markersize=7,
             color=color_avg, label="Avg rating", zorder=4)
    ax1.set_ylabel("Average rating", color=color_avg, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_avg)
    ax1.set_ylim(RATING_SCALE[0] - 0.5, RATING_SCALE[1] + 0.5)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(1))
    for xi, v in zip(x, trend["avg_rating"]):
        if pd.notna(v):
            ax1.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9, color=color_avg, fontweight="bold")

    # ── NSAT % (right axis, its own natural scale) ───────────────────────
    ax2 = ax1.twinx()
    ax2.spines["right"].set_position(("axes", 1.0))
    ax2.spines["right"].set_color(PALETTE["ink"])
    ax2.spines["right"].set_linewidth(1.0)
    ax2.set_facecolor("none")
    ax2.plot(x, trend["nsat"], marker="s", linewidth=2.5, markersize=7, linestyle="--",
             color=color_nsat, label="NSAT %", zorder=4)
    ax2.set_ylabel("NSAT %", color=color_nsat, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_nsat)
    # Push the NSAT axis range up so the line sits clearly ABOVE the rating
    # line, with a guaranteed gap — not just "somewhat higher". Anchored off
    # where the rating line actually falls on its own axis (rather than a
    # fixed multiplier), since a fixed multiplier still let them cross when
    # NSAT dipped and rating ticked up in the same month (e.g. Feb 2026).
    nsat_vals = trend["nsat"].dropna()
    if not nsat_vals.empty:
        nmin, nmax = float(nsat_vals.min()), float(nsat_vals.max())
        span = max(nmax - nmin, 1e-6)
        rating_bot_lim, rating_top_lim = RATING_SCALE[0] - 0.5, RATING_SCALE[1] + 0.5
        rating_max = float(trend["avg_rating"].max()) if trend["avg_rating"].notna().any() else RATING_SCALE[1]
        rating_max_frac = (rating_max - rating_bot_lim) / (rating_top_lim - rating_bot_lim)
        f_min = max(0.85, min(rating_max_frac + 0.12, 0.90))  # NSAT's lowest point sits here
        f_max = min(f_min + 0.12, 0.99)                        # NSAT's highest point sits here
        gap = f_max - f_min
        height = span / gap
        nsat_low = nmin - f_min * height
        nsat_high = nsat_low + height
        ax2.set_ylim(nsat_low, nsat_high)
    for xi, v in zip(x, trend["nsat"]):
        if pd.notna(v):
            ax2.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points",
                         xytext=(0, -16), ha="center", fontsize=9, color=color_nsat, fontweight="bold")
    for spine_name in ("top", "left", "bottom"):
        ax2.spines[spine_name].set_visible(False)

    # ── X-axis & layout ───────────────────────────────────────────────────
    ax1.set_xticks(x)
    ax1.set_xticklabels(month_labels, rotation=0, ha="center", fontsize=10)
    ax1.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.spines["right"].set_visible(False)
    for side in ("top", "left", "bottom"):
        ax1.spines[side].set_color(PALETTE["ink"])
        ax1.spines[side].set_linewidth(1.0)

    # Legend outside the axes, top-right — matching Rating distribution
    # and Sentiment split over time.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax_count.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
               loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=9, frameon=True)

    fig.tight_layout()
    return fig


def _rating_distribution_stacked(view: pd.DataFrame) -> plt.Figure:
    """
    Rating distribution across months — 100%-stacked bar per month, one
    segment per star level, percentage labeled inside each segment.
    Uses the SAME 5-color green-to-red scheme as _rating_distribution_bar
    (Monthly view) so the two stay visually consistent: ★5/★4 green
    shades, ★3 amber, ★2/★1 red shades. Labeling style (percentage inside
    each segment, same font/weight/color) matches the Sentiment split
    over time chart on the Sentiment page.
    """
    star_cols = ["rating_5", "rating_4", "rating_3", "rating_2", "rating_1"]
    data = view.dropna(subset=star_cols, how="all").copy()
    if data.empty:
        return None

    parsed = pd.to_datetime(data["month"], format="%Y-%m", errors="coerce")
    month_labels = parsed.dt.strftime("%b %Y") if parsed.notna().all() else data["month"]

    colors = {
        "rating_5": PALETTE["positive"],
        "rating_4": _with_alpha(PALETTE["positive"], 0.55),
        "rating_3": PALETTE["neutral"],
        "rating_2": _with_alpha(PALETTE["negative"], 0.55),
        "rating_1": PALETTE["negative"],
    }
    star_labels = {"rating_5": "5 star", "rating_4": "4 star", "rating_3": "3 star",
                   "rating_2": "2 star", "rating_1": "1 star"}

    totals = data[star_cols].sum(axis=1).replace(0, np.nan)
    pct = data[star_cols].div(totals, axis=0).fillna(0) * 100

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])

    x = np.arange(len(data))
    bottom = np.zeros(len(data))
    # Bottom-to-top order 1★→5★ so 5★ ends up on top, matching the reference.
    for col in ["rating_1", "rating_2", "rating_3", "rating_4", "rating_5"]:
        vals = pct[col].values
        ax.bar(x, vals, bottom=bottom, color=colors[col], width=0.6, label=star_labels[col])
        for xi, (b, v) in enumerate(zip(bottom, vals)):
            if v > 3:
                ax.text(xi, b + v / 2, f"{v:.1f}%", ha="center", va="center",
                         fontsize=9, color=PALETTE["paper"], fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, rotation=0, ha="center")
    ax.set_ylabel("Ratings (%)")
    ax.set_ylim(0, 100)
    handles, labels = ax.get_legend_handles_labels()
    # Legend in 5★-on-top reading order.
    order = ["5 star", "4 star", "3 star", "2 star", "1 star"]
    ordered = [handles[labels.index(l)] for l in order]
    ax.legend(ordered, order, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=True)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["ink"])
        spine.set_linewidth(1.0)
    fig.tight_layout()
    return fig


def _feature_negative_heatmap(feat_view: pd.DataFrame, top_n: int = 12) -> plt.Figure | None:
    """
    Feature x month heatmap, cell color = % negative mentions.
    Replaces the "features to compare" line chart (change #5) — the line
    chart got unreadable once more than 2-3 features were selected;
    a heatmap scales to many features/months at a glance instead.
    """
    if feat_view.empty:
        return None
    feat_view = feat_view.dropna(subset=["feature"])
    feat_view = feat_view[feat_view["feature"].astype(str).str.strip().str.lower() != "nan"]
    if feat_view.empty:
        return None

    grouped = feat_view.groupby(["month", "feature"]).agg(
        total_mentions=("total_mentions", "sum"),
        negative_mentions=("negative_mentions", "sum"),
    ).reset_index()
    grouped["negative_pct"] = (grouped["negative_mentions"] / grouped["total_mentions"] * 100).round(1)

    top_features = (
        grouped.groupby("feature")["total_mentions"].sum()
        .sort_values(ascending=False).head(top_n).index.tolist()
    )
    grouped = grouped[grouped["feature"].isin(top_features)]
    if grouped.empty:
        return None

    pivot = grouped.pivot_table(index="feature", columns="month", values="negative_pct")
    pivot = pivot.loc[top_features]  # keep ranked order, most-mentioned feature on top

    cmap = LinearSegmentedColormap.from_list(
        "neg_pct", [PALETTE["positive"], PALETTE["paper"], PALETTE["negative"]]
    )

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.5)))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0, ha="center")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if pd.notna(v):
                text_color = PALETTE["paper"] if v > 65 or v < 20 else PALETTE["ink"]
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                         fontsize=8, color=text_color)

    ax.set_title("Feature comparison across months — % negative mentions",
                  fontsize=12, loc="left", pad=10, color=PALETTE["ink"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("% negative", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


# ============================================================================
# SECTION 3 — SIDEBAR
# ============================================================================

def _sidebar():
    st.sidebar.title("Planning Portal")
    st.sidebar.caption("Feedback Pipeline Dashboard")
    st.sidebar.divider()

    months = load_available_months()
    if not months:
        st.sidebar.warning(
            "No data found. Run the pipeline first:\n\n`python run_pipeline.py`"
        )
        return

    latest_run = load_latest_run_date()
    if latest_run:
        st.sidebar.caption(f"Last pipeline run: **{latest_run}**")
    st.sidebar.caption(f"{len(months)} month(s) available: {months[0]} to {months[-1]}")
    st.sidebar.caption("DB: `data/metrics.db`")


def _aggregate_month_picker(months: list[str], key_prefix: str = "agg",
                             title: str | None = None) -> list[str] | None:
    """Month selector for an 'Aggregate' sub-tab, rendered in the sidebar
    (not inline in the page) so it's out of the way of the charts.
    key_prefix keeps widget keys unique when this is used in more than one
    tab at once. Returns None for 'All time' (no filter), or the (possibly
    empty) list of months selected — an empty list here is a deliberate
    "show nothing" state, not the same as "no filter".

    Both 'Feedback ratings > Aggregate' and 'Feedback comments > Aggregate'
    render on every script rerun regardless of which tab is visually
    active (Streamlit renders all tab bodies, it just hides the inactive
    ones), so both pickers end up in the sidebar at once — hence the
    `title` label so it's clear which section each one filters.
    """
    if title:
        st.sidebar.markdown(f"**{title}**")
    mode = st.sidebar.radio("View", ["All time", "Select months"], index=0, horizontal=True,
                             key=f"{key_prefix}_mode")
    if mode == "All time":
        st.sidebar.divider()
        return None
    selected = st.sidebar.multiselect("Select months", options=months, default=months,
                                       key=f"{key_prefix}_months")
    if not selected:
        st.sidebar.warning("No months selected — pick at least one to see data.")
    st.sidebar.divider()
    return selected


# ============================================================================
# SECTION 4 — PAGE FUNCTIONS (Aggregate view)
# ============================================================================

def page_overview(months: list[str] | None):
    st.header("📊 Overview")

    summary = load_monthly_summary()
    if summary.empty:
        st.warning("No data yet.")
        return

    view = filter_by_months(summary, months)
    if view.empty:
        st.warning("No data for the selected period.")
        return

    total_respondents = int(view["total_respondents"].sum())
    total_feedback = int(view["total_with_feedback"].sum())
    avg_rating = round(
        (view["avg_rating"] * view["total_respondents"]).sum() / max(total_respondents, 1), 2
    ) if view["avg_rating"].notna().any() else None
    avg_nsat = round(view["nsat"].dropna().mean(), 1) if view["nsat"].notna().any() else None

    col1, col2, col3, col4 = st.columns(4, gap="large")
    with col1:
        _kpi_card("Total responses", f"{total_respondents:,}")
    with col2:
        feedback_rate = round(total_feedback / total_respondents * 100, 2) if total_respondents else 0
        _kpi_card("Feedback rate", f"{feedback_rate}%")
    with col3:
        _kpi_card("Average rating", f"{avg_rating}" if avg_rating else "—")
    with col4:
        _kpi_card("NSAT", f"{avg_nsat}%" if avg_nsat is not None else "—")
    st.write("")

    st.divider()

    trend = view.sort_values("month")
    if len(trend) > 1:
        st.subheader("Ratings, NSAT, responses & comments over time")
        st.pyplot(_overlay_trend_chart(trend))

        st.divider()
        st.subheader("Rating distribution across months")
        rating_fig = _rating_distribution_stacked(trend)
        if rating_fig is not None:
            st.pyplot(rating_fig)
            plt.close(rating_fig)
        else:
            st.info("No rating breakdown recorded across these months yet.")
    else:
        st.info("Only one month of data so far — trend charts will appear "
               "once more months are collected.")


def page_sentiment(months: list[str] | None):
    st.header("💬 Sentiment")

    summary = load_monthly_summary()
    if summary.empty:
        st.warning("No data yet.")
        return

    view = filter_by_months(summary, months)
    if view.empty or view["positive_count"].isna().all():
        st.info("No NLP sentiment data yet for this period. "
               "Run the pipeline without SKIP_NLP to populate this.")
        return

    total_pos = int(view["positive_count"].sum())
    total_neg = int(view["negative_count"].sum())
    total_neu = int(view["neutral_count"].sum())
    total = total_pos + total_neg + total_neu

    st.write("")
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        _kpi_card("Positive", f"{round(total_pos/total*100,1)}%" if total else "—")
    with col2:
        _kpi_card("Negative", f"{round(total_neg/total*100,1)}%" if total else "—")
    with col3:
        _kpi_card("Neutral", f"{round(total_neu/total*100,1)}%" if total else "—")
    st.write("")

    st.divider()

    trend = summary.sort_values("month").copy()
    trend = trend[trend["positive_count"].notna()]
    if len(trend) > 1:
        st.subheader("Sentiment split over time")

        parsed = pd.to_datetime(trend["month"], format="%Y-%m", errors="coerce")
        month_labels = parsed.dt.strftime("%b %Y") if parsed.notna().all() else trend["month"]

        fig, ax = plt.subplots(figsize=(10, 4.8))
        fig.patch.set_facecolor(PALETTE["paper"])
        ax.set_facecolor(PALETTE["paper"])
        totals = trend["positive_count"] + trend["negative_count"] + trend["neutral_count"]
        pos_pct = (trend["positive_count"] / totals * 100).fillna(0)
        neg_pct = (trend["negative_count"] / totals * 100).fillna(0)
        neu_pct = (100 - pos_pct - neg_pct).clip(lower=0)

        x = np.arange(len(trend))
        ax.bar(x, neg_pct, color=SENTIMENT_COLORS["negative"], width=0.6, label="Negative")
        ax.bar(x, neu_pct, bottom=neg_pct, color=SENTIMENT_COLORS["neutral"], width=0.6, label="Neutral")
        ax.bar(x, pos_pct, bottom=neg_pct + neu_pct, color=SENTIMENT_COLORS["positive"], width=0.6, label="Positive")

        # Percentage labels inside each segment (matching the count labels
        # on the Rating distribution chart).
        bottoms = {"neg": np.zeros(len(trend)), "neu": neg_pct.values, "pos": (neg_pct + neu_pct).values}
        for key, vals in zip(["neg", "neu", "pos"], [neg_pct, neu_pct, pos_pct]):
            for xi, (b, v) in enumerate(zip(bottoms[key], vals.values)):
                if v > 3:
                    ax.text(xi, b + v / 2, f"{v:.1f}%", ha="center", va="center",
                            fontsize=9, color=PALETTE["paper"], fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(month_labels, rotation=0, ha="center")
        ax.set_ylabel("% of reviews")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=True)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(PALETTE["ink"])
            spine.set_linewidth(1.0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Only one month of sentiment data so far.")


def render_aggregate_view():
    months_all = load_available_months()
    months = _aggregate_month_picker(months_all, key_prefix="agg", title="Feedback ratings — months")
    page_overview(months)
    st.divider()
    page_sentiment(months)


# ============================================================================
# SECTION 4B — PAGE FUNCTIONS (Feedback comments tab — tag groups / features
# / errors / browse reviews; everything derived from free-text comments)
# ============================================================================

def page_tag_groups(months: list[str] | None):
    st.header("🏷️ Tag Groups")

    df = load_tag_group_trends()
    if df.empty:
        st.info("No tag data yet. Run the pipeline without SKIP_NLP.")
        return

    view = filter_by_months(df, months)
    view = view[~view["tag_group"].astype(str).str.strip().str.casefold().eq("user type")]
    if view.empty:
        st.warning("No data for the selected period.")
        return

    st.subheader("Volume + sentiment by tag group")
    st.pyplot(_sentiment_stacked_bar(view, "tag_group", "", pct=True))

    st.divider()
    st.subheader("Raw counts")
    display = view.groupby("tag_group").agg(
        total_reviews=("total_reviews", "sum"),
        positive=("positive_count", "sum"),
        negative=("negative_count", "sum"),
        neutral=("neutral_count", "sum"),
        avg_sentiment_score=("avg_sentiment_score", "mean"),
    ).sort_values("total_reviews", ascending=False).reset_index()
    st.dataframe(display, use_container_width=True)


def page_features(months: list[str] | None):
    st.header("🧩 Features")

    df = load_feature_trends()
    if df.empty:
        st.info("No feature data yet. Run the pipeline without SKIP_NLP.")
        return

    view = filter_by_months(df, months)
    view = view.dropna(subset=["feature"])
    view = view[view["feature"].astype(str).str.strip().str.lower() != "nan"]
    if view.empty:
        st.warning("No data for the selected period.")
        return

    agg = view.groupby("feature").agg(
        total_mentions=("total_mentions", "sum"),
        negative_mentions=("negative_mentions", "sum"),
        positive_mentions=("positive_mentions", "sum"),
    ).reset_index()
    agg["negative_pct"] = round(agg["negative_mentions"] / agg["total_mentions"] * 100, 1)
    agg = agg.sort_values("negative_pct", ascending=False)

    st.subheader("Most negatively-mentioned features")
    top = agg.head(15).set_index("feature")
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.4)))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    ax.barh(top.index[::-1], top["negative_pct"][::-1], color=SENTIMENT_COLORS["negative"])
    ax.set_xlabel("% negative mentions")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("All features")
    st.dataframe(agg, use_container_width=True)

    # ---- Feature comparison across months (heatmap — was the "features
    #      to compare" line chart, replaced per change #5) ----------------
    st.divider()
    st.subheader("Feature comparison across months")
    heatmap_fig = _feature_negative_heatmap(view)
    if heatmap_fig is None:
        st.info("Not enough data across months to build the comparison heatmap yet.")
    else:
        st.pyplot(heatmap_fig)
        plt.close(heatmap_fig)


def page_errors(months: list[str] | None):
    st.header("⚠️ Errors")

    df = load_error_trends()
    if df.empty:
        st.info("No error pattern data yet. Run the pipeline without SKIP_NLP.")
        return

    view = filter_by_months(df, months)
    if view.empty:
        st.warning("No data for the selected period.")
        return

    agg = view.groupby("error_pattern")["count"].sum().sort_values(ascending=False).head(15)

    st.subheader("Most frequent error patterns")
    fig, ax = plt.subplots(figsize=(10, max(4, len(agg) * 0.4)))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["paper"])
    ax.barh(agg.index[::-1], agg.values[::-1], color=PALETTE["primary"])
    ax.set_xlabel("Mentions")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def page_browse_reviews(months: list[str] | None):
    st.header("🔎 Browse Reviews")

    df = load_review_detail(months)
    if df.empty:
        st.info("No tagged reviews yet. Run the pipeline without SKIP_NLP.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        tag_options = ["All"] + sorted(df["primary_tag_group"].dropna().unique().tolist())
        tag_filter = st.selectbox("Tag group", tag_options)
    with col2:
        sent_values = sorted(df["grouping_sentiment"].dropna().unique().tolist())
        sent_options = ["All"] + sent_values
        sent_filter = st.selectbox(
            "Sentiment", sent_options,
            format_func=lambda v: v if v == "All" else f"{SENTIMENT_EMOJI.get(v, '')} {v.title()}",
        )
    with col3:
        search = st.text_input("Search feedback text")

    filtered = df.copy()
    if tag_filter != "All":
        filtered = filtered[filtered["primary_tag_group"] == tag_filter]
    if sent_filter != "All":
        filtered = filtered[filtered["grouping_sentiment"] == sent_filter]
    if search:
        filtered = filtered[filtered["feedback_clean"].str.contains(search, case=False, na=False)]

    st.caption(f"{len(filtered):,} reviews match")
    show_cols = [
        "respondent_id", "month", "rating",
        "primary_tag_group", "primary_tag", "grouping_sentiment", "feedback_clean",
    ]
    show_cols = [c for c in show_cols if c in filtered.columns]
    display_df = filtered[show_cols].copy()
    if "grouping_sentiment" in display_df.columns:
        # Icon only (no "Positive"/"Negative" text) sitting right next to the
        # feedback text — matching the compact inline-icon style of the
        # SurveyMonkey reference, rather than a verbose text label.
        display_df["grouping_sentiment"] = display_df["grouping_sentiment"].map(
            lambda v: SENTIMENT_EMOJI.get(v, "") if pd.notna(v) else ""
        )
    display_df = display_df.rename(columns={
        "respondent_id": "Respondent", "month": "Month", "rating": "Rating",
        "grouping_sentiment": "", "primary_tag_group": "Tag group",
        "primary_tag": "Tag", "feedback_clean": "Feedback",
    })
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600,
        column_config={"": st.column_config.TextColumn(width="small")},
    )


def _render_comment_themes_month(selected_month: str):
    """Thematic analysis + Feature & error analysis for one month — used by
    the Feedback comments tab's Month by month sub-tab."""
    tag_df = load_tag_group_trends()
    tag_month = tag_df[tag_df["month"] == selected_month] if not tag_df.empty else tag_df
    if not tag_month.empty:
        tag_month = tag_month[~tag_month["tag_group"].astype(str).str.strip().str.casefold().eq("user type")]

    st.subheader("Thematic analysis")
    if tag_month.empty:
        st.info("No tag data yet for this month. Run the pipeline without SKIP_NLP.")
    else:
        tc1, tc2 = st.columns(2)
        with tc1:
            vol = tag_month.groupby("tag_group")["total_reviews"].sum().sort_values(ascending=False)
            sentiment_by_group = tag_month.groupby("tag_group")["avg_sentiment_score"].mean()
            colors = [SENTIMENT_COLORS["positive"] if sentiment_by_group.get(g, 0) >= 0
                      else SENTIMENT_COLORS["negative"] for g in vol.index]
            fig, ax = plt.subplots(figsize=(10, max(4, len(vol) * 0.4)))
            fig.patch.set_facecolor(PALETTE["paper"])
            ax.set_facecolor(PALETTE["paper"])
            ax.barh(vol.index[::-1], vol.values[::-1], color=colors[::-1])
            ax.set_title("Tag group mentions — this month", fontsize=12, loc="left", color=PALETTE["ink"])
            ax.set_xlabel("Volume of comments per theme")
            ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with tc2:
            st.pyplot(_diverging_sentiment_bar(tag_month, "tag_group", "avg_sentiment_score",
                                                "Tag group sentiment — this month"))

    st.divider()

    st.subheader("Feature & error analysis")
    fc1, fc2 = st.columns(2)
    with fc1:
        feat_df = load_feature_trends()
        feat_month = feat_df[feat_df["month"] == selected_month] if not feat_df.empty else feat_df
        if not feat_month.empty:
            feat_month = feat_month.dropna(subset=["feature"])
            feat_month = feat_month[feat_month["feature"].astype(str).str.strip().str.lower() != "nan"]
        if feat_month.empty:
            st.info("No feature data yet for this month.")
        else:
            agg = feat_month.groupby("feature").agg(
                total_mentions=("total_mentions", "sum"),
                negative_mentions=("negative_mentions", "sum"),
            )
            agg["negative_pct"] = round(agg["negative_mentions"] / agg["total_mentions"] * 100, 1)
            top = agg.sort_values("negative_pct", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6, max(3, len(top) * 0.4)))
            fig.patch.set_facecolor(PALETTE["paper"])
            ax.set_facecolor(PALETTE["paper"])
            ax.barh(top.index[::-1], top["negative_pct"][::-1], color=SENTIMENT_COLORS["negative"])
            ax.set_title("Most negatively mentioned features", fontsize=11, loc="left", color=PALETTE["ink"])
            ax.set_xlabel("% of mentions that are negative")
            ax.set_xlim(0, 100)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    with fc2:
        err_df = load_error_trends()
        err_month = err_df[err_df["month"] == selected_month] if not err_df.empty else err_df
        if err_month.empty:
            st.info("No error pattern data yet for this month.")
        else:
            top_err = err_month.groupby("error_pattern")["count"].sum().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6, max(3, len(top_err) * 0.4)))
            fig.patch.set_facecolor(PALETTE["paper"])
            ax.set_facecolor(PALETTE["paper"])
            ax.barh(top_err.index[::-1], top_err.values[::-1], color=PALETTE["neutral"])
            ax.set_title("Top error patterns", fontsize=11, loc="left", color=PALETTE["ink"])
            ax.set_xlabel("Most frequently reported issues")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def render_feedback_comments_view():
    months = load_available_months()
    if not months:
        st.info("No data yet.")
        return

    sub_agg, sub_month = st.tabs(["Aggregate", "Month by month"])

    with sub_month:
        default_month = st.session_state.get("selected_month", months[-1])
        selected = st.selectbox(
            "Month", months, index=months.index(default_month) if default_month in months else len(months) - 1,
            key="comments_month_select",
        )
        st.divider()
        _render_comment_themes_month(selected)

    with sub_agg:
        agg_months = _aggregate_month_picker(months, key_prefix="comments_agg", title="Feedback comments — months")
        page_tag_groups(agg_months)
        st.divider()
        page_features(agg_months)
        st.divider()
        page_errors(agg_months)
        st.divider()
        page_browse_reviews(agg_months)


# ============================================================================
# SECTION 5 — MONTHLY SNAPSHOT VIEW (matches the preferred design)
# ============================================================================

def render_monthly_view():
    months = load_available_months()
    summary = load_monthly_summary().sort_values("month")

    # Month pill row
    default_idx = len(months) - 1
    n_pills = min(len(months), 8)
    recent_months = months[-n_pills:]
    if "selected_month" not in st.session_state:
        st.session_state.selected_month = months[default_idx]

    cols = st.columns(len(recent_months))
    for i, m in enumerate(recent_months):
        with cols[i]:
            is_selected = m == st.session_state.selected_month
            if st.button(m, key=f"pill_{m}", type="primary" if is_selected else "secondary",
                         use_container_width=True):
                st.session_state.selected_month = m
                st.rerun()

    selected_month = st.session_state.selected_month
    row = summary[summary["month"] == selected_month]
    if row.empty:
        st.warning("No data for the selected month.")
        return
    row = row.iloc[0]

    prior_months = summary[summary["month"] < selected_month]
    prior = prior_months.iloc[-1] if not prior_months.empty else None

    st.caption("THIS MONTH AT A GLANCE")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        delta = (row["total_respondents"] - prior["total_respondents"]) if prior is not None else None
        _kpi_card("Total responses", f"{int(row['total_respondents']):,}",
                  delta=delta, delta_suffix="")
    with c2:
        delta = (row["avg_rating"] - prior["avg_rating"]) if prior is not None and pd.notna(row["avg_rating"]) else None
        _kpi_card("Average rating", f"{row['avg_rating']:.1f}" if pd.notna(row["avg_rating"]) else "—",
                  delta=delta, delta_suffix="")
    with c3:
        delta = (row["nsat"] - prior["nsat"]) if prior is not None and pd.notna(row["nsat"]) else None
        _kpi_card("NSAT score", f"{row['nsat']:.1f}" if pd.notna(row["nsat"]) else "—",
                  delta=delta, delta_suffix="")
    with c4:
        left_fb = int(row["total_with_feedback"]) if pd.notna(row["total_with_feedback"]) else 0
        pct = round(left_fb / row["total_respondents"] * 100, 0) if row["total_respondents"] else 0
        _kpi_card("Left feedback", f"{left_fb:,}", sublabel=f"{pct:.0f}% of respondents")

    st.divider()

    # ---- Ratings section ------------------------------------------------
    st.subheader("Ratings")
    st.markdown("**Rating distribution — this month**")
    rating_cols = ["rating_5", "rating_4", "rating_3", "rating_2", "rating_1"]
    if all(c in row.index for c in rating_cols) and row[rating_cols].notna().any():
        counts = {s: row[f"rating_{s}"] for s in [5, 4, 3, 2, 1]}
        st.pyplot(_rating_distribution_bar(counts, ""))
    else:
        st.info("No rating breakdown recorded for this month yet.")

    st.divider()

    # ---- Sentiment section ------------------------------------------------
    # Note (change #1): the "NSAT — all months" and "Comment sentiment —
    # all months" trend charts that used to live here have been removed;
    # this section now only shows this month's sentiment split.
    st.subheader("Sentiment — comments only")
    if pd.notna(row.get("positive_count")):
        total = row["positive_count"] + row["negative_count"] + row["neutral_count"]
        if total:
            pos_pct = row["positive_count"] / total * 100
            neg_pct = row["negative_count"] / total * 100
            neu_pct = 100 - pos_pct - neg_pct
            st.pyplot(_single_sentiment_bar(pos_pct, neg_pct, neu_pct,""))
                                             # "Overall comment sentiment — this month"))
    else:
        st.info("No NLP sentiment data yet for this month. Run the pipeline without SKIP_NLP.")


# ============================================================================
# SECTION 6 — PAGE ROUTER
# ============================================================================

def render_feedback_ratings_view():
    """Ratings/NSAT/volume side of the dashboard — Monthly view + Aggregate
    view merged under one top-level tab, split the same way as Feedback
    comments (Month by month / Aggregate), for a consistent structure."""
    sub_agg, sub_month = st.tabs(["Aggregate", "Month by month"])
    with sub_month:
        render_monthly_view()
    with sub_agg:
        render_aggregate_view()


def main():
    _inject_css()

    if not db_exists():
        st.title("Planning Portal Feedback")
        st.warning(
            "No database found yet. Run the pipeline first:\n\n"
            "`python run_pipeline.py`\n\n"
            "or wait for the scheduled Monday-morning GitHub Action to run."
        )
        return

    _sidebar()

    st.title("🗺️ Planning Portal Feedback Dashboard")

    tab1, tab2 = st.tabs(["Feedback ratings", "Feedback comments"])
    with tab1:
        render_feedback_ratings_view()
    with tab2:
        render_feedback_comments_view()


if __name__ == "__main__":
    main()