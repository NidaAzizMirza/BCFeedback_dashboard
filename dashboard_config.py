# dashboard_config.py
# ── Shared styling for the Streamlit dashboard ─────────────────────────────
# Edit here to change colours/palette across every page at once.

PALETTE = {
    "paper":   "#FFFFFF",
    "ink":     "#1F2A33",
    "grid":    "#E4E7EB",
    "primary": "#2B6CB0",   # blue — neutral/volume
    "positive": "#2F855A",  # green
    "negative": "#A6443C",  # red
    "neutral": "#B7862C",   # amber
    "accent":  "#6B46C1",   # purple — highlights
    "muted":   "#8A97A6",
}

SERIES_COLORS = [
    PALETTE["primary"], PALETTE["accent"], PALETTE["neutral"],
    PALETTE["positive"], PALETTE["negative"], PALETTE["muted"],
]

SENTIMENT_COLORS = {
    "positive": PALETTE["positive"],
    "negative": PALETTE["negative"],
    "neutral":  PALETTE["neutral"],
}