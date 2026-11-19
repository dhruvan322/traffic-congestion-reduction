"""
UI Theme and Design System for Traffic Intelligence Operations Center.

Provides unified styling, color tokens, KPI card components,
Plotly dark-theme stylers, and Folium dark map factories.
"""

from typing import Optional, Dict, Any
import streamlit as st

# ──────────────────────────── Design Tokens ────────────────────────────
COLOR_BG = "#0A0E17"
COLOR_SIDEBAR = "#111827"
COLOR_CARD = "#131C2E"
COLOR_CARD_BORDER = "rgba(0, 240, 255, 0.15)"

COLOR_CYAN = "#00F0FF"
COLOR_TEAL = "#00E5BC"
COLOR_EMERALD = "#10B981"
COLOR_AMBER = "#F59E0B"
COLOR_RED = "#EF4444"
COLOR_BLUE = "#3B82F6"

COLOR_TEXT_PRIMARY = "#F8FAFC"
COLOR_TEXT_MUTED = "#94A3B8"
COLOR_TEXT_DIM = "#64748B"

CONGESTION_COLOR_MAP: Dict[str, str] = {
    "Free Flow": COLOR_EMERALD,
    "Low": COLOR_TEAL,
    "Moderate": COLOR_CYAN,
    "High": COLOR_AMBER,
    "Severe": COLOR_RED,
}


# ──────────────────────────── Theme Application ────────────────────────────
def apply_theme():
    """
    Inject global CSS rules to transform Streamlit into an
    enterprise Traffic Intelligence Operations Center.
    """
    custom_css = f"""
    <style>
        /* Base typography & Canvas */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif;
            color: {COLOR_TEXT_PRIMARY};
        }}

        .stApp {{
            background-color: {COLOR_BG};
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(0, 240, 255, 0.03) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(0, 229, 188, 0.02) 0%, transparent 40%);
        }}

        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: {COLOR_SIDEBAR} !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }}

        /* Equal-height Column Containers */
        [data-testid="column"] {{
            display: flex;
            flex-direction: column;
        }}
        [data-testid="column"] > div {{
            height: 100%;
            display: flex;
            flex-direction: column;
        }}

        /* Operations KPI Cards */
        .ops-kpi-card {{
            background: linear-gradient(135deg, rgba(19, 28, 46, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 1.1rem 1.15rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 128px;
            height: 100%;
            box-sizing: border-box;
        }}

        .ops-kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        }}

        .ops-kpi-card.border-cyan {{ border-top: 3px solid {COLOR_CYAN}; }}
        .ops-kpi-card.border-teal {{ border-top: 3px solid {COLOR_TEAL}; }}
        .ops-kpi-card.border-emerald {{ border-top: 3px solid {COLOR_EMERALD}; }}
        .ops-kpi-card.border-amber {{ border-top: 3px solid {COLOR_AMBER}; }}
        .ops-kpi-card.border-red {{ border-top: 3px solid {COLOR_RED}; }}
        .ops-kpi-card.border-blue {{ border-top: 3px solid {COLOR_BLUE}; }}

        .ops-kpi-top {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            min-height: 1.4rem;
        }}

        .ops-kpi-title {{
            font-size: 0.74rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {COLOR_TEXT_MUTED};
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin: 0;
            line-height: 1.2;
        }}

        .ops-kpi-icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            line-height: 1;
            font-family: 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif;
            vertical-align: middle;
        }}

        .ops-kpi-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.7rem;
            font-weight: 700;
            color: {COLOR_TEXT_PRIMARY};
            letter-spacing: -0.02em;
            line-height: 1.2;
            margin: 0.4rem 0 0.3rem 0;
        }}

        .ops-kpi-subtitle {{
            font-size: 0.75rem;
            color: {COLOR_TEXT_DIM};
            display: flex;
            align-items: center;
            gap: 0.35rem;
            margin: 0;
            min-height: 1.1rem;
            line-height: 1.2;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .ops-kpi-delta-pos {{
            color: {COLOR_EMERALD};
            font-weight: 600;
        }}

        .ops-kpi-delta-neg {{
            color: {COLOR_RED};
            font-weight: 600;
        }}

        /* Header block */
        .ops-header-container {{
            margin-bottom: 1.25rem;
            padding-bottom: 0.8rem;
            border-bottom: 1px solid rgba(0, 240, 255, 0.15);
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .ops-header-title {{
            font-size: 1.65rem;
            font-weight: 700;
            color: {COLOR_TEXT_PRIMARY};
            letter-spacing: -0.02em;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}

        .ops-header-subtitle {{
            font-size: 0.875rem;
            color: {COLOR_TEXT_MUTED};
            margin: 0.3rem 0 0 0;
        }}

        /* Status Badges */
        .ops-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.2rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}

        .ops-badge-cyan {{
            background: rgba(0, 240, 255, 0.12);
            color: {COLOR_CYAN};
            border: 1px solid rgba(0, 240, 255, 0.35);
        }}

        .ops-badge-teal {{
            background: rgba(0, 229, 188, 0.12);
            color: {COLOR_TEAL};
            border: 1px solid rgba(0, 229, 188, 0.35);
        }}

        .ops-badge-emerald {{
            background: rgba(16, 185, 129, 0.12);
            color: {COLOR_EMERALD};
            border: 1px solid rgba(16, 185, 129, 0.35);
        }}

        .ops-badge-amber {{
            background: rgba(245, 158, 11, 0.12);
            color: {COLOR_AMBER};
            border: 1px solid rgba(245, 158, 11, 0.35);
        }}

        .ops-badge-red {{
            background: rgba(239, 68, 68, 0.12);
            color: {COLOR_RED};
            border: 1px solid rgba(239, 68, 68, 0.35);
        }}

        /* Section Container Card */
        .ops-section-card {{
            background: rgba(19, 28, 46, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 10px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
        }}

        /* Section Titles */
        .ops-section-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: {COLOR_TEXT_PRIMARY};
            letter-spacing: -0.01em;
            margin: 0 0 0.85rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        /* Native metric widget refinement */
        [data-testid="stMetric"] {{
            background: rgba(19, 28, 46, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 8px !important;
            padding: 0.75rem 1rem !important;
        }}

        [data-testid="stMetricValue"] {{
            font-family: 'JetBrains Mono', monospace !important;
            color: {COLOR_TEXT_PRIMARY} !important;
            font-size: 1.5rem !important;
        }}

        [data-testid="stMetricLabel"] {{
            color: {COLOR_TEXT_MUTED} !important;
            font-size: 0.78rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }}

        /* Button refinement */
        button[kind="primary"] {{
            background: linear-gradient(135deg, {COLOR_CYAN} 0%, {COLOR_TEAL} 100%) !important;
            color: #050B14 !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 6px !important;
            box-shadow: 0 2px 10px rgba(0, 240, 255, 0.25) !important;
        }}

        button[kind="primary"]:hover {{
            box-shadow: 0 4px 18px rgba(0, 240, 255, 0.45) !important;
        }}

        /* Table & Dataframe refinement */
        [data-testid="stDataFrame"] {{
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            overflow: hidden !important;
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 7px;
            height: 7px;
        }}
        ::-webkit-scrollbar-track {{
            background: {COLOR_BG};
        }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {COLOR_CYAN};
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# ──────────────────────────── Reusable Components ────────────────────────────
def render_header(
    title: str,
    subtitle: Optional[str] = None,
    badge_text: Optional[str] = None,
    badge_type: str = "cyan",
):
    """
    Render a standardized Operations Center page/view header.
    """
    badge_html = ""
    if badge_text:
        badge_html = f'<span class="ops-badge ops-badge-{badge_type}">● {badge_text}</span>'

    subtitle_html = (
        f'<p class="ops-header-subtitle">{subtitle}</p>' if subtitle else ""
    )

    header_html = f"""
    <div class="ops-header-container">
        <div>
            <h1 class="ops-header-title">{title}</h1>
            {subtitle_html}
        </div>
        <div>
            {badge_html}
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_kpi_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    status: str = "cyan",
    icon: Optional[str] = None,
    subtitle: Optional[str] = None,
):
    """
    Render a high-contrast, glassmorphic Operations Center KPI card.

    :param title: Card label (e.g. 'AVERAGE SPEED')
    :param value: Displayed metric value (e.g. '34.2 km/h')
    :param delta: Optional trend or delta value (e.g. '+2.4 km/h')
    :param status: Accent color theme: 'cyan', 'teal', 'emerald', 'amber', 'red', 'blue'
    :param icon: Optional emoji or symbol
    :param subtitle: Contextual caption
    """
    icon_html = f'<span class="ops-kpi-icon">{icon}</span>' if icon else ""

    delta_html = ""
    if delta:
        delta_class = "ops-kpi-delta-pos" if not delta.startswith("-") else "ops-kpi-delta-neg"
        delta_html = f'<span class="{delta_class}">{delta}</span>'

    sub_parts = []
    if delta_html:
        sub_parts.append(delta_html)
    if subtitle:
        sub_parts.append(subtitle)

    subtitle_content = " \u2022 ".join(sub_parts) if sub_parts else "&nbsp;"

    card_html = f"""
    <div class="ops-kpi-card border-{status}">
        <div class="ops-kpi-top">
            <span class="ops-kpi-title">{icon_html}{title}</span>
        </div>
        <div class="ops-kpi-value">{value}</div>
        <div class="ops-kpi-subtitle">{subtitle_content}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def apply_plotly_theme(fig):
    """
    Apply the Operations Center deep navy and electric cyan palette
    to a Plotly Figure using modern title.font structure.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(10, 14, 23, 0.45)",
        font=dict(
            family="Inter, -apple-system, sans-serif",
            color=COLOR_TEXT_PRIMARY,
            size=12,
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(0, 240, 255, 0.15)",
            tickfont=dict(color=COLOR_TEXT_MUTED, size=11),
            title=dict(font=dict(color=COLOR_TEXT_MUTED, size=12)),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(0, 240, 255, 0.15)",
            tickfont=dict(color=COLOR_TEXT_MUTED, size=11),
            title=dict(font=dict(color=COLOR_TEXT_MUTED, size=12)),
        ),
        hoverlabel=dict(
            bgcolor="#0A0E17",
            bordercolor=COLOR_CYAN,
            font=dict(
                family="JetBrains Mono, monospace",
                size=12,
                color=COLOR_TEXT_PRIMARY,
            ),
        ),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def get_dark_folium_map(lat: float, lon: float, zoom: int = 12):
    """
    Create a pre-configured Folium map with CartoDB dark_matter tiles
    for consistent operations room aesthetic.
    """
    import folium

    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )
    return m
