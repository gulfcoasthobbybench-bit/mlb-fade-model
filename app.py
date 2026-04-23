import streamlit as st
import pandas as pd
from datetime import datetime
from main import run_model

st.set_page_config(
    page_title="MLB Fade Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# THEME / CSS
# =========================================================
st.markdown("""
<style>
:root {
    --bg: #07101d;
    --panel: #0f1a2b;
    --panel-2: #101d31;
    --border: #223049;
    --text: #edf2f7;
    --muted: #93a4bd;
    --green: #39d98a;
    --blue: #4da3ff;
    --purple: #9b6bff;
    --red: #ff5e72;
    --yellow: #f5b942;
    --shadow: 0 10px 24px rgba(0, 0, 0, 0.25);
}

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #07101d 0%, #0a1322 100%);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

.block-container {
    max-width: 1520px;
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #08111f 0%, #091321 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

.sidebar-title {
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: 0.2px;
    margin-bottom: 0.2rem;
}

.sidebar-sub {
    color: var(--muted);
    font-size: 0.85rem;
    margin-bottom: 1.1rem;
}

.nav-group-label {
    color: var(--muted);
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.15rem 0 0.45rem 0;
}

.page-title {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
}

.page-subtitle {
    color: var(--muted);
    font-size: 0.98rem;
    margin-top: 0.35rem;
}

.meta-badges {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.65rem;
    flex-wrap: wrap;
}

.meta-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.6rem;
    border-radius: 999px;
    font-size: 0.74rem;
    color: var(--text);
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.06);
}

.last-run {
    color: var(--muted);
    font-size: 0.9rem;
    white-space: nowrap;
    text-align: right;
    padding-top: 0.6rem;
}

div.stButton > button {
    background: linear-gradient(180deg, #2ecb7f 0%, #21b36f 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.72rem 1.35rem;
    font-weight: 700;
    font-size: 0.95rem;
    width: 100%;
}

.stRadio > div {
    gap: 0.45rem;
}

.card {
    background: linear-gradient(180deg, rgba(18,28,46,0.98), rgba(14,23,38,0.98));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem;
    min-height: 116px;
    box-shadow: var(--shadow);
}

.card-label {
    color: var(--muted);
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.7rem;
}

.card-value {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.55rem;
}

.card-sub {
    color: var(--muted);
    font-size: 0.88rem;
}

.panel-title {
    font-size: 1.02rem;
    font-weight: 800;
    margin-bottom: 0.65rem;
}

.panel-subtitle {
    color: var(--muted);
    font-size: 0.84rem;
    margin-top: -0.25rem;
    margin-bottom: 0.75rem;
}

.panel-shell {
    background: linear-gradient(180deg, rgba(18,28,46,0.98), rgba(13,22,38,0.98));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem;
    box-shadow: var(--shadow);
}

.parlay-card {
    border: 1px solid rgba(255,255,255,0.07);
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    padding: 0.85rem;
    margin-bottom: 0.75rem;
}

.parlay-score {
    color: var(--green);
    font-weight: 800;
    font-size: 1.35rem;
}

.small-note {
    color: var(--muted);
    font-size: 0.8rem;
}

.detail-chip-wrap {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin-top: 0.65rem;
}

.detail-chip {
    display: inline-block;
    padding: 0.35rem 0.55rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.06);
    font-size: 0.76rem;
    color: var(--text);
}

.footer-note {
    color: var(--muted);
    font-size: 0.82rem;
    margin-top: 0.85rem;
}

hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin: 0.6rem 0 0.8rem 0;
}

.dark-table-wrap {
    background: linear-gradient(180deg, rgba(18,28,46,0.98), rgba(13,22,38,0.98));
    border: 1px solid #223049;
    border-radius: 14px;
    padding: 0.4rem;
    box-shadow: 0 10px 24px rgba(0,0,0,0.25);
    overflow: hidden;
}

.dark-table-scroll {
    overflow-x: auto;
}

.dark-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    color: #edf2f7;
    background: transparent;
}

.dark-table thead th {
    text-align: left;
    padding: 0.85rem 0.7rem;
    font-size: 0.77rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #93a4bd;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.02);
    white-space: nowrap;
}

.dark-table tbody td {
    padding: 0.78rem 0.7rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    vertical-align: middle;
    white-space: nowrap;
}

.dark-table tbody tr:hover {
    background: rgba(77,163,255,0.08);
}

.metric-pill {
    display: inline-block;
    min-width: 62px;
    text-align: center;
    padding: 0.3rem 0.45rem;
    border-radius: 8px;
    font-weight: 700;
    color: white;
    font-size: 0.8rem;
}

.pill-blue { background: linear-gradient(180deg, #2d7ff9, #215fbd); }
.pill-purple { background: linear-gradient(180deg, #8f5cff, #6b42cc); }
.pill-green { background: linear-gradient(180deg, #2ecb7f, #209c60); }
.pill-yellow { background: linear-gradient(180deg, #d8a72d, #a87e1c); }
.pill-red { background: linear-gradient(180deg, #ff6a7d, #d14b5d); }

.reason-cell {
    max-width: 340px;
    white-space: normal !important;
    line-height: 1.35;
    color: #c9d5e6;
}

.tag-low { color: #39d98a; font-weight: 700; }
.tag-med { color: #f5b942; font-weight: 700; }
.tag-high { color: #ff5e72; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def props_to_df(props):
    rows = []
    for p in props:
        secondary = p.matchup
        if p.opposing_pitcher:
            secondary = f"{p.matchup} • vs {p.opposing_pitcher}"

        rows.append({
            "Prop": p.prop,
            "Entity": p.entity_name,
            "Matchup": secondary,
            "Model %": round(p.model_prob * 100, 1),
            "Confidence": round(p.confidence, 1),
            "Stability": round(p.stability, 1),
            "Parlay Fit": round(p.parlay_fit, 1),
            "Opportunity": round(p.opportunity_score, 1),
            "Volatility": p.volatility,
            "Reason": p.reason,
            "Game ID": str(p.game_id),
            "Prop Family": p.prop_family,
            "Script Tag": p.script_tag,
        })
    return pd.DataFrame(rows)


def apply_filters(props, min_conf, min_opp, prop_type, game_filter, volatility_filter):
    filtered = []
    for p in props:
        if p.confidence < min_conf:
            continue
        if p.opportunity_score < min_opp:
            continue

        if prop_type != "All":
            if prop_type == "Hitter" and p.prop_family != "HITTER_FADE":
                continue
            if prop_type == "Pitcher" and p.prop_family not in {
                "PITCHER_OUTS", "PITCHER_STRIKEOUTS", "PITCHER_HITS_ALLOWED", "PITCHER_WALKS_ALLOWED"
            }:
                continue
            if prop_type == "Team" and p.prop_family not in {"TEAM_TOTAL_UNDER", "FIRST5_TEAM_TOTAL_UNDER"}:
                continue

        if game_filter != "All" and str(p.game_id) != game_filter:
            continue

        if volatility_filter != "All" and p.volatility != volatility_filter:
            continue

        filtered.append(p)

    return filtered


def render_metric_card(label, value, sub, col):
    with col:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            <div class="card-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)


def render_empty(message="No props match the current filter settings."):
    st.markdown(f'<div class="panel-shell">{message}</div>', unsafe_allow_html=True)


def score_pill(value, kind="green"):
    cls = {
        "blue": "pill-blue",
        "purple": "pill-purple",
        "green": "pill-green",
        "yellow": "pill-yellow",
        "red": "pill-red",
    }.get(kind, "pill-green")
    return f'<span class="metric-pill {cls}">{value}</span>'


def volatility_tag(value):
    value = str(value).upper()
    if value == "LOW":
        return '<span class="tag-low">LOW</span>'
    if value == "MED":
        return '<span class="tag-med">MED</span>'
    return '<span class="tag-high">HIGH</span>'


def render_dark_table(df: pd.DataFrame, columns=None, max_rows=12):
    if df.empty:
        render_empty()
        return

    table_df = df.copy()
    if columns:
        existing = [c for c in columns if c in table_df.columns]
        table_df = table_df[existing]

    table_df = table_df.head(max_rows)

    html = ['<div class="dark-table-wrap"><div class="dark-table-scroll"><table class="dark-table">']
    html.append("<thead><tr>")
    for col in table_df.columns:
        html.append(f"<th>{col}</th>")
    html.append("</tr></thead><tbody>")

    for _, row in table_df.iterrows():
        html.append("<tr>")
        for col in table_df.columns:
            val = row[col]

            if col == "Confidence":
                cell = score_pill(f"{float(val):.1f}", "blue")
            elif col == "Stability":
                cell = score_pill(f"{float(val):.1f}", "purple")
            elif col == "Parlay Fit":
                cell = score_pill(f"{float(val):.1f}", "green")
            elif col == "Opportunity":
                cell = score_pill(f"{float(val):.1f}", "green")
            elif col == "Model %":
                cell = score_pill(f"{float(val):.1f}", "yellow")
            elif col == "Volatility":
                cell = volatility_tag(val)
            elif col == "Reason":
                cell = f'<span class="reason-cell">{val}</span>'
            else:
                cell = str(val)

            html.append(f"<td>{cell}</td>")
        html.append("</tr>")

    html.append("</tbody></table></div></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def prop_detail_panel(prop):
    if not prop:
        render_empty("No prop details available.")
        return

    reason_text = prop.reason if prop.reason else "No additional tags available."

    chips = f"""
    <div class="detail-chip-wrap">
        <span class="detail-chip">{prop.prop_family}</span>
        <span class="detail-chip">Conf {prop.confidence:.1f}</span>
        <span class="detail-chip">Opp {prop.opportunity_score:.1f}</span>
        <span class="detail-chip">Stable {prop.stability:.1f}</span>
        <span class="detail-chip">Parlay {prop.parlay_fit:.1f}</span>
        <span class="detail-chip">{prop.volatility}</span>
        <span class="detail-chip">{prop.script_tag}</span>
    </div>
    """

    st.markdown(f"""
    <div class="panel-shell">
        <div class="panel-title">{prop.prop}</div>
        <div class="panel-subtitle">{prop.entity_name} • {prop.matchup}</div>
        <div style="font-size:0.95rem; line-height:1.55;">{reason_text}</div>
        {chips}
    </div>
    """, unsafe_allow_html=True)


def parlay_cards(parlays, limit=4):
    if not parlays:
        render_empty("No parlay builds available from the current run.")
        return

    st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
    for parlay in parlays[:limit]:
        legs = "<br>".join([f"{leg.prop} — {leg.entity_name}" for leg in parlay.legs])
        leg_count = len(parlay.legs)

        st.markdown(f"""
        <div class="parlay-card">
            <div style="font-weight:700; margin-bottom:0.35rem;">{legs}</div>
            <hr>
            <div class="parlay-score">Score {round(parlay.final_score, 1)}</div>
            <div class="small-note">
                {leg_count}-Leg • Est. Hit % {parlay.estimated_hit_rate:.1f}% • Avg Conf {parlay.avg_conf:.1f} • Floor {parlay.min_conf:.1f} • Structure {parlay.structural:.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# SESSION STATE / MODEL RUN
# =========================================================
if "model_results" not in st.session_state:
    with st.spinner("Running model..."):
        st.session_state["model_results"] = run_model()
        st.session_state["last_run"] = datetime.now()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">MLB Fade Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Data. Models. Edge.</div>', unsafe_allow_html=True)

    st.markdown('<div class="nav-group-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Dashboard", "Top Bets", "Best Opportunity", "Most Stable", "Parlays", "Fade Zone"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="nav-group-label">Filters</div>', unsafe_allow_html=True)
    min_conf = st.slider("Min Confidence", 40, 95, 65, 1)
    min_opp = st.slider("Min Opportunity", 40, 100, 60, 1)
    prop_type = st.selectbox("Prop Type", ["All", "Hitter", "Pitcher", "Team"])
    volatility_filter = st.selectbox("Volatility", ["All", "LOW", "MED", "HIGH"])

    all_props_initial = st.session_state["model_results"]["all_props"]
    game_choices = ["All"] + sorted(
        list({str(getattr(p, "game_id", "")) for p in all_props_initial if getattr(p, "game_id", "")})
    )
    game_filter = st.selectbox("Game", game_choices)

# =========================================================
# HEADER
# =========================================================
results = st.session_state["model_results"]

left_head, mid_head, right_head = st.columns([3.2, 1.2, 1.4])

with left_head:
    meta = results.get("meta", {})
    games_found = meta.get("games_found", 0)
    source_mode = meta.get("source_mode", "unknown")
    source_label = "Confirmed Lineups" if source_mode == "confirmed_lineup" else "Preliminary Pool"

    st.markdown(f"""
    <div class="page-title">{page}</div>
    <div class="page-subtitle">Daily model run and top opportunities</div>
    <div class="meta-badges">
      <span class="meta-badge">{games_found} Games</span>
      <span class="meta-badge">{source_label}</span>
    </div>
    """, unsafe_allow_html=True)

with mid_head:
    run_clicked = st.button("Run Model")

with right_head:
    last_run = st.session_state["last_run"].strftime("%b %d, %Y %I:%M %p")
    st.markdown(f'<div class="last-run">Last Run: {last_run}</div>', unsafe_allow_html=True)

if run_clicked:
    with st.spinner("Running model..."):
        st.session_state["model_results"] = run_model()
        st.session_state["last_run"] = datetime.now()
        results = st.session_state["model_results"]

# =========================================================
# FILTERED DATA
# =========================================================
filtered_top = apply_filters(results["top_bets"], min_conf, min_opp, prop_type, game_filter, volatility_filter)
filtered_opportunity = apply_filters(results["best_opportunity"], min_conf, min_opp, prop_type, game_filter, volatility_filter)
filtered_stable = apply_filters(results["most_stable"], min_conf, min_opp, prop_type, game_filter, volatility_filter)
filtered_risk = apply_filters(results["high_risk"], min_conf, min_opp, prop_type, game_filter, volatility_filter)
filtered_all = apply_filters(results["all_props"], min_conf, min_opp, prop_type, game_filter, volatility_filter)

top_df = props_to_df(filtered_top)
opp_df = props_to_df(filtered_opportunity)
stable_df = props_to_df(filtered_stable)
risk_df = props_to_df(filtered_risk)
all_df = props_to_df(filtered_all)

# =========================================================
# KPI ROW
# =========================================================
metrics = results.get("metrics", {})
k1, k2, k3, k4, k5 = st.columns(5)

render_metric_card("Top Opportunities", str(metrics.get("top", 0)), "Confidence 70%+", k1)
render_metric_card("Avg Opportunity", f"{metrics.get('avg_opportunity', 0):.1f}", "Across all props", k2)
render_metric_card("Best Opportunity", f"{metrics.get('best_opportunity', 0):.1f}", "Today's strongest score", k3)
render_metric_card("Avg Confidence", f"{metrics.get('avg_conf', 0):.1f}%", "Across all props", k4)
render_metric_card("Total Props", f"{metrics.get('total', 0)}", "Analyzed today", k5)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":
    main_left, main_right = st.columns([2.25, 1.05], gap="large")

    with main_left:
        st.markdown('<div class="panel-title">Top Bets Today</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Highest-confidence props after filters</div>', unsafe_allow_html=True)
        render_dark_table(
            top_df,
            columns=["Prop", "Entity", "Matchup", "Model %", "Confidence", "Stability", "Parlay Fit", "Opportunity"],
            max_rows=12
        )

    with main_right:
        st.markdown('<div class="panel-title">Top Parlay Ideas</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Cleaner 2-leg and 3-leg structures</div>', unsafe_allow_html=True)
        parlay_cards(results.get("parlays", []), limit=4)

    st.markdown("<br>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns([1.3, 1.1, 1.1], gap="large")

    with b1:
        st.markdown('<div class="panel-title">Best Bet Detail</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Top-ranked prop with quick reasoning</div>', unsafe_allow_html=True)
        prop_detail_panel(filtered_top[0] if filtered_top else None)

    with b2:
        st.markdown('<div class="panel-title">Best Opportunity Props</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Strongest raw opportunity scores</div>', unsafe_allow_html=True)
        render_dark_table(
            opp_df,
            columns=["Prop", "Entity", "Matchup", "Opportunity", "Confidence", "Reason"],
            max_rows=8
        )

    with b3:
        st.markdown('<div class="panel-title">Most Stable Plays</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Lower-volatility support plays</div>', unsafe_allow_html=True)
        render_dark_table(
            stable_df,
            columns=["Prop", "Entity", "Matchup", "Stability", "Confidence", "Reason"],
            max_rows=8
        )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.25, 1.25], gap="large")
    summary_tables = results.get("summary_tables", {})

    with c1:
        st.markdown('<div class="panel-title">Game Breakdown</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Where the model is finding the most playable props</div>', unsafe_allow_html=True)
        by_game = summary_tables.get("by_game", pd.DataFrame())
        if by_game.empty:
            render_empty("No game summary available.")
        else:
            display = by_game.copy()
            for col in ["Avg_Confidence", "Avg_Opportunity", "Best_Prop"]:
                if col in display.columns:
                    display[col] = display[col].round(1)
            st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
            st.dataframe(display, use_container_width=True, height=300)
            st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel-title">Tier Buckets</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Quick grouping of the board into usable tiers</div>', unsafe_allow_html=True)
        tiered = summary_tables.get("tiered", pd.DataFrame())
        if tiered.empty:
            render_empty("No tier summary available.")
        else:
            display = tiered.copy()
            for col in ["Avg_Confidence", "Avg_Opportunity"]:
                if col in display.columns:
                    display[col] = display[col].round(1)
            st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
            st.dataframe(display, use_container_width=True, height=300)
            st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TOP BETS
# =========================================================
elif page == "Top Bets":
    st.markdown('<div class="panel-title">Top Bets Board</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Full board of strongest filtered plays</div>', unsafe_allow_html=True)

    render_dark_table(
        top_df,
        columns=["Prop", "Entity", "Matchup", "Model %", "Confidence", "Stability", "Parlay Fit", "Opportunity", "Volatility", "Reason"],
        max_rows=20
    )

# =========================================================
# BEST OPPORTUNITY
# =========================================================
elif page == "Best Opportunity":
    left, right = st.columns([1.95, 1.05], gap="large")

    with left:
        st.markdown('<div class="panel-title">Best Opportunity Board</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Sorted by opportunity score first</div>', unsafe_allow_html=True)
        render_dark_table(
            opp_df,
            columns=["Prop", "Entity", "Matchup", "Opportunity", "Confidence", "Stability", "Parlay Fit", "Reason"],
            max_rows=20
        )

    with right:
        st.markdown('<div class="panel-title">Lead Opportunity Detail</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Top raw edge currently on the board</div>', unsafe_allow_html=True)
        prop_detail_panel(filtered_opportunity[0] if filtered_opportunity else None)

# =========================================================
# MOST STABLE
# =========================================================
elif page == "Most Stable":
    left, right = st.columns([1.95, 1.05], gap="large")

    with left:
        st.markdown('<div class="panel-title">Most Stable Plays</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Sorted by stability with confidence support</div>', unsafe_allow_html=True)
        render_dark_table(
            stable_df,
            columns=["Prop", "Entity", "Matchup", "Stability", "Confidence", "Opportunity", "Parlay Fit", "Reason"],
            max_rows=20
        )

    with right:
        st.markdown('<div class="panel-title">Lead Stability Detail</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Best low-volatility profile</div>', unsafe_allow_html=True)
        prop_detail_panel(filtered_stable[0] if filtered_stable else None)

# =========================================================
# PARLAYS
# =========================================================
elif page == "Parlays":
    left, right = st.columns([1.2, 1.1], gap="large")

    with left:
        st.markdown('<div class="panel-title">Recommended Parlays</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Model-blended combinations with estimated hit rate</div>', unsafe_allow_html=True)
        parlay_cards(results.get("parlays", []), limit=6)

    with right:
        st.markdown('<div class="panel-title">Best Parlay Candidates</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Strong parlay-fit plays from the filtered board</div>', unsafe_allow_html=True)

        if all_df.empty:
            render_empty("No parlay candidates available.")
        else:
            parlay_df = all_df.sort_values(["Parlay Fit", "Confidence"], ascending=False).head(12)
            render_dark_table(
                parlay_df,
                columns=["Prop", "Entity", "Matchup", "Parlay Fit", "Confidence", "Stability", "Opportunity"],
                max_rows=12
            )

# =========================================================
# FADE ZONE
# =========================================================
elif page == "Fade Zone":
    left, right = st.columns([1.95, 1.05], gap="large")

    with left:
        st.markdown('<div class="panel-title">Fade Zone</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">High-risk or weaker-structure plays that need caution</div>', unsafe_allow_html=True)
        render_dark_table(
            risk_df,
            columns=["Prop", "Entity", "Matchup", "Volatility", "Confidence", "Opportunity", "Reason"],
            max_rows=20
        )

    with right:
        st.markdown('<div class="panel-title">Why It Lands Here</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">The highest-risk filtered play on the board</div>', unsafe_allow_html=True)
        prop_detail_panel(filtered_risk[0] if filtered_risk else None)

st.markdown(
    '<div class="footer-note">Model uses probability, stability, volatility-adjusted confidence, internal opportunity scoring, and blended parlay estimation.</div>',
    unsafe_allow_html=True,
)
