import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from pathlib import Path

from src.predict import predict_risk, load_model
from src.actions import get_conservation_actions




# =========================================================
# BUILT-IN CONSERVATION PRIORITY ENGINE
# =========================================================
def _safe_number(value, default=0.0):
    try:
        number = float(value)
        if pd.isna(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def assess_conservation_priority(input_data, high_risk_probability, risk_level):
    """
    Transparent prototype score for conservation planning.
    This is a decision-support heuristic, not a validated ecological index.
    """
    high_probability = max(0.0, min(_safe_number(high_risk_probability), 100.0))
    prior_loss = max(0.0, _safe_number(input_data.get("prior_loss_pct")))
    fire_count = max(0.0, _safe_number(input_data.get("fire_count_last_year")))
    tree_cover = max(0.0, min(_safe_number(input_data.get("tree_cover_pct"), 100.0), 100.0))
    road_distance = max(0.0, _safe_number(input_data.get("dist_to_road_km"), 30.0))
    settlement_distance = max(0.0, _safe_number(input_data.get("dist_to_settlement_km"), 35.0))

    components = {
        "Model High-risk probability": high_probability,
        "Previous forest loss": min(prior_loss / 40.0, 1.0) * 100.0,
        "Fire incidents": min(fire_count / 5.0, 1.0) * 100.0,
        "Tree-cover deficit": 100.0 - tree_cover,
        "Road / settlement proximity": (
            (1.0 - min(road_distance / 30.0, 1.0)) * 50.0
            + (1.0 - min(settlement_distance / 35.0, 1.0)) * 50.0
        )
    }

    weights = {
        "Model High-risk probability": 0.40,
        "Previous forest loss": 0.20,
        "Fire incidents": 0.15,
        "Tree-cover deficit": 0.15,
        "Road / settlement proximity": 0.10
    }

    weighted_points = {
        name: components[name] * weights[name]
        for name in components
    }
    score = round(sum(weighted_points.values()), 1)

    if score >= 75:
        level = "Critical"
        action = "Prioritize prompt field verification; review recent forest-loss evidence and coordinate an inspection."
    elif score >= 55:
        level = "High"
        action = "Schedule an early field assessment and review satellite imagery for recent changes."
    elif score >= 35:
        level = "Moderate"
        action = "Include this location in the next monitoring cycle and investigate the strongest contributing indicators."
    else:
        level = "Routine"
        action = "Continue routine monitoring and reassess when new environmental or land-use data becomes available."

    reasons = []
    if high_probability >= 50:
        reasons.append(f"Elevated model High-risk probability ({high_probability:.1f}%)")
    if prior_loss >= 10:
        reasons.append(f"Previous forest loss is notable ({prior_loss:.1f}%)")
    if fire_count >= 2:
        reasons.append(f"Multiple fire incidents recorded ({fire_count:.0f})")
    if tree_cover <= 40:
        reasons.append(f"Tree cover is relatively low ({tree_cover:.1f}%)")
    if components["Road / settlement proximity"] >= 60:
        reasons.append("Location is relatively close to roads and/or settlements")
    if not reasons:
        reasons.append("No individual indicator crossed the prototype explanation thresholds")

    return {
        "score": score,
        "level": level,
        "components": components,
        "weighted_points": weighted_points,
        "reasons": reasons,
        "recommended_action": action,
        "risk_level": risk_level
    }


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="ForestGuard | Green Lanterns",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# DESIGN SYSTEM
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --forest: #173f35;
    --forest-dark: #102d27;
    --leaf: #5b956e;
    --sage: #dce9dc;
    --cream: #f5f7f1;
    --ink: #1c3028;
    --muted: #6d7e73;
    --line: #e2e9df;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: linear-gradient(180deg, #f7f9f4 0%, #f2f6ef 100%);
    color: var(--ink);
}
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
[data-testid="stSidebar"] {
    background: #173f35;
    border-right: 1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] * {
    color: #edf5eb !important;
}
[data-testid="stSidebar"] .stCaption {
    color: #c0d4c3 !important;
}
.hero {
    background: linear-gradient(115deg, #12382f 0%, #1f5945 58%, #477b55 100%);
    border-radius: 24px;
    padding: 2rem 2.2rem;
    color: #f7fbf5;
    position: relative;
    overflow: hidden;
    box-shadow: 0 14px 35px rgba(23,63,53,.14);
    margin-bottom: 1.25rem;
}
.hero:after {
    content: "✳";
    position: absolute;
    right: 3.2rem;
    top: -2.4rem;
    font-size: 12rem;
    line-height: 1;
    color: rgba(231,245,225,.08);
}
.hero-kicker {
    text-transform: uppercase;
    letter-spacing: .16em;
    font-size: .72rem;
    font-weight: 700;
    color: #c7dfc5;
    margin-bottom: .8rem;
}
.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(2.2rem, 4vw, 3.5rem);
    line-height: 1.05;
    color: #ffffff;
    margin: 0 0 .7rem 0;
    font-weight: 800;
}
.hero p {
    color: #e0eee0;
    font-size: 1.03rem;
    max-width: 720px;
    margin: 0;
}
.hero-meta {
    margin-top: 1.25rem;
    color: #c8dfc8;
    font-size: .82rem;
    font-weight: 600;
}
.section-heading {
    background: linear-gradient(90deg, #e6f0e5, #f5f8f2);
    border-left: 5px solid #397c52;
    border-radius: 9px;
    padding: .62rem .9rem;
    font-family: 'Manrope', sans-serif;
    color: #173f35;
    font-size: 1.35rem;
    font-weight: 800;
    margin: 1.15rem 0 .25rem 0;
}
.section-sub {
    color: #718076;
    margin-bottom: 1rem;
    font-size: .92rem;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e4ebe1;
    border-radius: 18px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 5px 18px rgba(27,61,43,.045);
    min-height: 112px;
}
.metric-label {
    color: #718076;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
}
.metric-value {
    color: #173f35;
    font-family: 'Manrope', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    margin-top: .28rem;
}
.metric-note {
    color: #87948b;
    font-size: .76rem;
    margin-top: .1rem;
}

.form-section-title {
    background: linear-gradient(90deg, #e4efe4 0%, #f3f7f0 100%);
    border: 1px solid #c8ddca;
    border-left: 5px solid #3d8056;
    border-radius: 11px;
    padding: .75rem 1rem;
    margin: .8rem 0 .9rem 0;
    color: #000000 !important;
    font-family: 'Manrope', sans-serif;
    font-size: 1.03rem;
    font-weight: 800;

.form-section-caption {
    color: #000000 !important;
    font-size: .82rem;
    font-weight: 500;
    margin-top: .15rem;
}

.panel {
    background: #ffffff;
    border: 1px solid #e4ebe1;
    border-radius: 18px;
    padding: 1.15rem 1.25rem;
    box-shadow: 0 5px 18px rgba(27,61,43,.04);
    margin-bottom: .8rem;
}
.panel-title {
    color: #173f35;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 1rem;
    margin-bottom: .25rem;
}
.panel-copy {
    color: #728076;
    font-size: .86rem;
}
.risk-banner {
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    color: #173f35;
    border: 1px solid transparent;
    margin: .5rem 0 1rem 0;
}
.risk-high { background: #fff0ed; border-color: #f5c9c0; }
.risk-medium { background: #fff7e8; border-color: #f0dfb7; }
.risk-low { background: #eaf5ec; border-color: #c9e2ce; }
.risk-label {
    font-size: .75rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    font-weight: 800;
    opacity: .7;
}
.risk-value {
    font-family: 'Manrope', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin-top: .15rem;
}
.action-item {
    background: #f5f8f3;
    border: 1px solid #e4ebe1;
    border-radius: 12px;
    padding: .75rem .9rem;
    margin: .45rem 0;
    color: #304b3c;
    font-size: .9rem;
}
.small-note {
    color: #718076;
    font-size: .78rem;
}
div.stButton > button, div.stFormSubmitButton > button {
    background: #1d5a43;
    color: white;
    border: 0;
    border-radius: 11px;
    padding: .65rem 1rem;
    font-weight: 700;
}
div.stButton > button:hover, div.stFormSubmitButton > button:hover {
    background: #143f30;
    color: white;
    border: 0;
}
.stTabs [data-baseweb="tab-list"] {
    gap: .4rem;
    background: #eaf0e7;
    padding: .35rem;
    border-radius: 13px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: #496052;
    font-weight: 700;
    padding: .55rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #173f35 !important;
}
div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input {
    border-radius: 10px;
}
div[data-testid="stSelectbox"] > div {
    border-radius: 10px;
}
hr { border-color: #e3eae0; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, note=""):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
        </div>""",
        unsafe_allow_html=True
    )


def section_heading(title, subtitle=None):
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)



def visible_subheading(title, subtitle=None):
    subtitle_html = (
        f'<div class="form-section-caption">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f'<div class="form-section-title">{title}{subtitle_html}</div>',
        unsafe_allow_html=True
    )


def risk_class_name(risk):
    return {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}.get(risk, "risk-low")


# =========================================================
# MODEL LOADING
# =========================================================
try:
    model, encoder, features, threshold = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.info("From the project root, run: `python src/train.py`")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🌿 ForestGuard")
    st.caption("ENVIRONMENTAL INTELLIGENCE")
    st.markdown("---")
    st.markdown("### Project")
    st.markdown("**Team:** Green Lanterns")
    st.markdown("**Team ID:** U3M27AE4")
    st.markdown("**Selected model:** Logistic Regression")
    st.markdown("**Task:** Low / Medium / High risk")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("① Enter plot indicators")
    st.markdown("② Generate a risk assessment")
    st.markdown("③ Review location and map")
    st.markdown("④ Read conservation actions")
    st.markdown("⑤ Export the assessment")
    st.markdown("---")
    st.caption(
        "ForestGuard is an experimental decision-support prototype. "
        "Predictions require independent environmental and field verification."
    )


# =========================================================
# HERO / TOP METRICS
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-kicker">AI-powered conservation decision support</div>
    <h1>ForestGuard</h1>
    <p>Understand potential deforestation risk. Prioritize monitoring.
    Support informed forest conservation decisions.</p>
    <div class="hero-meta">GREEN LANTERNS &nbsp; • &nbsp; AI SANSAD &nbsp; • &nbsp; TEAM U3M27AE4</div>
</div>
""", unsafe_allow_html=True)

top1, top2, top3, top4 = st.columns(4)
with top1:
    metric_card("Dataset records", "1,000", "Organizer-provided dataset")
with top2:
    metric_card("Test accuracy", "88%", "Held-out test set")
with top3:
    metric_card("Test Macro F1", "0.7894", "All three classes")
with top4:
    metric_card("Risk categories", "03", "Low · Medium · High")

st.markdown(
    '<div class="small-note">Model metrics reflect evaluation on the supplied dataset; '
    'they are not a guarantee of real-world deforestation outcomes.</div>',
    unsafe_allow_html=True
)

# =========================================================
# MAIN NAVIGATION
# =========================================================
tab_assess, tab_batch, tab_map, tab_priority, tab_insights, tab_about = st.tabs([
    "🌱  Assess a region",
    "🗂️  Multi-region analysis",
    "🗺️  Risk map",
    "🎯  Priority engine",
    "📈  Model insights",
    "ℹ️  About ForestGuard"
])


# =========================================================
# TAB 1: SINGLE REGION ASSESSMENT
# =========================================================
with tab_assess:
    section_heading(
        "Regional risk assessment",
        "Enter the environmental and geographical indicators for the plot you want to assess."
    )

    with st.form("forestguard_assessment_form"):
        visible_subheading("📍 Location details", "Identify the forest, reserve or geographical plot being assessed.")
        loc1, loc2, loc3 = st.columns([1.3, 1, 1])
        with loc1:
            region_name = st.text_input("Region / forest name", value="My Forest Region")
        with loc2:
            latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0, format="%.6f")
        with loc3:
            longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0, format="%.6f")

        st.markdown("---")
        visible_subheading("🧾 Environmental assessment inputs", "Complete all three categories using the best available regional data.")
        geo_col, human_col, env_col = st.columns(3, gap="large")

        with geo_col:
            visible_subheading("🧭 Geography & terrain", "Distance, elevation and physical landscape characteristics.")
            road = st.number_input("Distance from road (km)", min_value=0.15, max_value=29.31, value=5.0)
            settlement = st.number_input("Distance from settlement (km)", min_value=0.07, max_value=34.87, value=8.0)
            forest_edge = st.number_input("Distance from forest edge (km)", min_value=0.03, max_value=23.12, value=3.0)
            slope = st.number_input("Slope (degrees)", min_value=0.3, max_value=55.0, value=12.0)
            elevation = st.number_input("Elevation (meters)", min_value=0.0, max_value=1546.0, value=500.0)

        with human_col:
            visible_subheading("🏘️ Human activity indicators", "Human presence and land-use pressure around the forest.")
            population = st.number_input("Population density", min_value=1.2, max_value=889.5, value=56.0)
            agriculture = st.number_input("Agricultural price index", min_value=1.2, max_value=889.5, value=100.0)
            fires = st.number_input("Fire incidents last year", min_value=0, max_value=5, value=1)
            protected = st.selectbox(
                "Protected area?",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with env_col:
            visible_subheading("🌳 Environmental indicators", "Forest condition, rainfall and previous disturbance indicators.")
            rainfall = st.number_input("Annual rainfall (mm)", min_value=58.0, max_value=2686.0, value=1000.0)
            tree_cover = st.number_input("Tree cover (%)", min_value=10.2, max_value=99.8, value=70.0)
            prior_loss = st.number_input("Previous forest loss (%)", min_value=0.02, max_value=47.35, value=8.0)

        st.markdown("---")
        submitted = st.form_submit_button("🌿 Generate risk assessment", use_container_width=True)

    if submitted:
        region_data = {
            "dist_to_road_km": road,
            "dist_to_settlement_km": settlement,
            "dist_to_forest_edge_km": forest_edge,
            "slope_deg": slope,
            "elevation_m": elevation,
            "pop_density": population,
            "agri_price_index": agriculture,
            "rainfall_mm": rainfall,
            "fire_count_last_year": fires,
            "protected_area": protected,
            "tree_cover_pct": tree_cover,
            "prior_loss_pct": prior_loss
        }

        try:
            result = predict_risk(region_data)
            recommendations = get_conservation_actions(result["risk_level"], region_data)
            priority = assess_conservation_priority(
                region_data,
                result["high_risk_probability"],
                result["risk_level"]
            )
            st.session_state["fg_latest"] = {
                "region_name": region_name,
                "latitude": latitude,
                "longitude": longitude,
                "region_data": region_data,
                "result": result,
                "recommendations": recommendations,
                "priority": priority
            }
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    latest = st.session_state.get("fg_latest")

    if latest:
        result = latest["result"]
        risk = result["risk_level"]
        probabilities = result["probabilities"]
        recommendations = latest["recommendations"]
        region_data = latest["region_data"]

        section_heading("Assessment result", f"Region: {latest['region_name'] or 'Unnamed region'}")
        st.markdown(
            f"""<div class="risk-banner {risk_class_name(risk)}">
            <div class="risk-label">Predicted deforestation risk</div>
            <div class="risk-value">{risk.upper()} RISK</div>
            <div>High-risk probability: <b>{result['high_risk_probability']:.2f}%</b>
            &nbsp; · &nbsp; Decision threshold: {result['threshold']:.2f}</div>
            </div>""",
            unsafe_allow_html=True
        )

        prob_cols = st.columns(3)
        color_map = {"High": "#c65e50", "Medium": "#d6a447", "Low": "#5b956e"}
        for col, label in zip(prob_cols, ["High", "Medium", "Low"]):
            with col:
                metric_card(f"{label} risk probability", f"{probabilities.get(label, 0):.2f}%", "Model-estimated class probability")

        section_heading("Probability distribution", "Compare the model's estimated probability across all three risk categories.")
        prob_df = pd.DataFrame({
            "Risk category": list(probabilities.keys()),
            "Probability (%)": list(probabilities.values())
        }).set_index("Risk category")
        st.bar_chart(prob_df, color="#5b956e", height=220)

        section_heading("Satellite location view", "The marker shows the coordinates entered above; it is not an automatically detected forest-loss event.")
        if not latest["region_name"].strip():
            st.warning("Add a region name to identify this assessment.")
        elif latest["latitude"] == 0.0 and latest["longitude"] == 0.0:
            st.info("Enter the actual latitude and longitude in the form to display the location map.")
        else:
            marker_color = {"High": "red", "Medium": "orange", "Low": "green"}.get(risk, "blue")
            satellite_map = folium.Map(
                location=[latest["latitude"], latest["longitude"]],
                zoom_start=11,
                tiles=None,
                control_scale=True
            )
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Tiles © Esri",
                name="Satellite imagery",
                overlay=False,
                control=True
            ).add_to(satellite_map)
            folium.Marker(
                location=[latest["latitude"], latest["longitude"]],
                popup=(
                    f"<b>{latest['region_name']}</b><br>"
                    f"Predicted risk: {risk}<br>"
                    f"Latitude: {latest['latitude']}<br>"
                    f"Longitude: {latest['longitude']}"
                ),
                tooltip=f"{latest['region_name']} · {risk} risk",
                icon=folium.Icon(color=marker_color, icon="leaf", prefix="fa")
            ).add_to(satellite_map)
            folium.LayerControl().add_to(satellite_map)
            st_folium(
                satellite_map,
                width=None,
                height=470,
                use_container_width=True,
                returned_objects=[]
            )
            st.caption("Basemap: Esri World Imagery. The map marker represents the manually entered assessment location.")

        section_heading("Conservation action plan", "Suggested follow-up actions based on the predicted risk and entered indicators.")
        priority = recommendations["priority"]
        priority_color = {"URGENT": "#c65e50", "MODERATE": "#d6a447", "ROUTINE": "#5b956e"}.get(priority, "#5b956e")
        st.markdown(
            f"""<div class="panel">
            <div class="panel-title">Conservation priority</div>
            <div style="font-size:1.35rem;font-weight:800;color:{priority_color};">{priority}</div>
            </div>""",
            unsafe_allow_html=True
        )
        action_col, alert_col = st.columns(2, gap="large")
        with action_col:
            st.markdown("**Recommended actions**")
            for action in recommendations["actions"]:
                st.markdown(f'<div class="action-item">✓ &nbsp; {action}</div>', unsafe_allow_html=True)
        with alert_col:
            st.markdown("**Environmental warning signals**")
            if recommendations["alerts"]:
                for alert in recommendations["alerts"]:
                    st.warning(alert)
            else:
                st.success("No rule-based warning signals were triggered by the entered indicators.")

        st.info(
            "Verification required: these are preliminary model estimates and rule-based suggestions. "
            "Confirm potential threats using reliable satellite-change analysis and field inspection."
        )

        section_heading("Assessment data", "The exact inputs used for this prediction.")
        summary_data = dict(region_data)
        summary_data.update({
            "region_name": latest["region_name"],
            "latitude": latest["latitude"],
            "longitude": latest["longitude"]
        })
        summary_df = pd.DataFrame(list(summary_data.items()), columns=["Indicator", "Value"])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        report = {
            "Region Name": latest["region_name"],
            "Risk Category": risk,
            "High Risk Probability (%)": result["high_risk_probability"],
            "Conservation Priority": priority,
            "Latitude": latest["latitude"],
            "Longitude": latest["longitude"],
            "Recommended Actions": " | ".join(recommendations["actions"]),
            "Environmental Alerts": " | ".join(recommendations["alerts"])
        }
        report.update(region_data)
        report_df = pd.DataFrame([report])
        st.download_button(
            "⬇️ Download conservation assessment (CSV)",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="forestguard_assessment.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">Ready to assess a forest region?</div>
            <div class="panel-copy">Complete the input fields above and select “Generate risk assessment”.
            Your prediction, probability breakdown, map, and suggested conservation actions will appear here.</div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# TAB 2: MULTI-REGION ANALYSIS
# =========================================================
with tab_batch:
    section_heading(
        "Multi-region analysis",
        "Upload a CSV containing the 12 model input columns to assess multiple plots. Add latitude and longitude columns to display the predictions on the Risk map tab."
    )
    st.markdown("""
    <div class="panel">
      <div class="panel-title">Expected input columns</div>
      <div class="panel-copy">The uploaded CSV must contain the same 12 feature columns used during model training.
      Additional columns are retained in the exported results. For mapping, include <b>latitude</b> and <b>longitude</b> columns (or <b>lat</b> and <b>lon</b>).</div>
    </div>
    """, unsafe_allow_html=True)

    st.code(", ".join(features), language="text")
    uploaded = st.file_uploader("Upload regional dataset (CSV)", type=["csv"], key="forestguard_batch_csv")

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing_features = [f for f in features if f not in batch_df.columns]
            if missing_features:
                st.error("Missing required feature columns: " + ", ".join(missing_features))
            elif batch_df.empty:
                st.warning("The uploaded CSV contains no records.")
            else:
                st.write(f"Loaded **{len(batch_df):,}** records.")
                if st.button("Run multi-region assessment", type="primary", use_container_width=True):
                    predictions = []
                    progress = st.progress(0)
                    status = st.empty()
                    for i, (_, row) in enumerate(batch_df.iterrows(), start=1):
                        input_values = {f: pd.to_numeric(pd.Series([row[f]]), errors="coerce").iloc[0] for f in features}
                        prediction = predict_risk(input_values)
                        priority = assess_conservation_priority(
                            input_values,
                            prediction["high_risk_probability"],
                            prediction["risk_level"]
                        )
                        predictions.append({
                            "Risk Category": prediction["risk_level"],
                            "High Risk Probability (%)": prediction["high_risk_probability"],
                            "Low Probability (%)": prediction["probabilities"].get("Low", 0),
                            "Medium Probability (%)": prediction["probabilities"].get("Medium", 0),
                            "High Probability (%)": prediction["probabilities"].get("High", 0),
                            "Conservation Priority Score": priority["score"],
                            "Priority Level": priority["level"],
                            "Priority Reasons": "; ".join(priority["reasons"]),
                            "Recommended Action": priority["recommended_action"]
                        })
                        progress.progress(i / len(batch_df))
                        status.caption(f"Assessing plot {i} of {len(batch_df)}")
                    output_df = pd.concat(
                        [batch_df.reset_index(drop=True), pd.DataFrame(predictions)],
                        axis=1
                    )
                    st.session_state["fg_batch_results"] = output_df
                    status.success("Multi-region assessment completed.")
                    progress.empty()

            batch_results = st.session_state.get("fg_batch_results")
            if batch_results is not None:
                st.markdown("---")
                section_heading("Batch assessment results", "Risk categories predicted for the uploaded records.")
                counts = batch_results["Risk Category"].value_counts()
                c1, c2, c3 = st.columns(3)
                with c1:
                    metric_card("High risk plots", f"{int(counts.get('High', 0)):,}")
                with c2:
                    metric_card("Medium risk plots", f"{int(counts.get('Medium', 0)):,}")
                with c3:
                    metric_card("Low risk plots", f"{int(counts.get('Low', 0)):,}")

                st.bar_chart(counts.rename_axis("Risk category"), color="#5b956e")
                st.dataframe(
                    batch_results.sort_values("High Risk Probability (%)", ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
                st.download_button(
                    "⬇️ Download multi-region results (CSV)",
                    data=batch_results.to_csv(index=False).encode("utf-8"),
                    file_name="forestguard_multi_region_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as error:
            st.error(f"Could not process the uploaded CSV: {error}")
    else:
        st.caption("No CSV uploaded yet. You can use the organizer-provided dataset/another compatible CSV.")



# =========================================================
# TAB 3: INTERACTIVE MULTI-REGION RISK MAP
# =========================================================
with tab_map:
    section_heading(
        "Forest risk map",
        "Explore assessed locations on an interactive satellite map. Marker colours represent model-predicted risk."
    )

    map_records = []

    # Include the latest single-region assessment, if it has usable coordinates.
    latest_for_map = st.session_state.get("fg_latest")
    if latest_for_map:
        lat = latest_for_map.get("latitude")
        lon = latest_for_map.get("longitude")
        if lat is not None and lon is not None and not (float(lat) == 0 and float(lon) == 0):
            map_records.append({
                "Region": latest_for_map.get("region_name") or "Single-region assessment",
                "Latitude": float(lat),
                "Longitude": float(lon),
                "Risk Category": latest_for_map["result"]["risk_level"],
                "High Risk Probability (%)": latest_for_map["result"]["high_risk_probability"],
                "Source": "Single-region assessment"
            })

    # Include batch results when the uploaded CSV contains coordinate columns.
    batch_for_map = st.session_state.get("fg_batch_results")
    if batch_for_map is not None and not batch_for_map.empty:
        lower_columns = {str(column).strip().lower(): column for column in batch_for_map.columns}
        lat_col = next((lower_columns[name] for name in ["latitude", "lat"] if name in lower_columns), None)
        lon_col = next((lower_columns[name] for name in ["longitude", "lon", "long"] if name in lower_columns), None)
        name_col = next((lower_columns[name] for name in ["region_name", "region", "plot_name", "plot_id"] if name in lower_columns), None)

        if lat_col and lon_col:
            for row_index, row in batch_for_map.iterrows():
                try:
                    lat_value = float(row[lat_col])
                    lon_value = float(row[lon_col])
                    if pd.notna(lat_value) and pd.notna(lon_value) and -90 <= lat_value <= 90 and -180 <= lon_value <= 180:
                        region_label = str(row[name_col]) if name_col and pd.notna(row[name_col]) else f"Plot {row_index + 1}"
                        map_records.append({
                            "Region": region_label,
                            "Latitude": lat_value,
                            "Longitude": lon_value,
                            "Risk Category": row["Risk Category"],
                            "High Risk Probability (%)": float(row["High Risk Probability (%)"]),
                            "Source": "Multi-region CSV"
                        })
                except (TypeError, ValueError, KeyError):
                    continue

    map_df = pd.DataFrame(map_records)

    if map_df.empty:
        st.info(
            "No mappable assessments yet. Assess one region using real coordinates, or upload a multi-region CSV "
            "that includes latitude and longitude columns in addition to the 12 model features."
        )
        st.markdown("**Coordinate columns accepted:** `latitude` and `longitude` (also accepts `lat` and `lon`).")
        template_columns = ["region_name", "latitude", "longitude"] + list(features)
        template_df = pd.DataFrame(columns=template_columns)
        st.download_button(
            "⬇️ Download CSV template with coordinate columns",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="forestguard_geographic_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        filter_col, info_col = st.columns([1, 2])
        with filter_col:
            risk_filter = st.selectbox(
                "Filter locations by risk",
                ["All", "High", "Medium", "Low"],
                key="forestguard_map_risk_filter"
            )
        with info_col:
            st.caption(
                f"{len(map_df)} mapped assessment(s). High / Medium / Low markers are shown in red / orange / green."
            )

        visible_map_df = map_df if risk_filter == "All" else map_df[map_df["Risk Category"] == risk_filter]

        if visible_map_df.empty:
            st.warning(f"No locations match the {risk_filter} risk filter.")
        else:
            center_lat = float(visible_map_df["Latitude"].mean())
            center_lon = float(visible_map_df["Longitude"].mean())

            risk_colours = {"High": "#d94841", "Medium": "#e6a23c", "Low": "#2e9d62"}
            risk_map = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=5 if len(visible_map_df) > 1 else 10,
                tiles=None,
                control_scale=True
            )
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Tiles © Esri",
                name="Satellite imagery",
                overlay=False,
                control=True
            ).add_to(risk_map)
            folium.TileLayer(
                "OpenStreetMap",
                name="Street map",
                overlay=False,
                control=True
            ).add_to(risk_map)

            for _, point in visible_map_df.iterrows():
                risk = str(point["Risk Category"])
                colour = risk_colours.get(risk, "#426b8a")
                popup_html = (
                    f"<b>{point['Region']}</b><br>"
                    f"Predicted risk: <b>{risk}</b><br>"
                    f"High-risk probability: {point['High Risk Probability (%)']:.2f}%<br>"
                    f"Latitude: {point['Latitude']:.5f}<br>"
                    f"Longitude: {point['Longitude']:.5f}<br>"
                    f"Source: {point['Source']}"
                )
                folium.CircleMarker(
                    location=[point["Latitude"], point["Longitude"]],
                    radius=9 if risk == "High" else 7,
                    color=colour,
                    weight=2,
                    fill=True,
                    fill_color=colour,
                    fill_opacity=0.85,
                    tooltip=f"{point['Region']} · {risk} risk",
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(risk_map)

            folium.LayerControl().add_to(risk_map)
            st_folium(
                risk_map,
                width=None,
                height=570,
                use_container_width=True,
                returned_objects=[]
            )

            st.markdown("**Mapped locations**")
            st.dataframe(
                visible_map_df.sort_values("High Risk Probability (%)", ascending=False),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Map interpretation: marker colours show model predictions at the supplied coordinates. "
                "They do not represent satellite-confirmed forest loss or official forest boundaries."
            )



# =========================================================
# TAB 4: CONSERVATION PRIORITY ENGINE
# =========================================================
with tab_priority:
    section_heading(
        "Conservation priority engine",
        "Turn model predictions into a transparent action queue. Higher scores indicate locations that may deserve earlier field assessment."
    )

    st.markdown("""
    <div class="panel">
      <div class="panel-title">How the priority score works</div>
      <div class="panel-copy">
        The score combines the model's estimated High-risk probability with previous forest loss,
        fire history, low tree cover and proximity to roads/settlements. It is a transparent
        planning heuristic for demonstration, not a scientifically validated ecological index.
      </div>
    </div>
    """, unsafe_allow_html=True)

    weight_cols = st.columns(5)
    weight_items = [
        ("Model High-risk probability", "40%"),
        ("Previous forest loss", "20%"),
        ("Fire incidents", "15%"),
        ("Tree-cover deficit", "15%"),
        ("Road / settlement proximity", "10%")
    ]
    for col, (label, value) in zip(weight_cols, weight_items):
        with col:
            metric_card(label, value, "Score contribution")

    st.caption(
        "Priority bands: Critical ≥ 75 · High ≥ 55 · Moderate ≥ 35 · Routine < 35. "
        "Weights and cut-offs are initial design choices and should be calibrated using expert feedback and field outcomes."
    )

    priority_records = []

    # Add the latest single-region assessment.
    latest_priority = st.session_state.get("fg_latest")
    if latest_priority:
        row = {
            "Region": latest_priority.get("region_name") or "Single-region assessment",
            "Risk Category": latest_priority["result"]["risk_level"],
            "High Risk Probability (%)": latest_priority["result"]["high_risk_probability"],
            "Conservation Priority Score": latest_priority["priority"]["score"],
            "Priority Level": latest_priority["priority"]["level"],
            "Priority Reasons": "; ".join(latest_priority["priority"]["reasons"]),
            "Recommended Action": latest_priority["priority"]["recommended_action"],
            "Source": "Single-region assessment"
        }
        row.update(latest_priority["region_data"])
        priority_records.append(row)

    # Add all uploaded multi-region predictions.
    batch_priority = st.session_state.get("fg_batch_results")
    if batch_priority is not None and not batch_priority.empty:
        for index, batch_row in batch_priority.iterrows():
            row = {
                "Region": str(
                    batch_row.get("region_name",
                    batch_row.get("region", batch_row.get("plot_id", f"Plot {index + 1}")))
                ),
                "Risk Category": batch_row.get("Risk Category", "Unknown"),
                "High Risk Probability (%)": batch_row.get("High Risk Probability (%)", 0),
                "Conservation Priority Score": batch_row.get("Conservation Priority Score", 0),
                "Priority Level": batch_row.get("Priority Level", "Not calculated"),
                "Priority Reasons": batch_row.get("Priority Reasons", ""),
                "Recommended Action": batch_row.get("Recommended Action", ""),
                "Source": "Multi-region CSV"
            }
            for feature_name in features:
                if feature_name in batch_row.index:
                    row[feature_name] = batch_row[feature_name]
            priority_records.append(row)

    if not priority_records:
        st.info(
            "No assessed regions yet. First assess one region in the 'Assess a region' tab "
            "or upload a CSV and run 'Multi-region analysis'."
        )
    else:
        priority_df = pd.DataFrame(priority_records)

        # Ensure scores are numeric and remove any malformed rows.
        priority_df["Conservation Priority Score"] = pd.to_numeric(
            priority_df["Conservation Priority Score"], errors="coerce"
        )
        priority_df = priority_df.dropna(subset=["Conservation Priority Score"])
        priority_df = priority_df.sort_values(
            "Conservation Priority Score", ascending=False
        ).reset_index(drop=True)
        priority_df.insert(0, "Rank", range(1, len(priority_df) + 1))

        # Summary metrics.
        critical_count = int((priority_df["Priority Level"] == "Critical").sum())
        high_count = int((priority_df["Priority Level"] == "High").sum())
        moderate_count = int((priority_df["Priority Level"] == "Moderate").sum())
        routine_count = int((priority_df["Priority Level"] == "Routine").sum())

        metric_cols = st.columns(4)
        with metric_cols[0]:
            metric_card("Critical priority", str(critical_count), "Score ≥ 75")
        with metric_cols[1]:
            metric_card("High priority", str(high_count), "Score 55–74.9")
        with metric_cols[2]:
            metric_card("Moderate priority", str(moderate_count), "Score 35–54.9")
        with metric_cols[3]:
            metric_card("Routine priority", str(routine_count), "Score < 35")

        st.markdown("---")
        section_heading(
            "Ranked conservation queue",
            "Start with the highest-ranked locations for verification and field assessment."
        )

        display_columns = [
            "Rank", "Region", "Risk Category", "High Risk Probability (%)",
            "Conservation Priority Score", "Priority Level", "Recommended Action"
        ]
        st.dataframe(
            priority_df[display_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Conservation Priority Score": st.column_config.ProgressColumn(
                    "Priority score",
                    min_value=0,
                    max_value=100,
                    format="%.1f"
                ),
                "High Risk Probability (%)": st.column_config.NumberColumn(
                    "High-risk probability",
                    format="%.2f%%"
                )
            }
        )

        chart_df = priority_df[["Region", "Conservation Priority Score"]].set_index("Region")
        st.markdown("#### Priority score comparison")
        st.bar_chart(chart_df, color="#5b956e", height=300)

        st.markdown("---")
        section_heading(
            "Why is this location prioritized?",
            "Select a region to inspect its score components and suggested next step."
        )
        selected_region = st.selectbox(
            "Choose a region",
            priority_df["Region"].astype(str).tolist(),
            key="forestguard_priority_region"
        )
        selected = priority_df[priority_df["Region"].astype(str) == selected_region].iloc[0]

        detail_left, detail_right = st.columns([1, 1.5], gap="large")
        with detail_left:
            st.metric("Conservation priority score", f"{selected['Conservation Priority Score']:.1f} / 100")
            st.metric("Priority band", str(selected["Priority Level"]))
            st.metric("Predicted risk", str(selected["Risk Category"]))
        with detail_right:
            st.markdown("**Main contributing indicators**")
            reasons = str(selected["Priority Reasons"]).split("; ")
            reasons = [reason for reason in reasons if reason]
            if reasons:
                for reason in reasons:
                    st.markdown(f"- {reason}")
            else:
                st.write("No single indicator crossed the explanation threshold; review the full input profile.")

            st.markdown("**Recommended next step**")
            st.info(str(selected["Recommended Action"]))

        with st.expander("View score methodology"):
            st.markdown("""
            | Component | Weight | Normalization used |
            |---|---:|---|
            | Model High-risk probability | 40% | Model probability from 0–100% |
            | Previous forest loss | 20% | Capped at 40% loss |
            | Fire incidents | 15% | Capped at 5 incidents |
            | Tree-cover deficit | 15% | 100% minus tree-cover percentage |
            | Road / settlement proximity | 10% | Inverse distance using 30 km and 35 km reference ranges |

            Each component is converted to a 0–100 subscore and multiplied by its weight.
            The weighted components are added to produce the final 0–100 priority score.
            """)
            st.caption(
                "The score is a transparent prototype heuristic. It is not a probability of deforestation, "
                "does not establish causation, and should not replace expert judgement or field verification."
            )

        export_columns = [
            "Rank", "Region", "Risk Category", "High Risk Probability (%)",
            "Conservation Priority Score", "Priority Level", "Priority Reasons",
            "Recommended Action", "Source"
        ]
        st.download_button(
            "⬇️ Download conservation priority queue (CSV)",
            data=priority_df[export_columns].to_csv(index=False).encode("utf-8"),
            file_name="forestguard_conservation_priority_queue.csv",
            mime="text/csv",
            use_container_width=True
        )


# =========================================================
# TAB 5: MODEL INSIGHTS
# =========================================================
with tab_insights:
    section_heading("Model performance", "Evaluation results from the ForestGuard training experiment.")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Accuracy", "88.0%", "Held-out test set")
    with m2:
        metric_card("Macro F1", "0.7894", "Held-out test set")
    with m3:
        metric_card("Macro recall", "0.8426", "Held-out test set")
    with m4:
        metric_card("Balanced accuracy", "0.8426", "Held-out test set")

    section_heading("Cross-validation model comparison", "Mean five-fold cross-validation Macro F1; higher is better.")
    comparison = pd.DataFrame({
        "Algorithm": ["Logistic Regression", "XGBoost", "Random Forest", "Decision Tree"],
        "Mean CV Macro F1": [0.7093, 0.6419, 0.6369, 0.5111],
        "CV standard deviation": [0.0380, 0.0592, 0.0435, 0.0519]
    })
    st.bar_chart(comparison.set_index("Algorithm")[["Mean CV Macro F1"]], color="#5b956e")
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.success("Selected model: Logistic Regression — highest mean cross-validation Macro F1 (0.7093).")

    section_heading("Per-class test performance", "Precision, recall and F1-score on the 200-record test set.")
    class_report = pd.DataFrame({
        "Risk category": ["High", "Low", "Medium"],
        "Precision": [0.60, 0.95, 0.70],
        "Recall": [0.86, 0.91, 0.76],
        "F1-score": [0.71, 0.93, 0.73],
        "Test support": [7, 152, 41]
    })
    st.dataframe(class_report, use_container_width=True, hide_index=True)

    section_heading("Feature importance", "Normalized absolute Logistic Regression coefficient magnitudes; these are not causal effects.")
    try:
        importance_values = getattr(model, "feature_importances_", None)
        if importance_values is None:
            st.warning("Feature-importance values were not saved with the current model artifact.")
        else:
            importance_df = pd.DataFrame({
                "Feature": features,
                "Relative importance": importance_values
            }).sort_values("Relative importance", ascending=True)
            fig, ax = plt.subplots(figsize=(10, 5.5))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#ffffff")
            ax.barh(importance_df["Feature"], importance_df["Relative importance"], color="#5b956e")
            ax.set_xlabel("Normalized absolute coefficient magnitude", color="#405449")
            ax.tick_params(colors="#405449")
            for spine in ax.spines.values():
                spine.set_color("#e2e9df")
            ax.grid(axis="x", alpha=.18)
            ax.set_axisbelow(True)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    except Exception as error:
        st.warning(f"Could not display feature importance: {error}")

    st.caption(
        "The High-risk class has only 7 examples in the test set. Its precision and recall estimates "
        "are therefore uncertain and should be validated on a larger independent dataset."
    )


# =========================================================
# TAB 6: ABOUT / LIMITATIONS
# =========================================================
with tab_about:
    section_heading("About ForestGuard", "A machine-learning prototype for environmental risk screening.")
    st.markdown("""
    <div class="panel">
      <div class="panel-title">What ForestGuard does</div>
      <div class="panel-copy">
      ForestGuard accepts environmental and geographical indicators for a candidate plot,
      predicts a Low, Medium or High risk category, displays model-estimated class probabilities,
      and suggests preliminary conservation follow-up actions.
      </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Current capabilities")
        for item in [
            "Three-class deforestation-risk classification",
            "Logistic Regression model selected through cross-validation",
            "Satellite basemap with a manually entered location marker",
            "Rule-based conservation suggestions and warning signals",
            "Single-region and CSV-based multi-region assessment",
            "CSV export of assessment results"
        ]:
            st.markdown(f'<div class="action-item">✓ &nbsp; {item}</div>', unsafe_allow_html=True)
    with right:
        st.markdown("#### Limitations to keep in mind")
        for item in [
            "Trained and tested on the 1,000-record organizer-provided dataset",
            "No independent real-world geographic validation yet",
            "Map does not automatically detect deforestation from imagery",
            "Risk categories are not guarantees of future forest loss",
            "Conservation suggestions require expert and field verification",
            "The High-risk test class is very small"
        ]:
            st.markdown(f'<div class="action-item">• &nbsp; {item}</div>', unsafe_allow_html=True)

    st.info(
        "Responsible use: ForestGuard is intended to support preliminary prioritization. "
        "Do not use its output alone to make enforcement, land-use or conservation decisions."
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:1rem 0 0.5rem 0;">
    <div style="font-family:Manrope,sans-serif;font-weight:800;color:#173f35;font-size:1.05rem;">
        🌿 GREEN LANTERNS
    </div>
    <div style="color:#718076;font-size:.82rem;margin-top:.25rem;">
        AI SANSAD &nbsp; • &nbsp; ForestGuard &nbsp; • &nbsp; Building technology for environmental conservation
    </div>
</div>
""", unsafe_allow_html=True)








