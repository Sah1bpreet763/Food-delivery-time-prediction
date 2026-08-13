import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="RouteCast — Delivery Time Forecast", page_icon="🛵", layout="wide")

# ---------- palette ----------
BG = "#F3F6F4"
SURFACE = "#FFFFFF"
INK = "#1F2A33"
TEAL = "#1B7A72"
MANGO = "#F2994A"
CORAL = "#E76F51"
MIST = "#DCEAE6"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {INK};
}}
.stApp {{
    background-color: {BG};
}}
h1, h2, h3 {{
    font-family: 'Fraunces', serif;
    color: {INK};
}}
section[data-testid="stSidebar"] {{
    background-color: {SURFACE};
    border-right: 1px solid {MIST};
}}
.rc-header {{
    background: linear-gradient(90deg, {TEAL} 0%, #226B63 100%);
    padding: 28px 32px;
    border-radius: 14px;
    margin-bottom: 24px;
}}
.rc-header h1 {{
    color: white;
    margin: 0;
    font-size: 2.1rem;
}}
.rc-header p {{
    color: #E4F1EE;
    margin: 6px 0 0 0;
    font-size: 1rem;
}}
.rc-card {{
    background-color: {SURFACE};
    border: 1px solid {MIST};
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
}}
.rc-card h3 {{
    margin-top: 0;
    font-size: 1.1rem;
}}
.rc-row {{
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px dashed {MIST};
    font-size: 0.95rem;
}}
.rc-row:last-child {{
    border-bottom: none;
}}
.rc-badge {{
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    color: white;
}}
.rc-footer {{
    color: #6B7B76;
    font-size: 0.82rem;
    margin-top: 8px;
}}
div.stButton > button {{
    background-color: {MANGO};
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1rem;
}}
div.stButton > button:hover {{
    background-color: #DB8438;
    color: white;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="rc-header">
    <h1>🛵 RouteCast</h1>
    <p>Estimate how long an order will take to arrive, before the courier even leaves the kitchen.</p>
</div>
""", unsafe_allow_html=True)

# ---------- load model ----------
@st.cache_resource
def load_model():
    with open("../models/food_delivery_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_benchmark():
    try:
        return pd.read_csv("../data/processed/model_results.csv", index_col=0)
    except FileNotFoundError:
        return None

bundle = load_model()
model = bundle["model"]
columns = bundle["columns"]
benchmark = load_benchmark()

TRAFFIC_MAP = {"Low": 0, "Medium": 1, "High": 2}
WEATHER_OPTIONS = ["Clear", "Windy", "Foggy", "Rainy", "Snowy"]
TIME_OPTIONS = ["Morning", "Afternoon", "Evening", "Night"]
VEHICLE_OPTIONS = ["Bike", "Scooter", "Car"]


def build_input_row(distance, weather, traffic, time_of_day, vehicle, prep_time, courier_exp):
    row = {col: 0 for col in columns}
    row["Distance_km"] = distance
    row["Traffic_Level"] = TRAFFIC_MAP[traffic]
    row["Preparation_Time_min"] = prep_time
    row["Courier_Experience_yrs"] = courier_exp

    weather_col = f"Weather_{weather}"
    if weather_col in row:
        row[weather_col] = 1

    time_col = f"Time_of_Day_{time_of_day}"
    if time_col in row:
        row[time_col] = 1

    vehicle_col = f"Vehicle_Type_{vehicle}"
    if vehicle_col in row:
        row[vehicle_col] = 1

    return pd.DataFrame([row])[columns]


# ---------- sidebar inputs ----------
st.sidebar.header("Order Details")

distance = st.sidebar.slider("Distance (km)", 0.5, 20.0, 7.5, 0.1)
weather = st.sidebar.selectbox("Weather", WEATHER_OPTIONS)
traffic = st.sidebar.selectbox("Traffic Level", ["Low", "Medium", "High"], index=1)
time_of_day = st.sidebar.selectbox("Time of Day", TIME_OPTIONS, index=1)
vehicle = st.sidebar.selectbox("Vehicle Type", VEHICLE_OPTIONS, index=1)
prep_time = st.sidebar.slider("Prep Time (min)", 0, 40, 15)
courier_exp = st.sidebar.slider("Courier Experience (yrs)", 0, 10, 4)

predict_clicked = st.sidebar.button("Forecast Delivery Time", type="primary", use_container_width=True)

if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.inputs = None

if predict_clicked:
    X_input = build_input_row(distance, weather, traffic, time_of_day, vehicle, prep_time, courier_exp)
    pred = float(model.predict(X_input)[0])
    st.session_state.prediction = max(pred, 0)
    st.session_state.inputs = dict(
        distance=distance, weather=weather, traffic=traffic,
        time_of_day=time_of_day, vehicle=vehicle,
        prep_time=prep_time, courier_exp=courier_exp,
        X_input=X_input,
    )

# ---------- main area ----------
left, right = st.columns([1.1, 1])

with left:
    st.markdown('<div class="rc-card">', unsafe_allow_html=True)
    st.markdown("<h3>Predicted Delivery Time</h3>", unsafe_allow_html=True)

    pred = st.session_state.prediction
    gauge_value = pred if pred is not None else 0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        number={"suffix": " min", "font": {"size": 44, "color": INK}},
        gauge={
            "axis": {"range": [0, 150], "tickcolor": INK},
            "bar": {"color": TEAL, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "#CDEDD9"},
                {"range": [35, 70], "color": "#FBE3C2"},
                {"range": [70, 150], "color": "#F7CFC4"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    if pred is None:
        st.caption("Set the order details on the left and click **Forecast Delivery Time**.")
    else:
        if pred < 35:
            label, color = "Fast", TEAL
        elif pred < 70:
            label, color = "Moderate", MANGO
        else:
            label, color = "Slow", CORAL
        st.markdown(
            f'<span class="rc-badge" style="background-color:{color};">{label} · {pred:.0f} min</span>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="rc-card">', unsafe_allow_html=True)
    st.markdown("<h3>Order Summary</h3>", unsafe_allow_html=True)

    if st.session_state.inputs is None:
        st.caption("No forecast yet.")
    else:
        i = st.session_state.inputs
        rows = [
            ("Distance", f"{i['distance']:.1f} km"),
            ("Weather", i["weather"]),
            ("Traffic", i["traffic"]),
            ("Time of Day", i["time_of_day"]),
            ("Vehicle", i["vehicle"]),
            ("Prep Time", f"{i['prep_time']} min"),
            ("Courier Experience", f"{i['courier_exp']} yrs"),
        ]
        for label, value in rows:
            st.markdown(f'<div class="rc-row"><span>{label}</span><b>{value}</b></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if benchmark is not None:
        st.markdown('<div class="rc-card">', unsafe_allow_html=True)
        st.markdown("<h3>Model Quality</h3>", unsafe_allow_html=True)
        best_row = benchmark.loc[benchmark["RMSE"].idxmin()]
        st.markdown(
            f'<div class="rc-row"><span>Model in use</span><b>{type(model).__name__}</b></div>'
            f'<div class="rc-row"><span>Avg. error (MAE)</span><b>{best_row["MAE"]:.1f} min</b></div>'
            f'<div class="rc-row"><span>R² on test set</span><b>{best_row["R2"]:.2f}</b></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- what's driving this estimate ----------
if st.session_state.inputs is not None:
    st.markdown('<div class="rc-card">', unsafe_allow_html=True)

    if hasattr(model, "coef_"):
        st.markdown("<h3>What's driving this estimate</h3>", unsafe_allow_html=True)
        st.caption("Minutes each factor adds to (or saves from) the base estimate, for this specific order.")
        X_input = st.session_state.inputs["X_input"].iloc[0]
        contributions = pd.Series(model.coef_, index=columns) * X_input
        contributions = contributions[contributions.abs() > 0.01].sort_values()

        bar_colors = [TEAL if v < 0 else MANGO for v in contributions.values]
        fig2 = go.Figure(go.Bar(
            x=contributions.values,
            y=contributions.index,
            orientation="h",
            marker_color=bar_colors,
        ))
        fig2.update_layout(
            height=max(260, 30 * len(contributions)),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="minutes",
        )
        st.plotly_chart(fig2, use_container_width=True)

    elif hasattr(model, "feature_importances_"):
        st.markdown("<h3>What matters most overall</h3>", unsafe_allow_html=True)
        importances = pd.Series(model.feature_importances_, index=columns).sort_values()
        fig2 = go.Figure(go.Bar(
            x=importances.values,
            y=importances.index,
            orientation="h",
            marker_color=TEAL,
        ))
        fig2.update_layout(
            height=max(260, 30 * len(importances)),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<p class="rc-footer">RouteCast is a course project — estimates are based on historical averages, not live traffic data.</p>',
    unsafe_allow_html=True,
)
