import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ─── PAGE CONFIG ───────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ─── LOAD MODEL ────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/house_price_best_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, feature_names

model, feature_names = load_model()

# ─── HEADER ────────────────────────────────────────────
st.title("🏠 California House Price Predictor")
st.markdown("Predict house prices using a trained **XGBoost** model on California Census data.")
st.divider()

# ─── INPUT SECTION ─────────────────────────────────────
st.subheader("📊 Enter House Details")

col1, col2, col3 = st.columns(3)

with col1:
    med_inc = st.slider(
        "Median Income (in $10,000s)",
        min_value=0.5, max_value=15.0,
        value=5.0, step=0.1,
        help="Median income of people in this block"
    )
    house_age = st.slider(
        "House Age (years)",
        min_value=1, max_value=52,
        value=20, step=1,
        help="Median age of houses in this block"
    )
    ave_rooms = st.slider(
        "Average Rooms",
        min_value=1.0, max_value=15.0,
        value=5.0, step=0.1,
        help="Average number of rooms per house"
    )
    ave_bedrms = st.slider(
        "Average Bedrooms",
        min_value=1.0, max_value=5.0,
        value=1.0, step=0.1,
        help="Average number of bedrooms per house"
    )

with col2:
    population = st.slider(
        "Population",
        min_value=3, max_value=5000,
        value=1000, step=10,
        help="Total population of this block"
    )
    ave_occup = st.slider(
        "Average Occupants",
        min_value=1.0, max_value=10.0,
        value=3.0, step=0.1,
        help="Average number of people per house"
    )
    latitude = st.slider(
        "Latitude",
        min_value=32.5, max_value=42.0,
        value=37.0, step=0.1,
        help="Geographic latitude of the block"
    )
    longitude = st.slider(
        "Longitude",
        min_value=-124.0, max_value=-114.0,
        value=-119.0, step=0.1,
        help="Geographic longitude of the block"
    )

with col3:
    st.markdown("### 🔢 Engineered Features")
    st.markdown("*Calculated automatically from your inputs*")

    rooms_per_person = ave_rooms / ave_occup
    bedroom_ratio = ave_bedrms / ave_rooms
    income_per_room = med_inc / ave_rooms

    st.metric("Rooms Per Person", f"{rooms_per_person:.2f}")
    st.metric("Bedroom Ratio", f"{bedroom_ratio:.2f}")
    st.metric("Income Per Room", f"{income_per_room:.2f}")

# ─── PREDICTION ────────────────────────────────────────
st.divider()

if st.button("🔮 Predict House Price", type="primary", use_container_width=True):

    # build input array in correct feature order
    input_data = pd.DataFrame([[
        med_inc, house_age, ave_rooms, ave_bedrms,
        population, ave_occup, latitude, longitude,
        rooms_per_person, bedroom_ratio, income_per_room
    ]], columns=feature_names)

    # predict
    prediction = model.predict(input_data)[0]
    price_dollars = prediction * 100000

    # show result
    st.success(f"### 🏡 Predicted House Price: **${price_dollars:,.0f}**")

    # show confidence range
    lower = price_dollars * 0.85
    upper = price_dollars * 1.15
    st.info(f"📊 Estimated range: **${lower:,.0f}** — **${upper:,.0f}** (±15%)")

# ─── FEATURE IMPORTANCE ────────────────────────────────
st.divider()
st.subheader("📈 Feature Importance")
st.markdown("*Which features affect house price the most?*")

try:
    # get feature importance from XGBoost
    regressor = model.named_steps["regressor"]
    importances = regressor.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature")["Importance"])
except:
    st.write("Feature importance not available for this model.")

# ─── FOOTER ────────────────────────────────────────────
st.divider()
st.markdown("Built with ❤️ using Streamlit + XGBoost | California Housing Dataset 1990")