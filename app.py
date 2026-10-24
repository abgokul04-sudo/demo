"""Streamlit app: predict a car's MPG from its specifications.

Run:  streamlit run app.py
"""
import hashlib
import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mpg_model import NUMERIC, TARGET, load_data, predict_mpg, save_model, train_and_select

st.set_page_config(page_title="MPG Predictor", page_icon="⛽", layout="wide")
st.title("How far will it go on a gallon?")
st.caption("Upload auto-mpg.csv, compare five algorithms, and predict a car's fuel efficiency.")


@st.cache_resource(show_spinner="Training and comparing models...")
def get_model(data: bytes, key: str):
    df = load_data(io.BytesIO(data))
    model, results, best, cols = train_and_select(df)
    save_model(model, cols)
    return df, model, results, best, cols


upload = st.sidebar.file_uploader("Dataset (auto-mpg.csv)", type=["csv", "tsv", "txt"])
local = Path("auto-mpg.csv")
if upload is not None:
    data = upload.getvalue()
elif local.exists():
    data = local.read_bytes()
else:
    st.info("Upload auto-mpg.csv in the sidebar to train the models.")
    st.stop()

try:
    df, model, results, best, cols = get_model(data, hashlib.md5(data).hexdigest())
except ValueError as e:
    st.error(str(e))
    st.stop()

has_origin = "origin" in cols
tab_pred, tab_models, tab_3d, tab_batch = st.tabs(
    ["Predict", "Model comparison", "3D explorer", "Predict a file"])

with tab_pred:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Vehicle specifications")
        c1, c2 = st.columns(2)
        rec = {
            "cylinders": c1.number_input("Cylinders", 3, 12, 4),
            "displacement": c2.number_input("Displacement (cu in)", 50.0, 500.0, 140.0),
            "horsepower": c1.number_input("Horsepower", 40.0, 300.0, 90.0),
            "weight": c2.number_input("Weight (lb)", 1000.0, 6000.0, 2600.0),
            "acceleration": c1.number_input("Acceleration (0-60 s)", 6.0, 30.0, 15.0, step=0.1),
            "model_year": c2.number_input("Model year (19xx)", 60, 99, 78),
        }
        if has_origin:
            rec["origin"] = {"USA": 1, "Europe": 2, "Japan": 3}[
                st.selectbox("Origin", ["USA", "Europe", "Japan"])]
        mpg = predict_mpg(model, cols, rec)
    with right:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=mpg, number={"suffix": " mpg"},
            gauge={"axis": {"range": [0, 50]}, "bar": {"color": "#e59a00"}}))
        fig.update_layout(height=300, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Predicted by: {best}  |  {235.215 / max(mpg, 1):.1f} L/100 km")
        st.subheader("Fuel cost and emissions")
        miles = st.number_input("Miles per year", 0, 100000, 12000, step=1000)
        price = st.number_input("Fuel price per gallon", 0.0, 50.0, 3.5, step=0.1)
        gal = miles / max(mpg, 1)
        k1, k2, k3 = st.columns(3)
        k1.metric("Fuel per year", f"{gal:,.0f} gal")
        k2.metric("Cost per year", f"{gal * price:,.0f}")
        k3.metric("CO2 per year", f"{gal * 8.887:,.0f} kg")

with tab_models:
    st.subheader("Algorithms compared")
    st.dataframe(results.style.format({c: "{:.3f}" for c in results.columns[1:]})
                 .highlight_max(subset=["CV R2", "Test R2"], color="#c9f0dd"),
                 use_container_width=True, hide_index=True)
    st.success(f"Deployed model: {best} (highest cross-validated R2, refit on all rows).")
    st.plotly_chart(px.bar(results, x="Model", y="Test R2", color="Model",
                           title="R2 on the held-out test set"), use_container_width=True)

with tab_3d:
    st.subheader("Weight x horsepower x MPG")
    plot = df.dropna(subset=["weight", "horsepower"])
    fig3 = px.scatter_3d(plot, x="weight", y="horsepower", z=TARGET, color=TARGET,
                         color_continuous_scale="RdYlGn", opacity=0.75)
    fig3.add_trace(go.Scatter3d(x=[rec["weight"]], y=[rec["horsepower"]], z=[mpg],
                                mode="markers", name="Your car",
                                marker=dict(size=9, color="#e59a00")))
    fig3.update_layout(height=620, margin=dict(t=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Drag to rotate, scroll to zoom. The amber dot is your prediction.")

with tab_batch:
    st.subheader("Predict a whole file")
    batch = st.file_uploader("CSV of cars (same columns; mpg optional)", type=["csv"], key="batch")
    if batch is not None:
        try:
            raw = pd.read_csv(batch, na_values=["?", "NA", ""])
            clean = raw.copy()
            clean.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in clean.columns]
            out = raw.copy()
            out["predicted_mpg"] = [round(predict_mpg(model, cols, r), 2)
                                    for r in clean.to_dict("records")]
            st.dataframe(out, use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index=False),
                               "predicted-mpg.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not predict this file: {e}")
