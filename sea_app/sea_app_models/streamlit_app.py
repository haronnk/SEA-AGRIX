import streamlit as st
import pandas as pd
import os

from sea_engine import SEAEngine
from model_loader import get_model_names, load_model_by_name
from alerts_engine import generate_alerts

st.set_page_config(page_title="SEA-AGRIX Dashboard", layout="wide")

st.title("🌾 SEA-AGRIX — Self-Evolving AutoML for Crop Yield")

tabs = st.tabs(["Overview", "Data", "Models", "SEA Monitor", "Alerts", "Deploy"])

# -------------------------------------------------------------------
with tabs[0]:
    st.header("Overview")
    st.write("""
    SEA-AGRIX is a self-evolving ML system that:
    - Detects data drift  
    - Retrains automatically  
    - Tracks RMSE over time  
    - Allows model selection  
    """)

# -------------------------------------------------------------------
with tabs[1]:
    st.header("Data Upload")

    up = st.file_uploader("Upload dataset CSV (must include `yield` column)", type=['csv'])

    if up:
        df = pd.read_csv(up)
        st.session_state['df'] = df
        df.to_csv("dataset_master.csv", index=False)
        st.success("Dataset loaded")
        st.write(df.head())

# -------------------------------------------------------------------
with tabs[2]:
    st.header("Model Explorer")

    models = get_model_names()
    st.write("Available models:", models)

    choice = st.selectbox("Select model", ["None"] + models)

    if choice != "None":
        if st.button("Test on first 5 rows"):
            df = st.session_state.get('df')
            if df is not None:
                m = load_model_by_name(choice)
                X = df.drop(columns=["yield"]).values[:5]
                preds = m.predict(X).reshape(-1)
                st.write(preds)

# -------------------------------------------------------------------
with tabs[3]:
    st.header("Run SEA")

    if st.button("Start SEA Training + Streaming"):
        engine = SEAEngine("dataset_master.csv", chunk_size=32)
        path, rmse = engine.train_initial()
        st.write("Initial model RMSE:", rmse)

        logpath = engine.simulate_stream()
        st.success("SEA Completed!")
        st.write(open(logpath).read())

# -------------------------------------------------------------------
with tabs[4]:
    st.header("System Alerts")
    for a in generate_alerts():
        st.info(a)

# -------------------------------------------------------------------
with tabs[5]:
    st.header("Deploy App")
    st.write("""
    Deploy on **Streamlit Cloud**:
    
    1. Push repo to GitHub  
    2. Visit https://share.streamlit.io  
    3. Select your repo  
    4. Set entrypoint = `sea_app/streamlit_app.py`  
    """)

