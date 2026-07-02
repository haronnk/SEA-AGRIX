import streamlit as st
import os
import time
import json

from sea_engine import SEAEngine
from data_feed import append_live_data

# -------------------------
# Paths
# -------------------------
DATASET_PATH = "dataset_master.csv"
BASE_PLOT_FILE = "sea_outputs/SEA_RMSE_plot.png"
LOG_FILE = "sea_outputs/SEA_log.json"

# -------------------------
# Session State
# -------------------------
if "live_running" not in st.session_state:
    st.session_state.live_running = False

if "live_cycle" not in st.session_state:
    st.session_state.live_cycle = 0

if "live_plots" not in st.session_state:
    st.session_state.live_plots = []

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="SEA-AGRIX",
    layout="wide"
)

st.title("🌱 SEA-AGRIX: Self-Evolving Agriculture Intelligence")

tabs = st.tabs([
    "Overview",
    "Run Simulation",
    "Run Live",
    "Results",
    "Logs"
])

# =========================================================
# OVERVIEW
# =========================================================
with tabs[0]:
    st.markdown("""
    **SEA-AGRIX** is a self-evolving ML system that:

    - Predicts crop yield  
    - Detects concept drift  
    - Automatically retrains itself  
    - Supports continuous live data streams  
    """)

# =========================================================
# RUN SIMULATION (STATIC)
# =========================================================
with tabs[1]:
    st.header("Run Simulation")

    if st.button("▶️ Run Simulation"):
        # Reset live state
        st.session_state.live_running = False
        st.session_state.live_cycle = 0
        st.session_state.live_plots = []

        engine = SEAEngine(
            csv_path=DATASET_PATH,
            chunk_size=32,
            drift_threshold=0.25
        )

        engine.train_initial()
        engine.run_stream()

        st.success("Simulation completed")

# =========================================================
# RUN LIVE SYSTEM (CONTINUOUS)
# =========================================================
with tabs[2]:
    st.header("Run Live System")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶️ Start Live System"):
            st.session_state.live_running = True
            st.success("Live system started")

    with col2:
        if st.button("⏹ Stop Live System"):
            st.session_state.live_running = False
            st.warning("Live system stopped")

    # -------------------------
    # LIVE LOOP
    # -------------------------
    if st.session_state.live_running:

        st.session_state.live_cycle += 1
        cycle_id = st.session_state.live_cycle

        st.info(f"🔁 Live Cycle {cycle_id}")

        engine = SEAEngine(
            csv_path=DATASET_PATH,
            chunk_size=32,
            drift_threshold=0.25
        )

        # Append new live data
        append_live_data(
            csv_path=DATASET_PATH,
            n_rows=engine.chunk_size
        )

        # Train + evaluate
        engine.train_initial()
        engine.run_stream()

        # Save plot uniquely (NO overwrite)
        if os.path.exists(BASE_PLOT_FILE):
            live_plot_path = f"sea_outputs/live_rmse_cycle_{cycle_id}.png"

            if os.path.exists(live_plot_path):
                os.remove(live_plot_path)

            os.rename(BASE_PLOT_FILE, live_plot_path)
            st.session_state.live_plots.append(live_plot_path)

        st.success("Live cycle completed")

        time.sleep(3)
        st.rerun()

# =========================================================
# RESULTS
# =========================================================
with tabs[3]:
    st.header("Results")

    if st.session_state.live_plots:
        st.subheader("Live System Results")

        for i, plot in enumerate(st.session_state.live_plots, start=1):
            st.markdown(f"**Live Cycle {i}**")
            st.image(plot)

    elif os.path.exists(BASE_PLOT_FILE):
        st.subheader("Simulation Result")
        st.image(BASE_PLOT_FILE)

    else:
        st.info("Run simulation or live system to see results")

# =========================================================
# LOGS
# =========================================================
with tabs[4]:
    st.header("Logs")

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            log = json.load(f)
        st.json(log)
    else:
        st.info("No logs available yet")