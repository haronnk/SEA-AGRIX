# =========================================================
# SEA-AGRIX : Main Controller
# Supports:
# 1) Simulation Mode (static dataset)
# 2) Live Mode (continuous data feed + retraining)
# =========================================================

import time
import argparse
import pandas as pd

from sea_engine import SEAEngine
from data_feed import append_live_data

DATASET_PATH = "dataset_master.csv"


# ---------------------------------------------------------
# SIMULATION MODE
# ---------------------------------------------------------
def run_simulation():
    """
    One-time execution using existing dataset.
    Used for:
    - Academic evaluation
    - Reproducible experiments
    - Teacher demo (safe mode)
    """
    print("\n🧪 SIMULATION MODE STARTED")

    engine = SEAEngine(
        csv_path=DATASET_PATH,
        chunk_size=32,
        drift_threshold=0.25
    )

    engine.train_initial()
    engine.run_stream()

    print("\n✅ SIMULATION MODE COMPLETED")


# ---------------------------------------------------------
# LIVE MODE
# ---------------------------------------------------------
def run_live(interval_seconds=10):
    """
    Continuous execution:
    - Appends new data periodically
    - Retrains when drift occurs
    - Never 'ends' unless user stops
    """
    print("\n📡 LIVE MODE STARTED")
    print("System is now running continuously.")
    print("Press CTRL+C to stop.\n")

    engine = SEAEngine(
        csv_path=DATASET_PATH,
        chunk_size=32,
        drift_threshold=0.25
    )

    engine.train_initial()

    cycle = 1

    try:
        while True:
            print(f"\n🔁 LIVE CYCLE {cycle}")

            # Step 1: Append new incoming data
            append_live_data(
                csv_path=DATASET_PATH,
                n_rows=engine.chunk_size
            )

            # Step 2: Reload dataset with new rows
            engine.df = pd.read_csv(DATASET_PATH)

            # Step 3: Run SEA logic on updated data
            engine.run_stream()

            print(f"⏳ Waiting {interval_seconds} seconds for next data batch...\n")
            time.sleep(interval_seconds)

            cycle += 1

    except KeyboardInterrupt:
        print("\n🛑 LIVE MODE STOPPED BY USER")
        print("System shut down safely.")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SEA-AGRIX Execution Controller")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulation", "live"],
        default="simulation",
        help="Run mode: simulation or live"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Seconds between live data updates (live mode only)"
    )

    args = parser.parse_args()

    if args.mode == "simulation":
        run_simulation()
    else:
        run_live(interval_seconds=args.interval)