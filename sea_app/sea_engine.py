# =========================================================
# SEA-AGRIX : Self-Evolving Agriculture Intelligence Engine
# FUTURE-PROOF BACKEND (Python 3.13 + Keras 3 SAFE)
# =========================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# ---------------------------------------------------------
# Setup folders
# ---------------------------------------------------------
os.makedirs("sea_models", exist_ok=True)
os.makedirs("sea_outputs", exist_ok=True)

LOG_FILE = "sea_outputs/SEA_log.json"
PLOT_FILE = "sea_outputs/SEA_RMSE_plot.png"
PRED_FILE = "sea_outputs/SEA_yield_predictions.csv"

# ---------------------------------------------------------
# SEA ENGINE
# ---------------------------------------------------------
class SEAEngine:

    def __init__(self, csv_path, chunk_size=32, drift_threshold=0.25):
        self.csv_path = csv_path
        self.chunk_size = chunk_size
        self.drift_threshold = drift_threshold
        self.model = None

        self.log = {
            "initial_rmse": None,
            "rmse_per_chunk": [],
            "drift_events": []
        }

        # ✅ FIXED indentation
        self.live_rmse_history = []

        self._load_dataset()

    # -------------------------
    # Load / Reload Dataset
    # -------------------------
    def _load_dataset(self):
        self.df = pd.read_csv(self.csv_path)
        self.features = [c for c in self.df.columns if c != "yield"]

    # -------------------------
    # Model Architecture
    # -------------------------
    def build_model(self, input_dim):
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    # -------------------------
    # Initial Training
    # -------------------------
    def train_initial(self):
        X = self.df[self.features].values
        y = self.df["yield"].values

        self.model = self.build_model(X.shape[1])
        self.model.fit(X, y, epochs=15, batch_size=32, verbose=0)

        preds = self.model.predict(X).reshape(-1)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))

        self.model.save("sea_models/initial_model.keras")
        self.log["initial_rmse"] = rmse

        print("\n✅ Initial model trained")
        print(f"📉 Initial RMSE: {rmse:.2f}")

    # -------------------------
    # Streaming + Drift Logic
    # -------------------------
    def run_stream(self):

        # reload dataset for live feed behavior
        self._load_dataset()

        rmse_history = []
        event_id = 0

        for i in range(0, len(self.df), self.chunk_size):

            chunk_id = i // self.chunk_size
            chunk = self.df.iloc[i:i + self.chunk_size]

            Xc = chunk[self.features].values
            yc = chunk["yield"].values

            preds = self.model.predict(Xc).reshape(-1)
            rmse = float(np.sqrt(mean_squared_error(yc, preds)))

            # ✅ live RMSE tracking
            self.live_rmse_history.append(rmse)
            self.plot_live_rmse(chunk_id)

            rmse_history.append(rmse)

            self.log["rmse_per_chunk"].append({
                "chunk": chunk_id,
                "rmse": rmse
            })

            baseline = np.mean(rmse_history[-3:-1]) if len(rmse_history) >= 3 else rmse

            if rmse > baseline * (1 + self.drift_threshold):

                print(f"\n🚨 Drift detected at chunk {chunk_id}")
                print(f"   RMSE before retrain: {rmse:.2f}")

                retrain_df = self.df.iloc[:i + self.chunk_size]
                Xr = retrain_df[self.features].values
                yr = retrain_df["yield"].values

                weights = np.linspace(0.3, 1.0, len(yr))

                self.model = self.build_model(Xr.shape[1])
                self.model.fit(
                    Xr, yr,
                    sample_weight=weights,
                    epochs=12,
                    batch_size=32,
                    verbose=0
                )

                preds_after = self.model.predict(Xc).reshape(-1)
                rmse_after = float(np.sqrt(mean_squared_error(yc, preds_after)))

                self.model.save(f"sea_models/retrained_{event_id}.keras")

                self.log["drift_events"].append({
                    "event": event_id,
                    "chunk": chunk_id,
                    "baseline": float(baseline),
                    "rmse_before": rmse,
                    "rmse_after": rmse_after
                })

                print(f"   RMSE after retrain:  {rmse_after:.2f}")
                event_id += 1

        self.save_outputs()
        self.plot_rmse()
        self.save_predictions()

    # -------------------------
    # Save Logs
    # -------------------------
    def save_outputs(self):
        with open(LOG_FILE, "w") as f:
            json.dump(self.log, f, indent=2)
        print(f"\n📁 Log saved to {LOG_FILE}")

    # -------------------------
    # Simulation RMSE Plot
    # -------------------------
    def plot_rmse(self):
        chunks = [r["chunk"] for r in self.log["rmse_per_chunk"]]
        rmse_vals = [r["rmse"] for r in self.log["rmse_per_chunk"]]

        drift_chunks = [d["chunk"] for d in self.log["drift_events"]]
        before = [d["rmse_before"] for d in self.log["drift_events"]]
        after = [d["rmse_after"] for d in self.log["drift_events"]]

        plt.figure(figsize=(12,6))
        plt.plot(chunks, rmse_vals, marker="o", label="RMSE per chunk")
        plt.scatter(drift_chunks, before, color="red", s=160, label="Before retrain")
        plt.scatter(drift_chunks, after, color="green", s=160, label="After retrain")

        for c in drift_chunks:
            plt.axvline(c, linestyle="--", color="gray", alpha=0.4)

        plt.title("SEA-AGRIX: Drift Detection & Self-Evolving Learning")
        plt.xlabel("Chunk Index")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=300)
        plt.close()

        print(f"📊 Simulation RMSE plot saved to {PLOT_FILE}")

    # -------------------------
    # LIVE RMSE Plot (NEW, FIXED)
    # -------------------------
    def plot_live_rmse(self, cycle_id):
        plt.figure(figsize=(10,5))
        plt.plot(
            range(1, len(self.live_rmse_history) + 1),
            self.live_rmse_history,
            marker="o",
            color="orange",
            linewidth=2
        )
        plt.title(f"LIVE RMSE Evolution (Cycle {cycle_id})")
        plt.xlabel("Live Step")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.tight_layout()

        live_plot_path = f"sea_outputs/live_rmse_cycle_{cycle_id}.png"
        plt.savefig(live_plot_path, dpi=300)
        plt.close()

        print(f"📈 Live RMSE graph updated → {live_plot_path}")

    # -------------------------
    # Final Predictions
    # -------------------------
    def save_predictions(self):
        final_chunk = self.df.iloc[-self.chunk_size:]
        Xf = final_chunk[self.features].values
        y_true = final_chunk["yield"].values
        y_pred = self.model.predict(Xf).reshape(-1)

        pred_df = pd.DataFrame({
            "Actual_Yield": y_true,
            "Predicted_Yield": y_pred,
            "Error": y_pred - y_true
        })

        pred_df.to_csv(PRED_FILE, index=False)

        print("\n🌾 SAMPLE YIELD PREDICTIONS")
        print(pred_df.head(10))
        print(f"\n📁 Predictions saved to {PRED_FILE}")

# =========================================================
# RUN
# =========================================================

engine = SEAEngine(
    csv_path="dataset_master.csv",
    chunk_size=32,
    drift_threshold=0.25
)

engine.train_initial()
engine.run_stream()

print("\n✅ SEA-AGRIX FUTURE-PROOF BACKEND COMPLETED")