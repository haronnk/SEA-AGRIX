import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

LOG_PATH = "SEA_log_stable.json"
MODELS_DIR = "sea_app_models"
os.makedirs(MODELS_DIR, exist_ok=True)


class SEAEngine:
    def __init__(self, dataset_path: str, chunk_size: int = 32, drift_threshold: float = 0.12):
        self.df = pd.read_csv(dataset_path)
        self.feature_cols = [c for c in self.df.columns if c != "yield"]
        self.chunk_size = chunk_size
        self.drift_threshold = drift_threshold
        self.log = {"rmse_history": [], "drift_events": []}
        self.current_model_path = None

    def _build(self, input_dim):
        m = Sequential([
            Dense(64, activation="relu", input_shape=(input_dim,)),
            Dropout(0.12),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    def train_initial(self, epochs=12):
        X = self.df[self.feature_cols].values
        y = self.df["yield"].values

        m = self._build(X.shape[1])
        m.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

        path = f"{MODELS_DIR}/sea_initial.h5"
        m.save(path)
        self.current_model_path = path

        rmse = float(np.sqrt(mean_squared_error(y, m.predict(X).reshape(-1))))
        self.log["rmse_history"].append({"initial": rmse})

        json.dump(self.log, open(LOG_PATH, "w"), indent=2)
        return path, rmse

    def simulate_stream(self, retrain_epochs=6):
        if not self.current_model_path:
            raise RuntimeError("Run train_initial() first")

        df = self.df.reset_index(drop=True)
        n = len(df)

        model = load_model(self.current_model_path)
        rmse_hist = []
        event_id = 0

        for i in range(0, n, self.chunk_size):
            chunk = df.iloc[i:i+self.chunk_size]
            Xc = chunk[self.feature_cols].values
            yc = chunk["yield"].values

            preds = model.predict(Xc).reshape(-1)
            rmse = float(np.sqrt(mean_squared_error(yc, preds)))
            rmse_hist.append(rmse)

            baseline = np.mean(rmse_hist[-4:])

            self.log["rmse_history"].append({"chunk": i, "rmse": rmse})

            if len(rmse_hist) > 3 and (rmse - baseline) / (baseline + 1e-9) > self.drift_threshold:
                # DRIFT TRIGGER
                self.log["drift_events"].append(
                    {"event_id": event_id, "chunk": i, "rmse": rmse, "baseline": baseline}
                )

                # RETRAIN ON ALL DATA SEEN
                sub_df = df.iloc[:i+self.chunk_size]
                Xr = sub_df[self.feature_cols].values
                yr = sub_df["yield"].values

                m2 = self._build(Xr.shape[1])
                m2.fit(Xr, yr, epochs=retrain_epochs, verbose=0)

                new_path = f"{MODELS_DIR}/sea_retrain_{event_id}.h5"
                m2.save(new_path)
                model = m2
                self.current_model_path = new_path

                event_id += 1

            json.dump(self.log, open(LOG_PATH, "w"), indent=2)

        return LOG_PATH
