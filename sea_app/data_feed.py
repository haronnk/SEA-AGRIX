# data_feed.py
# Schema-aware live data generator for SEA-AGRIX

import pandas as pd
import numpy as np


def generate_live_rows(base_df, n_rows=8):
    """
    Generate live rows dynamically based on dataset schema.
    Works for ANY numeric agricultural dataset.
    """

    rows = []

    feature_columns = [c for c in base_df.columns if c != "yield"]

    for _ in range(n_rows):
        row = {}

        for col in feature_columns:
            col_min = base_df[col].min()
            col_max = base_df[col].max()

            # Handle numeric columns safely
            if pd.api.types.is_numeric_dtype(base_df[col]):
                noise = np.random.normal(0, 0.05 * (col_max - col_min + 1e-6))
                value = np.random.uniform(col_min, col_max) + noise
                row[col] = max(value, 0)
            else:
                # If any categorical columns exist, sample from history
                row[col] = base_df[col].sample(1).values[0]

        # Yield is simulated observation (ground truth)
        row["yield"] = np.random.uniform(
            base_df["yield"].min(),
            base_df["yield"].max()
        )

        rows.append(row)

    return pd.DataFrame(rows)


def append_live_data(csv_path, n_rows=8):
    """
    Append schema-consistent live data to dataset
    """

    df = pd.read_csv(csv_path)
    new_rows = generate_live_rows(df, n_rows)

    df_updated = pd.concat([df, new_rows], ignore_index=True)
    df_updated.to_csv(csv_path, index=False)

    print(f"📡 Live data appended: {n_rows} rows")
    print(new_rows.head(3))

    return df_updated