import json

def generate_alerts(log_path="SEA_log_stable.json"):
    try:
        log = json.load(open(log_path))
    except:
        return ["No SEA log found"]

    alerts = []

    drifts = log.get("drift_events", [])
    if drifts:
        last = drifts[-1]
        alerts.append(
            f"Drift detected at chunk {last['chunk']} — model retrained."
        )
    else:
        alerts.append("No drift detected — Model stable.")

    last_rmse = log["rmse_history"][-1]
    alerts.append(f"Latest RMSE: {list(last_rmse.values())[0]:.3f}")

    return alerts
