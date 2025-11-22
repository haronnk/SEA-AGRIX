# SEA-AGRIX — Self-Evolving AutoML for Crop Yield Prediction

SEA-AGRIX is an end-to-end machine learning system designed to:

- Perform AutoML-style regression for crop yield
- Handle streaming data
- Detect drift using RMSE-based monitoring
- Retrain and update the model automatically
- Provide an interactive dashboard (Streamlit)
- Generate climate/agriculture alerts

## Features
- Self-evolving learning using SEA (Streaming Ensemble Adaptation)
- Auto model retraining
- Multi-model comparison
- Alert engine for drought & performance drops
- Simple front-end app using Streamlit

## How to run locally

```bash
pip install -r requirements.txt
streamlit run sea_app/streamlit_app.py
