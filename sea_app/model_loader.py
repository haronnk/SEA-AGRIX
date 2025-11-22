import os
from tensorflow.keras.models import load_model

MODELS_DIR = "sea_app_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def get_model_names():
    return [f for f in os.listdir(MODELS_DIR) if f.endswith(".h5")]

def load_model_by_name(name):
    return load_model(os.path.join(MODELS_DIR, name))
