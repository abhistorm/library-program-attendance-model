from pathlib import Path
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'model.joblib'
FEATURES = ["wait_time_min", "application_completeness", "service_load", "repeat_visits", "travel_distance_km", "staff_ratio", "response_delay_hours", "neighborhood_index", "weather_impact", "request_complexity"]
def get_bundle():
    if not MODEL_PATH.exists(): raise RuntimeError('Train first: python src/train.py --limit 20000')
    return joblib.load(MODEL_PATH)
def predict(values):
    bundle = get_bundle()
    row = {k: float(values.get(k, 0.0)) for k in FEATURES}
    return float(bundle['model'].predict(pd.DataFrame([row], columns=FEATURES))[0])

import streamlit as st
st.set_page_config(page_title='AI/ML Starter', layout='centered')
st.title('AI/ML prediction demo')
st.caption('Synthetic-data starter; not for real-world decisions.')
values = {f: st.number_input(f, value=0.0, format='%.4f') for f in FEATURES}
if st.button('Predict'):
    try: st.metric('Prediction', f'{predict(values):.5f}')
    except RuntimeError as exc: st.error(str(exc))
