import streamlit as st
import pandas as pd
import joblib
import os

st.title("Conversie Voorspeller")

# Controleer of modelbestand bestaat
model_path = 'streamlit_model_py313.pkl'
if not os.path.exists(model_path):
    st.error(f"❌ Kan modelbestand niet vinden: {model_path}")
    st.stop()

# Probeer model te laden
try:
    model, feature_names, label_encoder = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Fout bij laden van model: {e}")
    st.stop()

st.markdown("Voer hieronder de gegevens in om te voorspellen of een order een conversie wordt.")

# Genereer invoervelden
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", step=1.0)

if st.button("Voorspel"):
    try:
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)
        resultaat = label_encoder.inverse_transform(prediction)[0]
        st.success(f"→ Voorspelling: **{resultaat}**")
    except Exception as e:
        st.error(f"❌ Fout tijdens voorspellen: {e}")
