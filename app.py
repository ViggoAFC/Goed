
import streamlit as st
import pandas as pd
import joblib

# Laad het model, feature-namen en label encoder
model, feature_names, label_encoder = joblib.load("streamlit_model_py313.pkl")

# Titel van de app
st.title("Conversie Voorspeller")

st.markdown("Voer hieronder de gegevens in om te voorspellen of een order een conversie wordt.")

# Maak invoervelden op basis van feature names
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", step=1.0)

# Wanneer gebruiker op knop klikt
if st.button("Voorspel"):
    try:
        # Zet input om naar dataframe
        df = pd.DataFrame([input_data])
        # Maak voorspelling
        prediction = model.predict(df)
        resultaat = label_encoder.inverse_transform(prediction)[0]
        # Toon resultaat
        st.subheader("Voorspelling:")
        st.success(f"→ {resultaat}")
    except Exception as e:
        st.error(f"Er ging iets mis: {e}")
