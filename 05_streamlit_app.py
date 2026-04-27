import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.title("Maschinenbericht")
st.subheader("Sensordaten eingeben")

temp = st.slider("Temperatur (°C)", 50.0, 120.0, 85.0)
druck = st.slider("Druck (bar)", 1.0, 10.0, 5.0)
vibration = st.slider("Vibration (mm/s)", 0.0, 5.0, 1.0)
drehzahl = st.slider("Drehzahl (RPM)", 500.0, 3000.0, 1500.0)
strom = st.slider("Stromverbrauch (A)", 1.0, 20.0, 10.0)

if st.button("Bericht erstellen"):
    modell = joblib.load("model_lgb_v1.pkl")
    eingabe = np.array([[temp, druck, vibration, drehzahl, strom]])
    vorhersage = modell.predict(eingabe)[0]

    if vorhersage > 0.7:
        st.error("⛔ Status: KRITISCH")
    elif vorhersage > 0.4:
        st.warning("⚠️ Status: ERHÖHT")
    else:
        st.success("✅ Status: NORMAL")

    st.write(f"Risikowert: {vorhersage:.3f}")

# SHAP Erklärung
    st.subheader("Warum dieser Status?")
    explainer = shap.TreeExplainer(modell)
    feature_names = ["Temperatur", "Druck", "Vibration", "Drehzahl", "Strom"]
    import pandas as pd
    eingabe_df = pd.DataFrame(eingabe, columns=feature_names)
    shap_values = explainer(eingabe_df)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)