import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd

# Datenbankverbindung
conn = sqlite3.connect("maschinen.db")

st.title("Maschinenbericht")

# Tabs
tab1, tab2 = st.tabs(["Vorhersage", "Datenbank"])

with tab1:
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

        st.subheader("Warum dieser Status?")
        explainer = shap.TreeExplainer(modell)
        feature_names = ["Temperatur", "Druck", "Vibration", "Drehzahl", "Strom"]
        import pandas as pd
        eingabe_df = pd.DataFrame(eingabe, columns=feature_names)
        shap_values = explainer(eingabe_df)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

with tab2:
    st.subheader("Alle Maschinen")
    df = pd.read_sql("SELECT * FROM maschinen ORDER BY status", conn)
    st.dataframe(df)

    st.subheader("Neue Maschine hinzufügen")
    name = st.text_input("Name")
    temp_db = st.number_input("Temperatur (°C)", 50.0, 120.0, 85.0)
    druck_db = st.number_input("Druck (bar)", 1.0, 10.0, 5.0)
    vibration_db = st.number_input("Vibration (mm/s)", 0.0, 5.0, 1.0)
    status_db = st.selectbox("Status", ["NORMAL", "ERHÖHT", "KRITISCH"])

    if st.button("Maschine hinzufügen"):
        conn.execute(f"""
                INSERT INTO maschinen (name, temperatur, druck, vibration, status)
                VALUES ('{name}', {temp_db}, {druck_db}, {vibration_db}, '{status_db}')
            """)
        conn.commit()
        st.success(f"{name} wurde hinzugefügt!")
        st.rerun()