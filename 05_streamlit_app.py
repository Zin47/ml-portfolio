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
tab1, tab2, tab3, tab4 = st.tabs(["Vorhersage", "Datenbank", "Forecast", "Anomalien"])

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
with tab3:
    st.subheader("Temperatur Forecast")

    uploaded_file = st.file_uploader("CSV hochladen (Spalten: ds, y)", type="csv")

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        df_upload["ds"] = pd.to_datetime(df_upload["ds"])

        st.write("Daten geladen:")
        st.dataframe(df_upload.head())

        if st.button("Forecast erstellen"):
            from prophet import Prophet

            modell_ts = Prophet(daily_seasonality=False, yearly_seasonality=False)
            modell_ts.fit(df_upload)

            zukunft = modell_ts.make_future_dataframe(periods=14)
            forecast = modell_ts.predict(zukunft)

            import matplotlib.pyplot as plt

            fig = modell_ts.plot(forecast)
            plt.axhline(y=95, color="red", linestyle="--", label="Alarmgrenze 95°C")
            plt.title("Temperatur Forecast — 14 Tage")
            plt.legend()
            st.pyplot(fig)
with tab4:
    st.subheader("Anomaly Detection")

    uploaded_anomalie = st.file_uploader("CSV hochladen (Spalten: temperatur, druck, vibration)", type="csv",
                                         key="anomalie")

    if uploaded_anomalie is not None:
        df_anomalie = pd.read_csv(uploaded_anomalie)
        st.write("Daten geladen:")
        st.dataframe(df_anomalie.head())

        contamination = st.slider("Empfindlichkeit (% Anomalien erwartet)", 0.01, 0.1, 0.02)

        if st.button("Anomalien finden"):
            from sklearn.ensemble import IsolationForest

            modell_iso = IsolationForest(contamination=contamination, random_state=42)
            df_anomalie["label"] = modell_iso.fit_predict(df_anomalie[["temperatur", "druck", "vibration"]])
            df_anomalie["label"] = df_anomalie["label"].map({1: "Normal", -1: "Anomalie"})

            anzahl = len(df_anomalie[df_anomalie["label"] == "Anomalie"])
            st.warning(f"⚠️ {anzahl} Anomalien gefunden")

            fig, ax = plt.subplots()
            normal = df_anomalie[df_anomalie["label"] == "Normal"]
            anomalie = df_anomalie[df_anomalie["label"] == "Anomalie"]
            ax.scatter(normal["temperatur"], normal["vibration"], color="blue", alpha=0.4, s=20, label="Normal")
            ax.scatter(anomalie["temperatur"], anomalie["vibration"], color="red", s=100, marker="x", linewidths=2,
                       label="Anomalie")
            ax.set_xlabel("Temperatur (°C)")
            ax.set_ylabel("Vibration (mm/s)")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Anomale Maschinen")
            st.dataframe(df_anomalie[df_anomalie["label"] == "Anomalie"])