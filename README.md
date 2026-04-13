# ml-portfolio
# ML Portfolio — Hydraulikpumpen Predictive Maintenance

## Projekt
Vorhersage der Ausfallwahrscheinlichkeit von Hydraulikpumpen
basierend auf Sensordaten (Druck, Temperatur, Durchfluss, Vibration, Betriebsstunden).

## Daten
- 1.000 simulierte Hydraulikpumpen
- 5 Features: Druck, Temperatur, Durchfluss, Vibration, Betriebsstunden
- Zielwert: Ausfallwahrscheinlichkeit (0–1)

## Modelle & Ergebnisse

| Modell | RMSE | R² |
|--------|------|-----|
| XGBoost v1 (Default) | 0.0581 | 0.587 |
| XGBoost v2 (Early Stopping) | 0.0531 | 0.655 |

## Wichtigste Erkenntnisse
- Early Stopping verbessert RMSE um 8.6%
- Wichtigste Features: Druck, Temperatur, Durchfluss
- Nächster Schritt: LightGBM Vergleich + SHAP Erklärbarkeit

## Relevanz für die Industrie
Hydraulikpumpen wie bei Bosch Rexroth produzieren genau diese Sensordaten.
Ein solches Modell könnte ungeplante Ausfälle vorhersagen und Wartungskosten senken.