import numpy as np
import pandas as pd
import joblib
from keras.models import load_model

# =======================
# Cargar modelos
# =======================
scaler = joblib.load("modelos/scaler.pkl")                 # ✔ scaler REAL
threshold = joblib.load("modelos/threshold.pkl")           # ✔ número float
feature_means = joblib.load("modelos/promedios_normales.pkl")  # ✔ Series
autoencoder = load_model("modelos/autoencoder.h5", compile=False)

# Cargar columnas en el orden correcto
df = pd.read_csv("creditcard.csv").iloc[1:50000]
feature_columns = df.drop("Class", axis=1).columns


# =======================
# FUNCIÓN DE PRUEBA
# =======================
def mse_transaccion(hora, monto):
    d = feature_means.copy()             # es un pandas.Series
    d["Time"] = float(hora)
    d["Amount"] = float(monto)

    # Convertir a DataFrame en el ORDEN correcto
    X = pd.DataFrame([d[feature_columns]])

    # Aplicar scaler
    Xs = scaler.transform(X)             # ✔ ahora sí funciona

    # Pasar por el autoencoder
    rec = autoencoder.predict(Xs, verbose=0)

    mse = np.mean((Xs - rec) ** 2)
    return mse


# =======================
# PRUEBA AUTOMÁTICA
# =======================
for hora in [0, 5, 10, 20, 40]:
    for monto in [10, 100, 500, 1000, 2000, 2500]:
        m = mse_transaccion(hora, monto)
        print(f"Hora={hora}, Monto={monto} -> MSE={m:.6f} | {'ANÓMALA' if m > threshold else 'Normal'}")
