#Importar librerias
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
import os
import traceback


# Crear la app Flask
app = Flask(__name__)

# Rutas de los modelos guardados
SCALER_PATH = "modelos/scaler.pkl"
FEATURES_PATH = "modelos/feature_names.pkl"
ENCODER_PATH = "modelos/encoder.h5"
SVM_PATH = "modelos/svm.pkl"
AUTOENCODER_PATH = "modelos/autoencoder.h5"

# Cargar modelos
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)   #lista con las 30 columnas
encoder = load_model(ENCODER_PATH, compile=False)
svm_model = joblib.load(SVM_PATH)
autoencoder = load_model(AUTOENCODER_PATH, compile=False)

# Cargar threshold del autoencoder (basado en percentil 99 de error)
THRESHOLD_AE = joblib.load("modelos/threshold_ae.pkl")

# Cargar dataset para ejemplos y opciones
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "creditcard.csv")
if os.path.exists(CSV_PATH):
    # Se usa la misma porción del dataset con la que se entrenó el modelo
    df = pd.read_csv(CSV_PATH).iloc[1:50000].copy()

    #Variables originales (sin escalar) para ejemplos
    X_full = df.drop("Class", axis=1)
    y_full = df["Class"].values

    # 1) Escalar con el mismo scaler entrenado
    X_scaled_full = scaler.transform(X_full.values)

    # 2) Pasar por el encoder para obtener la representación latente
    latent_full = encoder.predict(X_scaled_full, verbose=0)

    # 3) SVM: scores y predicciones
    scores_full = svm_model.decision_function(latent_full)
    preds_full = svm_model.predict(latent_full)

    # 4) Autoencoder: reconstrucción y MSE en todas las muestras
    recon_full = autoencoder.predict(X_scaled_full, verbose=0)
    errors_full = np.mean((X_scaled_full - recon_full) ** 2, axis=1)

    # Elegir un NORMAL claro
    # Normal en etiquetas, SVM dice normal y AE lo ve dentro del umbral
    mask_normal_ok = (y_full == 0) & (preds_full == 0) & (errors_full <= THRESHOLD_AE)
    normal_idxs = np.where(mask_normal_ok)[0]

    if len(normal_idxs) > 0:
        # Tomamos el normal con menor error MSE (más "normal" para el AE)
        best_normal_local = np.argmin(errors_full[normal_idxs])
        best_normal_idx = normal_idxs[best_normal_local]
        normal_ejemplo = X_full.iloc[best_normal_idx]
    else:
        # Fallback: primer normal del dataset original
        normal_ejemplo = df[df["Class"] == 0].iloc[0].drop("Class")

       #Elegir un FRAUDE claro

    # Intentar: fraude que SVM ve como FRAUDE y AE ve ANÓMALO
    mask_fraud_strong = (y_full == 1) & (preds_full == 1) & (errors_full > THRESHOLD_AE)
    fraud_idxs = np.where(mask_fraud_strong)[0]

    # Si no hay ninguno, relajar condición: solo SVM = FRAUDE
    if len(fraud_idxs) == 0:
        mask_fraud_svm = (y_full == 1) & (preds_full == 1)
        fraud_idxs = np.where(mask_fraud_svm)[0]

    if len(fraud_idxs) > 0:
        # Tomamos el de score SVM más alto (más "fraudulento")
        best_fraud_local = np.argmax(scores_full[fraud_idxs])
        best_fraud_idx = fraud_idxs[best_fraud_local]
        fraude_ejemplo = X_full.iloc[best_fraud_idx]
    else:
        # Último recurso: primer fraude del dataset
        fraude_ejemplo = df[df["Class"] == 1].iloc[0].drop("Class")

else:
    df = None
    normal_ejemplo = None
    fraude_ejemplo = None

#Opciones por feature para el formulario con valores atípicos para selección rápida
feature_options = {}

if df is not None:
    for col in feature_names:
        serie = df[col].dropna()
        vals = np.unique(serie.values)

        # Reducir a máximo 15 valores representativos
        if len(vals) > 15:
            idxs = np.linspace(0, len(vals) - 1, 15, dtype=int)
            vals = vals[idxs]

        # Lo guardamos como lista de floats
        feature_options[col] = [float(v) for v in vals]
else:
    feature_options = {col: [0.0] for col in feature_names}


@app.route("/")
def home():
    """
    tipo = normal | fraude | manual
    """
    tipo = request.args.get("tipo", "normal")

    if df is not None:
        if tipo == "fraude" and fraude_ejemplo is not None:
            default_series = fraude_ejemplo
        elif tipo == "manual":
            default_series = None
        else:
            default_series = normal_ejemplo

        if default_series is not None:
            default_values = default_series.to_dict()
        else:
            default_values = {col: 0.0 for col in feature_names}
    else:
        tipo = "manual"
        default_values = {col: 0.0 for col in feature_names}

    return render_template(
        "index.html",
        # SVM
        resultado=None,
        score=None,
        confianza=None,
        # Autoencoder
        resultado_ae=None,
        mse_ae=None,
        threshold_ae=THRESHOLD_AE,
        # Otros
        default_values=default_values,
        feature_names=feature_names,
        feature_options=feature_options,
        ejemplo_tipo=tipo
    )

# Ruta para procesar el formulario de predicción
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Leer los valores del formulario
        input_dict = {}

        for col in feature_names:
            val_str = request.form.get(col, "").strip()
            if val_str == "":
                val = 0.0
            else:
                val = float(val_str)
            input_dict[col] = val

        input_df = pd.DataFrame([input_dict], columns=feature_names)

        # Esacalar con MinMaxScaler
        input_scaled = scaler.transform(input_df.values)

        # Autoencoder: reconstrucción y MSE
        reconstructed = autoencoder.predict(input_scaled, verbose=0)
        mse = np.mean((input_scaled - reconstructed) ** 2)

        if mse > THRESHOLD_AE:
            resultado_ae = "⚠️ Transacción Anómala Detectada"
        else:
            resultado_ae = "✓ Transacción Normal"

        # 3. Representación latente (encoder → para el SVM)
        latent = encoder.predict(input_scaled, verbose=0)

        # 4. Predicción con SVM
        score = svm_model.decision_function(latent)[0]
        pred = svm_model.predict(latent)[0]  # 0 = normal, 1 = fraude

        if pred == 1:
            resultado = "⚠️ Transacción Clasificada como FRAUDE"
        else:
            resultado = "✓ Transacción Clasificada como NORMAL"

        if score > 1:
            confianza = "Alta confianza en FRAUDE"
        elif score < -1:
            confianza = "Alta confianza en NORMAL"
        else:
            confianza = "Zona de incertidumbre (revisión recomendada)"

        return render_template(
            "index.html",
            # Resultados SVM
            resultado=resultado,
            score=f"{score:.4f}",
            confianza=confianza,
            # Resultados Autoencoder
            resultado_ae=resultado_ae,
            mse_ae=f"{mse:.6f}",
            threshold_ae=THRESHOLD_AE,
            # Otros
            default_values=input_dict,
            feature_names=feature_names,
            feature_options=feature_options,
            ejemplo_tipo="manual"
        )

    except Exception as e:
        print("ERROR en /predict:", e)
        traceback.print_exc()
        return render_template(
            "index.html",
            # SVM
            resultado="❌ Ocurrió un error al procesar la transacción.",
            score=None,
            confianza=None,
            # Autoencoder
            resultado_ae=None,
            mse_ae=None,
            threshold_ae=THRESHOLD_AE,
            # Otros
            default_values={col: 0.0 for col in feature_names},
            feature_names=feature_names,
            feature_options=feature_options,
            ejemplo_tipo="manual"
        )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

