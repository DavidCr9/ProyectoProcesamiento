# Imports generales
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Keras (Autoencoder)
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers

# Scikit-learn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# TensorFlow
import tensorflow as tf

# CONFIGURACIÓN PARA REPRODUCIBILIDAD TOTAL
SEED = 203  # Semilla global para garantizar resultados repetibles
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

sns.set(style="whitegrid")


# 1. Carga de Datos y Preprocesamiento
print("Iniciando carga y preprocesamiento de datos...")

df = pd.read_csv("creditcard.csv")

# Se toman 50,000 registros para acelerar experimentos educativos
df = df.iloc[1:50000].copy()

# Variables predictoras y etiqueta
X = df.drop("Class", axis=1)
y = df["Class"].values  # 0 = normal, 1 = fraude

# Escalado MinMax 
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# División entre normales y fraudes
x_norm = X_scaled[y == 0]
x_fraud = X_scaled[y == 1]

print(f"Datos NO fraude: {x_norm.shape[0]} | Datos fraude: {x_fraud.shape[0]}")

# Gráfica 0: distribución de clases
plt.figure(figsize=(4, 3))
sns.countplot(x=df["Class"])
plt.xticks([0, 1], ["Normal", "Fraude"])
plt.title("Distribución de clases en el dataset")
plt.tight_layout()
plt.show()

# 2. Autoencoder
# input_dim = 30 características [Time + V1..V28 + Amount]
input_dim = X.shape[1]

# Arquitectura del Autoencoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(100, activation='tanh',
                activity_regularizer=regularizers.l1(1e-4))(input_layer)
encoded = Dense(50, activation='relu')(encoded)  # Bottleneck (representación latente)

decoded = Dense(50, activation='tanh')(encoded)
decoded = Dense(100, activation='tanh')(decoded)
output_layer = Dense(input_dim, activation='relu')(decoded)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer="adadelta", loss="mse")

print("Entrenando autoencoder con 2000 muestras normales...")

history = autoencoder.fit(
    x_norm[:2000],          # Solo normales para aprender la estructura típica
    x_norm[:2000],
    batch_size=256,
    epochs=10,
    shuffle=True,
    validation_split=0.20,
    verbose=1
)

# Gráfica 1: curva de entrenamiento
plt.figure(figsize=(5, 3))
plt.plot(history.history["loss"], label="loss entrenamiento")
plt.plot(history.history["val_loss"], label="loss validación")
plt.xlabel("Época")
plt.ylabel("MSE")
plt.title("Curva de entrenamiento del Autoencoder")
plt.legend()
plt.tight_layout()
plt.show()


# 3. Modelo reducido (encoder) y representación latente
# Modelo reducido: solo parte codificadora del Autoencoder
hid_rep = Sequential([
    autoencoder.layers[0],  # Input
    autoencoder.layers[1],  # Dense 100
    autoencoder.layers[2]   # Bottleneck 50
])

# Proyección a espacio latente
norm_latent = hid_rep.predict(x_norm[:3000])
fraud_latent = hid_rep.predict(x_fraud)

# Dataset combinado en espacio latente
rep_x = np.vstack([norm_latent, fraud_latent])
rep_y = np.hstack([np.zeros(norm_latent.shape[0]),
                   np.ones(fraud_latent.shape[0])])

print(f"Representación latente total: {rep_x.shape}")


# Gráfica 2: representación latente (primeras 2 dimensiones)
plt.figure(figsize=(6, 5))
plt.scatter(norm_latent[:, 0], norm_latent[:, 1],
            alpha=0.3, s=10, label="Normal")
plt.scatter(fraud_latent[:, 0], fraud_latent[:, 1],
            alpha=0.7, s=10, label="Fraude")
plt.xlabel("Latente 1")
plt.ylabel("Latente 2")
plt.title("Representación latente (primeras 2 dimensiones)")
plt.legend()
plt.tight_layout()
plt.show()

# 4. Entrenamiento y evaluación inicial del SVM
train_x, val_x, train_y, val_y = train_test_split(
    rep_x, rep_y, test_size=0.25, random_state=42
)

svm_model = SVC(kernel='linear')  # SVM lineal sobre espacio latente
svm_model.fit(train_x, train_y)

pred_y = svm_model.predict(val_x)

print("\n=========== EVALUACIÓN INICIAL SVM ===========")
cm_holdout = confusion_matrix(val_y, pred_y)
print(cm_holdout)
print("Accuracy:", round(accuracy_score(val_y, pred_y), 4))

# Gráfica 3: matriz de confusión
plt.figure(figsize=(4, 3))
sns.heatmap(cm_holdout,
            annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Fraude"],
            yticklabels=["Normal", "Fraude"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de confusión SVM (hold-out)")
plt.tight_layout()
plt.show()


# Validación cruzada K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

recalls, precisions, accuracies = [], [], []

print("\n=========== VALIDACIÓN CRUZADA (K=5) ===========")

for fold, (tr_idx, tst_idx) in enumerate(kf.split(rep_x)):

    X_tr, X_tst = rep_x[tr_idx], rep_x[tst_idx]
    y_tr, y_tst = rep_y[tr_idx], rep_y[tst_idx]

    svm_k = SVC(kernel="linear")
    svm_k.fit(X_tr, y_tr)

    pred = svm_k.predict(X_tst)

    recalls.append(recall_score(y_tst, pred, zero_division=0))
    precisions.append(precision_score(y_tst, pred, zero_division=0))
    accuracies.append(accuracy_score(y_tst, pred))

    print(f"Fold {fold+1} → Recall={recalls[-1]:.4f} | Precision={precisions[-1]:.4f}")

print("\nPromedios:")
print("Recall:", np.mean(recalls).round(4))
print("Precision:", np.mean(precisions).round(4))
print("Accuracy:", np.mean(accuracies).round(4))


# 6. Función de inferencia 
def create_transaction(time_value, values_list, amount):
    """
    Crea una transacción con exactamente 30 características:
    [Time] + [V1..V28] + [Amount]
    """
    if len(values_list) != 28:
        raise ValueError("values_list debe contener exactamente 28 valores (V1..V28)")
    return np.array([[time_value] + values_list + [amount]])


def check_new_transaction(raw_data_array, scaler, encoder_model, svm_model):
    """
    Flujo completo:
    1. Escalado con MinMaxScaler
    2. Proyección al espacio latente (Autoencoder)
    3. Clasificación con SVM
    """
    scaled = scaler.transform(raw_data_array)
    latent = encoder_model.predict(scaled, verbose=0)

    pred = svm_model.predict(latent)[0]
    score = svm_model.decision_function(latent)[0]

    resultado = "FRAUDE" if pred == 1 else "NORMAL"

    print("\n--- RESULTADO DE LA TRANSACCIÓN ---")
    print("Predicción:", resultado)
    print("SVM Score:", round(score, 4))

    if score > 0.5:
        print("→ ALTA CONFIANZA EN FRAUDE")
    elif score < -0.5:
        print("→ ALTA CONFIANZA EN NORMAL")

    return resultado

# 7. Inferencias de ejemplo
print("\n=========== INFERENCIAS DE EJEMPLO ===========")

# Ejemplo de fraude real
fraude_data = np.array([[  
    41233.0, -10.64, 5.91, -11.67, 8.80, -7.97, -3.58, -13.61,
    6.42, -7.36, -11.55, 6.78, -12.28, 0.44, -13.63, 0.17, -9.07,
    -15.42, -5.30, -0.61, 0.69, 2.57, 0.20, -1.66, 0.55, -0.02,
    0.35, 0.27, -0.15, 0.0
]])

print("\n--- Transacción 1 (Fraude) ---")
check_new_transaction(fraude_data, scaler, hid_rep, svm_model)

# Ejemplo de transacción normal
normal_data = create_transaction(
    time_value=50000.0,
    values_list=[0]*28,
    amount=10.0
)

print("\n--- Transacción 2 (Normal) ---")
check_new_transaction(normal_data, scaler, hid_rep, svm_model)
