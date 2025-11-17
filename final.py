# Imports generales
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import joblib

#Configuración para reproducibilidad total
SEED = 203
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#Crear una carpeta para guardar modelos si no existe
os.makedirs("modelos", exist_ok=True)

#Csrgar y preprocesar datos
df = pd.read_csv("creditcard.csv")

# Se usan 50,000 muestras para acelerar entrenamiento 
df = df.iloc[1:50000].copy()

# X = variables numéricas (Time, V1..V28, Amount)
# y = etiqueta objetivo (0 = normal, 1 = fraude)
X = df.drop("Class", axis=1)
y = df["Class"].values

# Nombres de columnas para inferencias futuras
feature_names = X.columns.tolist()

#Scaling MinMax
# Escala cada feature a [0,1] 
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.values)

#Guardar el scaler y los nombres de las features
# scaler.pkl → necesario para transformar futuras transacciones
# feature_names.pkl → asegura orden correcto de columnas
joblib.dump(scaler, "modelos/scaler.pkl")
joblib.dump(feature_names, "modelos/feature_names.pkl")

print("✔ Guardado scaler.pkl y feature_names.pkl")

# Separamos normales y fraudulentos para el autoencoder y análisis latente
x_norm = X_scaled[y == 0]
x_fraud = X_scaled[y == 1]

print(f"Normales: {x_norm.shape[0]}  |  Fraudes: {x_fraud.shape[0]}")

#Autoencoder 
input_dim = X.shape[1]  # Deben ser 30 features

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(100, activation='tanh',
                activity_regularizer=regularizers.l1(1e-4))(input_layer)
encoded = Dense(50, activation='relu')(encoded)  # Bottleneck de 50 dimensiones

# Decoder
decoded = Dense(50, activation='tanh')(encoded)
decoded = Dense(100, activation='tanh')(decoded)
output_layer = Dense(input_dim, activation='relu')(decoded)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer="adadelta", loss="mse")

print("Entrenando autoencoder con 2000 muestras normales...")

# Entrenamos solo con normales → modelo aprende la "forma" de transacciones normales
history = autoencoder.fit(
    x_norm[:2000],     # 2000 normales seleccionadas
    x_norm[:2000],
    batch_size=256,
    epochs=10,
    shuffle=True,
    validation_split=0.20,
    verbose=1
)

# Guardado el modelo completo del autoencoder
# autoencoder.h5 → incluye encoder + decoder + pesos
autoencoder.save("modelos/autoencoder.h5")
print("✔ Guardado modelos/autoencoder.h5")


#Calcuar Threshold del autoencoder MSE mediante errores de reconstrucción en normales
print("Calculando errores de reconstrucción en transacciones normales...")

recon_norm = autoencoder.predict(x_norm, verbose=0)

# Error MSE por muestra
errors = np.mean((x_norm - recon_norm) ** 2, axis=1)

print(f"Errores normales → min: {errors.min():.6f}  max: {errors.max():.6f}")

# Umbral del autoencoder:
# percentil 99 = solo 1% de normales tienen mayor error
threshold_ae = np.percentile(errors, 99)
print(f"Threshold autoencoder (percentil 99): {threshold_ae:.6f}")

#Guardar el threshold del autoencoder
# threshold_ae.pkl → permite detección rápida sin entrenar SVM
joblib.dump(threshold_ae, "modelos/threshold_ae.pkl")

print("✔ Guardado modelos/threshold_ae.pkl")


#Extraer y guardar el encoder representación latente
# hid_rep = encoder extraído del modelo completo
hid_rep = Sequential([
    autoencoder.layers[0],  # Input
    autoencoder.layers[1],  # Dense 100
    autoencoder.layers[2]   # Bottleneck 50
])

# Guardamos el encoder por separado
# encoder.h5 → genera representaciones latentes 50-D
encoder_input = autoencoder.input
encoder_output = autoencoder.layers[2].output
encoder_model = Model(encoder_input, encoder_output)
encoder_model.save("modelos/encoder.h5")

print("✔ Guardado modelos/encoder.h5")

#Calcular el espacio latente para normales y fraudulentos
norm_latent = hid_rep.predict(x_norm, verbose=0)
fraud_latent = hid_rep.predict(x_fraud, verbose=0)

rep_x = np.vstack([norm_latent, fraud_latent])
rep_y = np.hstack([
    np.zeros(norm_latent.shape[0]),
    np.ones(fraud_latent.shape[0])
])


#Entrenamiento del modelo SVM 
# SVM entrenado sobre el espacio latente — mucho más eficiente que sobre datos brutos
svm_model = SVC(kernel='linear')
svm_model.fit(rep_x, rep_y)

# Guardado del modelo SVM
# svm.pkl → clasificador final basado en features latentes
joblib.dump(svm_model, "modelos/svm.pkl")

print("✔ Guardado modelos/svm.pkl")

#Lista de archivos generados
print("\nTodo listo ✔ Archivos generados en carpeta 'modelos/'")
print(os.listdir("modelos"))
