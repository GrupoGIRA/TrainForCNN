# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:43:17 2024

UNIVERSIDAD PEDAGOGICA Y TECNOLOGICA DE COLOMBIA
Ingeniería Electrónica
GRUPO DE INVESTIGACION EN ROBOTICA Y AUTOMATIZACION           
Autor: Jaime Andres Moya Africano

Descripcion: Este codigo permite entrenar una CNN tipo Lenet5 (). Esta red tiene 
dies clases de salida. utiliza un dataset Mnist para los numeros del 0 al 9 
con una resolucion de 32x32.

@author: Andres Moya
"""

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

# Carga del Dataset
image_dir = r'C:\Users\Andres Moya\Desktop\Grupo_Gira\17_CNN_MNIST_32X32_LENET5_V2'
Tamano = 32
image_size = (Tamano, Tamano)
batch_size = 256
ancho, alto = Tamano, Tamano
val = 0.2

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    r'C:\Users\Andres Moya\Desktop\Grupo_Gira\17_CNN_MNIST_32X32_LENET5_V2\Train_Mnist_32x32',
    validation_split=val,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Visualización del dataset
plt.figure(figsize=(32, 32))
for imagen, etiquetas in train_ds.take(1):
    for i in range(12):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(imagen[i].numpy().astype("uint8"))
        plt.title(int(etiquetas[i]))
        plt.axis("off")

# Asignación de datos para entrenamiento
datos_entrenamiento = []

for imagenes, etiquetas in train_ds.take(1):
    for i in range(len(imagenes)):
        imagen = cv2.resize(imagenes[i].numpy(), (ancho, alto))
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imagen = imagen.reshape(ancho, alto, 1)
        etiqueta = etiquetas[i].numpy()
        datos_entrenamiento.append([imagen, etiqueta])

# Características del dataset
datos_entrenamiento[0]
        
# Preparación de las variables X (entradas) y Y (etiquetas)
x = []
y = []

for imagen, etiqueta in datos_entrenamiento:
    x.append(imagen)
    y.append(etiqueta)

# Normalización de los datos de las X (imágenes)
x = np.array(x).astype(float)/ 255   

# Convertir etiquetas en arreglo simple
y = np.array(y)

# # Obtener una muestra de imágenes del conjunto de validación
num_images_to_show = 12

for x, y in val_ds.take(1):
    images = x[:num_images_to_show]
    labels = y[:num_images_to_show]

# # Visualizar las imágenes con sus etiquetas
# plt.figure(figsize=(12, 12))
# for i in range(num_images_to_show):
#     ax = plt.subplot(3, 4, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"), cmap="gray")
#     plt.title(f"etiqueta: {int(labels[i])}")
#     plt.axis("off")
# plt.show()

# Cantidad de datos de entrenamiento
print("Cantidad de datos, ancho, alto, Color", x.shape)

# Aumento de Datos
datagen = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True
)

datagen.fit(x)

# Arquitectura de la Red Neuronal
modeloCNN= tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(6, (5, 5), activation='relu', strides=(1,1), input_shape=(ancho, alto, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2)),
    
    tf.keras.layers.Conv2D(16, (5, 5), activation='relu', strides=(1,1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2)),
    
    tf.keras.layers.Flatten(),
    
    #Fully Connected
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(84, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
modeloCNN.summary()

# Compilación del modelo
modeloCNN.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Early Stopping para evitar sobreajuste
# early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# Entrenamiento del modelo
tensorboardCNN = TensorBoard(log_dir=r'C:\Users\Andres Moya\Desktop\Grupo_Gira\17_CNN_MNIST_32X32_LENET5_V2')
historial = modeloCNN.fit(x, y, batch_size=128,validation_split=val,epochs=50, callbacks=[tensorboardCNN])

# Guardado de modelos
modeloCNN.save('modelo.h5')
#modeloCNN.save_weights('PesosCNN.h5')
print("Modelo CNN guardado.")

# Validación del modelo
datos_entre = int(len(x) * (1 - val))

X_entrenamiento = x[:datos_entre]
X_validacion = x[datos_entre:]

y_entrenamiento = y[:datos_entre]
y_validacion = y[datos_entre:]


# Evaluamos el modelo
test_loss, test_acc = modeloCNN.evaluate(X_entrenamiento,y_entrenamiento)

print("-----------------------------------------------------------------------")
print(f"Precisión de las pruebas: {test_acc}")
print(f"Precisión de las pérdidas: {test_loss}")
print("-----------------------------------------------------------------------")

#--------------Cargar modelo y realizar predicción-----------------------------
modelo_cargado = load_model('modelo.h5')
print('Modelo cargado correctamente.')

# Realizar predicción
prediccion = modelo_cargado.predict(x)
prediccion = np.round(prediccion).flatten()

# Cálculo de la matriz de confusión con las etiquetas predichas en el conjunto de validación
y_pred_validacion = modeloCNN.predict(X_validacion)
y_pred_validacion = np.argmax(y_pred_validacion, axis=-1)  # Convierte las probabilidades en etiquetas

print("-------------------------------------------------------------------")
print("Valores de logits:")
print(y_pred_validacion)
print("Valores de labels:")
print(y_validacion)

print("-----------------------------------------------------------------------")
cm = confusion_matrix(y_validacion, y_pred_validacion)
print("Matriz de confusión sin cuantizar:")
print(cm)

#------------------- Evaluación del modelo Sin Cuantizar-----------------------
modelo_cargado.evaluate(X_validacion, y_validacion)

# Graficar la precisión de entrenamiento y validación
plt.figure()
plt.plot(historial.history['accuracy'], label='Entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Validación')
plt.title('Precisión Red Neuronal CNN "Lenet5"')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Graficar la pérdida de entrenamiento y validación
plt.figure()
plt.plot(historial.history['loss'], label='Entrenamiento')
plt.plot(historial.history['val_loss'], label='Validación')
plt.title('Pérdidas Red Neuronal CNN "Lenet5"')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()