import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Función para cargar un modelo CNN guardado
def load_cnn_model(model_path):
    return tf.keras.models.load_model(model_path)

# Cargar los modelos guardados
model_bathroom = load_cnn_model('model_bathroom.keras')
model_bedroom = load_cnn_model('model_bedroom.keras')
model_frontal = load_cnn_model('model_frontal.keras')
model_kitchen = load_cnn_model('model_kitchen.keras')

# Cargar el escalador y el modelo de regresión lineal guardados
scaler = joblib.load('scaler.joblib')
regressor = joblib.load('regressor.joblib')

# Función para cargar y procesar imágenes
def load_and_process_image(uploaded_file, target_size=(128, 128)):
    img = load_img(uploaded_file, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalizar
    return np.expand_dims(img_array, axis=0)  # Agregar una dimensión para el lote

# Extraer características de la imagen
def extract_features(model, image):
    return model.predict(image).flatten()

# Interfaz de Streamlit
st.title('Predicción de Precios de Casas')

# Formulario de entrada
bedrooms = st.number_input('Cantidad de Dormitorios', min_value=1, step=1)
bathrooms = st.number_input('Cantidad de Baños', min_value=1, step=1)
area = st.number_input('Área total del Terreno (en pies cuadrados)')
zipcode = st.number_input('Código Postal', step=1)

uploaded_bathroom = st.file_uploader('Sube una imagen del baño', type=['jpg', 'jpeg', 'png'])
uploaded_bedroom = st.file_uploader('Sube una imagen del dormitorio', type=['jpg', 'jpeg', 'png'])
uploaded_frontal = st.file_uploader('Sube una imagen frontal de la casa', type=['jpg', 'jpeg', 'png'])
uploaded_kitchen = st.file_uploader('Sube una imagen de la cocina', type=['jpg', 'jpeg', 'png'])

if st.button('Predecir'):
    if all([uploaded_bathroom, uploaded_bedroom, uploaded_frontal, uploaded_kitchen]):
        # Procesar y extraer características de las imágenes
        bathroom_image = load_and_process_image(uploaded_bathroom)
        bedroom_image = load_and_process_image(uploaded_bedroom)
        frontal_image = load_and_process_image(uploaded_frontal)
        kitchen_image = load_and_process_image(uploaded_kitchen)
        
        bathroom_features = extract_features(model_bathroom, bathroom_image)
        bedroom_features = extract_features(model_bedroom, bedroom_image)
        frontal_features = extract_features(model_frontal, frontal_image)
        kitchen_features = extract_features(model_kitchen, kitchen_image)

        # Combinar características de imágenes con datos de entrada
        features = np.hstack([bathroom_features, bedroom_features, frontal_features, kitchen_features])
        input_data = np.hstack([[bedrooms, bathrooms, area, zipcode], features])
        input_data = scaler.transform([input_data])  # Normalizar

        # Predecir el precio
        predicted_price = regressor.predict(input_data)[0]
        
        # Mostrar el precio predicho
        st.write(f'El precio predicho de la casa es: ${predicted_price:,.2f}')
    else:
        st.write('Por favor, sube todas las imágenes requeridas.')
