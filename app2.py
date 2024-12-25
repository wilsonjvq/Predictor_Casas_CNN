import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib  # Para guardar y cargar modelos de sklearn y escaladores

# Leer el archivo CSV y excluir las dos últimas filas
csv_path = r'C:\Users\ACER\Downloads\Houses-dataset-master\Houses-dataset-master\HousingInfo.csv'
df = pd.read_csv(csv_path)
df = df.iloc[:-2]  # Excluir las dos últimas filas
df = df.reset_index(drop=True)  # Resetear el índice

# Obtener las imágenes
def load_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalizar
    return img_array

# Crear arrays para las imágenes
def load_images(image_paths):
    images = [load_image(path) for path in image_paths]
    return np.array(images)

# Rutas a las imágenes
base_image_path = r'C:\Users\ACER\Downloads\Houses-dataset-master\Houses-dataset-master\Houses Dataset'
num_samples = len(df)  # Asegurarse de tener el mismo número de muestras
image_paths = {
    'bathroom': [base_image_path + f'/{i}_bathroom.jpg' for i in range(1, num_samples + 1)],
    'bedroom': [base_image_path + f'/{i}_bedroom.jpg' for i in range(1, num_samples + 1)],
    'frontal': [base_image_path + f'/{i}_frontal.jpg' for i in range(1, num_samples + 1)],
    'kitchen': [base_image_path + f'/{i}_kitchen.jpg' for i in range(1, num_samples + 1)],
}

# Cargar las imágenes
bathroom_images = load_images(image_paths['bathroom'])
bedroom_images = load_images(image_paths['bedroom'])
frontal_images = load_images(image_paths['frontal'])
kitchen_images = load_images(image_paths['kitchen'])

# Construir el modelo CNN mejorado
def build_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(1)  # Salida de un valor numérico
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Función de entrenamiento con EarlyStopping y guardado del modelo
def train_cnn_model(model, images, prices, model_path, batch_size=32, epochs=1):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(images, prices, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping])
    model.save(model_path)  # Guardar el modelo

# Entrenamiento de modelos para cada tipo de imagen
input_shape = (128, 128, 3)

# Modelo para imágenes de baño
model_bathroom = build_cnn_model(input_shape)
train_cnn_model(model_bathroom, bathroom_images, df['price'], 'model_bathroom.keras')

# Repetir para las otras imágenes
model_bedroom = build_cnn_model(input_shape)
train_cnn_model(model_bedroom, bedroom_images, df['price'], 'model_bedroom.keras')

model_frontal = build_cnn_model(input_shape)
train_cnn_model(model_frontal, frontal_images, df['price'], 'model_frontal.keras')

model_kitchen = build_cnn_model(input_shape)
train_cnn_model(model_kitchen, kitchen_images, df['price'], 'model_kitchen.keras')

# Función para cargar un modelo guardado
def load_cnn_model(model_path):
    return tf.keras.models.load_model(model_path)

# Cargar los modelos guardados
model_bathroom = load_cnn_model('model_bathroom.keras')
model_bedroom = load_cnn_model('model_bedroom.keras')
model_frontal = load_cnn_model('model_frontal.keras')
model_kitchen = load_cnn_model('model_kitchen.keras')

# Extraer características para cada tipo de imagen
def extract_features(model, image_paths):
    images = load_images(image_paths)
    features = model.predict(images)
    return features

# Extraer características
bathroom_features = extract_features(model_bathroom, image_paths['bathroom'])
bedroom_features = extract_features(model_bedroom, image_paths['bedroom'])
frontal_features = extract_features(model_frontal, image_paths['frontal'])
kitchen_features = extract_features(model_kitchen, image_paths['kitchen'])

# Combinar características de imágenes con datos de precios
features = np.hstack([bathroom_features, bedroom_features, frontal_features, kitchen_features])
X = np.hstack([df[['bedrooms', 'bathrooms', 'area', 'zipcode']].values, features])
y = df['price'].values

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Guardar el escalador
joblib.dump(scaler, 'scaler.joblib')

# Crear y entrenar el modelo de regresión lineal
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Guardar el modelo de regresión lineal
joblib.dump(regressor, 'regressor.joblib')

# Evaluar el modelo
score = regressor.score(X_test, y_test)
print(f'R^2 Score: {score}')

# Función para cargar el modelo de regresión lineal y el escalador
def load_regressor_and_scaler(regressor_path, scaler_path):
    regressor = joblib.load(regressor_path)
    scaler = joblib.load(scaler_path)
    return regressor, scaler

# Cargar el modelo de regresión lineal y el escalador
regressor, scaler = load_regressor_and_scaler('regressor.joblib', 'scaler.joblib')

# Transformar los datos de prueba usando el escalador cargado
X_test = scaler.transform(X_test)

# Evaluar el modelo cargado
loaded_score = regressor.score(X_test, y_test)
print(f'R^2 Score after loading: {loaded_score}')