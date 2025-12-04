import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os

# --- Load models ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
breed_model_path = os.path.join(BASE_DIR, "models", "breed_model.h5")
mood_model_path = os.path.join(BASE_DIR, "models", "mood_model.h5")

breed_model = tf.keras.models.load_model(breed_model_path)
mood_model = tf.keras.models.load_model(mood_model_path)


def predict_breed_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = breed_model.predict(img_array, verbose=0)
    class_idx = np.argmax(prediction)

    breed_names = {
        0: 'Abyssinian', 1: 'Bengal', 2: 'Birman', 3: 'Bombay', 
        4: 'British_Shorthair', 5: 'Egyptian_Mau', 6: 'Maine_Coon', 
        7: 'Persian', 8: 'Ragdoll', 9: 'Russian_Blue', 
        10: 'Siamese', 11: 'Sphynx'
    }

    return breed_names[class_idx]


def predict_mood_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = mood_model.predict(img_array, verbose=0)
    class_idx = np.argmax(prediction)

    mood_types = {
        0: 'curious', 1: 'eepy', 2: 'grumpy',
        3: 'happy', 4: 'zoomies'
    }

    return mood_types[class_idx]