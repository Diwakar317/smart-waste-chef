import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model once globally (not every time function is called)
model = MobileNetV2(weights='imagenet')

def load_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict_food(image_path):
    img = load_image(image_path)
    preds = model.predict(img)
    decoded = decode_predictions(preds, top=1)[0][0]
    return decoded[1]  # Return label name
