import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

def load_image(image_path):
    img = Image.open(image_path).resize((224, 224)).convert('RGB')
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict_food(image_path):
    model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5")
    image = load_image(image_path)
    predictions = model(image)
    predicted_id = np.argmax(predictions)

    with open("food_labels.txt", "r") as f:
        labels = f.read().splitlines()

    return labels[predicted_id % len(labels)]  # Simple matching (mocked)
