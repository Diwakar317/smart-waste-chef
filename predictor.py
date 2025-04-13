import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

# Load image and prepare it for the model
def load_image(image_path):
    img = Image.open(image_path).resize((224, 224)).convert('RGB')
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Load the pre-trained model
def predict_food(image_path):
    # Load the pre-trained model from TensorFlow Hub
    model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5")
    
    # Load and process the image
    image = load_image(image_path)
    
    # Run the model on the image
    predictions = model(image)

    # Ensure predictions are in the right format (convert to numpy if needed)
    predictions = predictions.numpy()  # Convert to numpy array for easier processing
    
    # Get the predicted class ID
    predicted_id = np.argmax(predictions, axis=-1)

    # Load labels from the text file
    with open("food_labels.txt", "r") as f:
        labels = f.read().splitlines()

    # Get the label corresponding to the prediction
    return labels[predicted_id[0]]  # Use the first prediction (batch size is 1)
