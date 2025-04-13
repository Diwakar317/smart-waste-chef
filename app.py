from flask import Flask, render_template, request
import os
import json
from predictor import predict_food

app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load recipes from the JSON file
with open('recipes.json', 'r') as f:
    RECIPES = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        file = request.files['image']
        if not file:
            return "No file uploaded", 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        
        # Save the uploaded file
        file.save(file_path)

        # Predict the food item
        predicted_item = predict_food(file_path)

        # Get the recipes related to the predicted item
        recipes = RECIPES.get(predicted_item, [])

        return render_template('result.html', recipes=recipes, predicted_item=predicted_item)
    
    except Exception as e:
        return f"Error during file upload or prediction: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
