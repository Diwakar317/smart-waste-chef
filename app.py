from flask import Flask, render_template, request
import os
import json
from predictor import predict_food

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

with open('recipes.json', 'r') as f:
    RECIPES = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    file = request.files['image']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    predicted_item = predict_food(file_path)
    recipes = RECIPES.get(predicted_item, [])

    return render_template('result.html', recipes=recipes, predicted_item=predicted_item)

if __name__ == '__main__':
    app.run(debug=True)
