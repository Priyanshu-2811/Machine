import os
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load the trained model
MODEL_PATH = 'models/lgp_clf.pkl'

def load_model():
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
            print("Model loaded successfully!")
            return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model when the application starts
model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        model = load_model()
        if model is None:
            return jsonify({"error": "Model could not be loaded"}), 500
    
    try:
        data = request.json
        
        # Ensure the total number of features is 121
        features = np.zeros(121)  # Default values for all features
        
        # Map input fields to their corresponding indices
        feature_mapping = {
            "Age": 0,
            "Sex": 1,
            "Height": 2,
            "Weight": 3,
            "Heart Rate": 4
        }

        # Fill only relevant fields
        for key, index in feature_mapping.items():
            if key in data:
                features[index] = float(data[key])  # Ensure values are float

        # Reshape for model input
        features = features.reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0].tolist()

        # Map prediction to disease name
        disease_names = ["Normal", "Ischemic changes", "Old Anterior Myocardial Infarction", 
                         "Old Inferior Myocardial Infarction", "Sinus tachycardy", 
                         "Sinus bradycardy", "Ventricular Premature Contraction", 
                         "Supraventricular Premature Contraction", 
                         "Left bundle branch block", "Right bundle branch block", 
                         "Left ventricule hypertrophy", "Atrial Fibrillation", 
                         "Other", "Undefined"]

        try:
            disease_name = disease_names[int(prediction)]  # Ensure integer indexing
        except IndexError:
            disease_name = f"Class {prediction}"

        return jsonify({
            "prediction": int(prediction),
            "disease": disease_name,
            "probabilities": probabilities
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/feature_names', methods=['GET'])
def get_feature_names():
    # List only relevant features for UI input
    feature_names = ["Age", "Sex", "Height", "Weight", "Heart Rate"]
    return jsonify({"feature_names": feature_names})

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    app.run(debug=True)
