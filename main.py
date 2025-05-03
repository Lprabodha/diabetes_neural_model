from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
loaded_model = load_model("diabetes_neural_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request as JSON
    data = request.get_json(force=True)
    
    # Extract data from JSON
    data_list = data['data']
    
    # Convert data to numpy array
    data_array = np.array(data_list)
    
    # Standardize the input data using the fitted scaler
    standardized_data = scaler.transform(data_array)
    
    # Make predictions
    predictions = loaded_model.predict(standardized_data)
    
    # Convert predictions to JSON format
    predictions_list = predictions.tolist()
    
    # Return the predictions
    return jsonify(predictions=predictions_list)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
