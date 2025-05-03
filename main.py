from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Set request timeout (e.g., 60 seconds)
app.config['TIMEOUT'] = 60

# Load the saved model
loaded_model = load_model("diabetes_neural_model.h5")

# Load the scaler used during training
scaler = StandardScaler()

# Fitting the scaler with training data before running the Flask app
# Assuming you have X_train, replace it with your actual training data
# scaler.fit(X_train)

# API endpoint for predictions
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
    # Run the Flask app
    app.run(debug=True, threaded=True, port=5000, host='0.0.0.0')