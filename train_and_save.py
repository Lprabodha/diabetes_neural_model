import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import joblib

# Replace this with your real training data
X_train = np.random.rand(100, 8)  # Example: 100 samples with 8 features
y_train = np.random.randint(0, 2, 100)  # Binary classification

# Fit the scaler on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the fitted scaler
joblib.dump(scaler, "scaler.pkl")

# Create and train a simple neural network
model = Sequential([
    Dense(16, input_shape=(8,), activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=10, batch_size=8, verbose=1)

# Save the trained model
model.save("diabetes_neural_model.h5")
