import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt
import joblib

# Data loading
data = pd.read_csv('data-processed-timestamps.csv', header=None, names=['date', 'time', 'energy_usage'])
data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='%d/%m/%y %H:%M')
data.set_index('timestamp', inplace=True)
data.drop(['date', 'time'], axis=1, inplace=True)

# Prepare the data for LSTM (use past 24 hours to predict the next hour)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 48  # 24 hours of half-hourly data means 48 data points to consider
X, y = create_sequences(data['energy_usage'].values, seq_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
y_train_scaled = scaler.transform(y_train.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

# Define the model-building function
def create_lstm_model(units=50, learning_rate=0.001):
    model = Sequential([
        LSTM(units, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# Create the KerasRegressor
model = KerasRegressor(build_fn=create_lstm_model, verbose=1, learning_rate=0.001, units=32)

# Define the hyperparameters to tune
param_grid = {
    'units': [32, 50, 64, 128],
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'epochs': [30, 50, 100]
}

# Create the grid search object
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Perform the grid search
grid_result = grid.fit(X_train_scaled, y_train_scaled)

# Print the best parameters and score
print("Best parameters: ", grid_result.best_params_)
print("Best score: ", grid_result.best_score_)

# Get the best model
best_model = grid_result.best_estimator_.model

# Make predictions with the best model
y_pred_scaled = best_model.predict(X_test_scaled)
y_pred = scaler.inverse_transform(y_pred_scaled)

# Calculate metrics
mse = np.mean((y_test - y_pred.flatten())**2)
r2 = 1 - (np.sum((y_test - y_pred.flatten())**2) / np.sum((y_test - np.mean(y_test))**2))

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Plot predictions vs actual
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('LSTM Model (Tuned): Actual vs Predicted Energy Usage')
plt.xlabel('Time Steps')
plt.ylabel('Energy Usage')
plt.legend()
plt.tight_layout()
plt.savefig('two-year-half-hourly/lstm/lstm_tuned_predictions.png')
plt.show()

# Save the best model and scaler
best_model.save('two-year-half-hourly/lstm/lstm_tuned_model.h5')
joblib.dump(scaler, 'two-year-half-hourly/lstm/lstm_tuned_scaler.joblib')