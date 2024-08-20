import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta
import tensorflow as tf

def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path, header=None, names=['date', 'time', 'energy_usage'])
    data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='%d/%m/%y %H:%M')
    data.set_index('timestamp', inplace=True)
    data.drop(['date', 'time'], axis=1, inplace=True)
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['day_of_year'] = data.index.dayofyear
    data['is_weekend'] = data.index.dayofweek.isin([5, 6]).astype(int)
    data['lag_1'] = data['energy_usage'].shift(1)
    data['lag_48'] = data['energy_usage'].shift(48)
    return data

def create_and_load_lstm_model(weights_path, input_shape):
    seq_length = 48  # 24 hours of half-hourly data means 48 data points to consider
    model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(64, return_sequences=True),
    LSTM(32, return_sequences=True),
    LSTM(24),
    Dense(32, activation='relu'),
    Dense(1)
])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.load_weights(weights_path)
    return model

def predict_svm(model, scaler, features):
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)

def predict_lstm(model, scaler, sequence):
    sequence_scaled = scaler.transform(sequence.reshape(-1, 1)).reshape(1, -1, 1)
    prediction_scaled = model.predict(sequence_scaled)
    return scaler.inverse_transform(prediction_scaled)[0, 0]

# Load data models and scalers
data = load_and_prepare_data('data-processed-timestamps.csv')
svm_model = joblib.load('model-testing/svm_model.joblib')
svm_scaler = joblib.load('model-testing/svm_scaler.joblib')
lstm_model = create_and_load_lstm_model('model-testing/four-layer-lstm_model.h5', input_shape=(48, 1))
lstm_scaler = joblib.load('model-testing/four-layer-lstm_scaler.joblib')

# XXX Date to predict goes here:
prediction_start = pd.Timestamp('2022-08-19 00:00:00') 

if prediction_start not in data.index:
    raise ValueError("Prediction start date not found in the data")

actual_data = data.loc[prediction_start:prediction_start + timedelta(hours=23.5)]

svm_features = actual_data[['hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend', 'lag_1', 'lag_48']]

lstm_sequence = data.loc[prediction_start - timedelta(hours=24):prediction_start - timedelta(minutes=30), 'energy_usage'].values

# Make predictions
svm_predictions = predict_svm(svm_model, svm_scaler, svm_features)
lstm_predictions = []
for _ in range(48):
    next_pred = predict_lstm(lstm_model, lstm_scaler, lstm_sequence)
    lstm_predictions.append(next_pred)
    lstm_sequence = np.roll(lstm_sequence, -1)
    lstm_sequence[-1] = next_pred

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Convert predictions to numpy arrays if they aren't already
actual_values = actual_data['energy_usage'].values
svm_pred_values = np.array(svm_predictions).flatten()
lstm_pred_values = np.array(lstm_predictions).flatten()

# Ensure all arrays have the same length
min_length = min(len(actual_values), len(svm_pred_values), len(lstm_pred_values))
actual_values = actual_values[:min_length]
svm_pred_values = svm_pred_values[:min_length]
lstm_pred_values = lstm_pred_values[:min_length]

# Get the datetime index as a numpy array
time_index = actual_data.index.to_numpy()[:min_length]

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(time_index, actual_values, label='Actual', marker='o')
plt.plot(time_index, svm_pred_values, label='SVM Prediction', marker='s')
plt.plot(time_index, lstm_pred_values, label='LSTM Prediction', marker='^')

plt.title('Energy Usage: Actual vs Predictions')
plt.xlabel('Time')
plt.ylabel('Energy Usage')
plt.legend()

# Format x-axis labels
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

plt.tight_layout()
plt.savefig('energy_predictions_comparison.png')
plt.show()

# Print results
for i in range(min_length):
    print(f"Time: {time_index[i]}, Actual: {actual_values[i]}, SVM: {svm_pred_values[i]}, LSTM: {lstm_pred_values[i]}")