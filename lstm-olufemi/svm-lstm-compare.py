import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta
import tensorflow as tf

def load_and_prepare_data(file_path):
    # Load the dataset with appropriate column names
    data = pd.read_csv(file_path, header=0, names=['index', 'datetime', 'col3', 'col4', 'power_usage'])
    
    # Convert datetime column to pandas datetime type and set as index
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    
    # Create additional features as needed
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['day_of_year'] = data.index.dayofyear
    data['is_weekend'] = data.index.dayofweek.isin([5, 6]).astype(int)
    data['lag_1'] = data['power_usage'].shift(1)
    data['lag_48'] = data['power_usage'].shift(48)
    
    return data.dropna()

def prepare_lstm_input(data, prediction_start):
    features = ['power_usage', 'hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend', 'lag_1', 'lag_48']
    
    # Get the row for the prediction start time
    row = data.loc[prediction_start, features].values
    
    # Reshape to match the expected input shape (1, 1, 8)
    return row.reshape(1, 1, -1)

def create_and_load_lstm_model():
    model = Sequential()
    model.add(LSTM(100, input_shape=(1, 8)))  # Change to 8 features
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def predict_svm(model, scaler, features):
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)

def predict_lstm(model, sequence):
    prediction = model.predict(sequence)
    return prediction[0, 0]

# Load data models and scalers
data = load_and_prepare_data('lstm-olufemi/2022-2024-report-combined.csv')
svm_model = joblib.load('lstm-olufemi/models/svm_model.joblib')
svm_scaler = joblib.load('lstm-olufemi/models/svm_scaler.joblib')

# Load and compile the new LSTM model
lstm_model = create_and_load_lstm_model()
lstm_model.load_weights('lstm-olufemi/models/lstm-olufemi.h5')

# XXX Date to predict goes here:
prediction_start = pd.Timestamp('2022-08-19 00:00:00')

if prediction_start not in data.index:
    raise ValueError("Prediction start date not found in the data")

actual_data = data.loc[prediction_start:prediction_start + timedelta(hours=23.5)]

svm_features = actual_data[['hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend', 'lag_1', 'lag_48']]

lstm_sequence = data.loc[prediction_start - timedelta(hours=24):prediction_start - timedelta(minutes=30), 'power_usage'].values

# Make predictions
svm_predictions = predict_svm(svm_model, svm_scaler, svm_features)

lstm_predictions = []
current_time = prediction_start
for _ in range(48):
    lstm_sequence = prepare_lstm_input(data, current_time)
    next_pred = predict_lstm(lstm_model, lstm_sequence)
    lstm_predictions.append(next_pred)
    
    # Move to the next time step
    current_time += timedelta(minutes=30)
    
    # Update the data for the next prediction
    data.loc[current_time, 'power_usage'] = next_pred
    data.loc[current_time, 'hour'] = current_time.hour
    data.loc[current_time, 'day_of_week'] = current_time.dayofweek
    data.loc[current_time, 'month'] = current_time.month
    data.loc[current_time, 'day_of_year'] = current_time.dayofyear
    data.loc[current_time, 'is_weekend'] = int(current_time.dayofweek in [5, 6])
    data.loc[current_time, 'lag_1'] = data.loc[current_time - timedelta(minutes=30), 'power_usage']
    data.loc[current_time, 'lag_48'] = data.loc[current_time - timedelta(hours=24), 'power_usage']

# Create a DataFrame with all the data
results = pd.DataFrame({
    'Actual': actual_data['power_usage'],
    'SVM Prediction': svm_predictions,
    'LSTM Prediction': lstm_predictions
}, index=actual_data.index)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(results.index, results['Actual'], label='Actual', marker='o')
plt.plot(results.index, results['SVM Prediction'], label='SVM Prediction', marker='s')
plt.plot(results.index, results['LSTM Prediction'], label='LSTM Prediction', marker='^')
plt.title('Energy Usage: Actual vs Predictions')
plt.xlabel('Time')
plt.ylabel('Energy Usage')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('energy_predictions_comparison.png')
plt.show()

print(results)
