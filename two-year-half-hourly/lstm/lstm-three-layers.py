import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
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

# Define the model with the specified parameters
def create_model(lstm_units_1=128, lstm_units_2=64, lstm_units_3=32, dense_units=32, learning_rate=0.001):
    model = Sequential([
        LSTM(lstm_units_1, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(lstm_units_2, activation='relu', return_sequences=True),
        LSTM(lstm_units_3, activation='relu'),
        Dense(dense_units, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# Create and train the model
model = create_model()

history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Make predictions
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler.inverse_transform(y_pred_scaled)

# Calculate metrics
mse = np.mean((y_test - y_pred.flatten())**2)
r2 = 1 - (np.sum((y_test - y_pred.flatten())**2) / np.sum((y_test - np.mean(y_test))**2))

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Save the final model and scaler
model.save('lstm_model_tuned.h5')
joblib.dump(scaler, 'lstm_scaler_tuned.joblib')

# Plot predictions vs actual
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('LSTM Model: Actual vs Predicted Energy Usage')
plt.xlabel('Time Steps')
plt.ylabel('Energy Usage')
plt.legend()
plt.tight_layout()
plt.savefig('two-year-half-hourly/lstm/lstm_predictions.png')
plt.show()

model.save('two-year-half-hourly/lstm/hyperparam-tuned-lstm_model.h5')
joblib.dump(scaler, 'two-year-half-hourly/lstm/lstm_scaler.joblib')