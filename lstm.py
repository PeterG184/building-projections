import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load the data
data = pd.read_csv('hourly-data.csv')

# Combine date and time into a single datetime column with the correct format
data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='%d/%m/%Y %H:%M:%S')
data.set_index('timestamp', inplace=True)
data.sort_index(inplace=True)

# Drop the now redundant date and time columns
data.drop(columns=['date', 'time'], inplace=True)

# If there are missing values, fill them or handle accordingly
data = data.fillna(method='ffill')

# Create lag features for the past 24 hours (48 readings)
for lag in range(1, 49):
    data[f'lag_{lag}'] = data['usage'].shift(lag)
data.dropna(inplace=True)

# Features and target
X = data[[f'lag_{lag}' for lag in range(1, 49)]]
y = data['usage']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape the input data to be 3D [samples, timesteps, features]
X_train_reshaped = np.reshape(X_train.values, (X_train.shape[0], 48, 1))
X_test_reshaped = np.reshape(X_test.values, (X_test.shape[0], 48, 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(48, 1)))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions
y_pred_train = model.predict(X_train_reshaped)
y_pred_test = model.predict(X_test_reshaped)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')
