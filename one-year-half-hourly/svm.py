import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('hourly-data.csv')

# Combine date and time into a single datetime column
data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='%d/%m/%Y %H:%M:%S')
data.set_index('timestamp', inplace=True)
data.sort_index(inplace=True)

# Drop the now redundant date and time columns
data.drop(columns=['date', 'time'], inplace=True)

# If there are missing values, fill them or handle accordingly
data = data.ffill()

print(data)

# Create lag features for the past 24 hours (48 readings)
for lag in range(1, 49):
    data[f'lag_{lag}'] = data['usage'].shift(lag)
data.dropna(inplace=True)

print(data)

# Features and target
X = data[[f'lag_{lag}' for lag in range(1, 49)]]
y = data['usage']

print(X)
print(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(X_train, X_test, y_train, y_test)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train, X_test)

# Initialize and train the model
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')
