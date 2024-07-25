import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Data preprocessing
data = pd.read_csv('two-year-half-hourly/data-processed-timestamps.csv', header=None, names=['date', 'time', 'energy_usage'])
data['timestamp'] = pd.to_datetime(data['date'] + ' ' + data['time'], format='%d/%m/%y %H:%M')
data.set_index('timestamp', inplace=True)
data.drop(['date', 'time'], axis=1, inplace=True)

# Create features
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month
data['day_of_year'] = data.index.dayofyear
data['is_weekend'] = data.index.dayofweek.isin([5, 6]).astype(int)
data['lag_1'] = data['energy_usage'].shift(1)
data['lag_48'] = data['energy_usage'].shift(48)

# Drop rows with NaN values created by lag features
data.dropna(inplace=True)

X = data[['hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend', 'lag_1', 'lag_48']]
y = data['energy_usage']

# Split data into training and testing sets and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
svm_model = SVR(kernel='rbf', C=100, epsilon=0.1)
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Plot predictions vs actual
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.title('SVM Model: Actual vs Predicted Energy Usage')
plt.xlabel('Timestamp')
plt.ylabel('Energy Usage')
plt.legend()
plt.tight_layout()
plt.savefig('two-year-half-hourly/svm/svm_predictions.png')
plt.show()

joblib.dump(svm_model, 'two-year-half-hourly/svm/svm_model.joblib')
joblib.dump(scaler, 'two-year-half-hourly/svm/svm_scaler.joblib')