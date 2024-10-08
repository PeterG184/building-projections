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
import optuna

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

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameters to tune
    lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 256)
    lstm_units_2 = trial.suggest_int('lstm_units_2', 16, 128)
    dense_units = trial.suggest_int('dense_units', 8, 64)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    # Build the LSTM model with the suggested hyperparameters
    model = Sequential([
        LSTM(lstm_units_1, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(lstm_units_2, activation='relu'),
        Dense(dense_units, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    # Train the model
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=25,  # You might want to reduce this for faster tuning
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate the model
    val_loss = history.history['val_loss'][-1]  # Get the last validation loss
    return val_loss

# Create an Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)  # Adjust n_trials as needed

# Get the best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train the final model with the best hyperparameters
final_model = Sequential([
    LSTM(best_params['lstm_units_1'], activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(best_params['lstm_units_2'], activation='relu'),
    Dense(best_params['dense_units'], activation='relu'),
    Dense(1)
])

final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mse')

history = final_model.fit(
    X_train_scaled, y_train_scaled,
    epochs=1000, 
    batch_size=best_params['batch_size'],
    validation_split=0.1,
    verbose=1
)

# Make predictions
y_pred_scaled = final_model.predict(X_test_scaled)
y_pred = scaler.inverse_transform(y_pred_scaled)

# Calculate metrics
mse = np.mean((y_test - y_pred.flatten())**2)
r2 = 1 - (np.sum((y_test - y_pred.flatten())**2) / np.sum((y_test - np.mean(y_test))**2))

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Save the final model and scaler
final_model.save('lstm_model_tuned.h5')
joblib.dump(scaler, 'lstm_scaler_tuned.joblib')

# Plot the optimization history
optuna.visualization.plot_optimization_history(study)
plt.savefig('optimization_history.png')
plt.show()

# Plot the parameter importances
optuna.visualization.plot_param_importances(study)
plt.savefig('param_importances.png')
plt.show()

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

final_model.save('two-year-half-hourly/lstm/hyperparam-tuned-lstm_model.h5')
joblib.dump(scaler, 'two-year-half-hourly/lstm/hyperparam-lstm_scaler.joblib')