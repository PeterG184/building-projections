import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import timedelta

def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    data['date_time'] = pd.to_datetime(data['date_time'])
    data.set_index('date_time', inplace=True)
    return data

def prepare_features(data):
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['day_of_year'] = data.index.dayofyear
    data['is_weekend'] = data.index.dayofweek.isin([5, 6]).astype(int)
    data['lag_1'] = data['total'].shift(1)
    data['lag_48'] = data['total'].shift(48)
    return data.dropna()

def predict_day(model, data, prediction_date):
    start_date = pd.Timestamp(prediction_date)
    end_date = start_date + timedelta(days=1)
    
    features = ['total', 'hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend', 'lag_1', 'lag_48']
    
    predictions = []
    current_time = start_date
    
    while current_time < end_date:
        # Prepare input for prediction
        input_data = data.loc[current_time, features].values.reshape(1, 1, -1)
        
        # Make prediction
        pred = model.predict(input_data)
        predictions.append(pred[0, 0])
        
        # Move to next time step
        current_time += timedelta(minutes=30)
        
        # Update data for next prediction
        data.loc[current_time, 'total'] = pred[0, 0]
        data.loc[current_time, 'hour'] = current_time.hour
        data.loc[current_time, 'day_of_week'] = current_time.dayofweek
        data.loc[current_time, 'month'] = current_time.month
        data.loc[current_time, 'day_of_year'] = current_time.dayofyear
        data.loc[current_time, 'is_weekend'] = int(current_time.dayofweek in [5, 6])
        data.loc[current_time, 'lag_1'] = data.loc[current_time - timedelta(minutes=30), 'total']
        data.loc[current_time, 'lag_48'] = data.loc[current_time - timedelta(hours=24), 'total']
    
    return predictions

# Main script
if __name__ == "__main__":
    # Load pre-trained model
    model = load_model('lstm-olufemi/models/lstm-olufemi.h5')
    
    # Load and prepare data
    data = load_and_prepare_data('2022-2024-report-combined.csv')
    data = prepare_features(data)
    
    # Predict for a specific day
    prediction_date = '2023-08-19'
    predictions = predict_day(model, data, prediction_date)
    
    # Plot results
    actual_data = data.loc[prediction_date:pd.Timestamp(prediction_date) + timedelta(days=1), 'total']
    plt.figure(figsize=(12, 6))
    plt.plot(actual_data.index, actual_data, label='Actual', marker='o')
    plt.plot(actual_data.index, predictions, label='Predicted', marker='s')
    plt.title(f'Energy Usage Prediction for {prediction_date}')
    plt.xlabel('Time')
    plt.ylabel('Energy Usage (kWh)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('lstm_prediction_comparison.png')
    plt.show()
    
    # Print results
    results = pd.DataFrame({'Actual': actual_data.values, 'Predicted': predictions}, index=actual_data.index)
    print(results)
    
    # Save results to CSV
    results.to_csv('lstm_prediction_results.csv')