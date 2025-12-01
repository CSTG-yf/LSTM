import torch
import yaml
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta

from src.data.dataset import load_and_preprocess_data
from src.models.lstm import LSTMRegressor

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model(model_path, config, device):
    """Load trained model"""
    model = LSTMRegressor(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        output_size=config['model']['output_size'],
        dropout=config['model']['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_next_days(model, last_sequence, scaler, device, num_days=7):
    """Predict visitor numbers for the next num_days"""
    predictions = []
    current_sequence = last_sequence.clone().to(device)
    
    with torch.no_grad():
        for _ in range(num_days):
            # Predict next day
            prediction = model(current_sequence.unsqueeze(0))
            predictions.append(prediction.cpu().numpy()[0])
            
            # Update sequence for next prediction
            # Remove first element and add prediction at the end
            current_sequence = torch.cat([current_sequence[1:], prediction.squeeze()])
    
    # Inverse transform predictions to original scale
    predictions = np.array(predictions)
    # We need to reshape for inverse transform (scaler expects 2D array)
    dummy_features = np.zeros((predictions.shape[0], scaler.scale_.shape[0]))
    dummy_features[:, 0] = predictions.flatten()  # Assuming first column is visitor count
    predictions_original_scale = scaler.inverse_transform(dummy_features)[:, 0]
    
    return predictions_original_scale

def main():
    # Load configuration
    config = load_config('configs/model_config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load trained model
    model = load_model('outputs/best_model.pth', config, device)
    print('Model loaded successfully')
    
    # Load and preprocess data
    print('Loading and preprocessing data...')
    data, scaler, dates = load_and_preprocess_data('Dataset')
    print(f'Data shape: {data.shape}')
    
    # Get the last sequence from the data for prediction
    seq_length = config['training']['sequence_length']
    last_sequence = torch.FloatTensor(data[-seq_length:])
    
    # Predict next 7 days
    print('Predicting next 7 days...')
    predictions = predict_next_days(model, last_sequence, scaler, device, num_days=7)
    
    # Print predictions
    last_date = pd.to_datetime(dates[-1])
    print("\nPredictions for the next 7 days:")
    for i, pred in enumerate(predictions):
        date = last_date + timedelta(days=i+1)
        print(f"{date.strftime('%Y-%m-%d')}: {pred:.2f} visitors")
    
    # Save predictions to file
    pred_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(predictions))]
    pred_df = pd.DataFrame({
        'date': pred_dates,
        'predicted_visitors': predictions
    })
    pred_df.to_csv('outputs/predictions.csv', index=False)
    print('\nPredictions saved to outputs/predictions.csv')

if __name__ == '__main__':
    main()