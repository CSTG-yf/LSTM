import torch
import torch.nn as nn
import yaml
import os
import numpy as np
from datetime import datetime

from src.data.dataset import load_and_preprocess_data, create_data_loaders
from src.models.lstm import LSTMRegressor

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(model, train_loader, val_loader, config, device):
    """Train the LSTM model"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10  # Early stopping patience
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config['training']['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'outputs/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return train_losses, val_losses

def main():
    # Load configuration
    config = load_config('configs/model_config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and preprocess data
    print('Loading and preprocessing data...')
    data, scaler, dates = load_and_preprocess_data('Dataset')
    print(f'Data shape: {data.shape}')
    
    # Create data loaders
    print('Creating data loaders...')
    train_loader, val_loader, test_loader = create_data_loaders(
        data,
        config['training']['sequence_length'],
        config['data']['train_ratio'],
        config['data']['val_ratio'],
        config['training']['batch_size']
    )
    
    # Initialize model
    model = LSTMRegressor(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        output_size=config['model']['output_size'],
        dropout=config['model']['dropout']
    ).to(device)
    
    print('Model architecture:')
    print(model)
    
    # Train model
    print('Starting training...')
    train_losses, val_losses = train_model(model, train_loader, val_loader, config, device)
    
    # Save final model
    torch.save(model.state_dict(), 'outputs/final_model.pth')
    print('Model saved to outputs/final_model.pth')
    
    # Save scaler for future use in prediction
    import pickle
    with open('outputs/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print('Training completed!')

if __name__ == '__main__':
    main()