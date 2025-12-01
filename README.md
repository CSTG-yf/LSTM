# LSTM Visitor Prediction

This project uses LSTM neural networks to predict visitor numbers based on historical data including visitor counts, order numbers, weather conditions, and holiday information.

## Project Structure

```
├── configs/                 # Configuration files
├── Dataset/                 # Data files
├── outputs/                 # Output files (models, predictions)
├── src/
│   ├── data/                # Data processing scripts
│   ├── models/              # Model definitions
│   ├── train.py            # Training script
│   └── predict.py          # Prediction script
├── train.py                # Main entry point
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model, run:
```bash
python train.py --mode train
```

This will:
- Load and preprocess all data files
- Split data into train/validation/test sets
- Train the LSTM model
- Save the best model to `outputs/best_model.pth`
- Save the final model to `outputs/final_model.pth`

### Making Predictions

To make predictions using the trained model, run:
```bash
python train.py --mode predict
```

This will:
- Load the trained model from `outputs/best_model.pth`
- Use the last sequence of data to predict visitor numbers for the next 7 days
- Save predictions to `outputs/predictions.csv`

## Configuration

The model configuration can be adjusted in `configs/model_config.yaml`:
- `input_size`: Number of features in the input data
- `hidden_size`: Number of features in the hidden state of the LSTM
- `num_layers`: Number of recurrent layers
- `output_size`: Number of output features
- `dropout`: Dropout rate for regularization
- Training parameters such as learning rate, batch size, and number of epochs

## Data Files

The Dataset directory contains the following CSV files:
- `visitor-numbers.csv`: Daily visitor counts for multiple locations
- `order_numbers.csv`: Daily order numbers for multiple locations
- `japan_2024_2025_workdays_binary.csv`: Binary indicator for holidays/workdays
- Weather data files for different regions

## Output Files

After training and prediction, the outputs directory will contain:
- `best_model.pth`: The best performing model during training
- `final_model.pth`: The final model after training completes
- `scaler.pkl`: Scaler used for data normalization (needed for prediction)
- `predictions.csv`: Predicted visitor numbers for the next 7 days