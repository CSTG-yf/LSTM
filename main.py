# Main entry point for training and prediction

import argparse
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description='LSTM Visitor Prediction')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help='Mode: train or predict')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting training process...")
        from src.train import main as train_main
        train_main()
    elif args.mode == 'predict':
        print("Starting prediction process...")
        from src.predict import main as predict_main
        predict_main()

if __name__ == "__main__":
    main()