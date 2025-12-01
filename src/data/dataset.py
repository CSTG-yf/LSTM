import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class LSTMDataset(Dataset):
    def __init__(self, data, seq_length, target_col_index=0):
        self.data = data
        self.seq_length = seq_length
        self.target_col_index = target_col_index
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, index):
        # Return a sequence of data and the corresponding target (only the visitor count)
        return (
            torch.FloatTensor(self.data[index:index+self.seq_length]),
            torch.FloatTensor([self.data[index+self.seq_length, self.target_col_index]])
        )

def load_and_preprocess_data(data_dir):
    """
    Load and preprocess all data files with proper handling of date formats and missing data
    """
    print("[INFO] Loading data files...")
    
    # Load visitor numbers (date format: 2024/1/1)
    visitor_df = pd.read_csv(os.path.join(data_dir, 'visitor-numbers.csv'))
    visitor_df['date'] = pd.to_datetime(visitor_df['date'], format='%Y/%m/%d')
    visitor_df = visitor_df.sort_values('date').reset_index(drop=True)
    print(f"[INFO] Visitor data loaded: {len(visitor_df)} records")
    
    # Load order numbers (date format: 2024/1/1)
    order_df = pd.read_csv(os.path.join(data_dir, 'order_numbers.csv'))
    order_df['date'] = pd.to_datetime(order_df['date'], format='%Y/%m/%d')
    order_df = order_df.sort_values('date').reset_index(drop=True)
    print(f"[INFO] Order data loaded: {len(order_df)} records")
    
    # Load holiday data (date format: 2024/1/1)
    holiday_df = pd.read_csv(os.path.join(data_dir, 'japan_2024_2025_workdays_binary.csv'))
    holiday_df.rename(columns={'Date': 'date'}, inplace=True)
    holiday_df['date'] = pd.to_datetime(holiday_df['date'], format='%Y/%m/%d')
    holiday_df = holiday_df.sort_values('date').reset_index(drop=True)
    holiday_df.rename(columns={'IsWorkingDay': 'is_working_day'}, inplace=True)
    print(f"[INFO] Holiday data loaded: {len(holiday_df)} records")
    
    # Load weather data - Mt Fuji
    weather_fuji_df = pd.read_csv(os.path.join(data_dir, 'weather_mt_fuji.csv'))
    weather_fuji_df['date'] = pd.to_datetime(weather_fuji_df['date'], format='%Y/%m/%d')
    weather_fuji_df = weather_fuji_df.sort_values('date').reset_index(drop=True)
    # One-hot encode weather_type for mt_fuji
    weather_fuji_weather = pd.get_dummies(weather_fuji_df['weather_type'], prefix='mt_fuji_weather')
    weather_fuji_df = pd.concat([weather_fuji_df[['date']], weather_fuji_weather], axis=1)
    print(f"[INFO] Mt Fuji weather data loaded: {len(weather_fuji_df)} records")
    
    # Load weather data - Kyoto/Nara
    weather_kyoto_df = pd.read_csv(os.path.join(data_dir, 'weather_kyoto_nara.csv'))
    weather_kyoto_df['date'] = pd.to_datetime(weather_kyoto_df['date'], format='%Y/%m/%d')
    weather_kyoto_df = weather_kyoto_df.sort_values('date').reset_index(drop=True)
    # One-hot encode weather_type for kyoto_nara
    weather_kyoto_weather = pd.get_dummies(weather_kyoto_df['weather_type'], prefix='kyoto_nara_weather')
    weather_kyoto_df = pd.concat([weather_kyoto_df[['date']], weather_kyoto_weather], axis=1)
    print(f"[INFO] Kyoto/Nara weather data loaded: {len(weather_kyoto_df)} records")
    
    # Load weather data - Sea of Kyoto
    weather_sea_df = pd.read_csv(os.path.join(data_dir, 'weather_sea_of_kyoto.csv'))
    weather_sea_df['date'] = pd.to_datetime(weather_sea_df['date'], format='%Y/%m/%d')
    weather_sea_df = weather_sea_df.sort_values('date').reset_index(drop=True)
    # One-hot encode weather_type for sea_of_kyoto
    weather_sea_weather = pd.get_dummies(weather_sea_df['weather_type'], prefix='sea_of_kyoto_weather')
    weather_sea_df = pd.concat([weather_sea_df[['date']], weather_sea_weather], axis=1)
    print(f"[INFO] Sea of Kyoto weather data loaded: {len(weather_sea_df)} records")
    
    # Merge all dataframes on date
    merged_df = visitor_df.merge(order_df, on='date', how='outer')
    merged_df = merged_df.merge(holiday_df[['date', 'is_working_day']], on='date', how='outer')
    merged_df = merged_df.merge(weather_fuji_df, on='date', how='outer')
    merged_df = merged_df.merge(weather_kyoto_df, on='date', how='outer')
    merged_df = merged_df.merge(weather_sea_df, on='date', how='outer')
    
    # Sort by date and reset index
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    print(f"[INFO] Merged data shape: {merged_df.shape}")
    
    # Remove rows with all NaN (incomplete rows at the end)
    merged_df = merged_df.dropna(how='all')
    print(f"[INFO] After removing empty rows: {len(merged_df)} records")
    
    # Handle missing values: forward fill then backward fill
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].ffill().bfill()
    print(f"[INFO] Missing values handled")
    
    # Select relevant features for LSTM (exclude non-numeric and ID columns)
    # Keep all visitor and order data plus key weather features and working day indicator
    exclude_cols = ['date']
    feature_columns = [col for col in merged_df.columns if col not in exclude_cols]
    features = merged_df[feature_columns].values
    
    print(f"[INFO] Selected {len(feature_columns)} features for training")
    print(f"[INFO] Feature columns: {feature_columns}")
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, scaler, merged_df['date'].values

def create_data_loaders(data, seq_length, train_ratio, val_ratio, batch_size):
    """
    Create train, validation and test data loaders
    """
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Create datasets (assuming visitor count is the first column, index 0)
    train_dataset = LSTMDataset(train_data, seq_length, target_col_index=0)
    val_dataset = LSTMDataset(val_data, seq_length, target_col_index=0)
    test_dataset = LSTMDataset(test_data, seq_length, target_col_index=0)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader