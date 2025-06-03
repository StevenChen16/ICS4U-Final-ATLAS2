import pandas as pd
import numpy as np
import talib as ta
from src.data import (
    rolling_normalize, 
    calculate_connors_rsi, 
    apply_kalman_filter, 
    apply_fft_filter_rolling
)

def load_and_prepare_local_data(file_path, window=21):
    """
    Load and prepare local stock data with technical indicators and normalization
    
    Parameters:
    file_path (str): Path to the CSV file
    window (int): Size of rolling window for normalization
    
    Returns:
    pd.DataFrame: Processed stock data with technical indicators
    """
    
    # Load the CSV file
    print(f"Loading data from {file_path}...")
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")
    
    # Check if required columns exist
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert Date column to datetime and set as index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Sort by date to ensure chronological order
    data = data.sort_index()
    
    # Keep only the required columns for processing
    # Note: You can add other columns from your CSV if needed
    price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # If you want to keep additional columns from your CSV, uncomment and modify:
    # additional_columns = ['Money', 'Avg', 'High_limit', 'Low_limit', 'Pre_close', 'Factor']
    # available_additional = [col for col in additional_columns if col in data.columns]
    # price_columns.extend(available_additional)
    
    stock = data[price_columns].copy()
    
    # Ensure we have data
    if stock.empty:
        raise ValueError("No data available after loading")
    
    # Convert price columns to float64
    stock = stock.astype({col: np.float64 for col in price_columns})
    
    # Remove any rows with NaN values in critical columns
    stock = stock.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    if stock.empty:
        raise ValueError("No valid data after removing NaN values")
    
    print(f"Data loaded successfully. Shape: {stock.shape}")
    print(f"Date range: {stock.index.min()} to {stock.index.max()}")
    
    # Calculate basic technical indicators
    print("Calculating technical indicators...")
    
    try:
        stock['MA5'] = ta.SMA(stock['Close'].values, timeperiod=5)
        stock['MA20'] = ta.SMA(stock['Close'].values, timeperiod=20)
        stock['MACD'], stock['MACD_Signal'], stock['MACD_Hist'] = ta.MACD(stock['Close'].values)
        stock['RSI'] = ta.RSI(stock['Close'].values)
        stock['Upper'], stock['Middle'], stock['Lower'] = ta.BBANDS(stock['Close'].values)
        stock['Volume_MA5'] = ta.SMA(stock['Volume'].values, timeperiod=5)
        
        # Add Connors RSI
        stock['CRSI'] = calculate_connors_rsi(stock)
        
        # Add Kalman Filter estimates
        stock['Kalman_Price'], stock['Kalman_Trend'] = apply_kalman_filter(stock)
        
        # Add FFT filtered prices
        stock['FFT_21'] = apply_fft_filter_rolling(stock, 21)
        stock['FFT_63'] = apply_fft_filter_rolling(stock, 63)
        
    except Exception as e:
        print(f"Warning: Error calculating some technical indicators: {str(e)}")
        # Continue with available indicators
    
    # Forward fill any NaN values from indicators
    stock = stock.ffill()
    
    # Backward fill any remaining NaN values at the beginning
    stock = stock.bfill()
    
    # Apply rolling window normalization to all columns
    print("Applying rolling window normalization...")
    columns_to_normalize = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA5', 'MA20', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'RSI', 'Upper', 'Middle', 'Lower', 'Volume_MA5',
        'CRSI', 'Kalman_Price', 'Kalman_Trend',
        'FFT_21', 'FFT_63'
    ]
    
    for col in columns_to_normalize:
        if col in stock.columns:
            stock[col] = rolling_normalize(stock[col], window=window)
    
    print("Data preparation completed!")
    return stock

def main():
    """
    Main function to process the local CSV file
    """
    # File path
    file_path = 'data_minute/000001.XSHE.csv'
    
    try:
        # Process the data
        processed_data = load_and_prepare_local_data(file_path, window=21)
        
        # Display basic information
        print("\n" + "="*50)
        print("PROCESSED DATA SUMMARY")
        print("="*50)
        print(f"Shape: {processed_data.shape}")
        print(f"Columns: {list(processed_data.columns)}")
        print("\nFirst 5 rows:")
        print(processed_data.head())
        print("\nLast 5 rows:")
        print(processed_data.tail())
        print("\nData types:")
        print(processed_data.dtypes)
        print("\nBasic statistics:")
        print(processed_data.describe())
        
        # Save processed data
        output_file = 'processed_000001_XSHE.csv'
        processed_data.to_csv(output_file)
        print(f"\nProcessed data saved to: {output_file}")
        
        return processed_data
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None

if __name__ == "__main__":
    # Run the main function
    result = main()