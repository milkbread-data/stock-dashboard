"""
Unit tests for the stock dashboard application.
"""
import pandas as pd
from unittest.mock import patch

# Import the functions to test
# app.df will be the global DataFrame loaded when app.py is imported.
from app import (
    is_valid_ticker,
    filter_dataframe,
    calculate_performance,
    prepare_export_data,
    get_ticker_data,
    get_cache_timestamp,
    RateLimiter,
    df as app_df # Import the global df for patching if necessary, or rely on app.df
)

def test_is_valid_ticker():
    """Test the ticker validation function."""
    # Valid tickers
    assert is_valid_ticker("AAPL") is True
    assert is_valid_ticker("MSFT") is True
    assert is_valid_ticker("BRK.B") is True  # Ticker with dot
    assert is_valid_ticker("BF-B") is False  # Ticker with hyphen - invalid by current regex
    
    # Invalid tickers
    assert is_valid_ticker("") is False  # Empty string
    assert is_valid_ticker("A" * 11) is False  # Too long
    assert is_valid_ticker("123") is False  # Starts with number
    assert is_valid_ticker("A..B") is False  # Consecutive dots
    assert is_valid_ticker("A B") is False  # Contains space
    assert is_valid_ticker("A@B") is False  # Invalid character
    assert is_valid_ticker("SYSTEM") is False  # In blocklist
    assert is_valid_ticker("CON") is False  # In blocklist
    assert is_valid_ticker(None) is False  # None value

def test_filter_dataframe(sample_stock_data):
    """Test dataframe filtering functionality."""    
    # Test date filtering
    start_date = '2023-01-03'
    end_date = '2023-01-07'
    tickers = ['AAPL', 'MSFT']
        
    # Patch the global df used by app.filter_dataframe
    with patch('app.df', sample_stock_data):
        # Call the actual filter_dataframe function from app.py
        filtered = filter_dataframe(start_date, end_date, tickers)
        
        # Check if filtered by date range
        assert not filtered.empty
        assert filtered['date'].min() >= pd.to_datetime(start_date)
        assert filtered['date'].max() <= pd.to_datetime(end_date)
        
        # Check if filtered by tickers
        assert set(filtered['Ticker'].unique()).issubset(set(tickers))
        
        # Test with no tickers (should return empty dataframe as per app.py logic)
        # app.filter_dataframe will filter by date first, then by tickers.
        # If tickers is empty, it will return data for all tickers within the date range.
        empty_ticker_filtered = filter_dataframe(start_date, end_date, [])
        assert not empty_ticker_filtered.empty # Should contain all tickers in sample_stock_data for the date range
        assert len(empty_ticker_filtered['Ticker'].unique()) == len(sample_stock_data['Ticker'].unique())

def test_calculate_performance():
    """Test performance calculation."""
    # Create a simple test dataframe
    dates = pd.date_range(start='2023-01-01', periods=5)
    tickers = ['AAPL', 'MSFT']
    
    data = []
    for i, ticker in enumerate(tickers):
        for j, date in enumerate(dates):
            data.append({
                'date': date,
                'Ticker': ticker,
                'close': 100 + (i * 10) + (j * 2)  # Each ticker starts at different base price
            })
    
    df = pd.DataFrame(data)
    # Calculate performance using the function from app.py
    performance_df = calculate_performance(df, tickers) 
    
    # Check if performance is calculated correctly
    assert isinstance(performance_df, pd.DataFrame)
    assert not performance_df.empty
    assert 'pct_change' in performance_df.columns

    for ticker in tickers:
        ticker_data = df[df['Ticker'] == ticker].sort_values('date')
        first_close = ticker_data.iloc[0]['close']
        last_close = ticker_data.iloc[-1]['close']
        expected_percent = ((last_close - first_close) / first_close) * 100
        
        # Get the actual last pct_change for the ticker from the result DataFrame
        ticker_perf_data = performance_df[performance_df['Ticker'] == ticker].sort_values('date')
        assert not ticker_perf_data.empty, f"No performance data found for {ticker} in result"
        actual_percent = ticker_perf_data['pct_change'].iloc[-1]
        
        assert abs(actual_percent - expected_percent) < 0.001  # Allow for floating point errors

def test_prepare_export_data(sample_stock_data):
    """Test data preparation for export."""
    start_date = '2023-01-03'
    end_date = '2023-01-07'
    tickers = ['AAPL', 'MSFT']
    
    with patch('app.df', sample_stock_data):
        # Call the actual prepare_export_data function from app.py
        result_df = prepare_export_data(start_date, end_date, tickers)
    
    # Check if result is a DataFrame
    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty
    assert 'daily_change' in result_df.columns # Check for one of the added columns
    assert result_df['date'].dtype == 'object' # Dates should be strings

    # Test with a date range that yields no data from sample_stock_data
    # This will likely cause app.prepare_export_data to fail with a KeyError
    # if not handled, which is a valid test outcome (revealing a bug in app.py).
    with patch('app.df', sample_stock_data):
        empty_result_df = prepare_export_data('2100-01-01', '2100-01-02', tickers)
        # If app.prepare_export_data handles empty filtered_df gracefully (e.g., returns empty df):
        assert empty_result_df.empty
        # Optionally, verify the structure of the empty DataFrame if the fix in app.py guarantees columns
        expected_cols = ['date', 'Ticker', 'open', 'high', 'low', 'close', 'volume', 'daily_change', 'daily_range']
        assert all(col in empty_result_df.columns for col in expected_cols)

def test_rate_limiter():
    """Test the rate limiter functionality."""
    limiter = RateLimiter()
    key = 'test_key'
    
    # First request should not be rate limited
    assert limiter.is_rate_limited(key, limit=2, window=1) is False
    
    # Second request should not be rate limited
    assert limiter.is_rate_limited(key, limit=2, window=1) is False
    
    # Third request should be rate limited
    assert limiter.is_rate_limited(key, limit=2, window=1) is True
    
    # Test with different keys
    assert limiter.is_rate_limited('another_key', limit=1, window=1) is False
    assert limiter.is_rate_limited('another_key', limit=1, window=1) is True

# Test the get_ticker_data function with mocks
@patch('app.get_cache_timestamp', return_value=12345)
def test_get_ticker_data(mock_timestamp, mock_yfinance):
    """Test the get_ticker_data function with mocks."""
    mock_ticker, _ = mock_yfinance
    
    # Call the function
    result = get_ticker_data('AAPL', 12345)
    
    # Check if the result contains expected keys
    assert isinstance(result, dict)
    assert 'current_price' in result
    assert 'recommendations' in result
    assert 'target_price' in result
    
    # Check if the mock was called correctly
    mock_ticker.assert_called_once_with('AAPL')
    
    # Check if the recommendations were processed correctly
    assert isinstance(result['recommendations'], pd.DataFrame)
    assert not result['recommendations'].empty
    assert 'Firm' in result['recommendations'].columns
    assert 'To Grade' in result['recommendations'].columns
    
    # Check if the income statement was processed correctly
    assert 'earnings' in result
    if result['earnings'] is not None:
        assert 'Net Income' in result['earnings'].columns
        assert 'EPS' in result['earnings'].columns

# Test the get_cache_timestamp function
def test_get_cache_timestamp():
    """Test the cache timestamp function."""
    timestamp1 = get_cache_timestamp()
    timestamp2 = get_cache_timestamp()
    
    # The timestamps should be the same within the same hour
    assert timestamp1 == timestamp2
    
    # The timestamp should be an integer
    assert isinstance(timestamp1, int)
