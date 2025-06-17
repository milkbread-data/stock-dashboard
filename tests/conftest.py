"""
Test configuration and fixtures for the stock dashboard application.
"""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set test environment variables
os.environ['FLASK_ENV'] = 'testing'
os.environ['TESTING'] = 'True'

# Sample test data
@pytest.fixture
def sample_stock_data():
    """Generate sample stock data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    data = []
    for ticker in tickers:
        for date in dates:
            close = 100 + np.random.rand() * 100  # Random price between 100 and 200
            data.append({
                'date': date,
                'Ticker': ticker,
                'open': close * 0.99,
                'high': close * 1.01,
                'low': close * 0.98,
                'close': close,
                'volume': int(1e6 * np.random.rand() + 1e5)
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def mock_yfinance():
    """Mock yfinance module for testing."""
    with patch('yfinance.Ticker') as mock_ticker, \
         patch('yfinance.download') as mock_download:
        
        # Mock yf.download
        mock_df = pd.DataFrame({
            'Open': [150.0, 151.0, 152.0],
            'High': [152.0, 153.0, 154.0],
            'Low': [149.0, 150.0, 151.0],
            'Close': [151.0, 152.0, 153.0],
            'Volume': [1000000, 1200000, 1100000]
        }, index=pd.date_range(start='2023-01-01', periods=3))
        
        mock_download.return_value = mock_df
        
        # Mock Ticker object
        mock_ticker.return_value.info = {
            'currentPrice': 150.0,
            'targetMeanPrice': 160.0,
            'recommendationMean': 1.5,
            'recommendationKey': 'buy',
            'longBusinessSummary': 'A sample business summary.',
            'trailingPE': 25.0,
            'dividendYield': 0.005
        }
        
        # Mock recommendations
        mock_recs = pd.DataFrame({
            'Firm': ['Firm1', 'Firm2', 'Firm3', 'Firm4', 'Firm5'],
            'To Grade': ['Buy', 'Hold', 'Buy', 'Strong Buy', 'Sell'],
            'Action': ['maintains', 'maintains', 'upgrades', 'maintains', 'downgrades']
        }, index=pd.DatetimeIndex(pd.date_range(end=datetime.now().normalize(), periods=5, freq='B', name='Date')))
        # yfinance often returns a DatetimeIndex for recommendations
        # Using normalize() to remove time part, and freq='B' for business days as an example

        mock_ticker.return_value.recommendations = mock_recs
        
        # Mock income statement
        # yfinance income_stmt index is typically the fiscal year end period (datetime or string)
        # For simplicity, using strings representing years.
        mock_income = pd.DataFrame({
            'Net Income': [100e9, 110e9, 120e9],
            'Basic EPS': [5.0, 5.5, 6.0]
        }, index=pd.Index([datetime(2021, 12, 31), datetime(2022, 12, 31), datetime(2023, 12, 31)], name='Period'))
        # Or, if your yfinance version returns strings:
        # }, index=pd.Index(['2021', '2022', '2023']))
        mock_ticker.return_value.income_stmt = mock_income
        
        yield mock_ticker, mock_download

@pytest.fixture
def test_app():
    """Create a test instance of the Dash app."""
    # Import here to avoid circular imports
    from app import app as dash_app
    
    # Configure test app
    dash_app.config.update({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False
    })
    
    # Create a test client using the Flask application configured for testing
    with dash_app.test_client() as testing_client:
        with dash_app.app_context():
            yield testing_client  # this is where the testing happens!
