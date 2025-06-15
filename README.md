# Stock Market Dashboard

An interactive dashboard for visualizing stock market data with real-time data from Yahoo Finance.

The dashboard features multiple interactive charts including:
- Stock price charts (line and candlestick)
- Volume analysis
- Performance comparison
- Analyst recommendations
- Company information and financial data

## Features

- **Real-time Stock Data**: Fetches the latest data from Yahoo Finance
- **Interactive Visualizations**: Price charts, volume analysis, and performance comparisons
- **Custom Ticker Search**: Add any stock symbol from Yahoo Finance
- **Analyst Recommendations**: View analyst ratings and price targets
- **Company Information**: Business summaries and financial performance data
- **Data Export**: Download data in CSV or Excel format

## Installation

```bash
# Clone the repository
git clone https://github.com/milkbread-data/stock-dashboard.git
cd stock-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## Usage

1. Select date range using the date picker
2. Choose stocks from the dropdown menu
3. Add custom ticker symbols as needed
4. Switch between line chart and candlestick views
5. View analyst recommendations for selected stocks
6. Download data in CSV or Excel format

## Data Source

All stock data is fetched from Yahoo Finance using the `yfinance` library.

## Dependencies

- dash==3.0.4
- dash-bootstrap-components==2.0.3
- pandas==2.3.0
- plotly==6.1.2
- numpy==2.3.0
- yfinance==0.2.63
- openpyxl==3.1.5
- gunicorn==23.0.0

## Deployment

### Deploying to Render (Recommended)

This app is optimized for deployment on Render's free tier:

1. Create a Render account at [render.com](https://render.com)
2. Connect your GitHub repository
3. Create a new Web Service and select this repository
4. Render will automatically detect the configuration from `render.yaml`
5. Click "Create Web Service"

Alternatively, you can manually configure:
- **Environment**: Python
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:server`
- **Plan**: Free

### Other Deployment Options

The app can also be deployed to platforms like Heroku, AWS, or Google Cloud:

1. Make sure the `Procfile` contains: `web: gunicorn app:server`
2. Verify `server = app.server` is in your app.py file

## License

MIT

## Author

milkbread
