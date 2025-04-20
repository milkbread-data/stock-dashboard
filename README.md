# Stock Market Dashboard

An interactive dashboard for visualizing stock market data with real-time data from Yahoo Finance.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Stock+Dashboard+Preview)

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
git clone https://github.com/yourusername/stock-dashboard.git
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

- dash
- dash-bootstrap-components
- pandas
- plotly
- numpy
- yfinance
- openpyxl

## Deployment

This app can be deployed to platforms like Heroku, AWS, or Google Cloud. For Heroku deployment, make sure to:

1. Add a `Procfile` with: `web: gunicorn app:server`
2. Add `server = app.server` to your app.py file

## License

MIT

## Author

Your Name