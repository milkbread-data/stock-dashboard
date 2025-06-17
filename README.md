# 📊 Stock Market Dashboard

[![Python](https://img.shields.io/badge/python-3.12.0-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/milkbread-data/stock-dashboard)
[![Open in GitHub Codespaces](https://img.shields.io/badge/Open%20in-Codespaces-blue?logo=github)](https://github.com/features/codespaces)

An interactive, real-time stock market dashboard built with Python and Dash. Visualize stock prices, trading volumes, and analyst recommendations with an intuitive web interface.

## ✨ Features

- **Real-time Stock Data**: Fetches the latest market data from Yahoo Finance
- **Interactive Visualizations**:
  - Interactive price charts (line and candlestick)
  - Volume analysis
  - Performance comparison across multiple stocks
  - 52-week high/low indicators
- **Comprehensive Stock Analysis**:
  - Analyst ratings and price targets
  - Company information and financials
  - Key statistics and performance metrics
- **User Experience**:
  - Custom ticker search
  - Date range selection
  - Responsive design for all devices
  - Dark/light theme support
- **Data Management**:
  - Export data to CSV/Excel
  - Caching for improved performance
  - Rate limiting for API protection

## 🚀 Quick Start

### Prerequisites

- Python 3.12.0
- pip (Python package manager)
- Git (for version control)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/milkbread-data/stock-dashboard.git
   cd stock-dashboard
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the dashboard**
   Open your browser and navigate to: http://127.0.0.1:8050/

## 🌐 Deployment

### Deploy to Render

1. **One-Click Deployment**
   - Click the [Deploy to Render](https://render.com/deploy?repo=https://github.com/milkbread-data/stock-dashboard) button above
   - Sign up or log in to your Render account
   - Follow the on-screen instructions to deploy

2. **Manual Deployment**
   - Fork this repository to your GitHub account
   - Create a new Web Service on Render
   - Connect your GitHub repository
   - Configure the following settings:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:server`
   - Set the following environment variables:
     - `PYTHON_VERSION`: 3.12.0
     - `DASH_DEBUG`: false
   - Click "Create Web Service"

### Environment Variables

For production deployment, you may want to set these environment variables:

```
DASH_DEBUG=false  # Set to false in production
PYTHONUNBUFFERED=true  # Recommended for Python applications
GUNICORN_WORKERS=4  # Adjust based on your Render plan
GUNICORN_TIMEOUT=120  # Timeout in seconds
```

## 📚 Usage Guide

### Basic Navigation

1. **Select Date Range**
   - Use the date picker to select your desired time period
   - Choose from preset timeframes (1M, 3M, 6M, YTD, 1Y, 5Y, MAX)

2. **Select Stocks**
   - Choose from popular stocks in the dropdown
   - Add custom ticker symbols using the search box
   - Compare up to 5 stocks simultaneously

3. **Chart Types**
   - Toggle between line and candlestick charts
   - Hover over data points for detailed information
   - Use the range slider to zoom in/out

4. **Analyst Data**
   - View analyst recommendations for selected stocks
   - Check price targets and consensus ratings
   - See earnings estimates and company information

### Data Export

1. **Export Options**
   - Download data in CSV or Excel format
   - Export includes OHLCV data and technical indicators
   - Select specific date ranges before exporting

## 🛠 API Documentation

### Endpoints

- `/` - Main dashboard interface
- `/api/stock-data` - JSON API for stock data (protected by rate limiting)
- `/api/analyst-data` - Analyst recommendations and ratings

### Rate Limiting

- 60 requests per minute per IP address
- Rate limit headers are included in responses
- Caching is implemented to reduce API calls

## 🧪 Testing

Run the test suite with:

```bash
pytest -v
```

Test coverage includes:
- Ticker validation
- Data filtering
- Performance calculations
- Rate limiting
- Cache functionality

## 🚀 Deployment

### Render (Recommended)

1. Click the button below to deploy to Render:
   
   [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

2. Or follow manual deployment:
   - Connect your GitHub repository
   - Select "Web Service"
   - Configure environment variables if needed
   - Deploy!

### Other Platforms

#### Heroku
```bash
# Set up Heroku CLI
heroku create your-app-name

# Deploy
git push heroku main
```

#### Docker
```bash
# Build the image
docker build -t stock-dashboard .

# Run the container
docker run -p 8050:8050 stock-dashboard
```

## 🐛 Troubleshooting

### Common Issues

1. **Data not loading**
   - Check your internet connection
   - Verify Yahoo Finance API status
   - Clear browser cache

2. **Installation errors**
   - Ensure Python 3.8+ is installed
   - Try updating pip: `pip install --upgrade pip`
   - Check for dependency conflicts

3. **Performance issues**
   - Reduce the date range
   - Select fewer stocks
   - Clear browser cache

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- Built with [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/)
- Data provided by [Yahoo Finance](https://finance.yahoo.com/)
- Icons by [Font Awesome](https://fontawesome.com/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

Made with ❤️ by milkbread

