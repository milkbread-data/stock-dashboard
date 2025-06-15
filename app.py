import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import openpyxl  # Required for Excel export
import yfinance as yf
import os  # For environment variables
import re  # For input validation
from functools import lru_cache  # For caching
import time  # For cache expiration
import logging
from werkzeug.middleware.proxy_fix import ProxyFix
from flask import Flask, request, abort
from flask_talisman import Talisman
import secrets

# Define stock tickers to download (with validation)
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
stock_tickers = [ticker for ticker in DEFAULT_TICKERS if is_valid_ticker(ticker)]

# Download stock data for a shorter period initially to speed up startup
end_date = datetime.now()
start_date = end_date - timedelta(days=90)  # Start with 90 days of data for faster loading

# Download data with error handling and rate limiting
print("Downloading stock data...")
try:
    stock_data = yf.download(
        tickers=stock_tickers,
        start=start_date,
        end=end_date,
        group_by='ticker',
        auto_adjust=True,
        progress=False,  # Disable progress bar for cleaner logs
        threads=True  # Use threads for parallel downloads
    )
except Exception as e:
    logger.error(f"Error downloading stock data: {str(e)}")
    stock_data = pd.DataFrame()  # Return empty DataFrame on error

# Process the data into a format suitable for our dashboard
df_list = []

for ticker in stock_tickers:
    # Extract data for this ticker
    ticker_data = stock_data[ticker].copy()
    ticker_data.reset_index(inplace=True)
    
    # Add ticker column
    ticker_data['Ticker'] = ticker
    
    # Rename columns
    ticker_data.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    df_list.append(ticker_data)

# Combine all tickers into one dataframe
df = pd.concat(df_list, ignore_index=True)

# Add caching to reduce API calls
# Cache for 1 hour (3600 seconds)
@lru_cache(maxsize=32)
def get_ticker_data(ticker, timestamp):
    """Get data for a ticker with caching. The timestamp parameter is used to invalidate the cache periodically."""
    print(f"Fetching fresh data for {ticker}")
    try:
        stock = yf.Ticker(ticker)
        
        # Get analyst recommendations
        try:
            recommendations = stock.recommendations
            if recommendations is not None and not recommendations.empty:
                if isinstance(recommendations.index, pd.DatetimeIndex):
                    recommendations = recommendations.reset_index()
                
                if 'Date' in recommendations.columns:
                    recommendations = recommendations.sort_values('Date', ascending=False).head(10)
        except Exception as rec_error:
            print(f"Error processing recommendations for {ticker}: {rec_error}")
            recommendations = None
        
        # Get current price
        try:
            current_price = stock.info.get('currentPrice', None)
            
            if current_price is None and not df.empty:
                ticker_df = df[df['Ticker'] == ticker].sort_values('date', ascending=False)
                if not ticker_df.empty:
                    current_price = ticker_df.iloc[0]['close']
        except Exception as price_error:
            print(f"Error getting current price for {ticker}: {price_error}")
            current_price = None
        
        # Get other data
        try:
            target_price = stock.info.get('targetMeanPrice', None)
            recommendation_mean = stock.info.get('recommendationMean', None)
            recommendation_key = stock.info.get('recommendationKey', 'N/A')
            business_summary = stock.info.get('longBusinessSummary', None)
        except Exception as info_error:
            print(f"Error getting info data for {ticker}: {info_error}")
            target_price = None
            recommendation_mean = None
            recommendation_key = 'N/A'
            business_summary = None
        
        # Get income statement data
        try:
            income_stmt = stock.income_stmt
            if income_stmt is not None and not income_stmt.empty:
                net_income = income_stmt.loc['Net Income']
                try:
                    eps = income_stmt.loc['Basic EPS']
                except:
                    eps = None
                
                earnings_data = {
                    'Year': net_income.index.year,
                    'Net Income': net_income.values,
                    'EPS': eps.values if eps is not None else None
                }
                earnings = pd.DataFrame(earnings_data)
            else:
                earnings = None
        except Exception as earnings_error:
            print(f"Error getting income statement for {ticker}: {earnings_error}")
            earnings = None
        
        return {
            'recommendations': recommendations,
            'target_price': target_price,
            'current_price': current_price,
            'recommendation_mean': recommendation_mean,
            'recommendation_key': recommendation_key,
            'business_summary': business_summary,
            'earnings': earnings
        }
    except Exception as e:
        print(f"Error getting analyst data for {ticker}: {e}")
        return {
            'recommendations': None,
            'target_price': None,
            'current_price': None,
            'recommendation_mean': None,
            'recommendation_key': 'N/A',
            'business_summary': None,
            'earnings': None
        }

# Get cache timestamp (refreshes every hour)
def get_cache_timestamp():
    """Return a timestamp that changes every hour to refresh the cache"""
    return int(time.time() / 3600)  # Changes every hour

# Get analyst recommendations and other information for each ticker
print("Getting analyst data...")
analyst_data = {}
current_timestamp = get_cache_timestamp()

for ticker in stock_tickers:
    analyst_data[ticker] = get_ticker_data(ticker, current_timestamp)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate ticker symbol format
def is_valid_ticker(ticker):
    """Validate that the ticker symbol contains only letters, numbers, and dots"""
    return bool(re.match(r'^[A-Z0-9.]{1,10}$', str(ticker).upper()))

# Initialize the Flask server
server = Flask(__name__)

# Security headers and middleware
server.wsgi_app = ProxyFix(server.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Initialize the Dash app with security configurations
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
    ],
    meta_tags=[
        {'name': 'viewport', 'content': 'width=device-width, initial-scale=1'},
        {'http-equiv': 'Content-Security-Policy', 'content': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline' https://use.fontawesome.com https://cdn.jsdelivr.net; font-src 'self' https://use.fontawesome.com https://cdn.jsdelivr.net; img-src 'self' data: https:; connect-src 'self' https://query1.finance.yahoo.com;"}
    ]
)

# Security headers
app.server.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30).total_seconds(),
    SECRET_KEY=os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
)

# Apply Talisman security headers
Talisman(
    app.server,
    force_https=True,
    strict_transport_security=True,
    session_cookie_secure=True,
    content_security_policy={
        'default-src': "'self'",
        'script-src': ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
        'style-src': ["'self'", "'unsafe-inline'"],
        'img-src': ["'self'", "data:", "https:"],
        'connect-src': ["'self'", "https://query1.finance.yahoo.com"],
    }
)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Stock Market Dashboard", className="text-center my-4"),
            html.P("Interactive dashboard showing real stock market data from Yahoo Finance", className="text-center text-muted")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    html.P("Select Date Range:"),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        start_date=df['date'].min(),
                        end_date=df['date'].max(),
                        display_format='YYYY-MM-DD'
                    ),
                    html.Div(style={"height": "20px"}),
                    html.P("Select Stocks:"),
                    dcc.Dropdown(
                        id='ticker-dropdown',
                        options=[{'label': ticker, 'value': ticker} for ticker in stock_tickers],
                        value=stock_tickers[:5],  # Default to first 5 stocks
                        multi=True
                    ),
                    html.Div(style={"height": "20px"}),
                    html.P("Add Custom Ticker:"),
                    dbc.InputGroup([
                        dbc.Input(id="custom-ticker-input", placeholder="Enter ticker symbol (e.g., NFLX)"),
                        dbc.Button("Add", id="add-ticker-button", color="primary", className="ms-2")
                    ]),
                    html.Div(id="ticker-feedback", className="mt-2"),
                    html.Div(style={"height": "20px"}),
                    html.P("Chart Type:"),
                    dcc.RadioItems(
                        id='chart-type',
                        options=[
                            {'label': 'Line Chart', 'value': 'line'},
                            {'label': 'Candlestick', 'value': 'candlestick'}
                        ],
                        value='line',
                        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                    )
                ])
            ]),
            dbc.Card([
                dbc.CardHeader("Download Data"),
                dbc.CardBody([
                    html.P("Export current selection:"),
                    dbc.Button("Download CSV", id="btn-download-csv", color="primary", className="me-2"),
                    dbc.Button("Download Excel", id="btn-download-excel", color="success"),
                    dcc.Download(id="download-dataframe-csv"),
                    dcc.Download(id="download-dataframe-excel")
                ])
            ], className="mt-4")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Stock Price Chart"),
                dbc.CardBody([
                    dcc.Graph(id='price-graph')
                ])
            ])
        ], width=9)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trading Volume"),
                dbc.CardBody([
                    dcc.Graph(id='volume-graph')
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Price Performance (% Change)"),
                dbc.CardBody([
                    dcc.Graph(id='performance-graph')
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Analyst Recommendations"),
                dbc.CardBody([
                    html.Div([
                        html.P("Select Stock:"),
                        dcc.Dropdown(
                            id='analyst-ticker-dropdown',
                            options=[{'label': ticker, 'value': ticker} for ticker in stock_tickers],
                            value=stock_tickers[0],  # Default to first stock
                            clearable=False
                        ),
                    ], className="mb-3"),
                    html.Div(id='analyst-recommendations')
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Company Overview & Earnings"),
                dbc.CardBody([
                    html.Div(id='company-overview'),
                    html.Div(id='earnings-data', className="mt-4")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Footer row with credits
    dbc.Row([
        dbc.Col([
            html.P("Data source: Yahoo Finance", className="text-center text-muted")
        ], width=12)
    ])
], fluid=True)

# Define a function to filter data based on user selections
def filter_dataframe(start_date, end_date, tickers):
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    if tickers:
        filtered_df = filtered_df[filtered_df['Ticker'].isin(tickers)]
    
    return filtered_df

# Calculate percentage change from first day for performance comparison
def calculate_performance(filtered_df, tickers):
    performance_data = []
    
    for ticker in tickers:
        ticker_data = filtered_df[filtered_df['Ticker'] == ticker].sort_values('date')
        if not ticker_data.empty:
            # Get the first close price
            first_close = ticker_data['close'].iloc[0]
            
            # Calculate percentage change
            ticker_data = ticker_data.copy()
            ticker_data['pct_change'] = ((ticker_data['close'] - first_close) / first_close) * 100
            
            performance_data.append(ticker_data)
    
    if performance_data:
        return pd.concat(performance_data, ignore_index=True)
    else:
        return pd.DataFrame()

# Callbacks
@app.callback(
    [Output('price-graph', 'figure'),
     Output('volume-graph', 'figure'),
     Output('performance-graph', 'figure')],
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('ticker-dropdown', 'value'),
     Input('chart-type', 'value')]
)
def update_graphs(start_date, end_date, tickers, chart_type):
    # Filter data based on date range and tickers
    filtered_df = filter_dataframe(start_date, end_date, tickers)
    
    # Price graph
    if chart_type == 'line':
        # Line chart for closing prices
        price_fig = px.line(filtered_df, x='date', y='close', color='Ticker',
                          title='Stock Price Trend',
                          labels={'close': 'Close Price ($)', 'date': 'Date', 'Ticker': 'Stock'})
    else:
        # Candlestick chart
        price_fig = go.Figure()
        for ticker in tickers:
            ticker_df = filtered_df[filtered_df['Ticker'] == ticker]
            price_fig.add_trace(
                go.Candlestick(
                    x=ticker_df['date'],
                    open=ticker_df['open'],
                    high=ticker_df['high'],
                    low=ticker_df['low'],
                    close=ticker_df['close'],
                    name=ticker
                )
            )
        price_fig.update_layout(
            title='Stock Price Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=False
        )
    
    # Volume graph
    volume_fig = px.bar(filtered_df, x='date', y='volume', color='Ticker',
                      title='Trading Volume',
                      labels={'volume': 'Volume', 'date': 'Date', 'Ticker': 'Stock'})
    
    # Performance comparison graph (% change from first day)
    performance_df = calculate_performance(filtered_df, tickers)
    performance_fig = px.line(performance_df, x='date', y='pct_change', color='Ticker',
                            title='Price Performance (% Change)',
                            labels={'pct_change': 'Change (%)', 'date': 'Date', 'Ticker': 'Stock'})
    performance_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    return price_fig, volume_fig, performance_fig

# Helper function to prepare data for export
def prepare_export_data(start_date, end_date, tickers):
    # Filter data based on current selections
    filtered_df = filter_dataframe(start_date, end_date, tickers)
    
    # Create a copy of the dataframe for export
    export_df = filtered_df.copy()
    
    # Format date column
    export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
    
    # Handle numeric columns safely
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        # Replace NaN with None
        export_df[col] = export_df[col].apply(lambda x: None if pd.isna(x) else x)
        # Round numeric values
        if col != 'volume':  # Don't round volume
            export_df[col] = export_df[col].apply(lambda x: round(x, 2) if x is not None else None)
    
    # Calculate additional metrics for export safely
    export_df['daily_change'] = export_df.apply(
        lambda row: round(((row['close'] - row['open']) / row['open'] * 100), 2) 
        if not pd.isna(row['close']) and not pd.isna(row['open']) and row['open'] != 0 
        else None, 
        axis=1
    )
    
    export_df['daily_range'] = export_df.apply(
        lambda row: round(((row['high'] - row['low']) / row['low'] * 100), 2) 
        if not pd.isna(row['high']) and not pd.isna(row['low']) and row['low'] != 0 
        else None, 
        axis=1
    )
    
    return export_df

# Callback for downloading data as CSV
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    [State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date'),
     State('ticker-dropdown', 'value')],
    prevent_initial_call=True,
)
def download_csv(n_clicks, start_date, end_date, tickers):
    if n_clicks:
        # Get prepared data
        export_df = prepare_export_data(start_date, end_date, tickers)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return dcc.send_data_frame(
            export_df.to_csv, 
            f"stock_market_data_{timestamp}.csv",
            index=False
        )

# Callback for downloading data as Excel
@app.callback(
    Output("download-dataframe-excel", "data"),
    Input("btn-download-excel", "n_clicks"),
    [State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date'),
     State('ticker-dropdown', 'value')],
    prevent_initial_call=True,
)
def download_excel(n_clicks, start_date, end_date, tickers):
    if n_clicks:
        # Get prepared data
        export_df = prepare_export_data(start_date, end_date, tickers)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return dcc.send_data_frame(
            export_df.to_excel, 
            f"stock_market_data_{timestamp}.xlsx",
            sheet_name="Stock Data",
            index=False
        )

# Callback for analyst recommendations
@app.callback(
    [Output('analyst-recommendations', 'children'),
     Output('company-overview', 'children'),
     Output('earnings-data', 'children')],
    [Input('analyst-ticker-dropdown', 'value')]
)
def update_analyst_data(ticker):
    if not ticker or ticker not in analyst_data:
        return html.P("No data available"), html.P("No data available"), html.P("No data available")
    
    ticker_data = analyst_data[ticker]
    
    # Analyst Recommendations Section
    analyst_section = []
    
    # Price Target
    if ticker_data['target_price'] and ticker_data['current_price']:
        target_price = ticker_data['target_price']
        current_price = ticker_data['current_price']
        upside = ((target_price - current_price) / current_price) * 100
        
        price_color = "success" if upside > 0 else "danger"
        
        analyst_section.append(html.Div([
            html.H5(f"Price Target Analysis for {ticker}"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"${current_price:.2f}", className="card-title"),
                            html.P("Current Price", className="card-text text-muted")
                        ])
                    ], className="text-center")
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"${target_price:.2f}", className="card-title"),
                            html.P("Mean Target Price", className="card-text text-muted")
                        ])
                    ], className="text-center")
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3([
                                f"{upside:.2f}%",
                                html.I(className=f"ms-2 fas fa-arrow-{'up' if upside > 0 else 'down'}")
                            ], className=f"card-title text-{price_color}"),
                            html.P("Potential Upside/Downside", className="card-text text-muted")
                        ])
                    ], className="text-center")
                ], width=4)
            ], className="mb-4")
        ]))
    
    # Analyst Rating
    if ticker_data['recommendation_mean'] and ticker_data['recommendation_key']:
        rec_mean = ticker_data['recommendation_mean']
        rec_key = ticker_data['recommendation_key'].capitalize()
        
        # Determine color based on recommendation
        rec_color = "secondary"
        if rec_key == "Buy" or rec_key == "Strong_buy":
            rec_color = "success"
        elif rec_key == "Sell" or rec_key == "Strong_sell":
            rec_color = "danger"
        elif rec_key == "Hold":
            rec_color = "warning"
        
        analyst_section.append(html.Div([
            html.H5("Analyst Consensus"),
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H3(f"{rec_mean:.2f}/5", className="mb-0")
                        ], width=4, className="text-center"),
                        dbc.Col([
                            html.H3(rec_key.replace('_', ' '), className=f"text-{rec_color} mb-0")
                        ], width=4, className="text-center"),
                        dbc.Col([
                            # Visual rating (stars or circles)
                            html.Div([
                                html.I(className=f"fas fa-circle {('text-' + rec_color) if i < int(rec_mean) else 'text-light'} me-1")
                                for i in range(5)
                            ], className="d-flex justify-content-center")
                        ], width=4)
                    ])
                ])
            ], className="mb-4")
        ]))
    
    # Recent Recommendations Table
    if ticker_data['recommendations'] is not None and not ticker_data['recommendations'].empty:
        try:
            # Make a copy and ensure all data is properly formatted
            rec_df = ticker_data['recommendations'].copy()
            
            # Convert datetime columns to string to avoid serialization issues
            for col in rec_df.columns:
                if pd.api.types.is_datetime64_any_dtype(rec_df[col]):
                    rec_df[col] = rec_df[col].dt.strftime('%Y-%m-%d')
            
            # Check if required columns exist
            required_columns = ['Date', 'Firm', 'To Grade', 'From Grade', 'Action']
            missing_columns = [col for col in required_columns if col not in rec_df.columns]
            
            if not missing_columns:
                # Create a table of recent recommendations
                analyst_section.append(html.Div([
                    html.H5("Recent Analyst Actions"),
                    dash_table.DataTable(
                        data=rec_df.to_dict('records'),
                        columns=[
                            {'name': 'Date', 'id': 'Date'},
                            {'name': 'Firm', 'id': 'Firm'},
                            {'name': 'To Grade', 'id': 'To Grade'},
                            {'name': 'From Grade', 'id': 'From Grade'},
                            {'name': 'Action', 'id': 'Action'}
                        ],
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px'
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'To Grade', 'filter_query': '{To Grade} contains "Buy" || {To Grade} contains "Outperform"'},
                                'backgroundColor': 'rgba(0, 255, 0, 0.1)',
                                'color': 'green'
                            },
                            {
                                'if': {'column_id': 'To Grade', 'filter_query': '{To Grade} contains "Sell" || {To Grade} contains "Underperform"'},
                                'backgroundColor': 'rgba(255, 0, 0, 0.1)',
                                'color': 'red'
                            }
                        ],
                        page_size=5
                    )
                ]))
            else:
                # Display available columns
                available_columns = rec_df.columns.tolist()
                
                # Create a simplified table with available columns
                table_columns = [{'name': col, 'id': col} for col in available_columns]
                
                analyst_section.append(html.Div([
                    html.H5("Recent Analyst Actions"),
                    html.P(f"Note: Some expected columns are missing. Showing available data."),
                    dash_table.DataTable(
                        data=rec_df.to_dict('records'),
                        columns=table_columns,
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px'
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        page_size=5
                    )
                ]))
        except Exception as e:
            print(f"Error processing recommendations table for {ticker}: {e}")
            analyst_section.append(html.Div([
                html.H5("Recent Analyst Actions"),
                html.P(f"Error processing recommendations data: {str(e)}")
            ]))
    
    if not analyst_section:
        analyst_section = [html.P("No analyst recommendations data available for this stock.")]
    
    # Company Overview Section
    company_section = []
    
    if ticker_data['business_summary']:
        company_section.append(html.Div([
            html.H5(f"About {ticker}"),
            html.P(ticker_data['business_summary'])
        ]))
    else:
        company_section = [html.P("No company overview available for this stock.")]
    
    # Earnings Data Section
    earnings_section = []
    
    if ticker_data['earnings'] is not None and not ticker_data['earnings'].empty:
        try:
            # Make a copy to avoid modifying the original data
            earnings_df = ticker_data['earnings'].copy()
            
            # Ensure Year is a string to avoid serialization issues
            earnings_df['Year'] = earnings_df['Year'].astype(str)
            
            # Create a bar chart of earnings
            earnings_fig = go.Figure()
            
            # Add Net Income bars
            earnings_fig.add_trace(go.Bar(
                x=earnings_df['Year'],
                y=earnings_df['Net Income'],
                name='Net Income',
                marker_color='rgb(26, 118, 255)'
            ))
            
            # Format y-axis to show in millions/billions
            def format_y_axis(y):
                if y >= 1e9:
                    return f'${y/1e9:.1f}B'
                elif y >= 1e6:
                    return f'${y/1e6:.1f}M'
                else:
                    return f'${y:.0f}'
            
            earnings_fig.update_layout(
                title=f'{ticker} Annual Net Income',
                xaxis_title='Year',
                yaxis_title='Net Income',
                barmode='group',
                bargap=0.15,
                bargroupgap=0.1,
                yaxis=dict(
                    tickformat='$,.0f'
                )
            )
            
            # Add EPS chart if available
            if 'EPS' in earnings_df.columns and earnings_df['EPS'].notna().any():
                eps_fig = go.Figure()
                eps_fig.add_trace(go.Bar(
                    x=earnings_df['Year'],
                    y=earnings_df['EPS'],
                    name='EPS',
                    marker_color='rgb(55, 83, 109)'
                ))
                
                eps_fig.update_layout(
                    title=f'{ticker} Earnings Per Share',
                    xaxis_title='Year',
                    yaxis_title='EPS ($)',
                    bargap=0.15,
                    yaxis=dict(
                        tickformat='$,.2f'
                    )
                )
                
                earnings_section.append(html.Div([
                    html.H5("Financial Performance"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(figure=earnings_fig)
                        ], width=12, lg=6),
                        dbc.Col([
                            dcc.Graph(figure=eps_fig)
                        ], width=12, lg=6)
                    ])
                ]))
            else:
                # Just show Net Income if EPS not available
                earnings_section.append(html.Div([
                    html.H5("Financial Performance"),
                    dcc.Graph(figure=earnings_fig)
                ]))
            
            # Also add a data table with the raw numbers
            # Prepare data for the table to avoid serialization issues
            table_data = []
            for _, row in earnings_df.iterrows():
                record = {'Year': row['Year']}
                
                # Format Net Income
                if pd.notna(row['Net Income']):
                    record['Net Income'] = row['Net Income']
                else:
                    record['Net Income'] = None
                
                # Format EPS if available
                if 'EPS' in earnings_df.columns and pd.notna(row['EPS']):
                    record['EPS'] = row['EPS']
                elif 'EPS' in earnings_df.columns:
                    record['EPS'] = None
                
                table_data.append(record)
            
            earnings_section.append(html.Div([
                html.H5("Annual Financial Data", className="mt-4"),
                dash_table.DataTable(
                    data=table_data,
                    columns=[
                        {'name': 'Year', 'id': 'Year'},
                        {'name': 'Net Income', 'id': 'Net Income', 'type': 'numeric', 'format': {'specifier': '$,.0f'}},
                    ] + ([{'name': 'EPS', 'id': 'EPS', 'type': 'numeric', 'format': {'specifier': '$,.2f'}}] 
                        if 'EPS' in earnings_df.columns else []),
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'right', 'padding': '10px'},
                    style_cell_conditional=[
                        {'if': {'column_id': 'Year'}, 'textAlign': 'left'}
                    ],
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{Net Income} < 0'},
                            'color': 'red'
                        },
                        {
                            'if': {'filter_query': '{EPS} < 0'},
                            'color': 'red'
                        }
                    ]
                )
            ]))
            
        except Exception as e:
            print(f"Error processing earnings data for {ticker}: {e}")
            earnings_section = [html.P(f"Error processing earnings data: {str(e)}")]
    else:
        earnings_section = [html.P("No earnings data available for this stock.")]
    
    return analyst_section, company_section, earnings_section

# Rate limiting dictionary to prevent abuse
from collections import defaultdict
from datetime import datetime, timedelta

# Simple in-memory rate limiter
rate_limits = defaultdict(list)

def is_rate_limited(ip, limit=5, window=60):
    """Check if the IP has exceeded the rate limit"""
    now = datetime.now()
    # Remove old entries
    rate_limits[ip] = [t for t in rate_limits[ip] if now - t < timedelta(seconds=window)]
    
    if len(rate_limits[ip]) >= limit:
        return True
        
    rate_limits[ip].append(now)
    return False

# Callback for adding custom ticker with input validation and rate limiting
@app.callback(
    [Output('ticker-dropdown', 'options'),
     Output('ticker-dropdown', 'value'),
     Output('ticker-input', 'value'),
     Output('analyst-tabs', 'children'),
     Output('alert-container', 'children')],  # Add alert output
    [Input('add-ticker-button', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('ticker-dropdown', 'options'),
     State('ticker-dropdown', 'value'),
     State('analyst-tabs', 'children')],
    prevent_initial_call=True
)
def add_custom_ticker(n_clicks, ticker_input, current_options, current_values, analyst_options):
    # Get client IP for rate limiting
    client_ip = request.remote_addr or '127.0.0.1'
    
    # Check rate limit
    if is_rate_limited(client_ip):
        alert = dbc.Alert(
            "Too many requests. Please wait before adding more tickers.",
            color="danger",
            dismissable=True,
            duration=4000
        )
        return dash.no_update, dash.no_update, "", dash.no_update, alert
    
    if n_clicks is None or not ticker_input:
        return current_options, current_values, "", analyst_options, dash.no_update
    
    # Convert ticker to uppercase and strip whitespace
    ticker = str(ticker_input).strip().upper()
    
    # Validate ticker format
    if not is_valid_ticker(ticker):
        alert = dbc.Alert(
            f"Invalid ticker symbol: {ticker}. Only letters, numbers, and dots are allowed.",
            color="danger",
            dismissable=True,
            duration=4000
        )
        return dash.no_update, dash.no_update, "", dash.no_update, alert
    
    # Check if ticker is already in the list
    if ticker in [opt['value'] for opt in current_options]:
        alert = dbc.Alert(
            f"{ticker} is already in the list.",
            color="warning",
            dismissable=True,
            duration=4000
        )
        return current_options, current_values, "", analyst_options, alert
    
    try:
        # Try to fetch data for the ticker to verify it exists
        test_data = yf.Ticker(ticker)
        if test_data.info is None or not test_data.history(period='1d').empty:
            raise ValueError("Invalid ticker or no data available")
            
        # Add new ticker to the dropdown options
        new_options = current_options + [{'label': ticker, 'value': ticker}]
        
        # Add the new ticker to the selected values
        new_values = current_values + [ticker] if current_values else [ticker]
        
        # Add a new tab for the analyst data
        new_tab = dcc.Tab(
            label=ticker,
            value=f'tab-{ticker}',
            children=[
                html.Div(id=f'analyst-content-{ticker}')
            ]
        )
        
        # Add the new tab to the existing tabs
        new_analyst_options = analyst_options + [new_tab] if analyst_options is not None else [new_tab]
        
        alert = dbc.Alert(
            f"Successfully added {ticker} to the dashboard.",
            color="success",
            dismissable=True,
            duration=4000
        )
        
        return new_options, new_values, "", new_analyst_options, alert
        
    except Exception as e:
        logger.error(f"Error adding ticker {ticker}: {str(e)}")
        alert = dbc.Alert(
            f"Failed to add {ticker}. Please check the ticker symbol and try again.",
            color="danger",
            dismissable=True,
            duration=4000
        )
        return dash.no_update, dash.no_update, "", dash.no_update, alert

# Memory cleanup function
def cleanup_memory():
    """Clean up memory by clearing cache periodically"""
    import gc
    gc.collect()
    print("Memory cleanup performed")

# Security middleware to set headers
@app.server.after_request
def add_security_headers(response):
    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    # Enable XSS protection
    response.headers['X-XSS-Protection'] = '1; mode=block'
    # Disable caching for sensitive pages
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

# Error handlers
@app.server.errorhandler(404)
def not_found_error(error):
    return "404 - Page not found", 404

@app.server.errorhandler(500)
def internal_error(error):
    logger.error(f"Server error: {error}")
    return "500 - Internal server error", 500

# Run the app
if __name__ == '__main__':
    # Get configuration from environment variables
    port = int(os.environ.get('PORT', 8050))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    # Never run with debug=True in production
    if not debug:
        import logging
        from logging.handlers import RotatingFileHandler
        
        # Configure logging to file
        file_handler = RotatingFileHandler('stock_dashboard.log', maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
        file_handler.setLevel(logging.INFO)
        app.server.logger.addHandler(file_handler)
        app.server.logger.setLevel(logging.INFO)
        app.server.logger.info('Stock Dashboard startup')
    
    # Run the app
    app.run_server(
        host='0.0.0.0',
        port=port,
        debug=debug,
        use_reloader=debug,
        ssl_context='adhoc' if not debug else None  # Enable HTTPS in production
    )