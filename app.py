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

# Define stock tickers to download
stock_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']

# Download stock data for the past 2 years
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # 2 years of data

# Download data
print("Downloading stock data...")
stock_data = yf.download(
    tickers=stock_tickers,
    start=start_date,
    end=end_date,
    group_by='ticker',
    auto_adjust=True
)

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

# Get analyst recommendations and other information for each ticker
print("Getting analyst data...")
analyst_data = {}
for ticker in stock_tickers:
    try:
        # Create Ticker object
        stock = yf.Ticker(ticker)
        
        # Get analyst recommendations - handle potential format issues
        try:
            recommendations = stock.recommendations
            if recommendations is not None and not recommendations.empty:
                # Check if 'Date' is in index or columns
                if isinstance(recommendations.index, pd.DatetimeIndex):
                    recommendations = recommendations.reset_index()
                
                # Sort by date if available
                if 'Date' in recommendations.columns:
                    recommendations = recommendations.sort_values('Date', ascending=False).head(10)
        except Exception as rec_error:
            print(f"Error processing recommendations for {ticker}: {rec_error}")
            recommendations = None
        
        # Get current price from historical data if not in info
        current_price = None
        try:
            # Try to get from info
            current_price = stock.info.get('currentPrice', None)
            
            # If not available, use the most recent close price
            if current_price is None and not df.empty:
                ticker_df = df[df['Ticker'] == ticker].sort_values('date', ascending=False)
                if not ticker_df.empty:
                    current_price = ticker_df.iloc[0]['close']
        except Exception as price_error:
            print(f"Error getting current price for {ticker}: {price_error}")
        
        # Get other data with error handling
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
        
        # Get income statement data instead of deprecated earnings
        try:
            income_stmt = stock.income_stmt
            # Create a simplified earnings dataframe from income statement
            if income_stmt is not None and not income_stmt.empty:
                # Extract Net Income and EPS data
                net_income = income_stmt.loc['Net Income']
                # Try to get EPS if available
                try:
                    eps = income_stmt.loc['Basic EPS']
                except:
                    # If EPS not available, calculate a simple version from net income
                    # This is just an approximation
                    eps = None
                
                # Create a dataframe with the data
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
        
        # Store all data
        analyst_data[ticker] = {
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
        analyst_data[ticker] = {
            'recommendations': None,
            'target_price': None,
            'current_price': None,
            'recommendation_mean': None,
            'recommendation_key': 'N/A',
            'business_summary': None,
            'earnings': None
        }

# Initialize the Dash app with Bootstrap theme and Font Awesome
app = dash.Dash(__name__, 
                external_stylesheets=[
                    dbc.themes.BOOTSTRAP,
                    'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
                ],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])

# Add server variable for deployment platforms like Heroku
server = app.server

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

# Callback for adding custom ticker
@app.callback(
    [Output('ticker-dropdown', 'options'),
     Output('ticker-dropdown', 'value'),
     Output('ticker-feedback', 'children'),
     Output('analyst-ticker-dropdown', 'options')],
    [Input('add-ticker-button', 'n_clicks')],
    [State('custom-ticker-input', 'value'),
     State('ticker-dropdown', 'options'),
     State('ticker-dropdown', 'value'),
     State('analyst-ticker-dropdown', 'options')],
    prevent_initial_call=True
)
def add_custom_ticker(n_clicks, ticker_input, current_options, current_values, analyst_options):
    if not ticker_input:
        return current_options, current_values, html.Div("Please enter a ticker symbol", className="text-danger"), analyst_options
    
    # Convert to uppercase
    ticker_input = ticker_input.strip().upper()
    
    # Check if ticker already exists in options
    if any(opt['value'] == ticker_input for opt in current_options):
        # If ticker exists but not selected, add it to selected values
        if ticker_input not in current_values:
            current_values.append(ticker_input)
        return current_options, current_values, html.Div(f"Ticker {ticker_input} already exists and is now selected", className="text-info"), analyst_options
    
    # Try to validate the ticker by fetching data
    try:
        # Check if ticker exists in Yahoo Finance
        ticker_info = yf.Ticker(ticker_input)
        
        # Try to get some basic info to validate
        history = ticker_info.history(period="1mo")
        
        if history.empty:
            return current_options, current_values, html.Div(f"Ticker {ticker_input} not found or has no data", className="text-danger"), analyst_options
        
        # Add the new ticker to global data
        try:
            # Download data for the new ticker
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 years of data
            
            print(f"Downloading data for {ticker_input}...")
            new_data = yf.download(
                tickers=ticker_input,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if not new_data.empty:
                # Process the data
                new_data.reset_index(inplace=True)
                new_data['Ticker'] = ticker_input
                
                # Rename columns
                new_data.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }, inplace=True)
                
                # Add to global dataframe
                global df
                df = pd.concat([df, new_data], ignore_index=True)
                
                # Get analyst data for the new ticker
                try:
                    # Create Ticker object
                    stock = yf.Ticker(ticker_input)
                    
                    # Get analyst recommendations - handle potential format issues
                    try:
                        recommendations = stock.recommendations
                        if recommendations is not None and not recommendations.empty:
                            # Check if 'Date' is in index or columns
                            if isinstance(recommendations.index, pd.DatetimeIndex):
                                recommendations = recommendations.reset_index()
                            
                            # Sort by date if available
                            if 'Date' in recommendations.columns:
                                recommendations = recommendations.sort_values('Date', ascending=False).head(10)
                    except Exception as rec_error:
                        print(f"Error processing recommendations for {ticker_input}: {rec_error}")
                        recommendations = None
                    
                    # Get current price from historical data if not in info
                    current_price = None
                    try:
                        # Try to get from info
                        current_price = stock.info.get('currentPrice', None)
                        
                        # If not available, use the most recent close price
                        if current_price is None and not new_data.empty:
                            current_price = new_data.iloc[-1]['close']
                    except Exception as price_error:
                        print(f"Error getting current price for {ticker_input}: {price_error}")
                    
                    # Get other data with error handling
                    try:
                        target_price = stock.info.get('targetMeanPrice', None)
                        recommendation_mean = stock.info.get('recommendationMean', None)
                        recommendation_key = stock.info.get('recommendationKey', 'N/A')
                        business_summary = stock.info.get('longBusinessSummary', None)
                    except Exception as info_error:
                        print(f"Error getting info data for {ticker_input}: {info_error}")
                        target_price = None
                        recommendation_mean = None
                        recommendation_key = 'N/A'
                        business_summary = None
                    
                    # Get income statement data
                    try:
                        income_stmt = stock.income_stmt
                        # Create a simplified earnings dataframe from income statement
                        if income_stmt is not None and not income_stmt.empty:
                            # Extract Net Income and EPS data
                            net_income = income_stmt.loc['Net Income']
                            # Try to get EPS if available
                            try:
                                eps = income_stmt.loc['Basic EPS']
                            except:
                                # If EPS not available, calculate a simple version from net income
                                eps = None
                            
                            # Create a dataframe with the data
                            earnings_data = {
                                'Year': net_income.index.year,
                                'Net Income': net_income.values,
                                'EPS': eps.values if eps is not None else None
                            }
                            earnings = pd.DataFrame(earnings_data)
                        else:
                            earnings = None
                    except Exception as earnings_error:
                        print(f"Error getting income statement for {ticker_input}: {earnings_error}")
                        earnings = None
                    
                    # Store all data
                    global analyst_data
                    analyst_data[ticker_input] = {
                        'recommendations': recommendations,
                        'target_price': target_price,
                        'current_price': current_price,
                        'recommendation_mean': recommendation_mean,
                        'recommendation_key': recommendation_key,
                        'business_summary': business_summary,
                        'earnings': earnings
                    }
                except Exception as e:
                    print(f"Error getting analyst data for {ticker_input}: {e}")
                    analyst_data[ticker_input] = {
                        'recommendations': None,
                        'target_price': None,
                        'current_price': None,
                        'recommendation_mean': None,
                        'recommendation_key': 'N/A',
                        'business_summary': None,
                        'earnings': None
                    }
            
            # Add to options and selected values
            new_options = current_options + [{'label': ticker_input, 'value': ticker_input}]
            new_values = current_values + [ticker_input]
            new_analyst_options = analyst_options + [{'label': ticker_input, 'value': ticker_input}]
            
            return new_options, new_values, html.Div([
                html.Span("Success! ", className="text-success fw-bold"),
                f"Added {ticker_input} successfully!"
            ]), new_analyst_options
            
        except Exception as e:
            print(f"Error adding ticker data for {ticker_input}: {e}")
            return current_options, current_values, html.Div(f"Error adding ticker data: {str(e)}", className="text-danger"), analyst_options
        
    except Exception as e:
        print(f"Error validating ticker {ticker_input}: {e}")
        return current_options, current_values, html.Div(f"Invalid ticker symbol: {str(e)}", className="text-danger"), analyst_options

# Run the app
if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=True, host='0.0.0.0', port=port)