"""
ATLAS - Stock Analysis Dashboard
Author: Steven Chen
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import warnings
import time
import json

warnings.filterwarnings('ignore')

# ========================================
# Configuration Parameters
# ========================================

REFRESH_INTERVAL = 120000  # Update every 2 minutes to avoid frequent requests
DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
COLORS = {
    'background': '#0e1117',
    'card': '#262730', 
    'text': '#ffffff',
    'primary': '#00d4ff',
    'success': '#00ff88',
    'warning': '#ffaa00',
    'danger': '#ff4444',
    'up': '#00ff88',
    'down': '#ff4444'
}

# Initialize Dash application
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "ATLAS Multi-Stock Dashboard"

# Global data cache
data_cache = {}
last_update_time = {}

# ========================================
# Data Retrieval and Processing Functions
# ========================================

def get_stock_data_safe(symbol, period='1d', interval='1m'):
    """Safe data fetching with rate limiting protection"""
    print(f"Fetching {symbol} data...")
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            print(f"Warning: {symbol} data is empty")
            return None
            
        # Calculate technical indicators
        data = calculate_technical_indicators(data)
        
        print(f"✅ Successfully fetched {symbol} data, {len(data)} records")
        return data
        
    except Exception as e:
        print(f"❌ Failed to fetch {symbol} data: {e}")
        # If failed, try to use cached data
        if symbol in data_cache:
            print(f"📦 Using cached data for {symbol}")
            return data_cache[symbol]['df']
        return None

def calculate_connors_rsi(data, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI (simplified version)"""
    try:
        # Simplified calculation to avoid complex streak computation
        rsi = data['Close'].diff().rolling(rsi_period).apply(
            lambda x: 100 - (100 / (1 + (x[x > 0].mean() / abs(x[x < 0].mean())))) if len(x[x < 0]) > 0 else 50
        )
        return rsi.fillna(50)
    except:
        return pd.Series([50] * len(data), index=data.index)

def apply_kalman_filter_simple(data):
    """Simplified Kalman filter"""
    try:
        prices = data['Close'].values
        filtered = []
        trend = []
        
        # Simple exponential smoothing as Kalman filter approximation
        alpha = 0.3
        filtered_price = prices[0]
        
        for price in prices:
            filtered_price = alpha * price + (1 - alpha) * filtered_price
            filtered.append(filtered_price)
            trend.append(filtered_price - filtered[-2] if len(filtered) > 1 else 0)
        
        return pd.Series(filtered, index=data.index), pd.Series(trend, index=data.index)
    except:
        return pd.Series(data['Close']), pd.Series([0] * len(data), index=data.index)

def apply_fft_filter_simple(data, cutoff_period=21):
    """Simplified FFT filtering"""
    try:
        from scipy.signal import savgol_filter
        # Use Savitzky-Golay filter as simplified version of FFT
        if len(data) > cutoff_period:
            window_length = min(cutoff_period, len(data))
            if window_length % 2 == 0:
                window_length -= 1
            if window_length < 3:
                return pd.Series(data['Close'])
            filtered = savgol_filter(data['Close'], window_length, 3)
            return pd.Series(filtered, index=data.index)
        else:
            return pd.Series(data['Close'])
    except:
        # If scipy is not available, use moving average instead
        return data['Close'].rolling(window=cutoff_period).mean().fillna(data['Close'])

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    if df.empty:
        return df
    
    # Moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Connors RSI
    df['CRSI'] = calculate_connors_rsi(df)
    
    # Kalman filter
    df['Kalman_Price'], df['Kalman_Trend'] = apply_kalman_filter_simple(df)
    
    # FFT filtering
    df['FFT_21'] = apply_fft_filter_simple(df, 21)
    df['FFT_63'] = apply_fft_filter_simple(df, 63)
    
    # Volume moving average
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
    
    # Price change rates
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5'] = df['Close'].pct_change(5)
    
    # Fill NaN values
    df = df.ffill().bfill()
    
    return df

def get_multiple_stocks_data(symbols, period='1d', interval='1m'):
    """Sequential data fetching to avoid rate limiting"""
    all_data = {}
    
    print(f"🔄 Fetching data for {len(symbols)} symbols sequentially...")
    
    for i, symbol in enumerate(symbols):
        print(f"📊 [{i+1}/{len(symbols)}] Fetching {symbol}...")
        
        # Sequential fetching with delays
        df = get_stock_data_safe(symbol, period, interval)
        if df is not None:
            all_data[symbol] = df
            # Update cache
            data_cache[symbol] = {
                'df': df,
                'timestamp': datetime.now()
            }
            print(f"✅ {symbol} data fetched successfully")
        else:
            print(f"❌ Failed to fetch {symbol} data")
        
        # Add delay between requests to avoid rate limiting
        if i < len(symbols) - 1:  # Don't delay after the last request
            time.sleep(1.5)  # 1.5 second delay between requests
    
    return all_data

def calculate_simple_prediction(df):
    """Comprehensive prediction algorithm based on multiple technical indicators"""
    if df.empty or len(df) < 20:
        return "HOLD", 0.5, 0
    
    # Get latest values (handle NaN)
    latest_rsi = df['RSI'].dropna().iloc[-1] if not df['RSI'].dropna().empty else 50
    latest_macd = df['MACD'].dropna().iloc[-1] if not df['MACD'].dropna().empty else 0
    latest_signal = df['MACD_Signal'].dropna().iloc[-1] if not df['MACD_Signal'].dropna().empty else 0
    latest_crsi = df['CRSI'].dropna().iloc[-1] if not df['CRSI'].dropna().empty else 50
    latest_kalman_trend = df['Kalman_Trend'].dropna().iloc[-1] if not df['Kalman_Trend'].dropna().empty else 0
    
    # Price trend (5-day)
    price_trend = df['Price_Change_5'].dropna().iloc[-1] if not df['Price_Change_5'].dropna().empty else 0
    
    # Volume trend
    volume_ratio = (df['Volume'].iloc[-1] / df['Volume_MA'].iloc[-1]) if not pd.isna(df['Volume_MA'].iloc[-1]) else 1
    
    # Bollinger Band position
    latest_close = df['Close'].iloc[-1]
    bb_upper = df['BB_Upper'].dropna().iloc[-1] if not df['BB_Upper'].dropna().empty else latest_close
    bb_lower = df['BB_Lower'].dropna().iloc[-1] if not df['BB_Lower'].dropna().empty else latest_close
    bb_position = (latest_close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5
    
    # Scoring system
    score = 0
    
    # RSI score (-2 to +2)
    if latest_rsi < 25:
        score += 2  # Severely oversold
    elif latest_rsi < 35:
        score += 1  # Oversold
    elif latest_rsi > 75:
        score -= 2  # Severely overbought
    elif latest_rsi > 65:
        score -= 1  # Overbought
    
    # Connors RSI score (-2 to +2)
    if latest_crsi < 20:
        score += 2  # Severely oversold
    elif latest_crsi < 30:
        score += 1  # Oversold
    elif latest_crsi > 80:
        score -= 2  # Severely overbought
    elif latest_crsi > 70:
        score -= 1  # Overbought
    
    # MACD score (-1 to +1)
    if latest_macd > latest_signal and latest_macd > 0:
        score += 1  # Strong bullish crossover
    elif latest_macd < latest_signal and latest_macd < 0:
        score -= 1  # Weak bearish crossover
    
    # Kalman trend score (-1 to +1)
    if latest_kalman_trend > 0.5:
        score += 1  # Upward trend
    elif latest_kalman_trend < -0.5:
        score -= 1  # Downward trend
    
    # Bollinger Band position score (-1 to +1)
    if bb_position < 0.2:
        score += 1  # Near lower band, possible bounce
    elif bb_position > 0.8:
        score -= 1  # Near upper band, possible pullback
    
    # Price trend score (-2 to +2)
    if price_trend > 0.05:
        score += 2  # Strong upward movement
    elif price_trend > 0.02:
        score += 1  # Moderate upward movement
    elif price_trend < -0.05:
        score -= 2  # Significant decline
    elif price_trend < -0.02:
        score -= 1  # Minor decline
    
    # Volume confirmation (-1 to +1)
    if volume_ratio > 1.5:
        if score > 0:
            score += 1  # Volume confirms upward movement
        elif score < 0:
            score -= 1  # Volume confirms downward movement
    
    # Convert to prediction result
    if score >= 4:
        prediction = "STRONG_BUY"
        confidence = min(0.90, 0.65 + abs(score) * 0.04)
    elif score >= 2:
        prediction = "BUY"
        confidence = min(0.80, 0.60 + abs(score) * 0.04)
    elif score >= 1:
        prediction = "WEAK_BUY"
        confidence = min(0.70, 0.55 + abs(score) * 0.04)
    elif score <= -4:
        prediction = "STRONG_SELL"
        confidence = min(0.90, 0.65 + abs(score) * 0.04)
    elif score <= -2:
        prediction = "SELL"
        confidence = min(0.80, 0.60 + abs(score) * 0.04)
    elif score <= -1:
        prediction = "WEAK_SELL"
        confidence = min(0.70, 0.55 + abs(score) * 0.04)
    else:
        prediction = "HOLD"
        confidence = 0.5
    
    return prediction, confidence, score

# ========================================
# Dashboard Layout Components
# ========================================

def create_header():
    """Create page header"""
    return dbc.NavbarSimple(
        children=[
            dbc.Badge("Multi-Stock", color="info", className="me-2"),
            dbc.Badge("Live", color="success"),
        ],
        brand="📊 ATLAS Dashboard Pro",
        color="dark",
        dark=True,
        className="mb-3"
    )

def create_stock_selector():
    """Create stock selector"""
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Stocks (Multiple):", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='stock-selector',
                        options=[{'label': f'{symbol} - {get_stock_name(symbol)}', 'value': symbol} 
                                for symbol in DEFAULT_SYMBOLS],
                        value=['AAPL', 'TSLA', 'NVDA'],  # Default selection
                        multi=True,
                        style={'color': '#000'},
                        placeholder="Select stock symbols..."
                    )
                ], width=4),
                dbc.Col([
                    html.Label("Time Period:", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='timeframe-selector',
                        options=[
                            {'label': '1 Day (1min)', 'value': '1d'},
                            {'label': '5 Days (5min)', 'value': '5d'},
                            {'label': '1 Month (1hr)', 'value': '1mo'}
                        ],
                        value='1d',
                        style={'color': '#000'}
                    )
                ], width=2),
                dbc.Col([
                    html.Label("Technical Indicators:", className="fw-bold mb-2"),
                    dcc.Checklist(
                        id='indicator-selector',
                        options=[
                            {'label': ' Moving Averages', 'value': 'MA'},
                            {'label': ' Bollinger Bands', 'value': 'BB'},
                            {'label': ' Volume', 'value': 'VOL'},
                            {'label': ' RSI', 'value': 'RSI'},
                            {'label': ' MACD', 'value': 'MACD'},
                            {'label': ' Kalman Filter', 'value': 'KALMAN'},
                            {'label': ' FFT Filter', 'value': 'FFT'},
                            {'label': ' Connors RSI', 'value': 'CRSI'}
                        ],
                        value=['MA'],
                        inline=True,
                        style={'color': COLORS['text']}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Actions:", className="fw-bold mb-2"),
                    html.Div([
                        dbc.Button("🔄 Refresh", id="refresh-btn", color="primary", size="sm", className="me-2"),
                        dbc.Button("📊 Select All", id="select-all-btn", color="secondary", size="sm")
                    ])
                ], width=2),
                dbc.Col([
                    html.Div(id="last-update", className="text-muted small mt-3")
                ], width=1)
            ])
        ])
    ], className="mb-3")

def get_stock_name(symbol):
    """Get stock name"""
    stock_names = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft',
        'TSLA': 'Tesla',
        'NVDA': 'NVIDIA',
        'AMZN': 'Amazon',
        'META': 'Meta',
        'NFLX': 'Netflix',
        'GOOG': 'Alphabet',
        'TSM': 'Taiwan Semiconductor Manufacturing',
        'BABA': 'Alibaba Group Holding',
        'BIDU': 'Baidu',
        'JD': 'JD.com',
        'AMD': 'Advanced Micro Devices',
        'INTC': 'Intel',
        'CSCO': 'Cisco Systems',
        'IBM': 'IBM',
        'ORCL': 'Oracle',
        'VMW': 'VMware',
        'ADBE': 'Adobe',
        'SAP': 'SAP',
        'HON': 'Honeywell International',
        'QCOM': 'Qualcomm',
        'AMGN': 'Amgen',
        'CRM': 'Salesforce',
        'PLTR': 'Palantir',
        'TXN': 'Texas Instruments',
    }
    return stock_names.get(symbol, symbol)

# Main layout
app.layout = dbc.Container([
    # Auto-refresh component
    dcc.Interval(
        id='interval-component',
        interval=REFRESH_INTERVAL,
        n_intervals=0
    ),
    
    # Data storage
    dcc.Store(id='multi-stock-data-store'),
    
    # Page layout
    create_header(),
    create_stock_selector(),
    
    # Stock overview cards
    html.Div(id="stock-overview-cards", className="mb-3"),
    
    # Main chart area
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 Multi-Stock Price Comparison"),
                dbc.CardBody([
                    dcc.Graph(id="price-comparison-chart", style={'height': '500px'})
                ])
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🤖 AI Prediction Summary"),
                dbc.CardBody([
                    html.Div(id="predictions-summary")
                ])
            ])
        ], width=4)
    ], className="mb-3"),
    
    # Detailed technical analysis
    html.Div(id="detailed-analysis"),
    
    # Footer information
    html.Footer([
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.P("🚀 ATLAS Multi-Stock Dashboard", className="text-muted mb-0")
            ], width=6),
            dbc.Col([
                html.P("⚡ Powered by YFinance + Plotly Dash", className="text-muted mb-0 text-end")
            ], width=6)
        ])
    ])
], fluid=True)

# ========================================
# Callback functions
# ========================================

@app.callback(
    Output('stock-selector', 'value'),
    Input('select-all-btn', 'n_clicks'),
    State('stock-selector', 'value')
)
def select_all_stocks(n_clicks, current_selection):
    """Select all stocks button"""
    if n_clicks:
        return DEFAULT_SYMBOLS
    return current_selection

@app.callback(
    [Output('multi-stock-data-store', 'data'),
     Output('last-update', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('refresh-btn', 'n_clicks'),
     Input('stock-selector', 'value'),
     Input('timeframe-selector', 'value')]
)
def update_multi_stock_data(n_intervals, n_clicks, selected_symbols, timeframe):
    """Update multi-stock data"""
    if not selected_symbols:
        return {}, "Please select stocks"
    
    # Limit to maximum 8 stocks
    if len(selected_symbols) > 8:
        selected_symbols = selected_symbols[:8]
    
    print(f"🔄 Starting to fetch data for {len(selected_symbols)} stocks...")
    
    # Fetch multi-stock data
    all_stock_data = get_multiple_stocks_data(selected_symbols, period=timeframe)
    
    # Prepare return data
    processed_data = {}
    
    for symbol, df in all_stock_data.items():
        if df is not None and not df.empty:
            # Calculate prediction
            prediction, confidence, score = calculate_simple_prediction(df)
            
            processed_data[symbol] = {
                'data': df.reset_index().to_dict('records'),
                'last_price': float(df['Close'].iloc[-1]),
                'change': float(df['Close'].iloc[-1] - df['Close'].iloc[-2]) if len(df) > 1 else 0,
                'change_pct': float((df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100) if len(df) > 1 else 0,
                'volume': int(df['Volume'].iloc[-1]) if not pd.isna(df['Volume'].iloc[-1]) else 0,
                'rsi': float(df['RSI'].dropna().iloc[-1]) if not df['RSI'].dropna().empty else 50,
                'prediction': prediction,
                'confidence': confidence,
                'score': score
            }
    
    current_time = datetime.now().strftime("%H:%M:%S")
    success_count = len(processed_data)
    
    print(f"✅ Successfully fetched {success_count}/{len(selected_symbols)} stocks data")
    
    return {
        'stocks': processed_data,
        'timeframe': timeframe,
        'update_time': current_time
    }, f"Updated: {current_time} ({success_count}/{len(selected_symbols)})"

@app.callback(
    Output('stock-overview-cards', 'children'),
    Input('multi-stock-data-store', 'data')
)
def update_stock_overview_cards(data):
    """Update stock overview cards"""
    if not data or 'stocks' not in data:
        return html.Div("Select stocks to view data", className="text-center text-muted p-4")
    
    cards = []
    stocks = data['stocks']
    
    for symbol, stock_data in stocks.items():
        price = stock_data['last_price']
        change = stock_data['change']
        change_pct = stock_data['change_pct']
        prediction = stock_data['prediction']
        confidence = stock_data['confidence']
        
        # Determine colors
        if change >= 0:
            price_color = 'text-success'
            arrow = '↗️'
        else:
            price_color = 'text-danger'
            arrow = '↘️'
        
        # Prediction colors
        pred_colors = {
            # English prediction results
            'STRONG_BUY': 'success',
            'BUY': 'info', 
            'HOLD': 'warning',
            'SELL': 'danger',
            'STRONG_SELL': 'dark'
        }
        pred_color = pred_colors.get(prediction, 'secondary')
        
        card = dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5(f"{symbol}", className="card-title mb-1"),
                    html.H4(f"${price:.2f}", className=f"mb-1 {price_color}"),
                    html.P(f"{arrow} {change:+.2f} ({change_pct:+.2f}%)", 
                          className=f"small mb-2 {price_color}"),
                    dbc.Badge(
                        f"{prediction} ({confidence:.0%})", 
                        color=pred_color, 
                        className="small"
                    )
                ], className="text-center")
            ], className="h-100")
        ], width=12//min(len(stocks), 6))  # Auto-responsive column width
        
        cards.append(card)
    
    return dbc.Row(cards)

@app.callback(
    Output('price-comparison-chart', 'figure'),
    [Input('multi-stock-data-store', 'data'),
     Input('indicator-selector', 'value')]
)
def update_price_comparison_chart(data, indicators):
    """Update price comparison chart"""
    if not data or 'stocks' not in data:
        return go.Figure().add_annotation(
            text="Select stocks to view price chart",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
    
    stocks = data['stocks']
    
    # Create subplots: price + volume (if selected)
    subplot_rows = 2 if 'VOL' in indicators else 1
    row_heights = [0.7, 0.3] if 'VOL' in indicators else [1.0]
    
    fig = make_subplots(
        rows=subplot_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=['Multi-Stock Price Comparison'] + (['Volume Comparison'] if 'VOL' in indicators else [])
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, (symbol, stock_data) in enumerate(stocks.items()):
        df = pd.DataFrame(stock_data['data'])
        if df.empty:
            continue
            
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        color = colors[i % len(colors)]
        
        # Main price line
        fig.add_trace(
            go.Scatter(
                x=df['Datetime'], 
                y=df['Close'],
                name=f"{symbol} (${stock_data['last_price']:.2f})",
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{symbol}</b><br>" +
                            "Time: %{x}<br>" +
                            "Price: $%{y:.2f}<br>" +
                            "<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'MA' in indicators:
            for ma_period, width, opacity in [(5, 1, 0.8), (20, 1, 0.6), (50, 1, 0.4)]:
                ma_col = f'MA{ma_period}'
                if ma_col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['Datetime'], 
                            y=df[ma_col],
                            name=f"{symbol} MA{ma_period}",
                            line=dict(color=color, width=width, dash='dash'),
                            opacity=opacity,
                            showlegend=False,
                            hovertemplate=f"<b>{symbol} MA{ma_period}</b><br>" +
                                        "Time: %{x}<br>" +
                                        "Price: $%{y:.2f}<br>" +
                                        "<extra></extra>"
                        ),
                        row=1, col=1
                    )
        
        # Kalman filter
        if 'KALMAN' in indicators and 'Kalman_Price' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Datetime'], 
                    y=df['Kalman_Price'],
                    name=f"{symbol} Kalman Filter",
                    line=dict(color=color, width=2, dash='dot'),
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate=f"<b>{symbol} Kalman Filter</b><br>" +
                                "Time: %{x}<br>" +
                                "Price: $%{y:.2f}<br>" +
                                "<extra></extra>"
                ),
                row=1, col=1
            )
        
        # FFT filtering
        if 'FFT' in indicators:
            for fft_period, dash_style in [(21, 'dashdot'), (63, 'longdash')]:
                fft_col = f'FFT_{fft_period}'
                if fft_col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['Datetime'], 
                            y=df[fft_col],
                            name=f"{symbol} FFT{fft_period}",
                            line=dict(color=color, width=1, dash=dash_style),
                            opacity=0.6,
                            showlegend=False,
                            hovertemplate=f"<b>{symbol} FFT{fft_period}</b><br>" +
                                        "Time: %{x}<br>" +
                                        "Price: $%{y:.2f}<br>" +
                                        "<extra></extra>"
                        ),
                        row=1, col=1
                    )
        
        # Bollinger Bands
        if 'BB' in indicators and 'BB_Upper' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Datetime'], y=df['BB_Upper'],
                    name=f"{symbol} BB Upper", line=dict(color=color, width=1),
                    opacity=0.3, showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['Datetime'], y=df['BB_Lower'],
                    name=f"{symbol} BB Lower", line=dict(color=color, width=1),
                    opacity=0.3, showlegend=False,
                    fill='tonexty', fillcolor=f'rgba({",".join(map(str, px.colors.hex_to_rgb(color)))}, 0.1)'
                ),
                row=1, col=1
            )
        
        # Volume
        if 'VOL' in indicators:
            fig.add_trace(
                go.Bar(
                    x=df['Datetime'], 
                    y=df['Volume'],
                    name=f"{symbol} Volume",
                    marker_color=color,
                    opacity=0.6,
                    yaxis='y2'
                ),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

@app.callback(
    Output('predictions-summary', 'children'),
    Input('multi-stock-data-store', 'data')
)
def update_predictions_summary(data):
    """Update prediction summary"""
    if not data or 'stocks' not in data:
        return html.Div("No prediction data available", className="text-center text-muted")
    
    stocks = data['stocks']
    predictions = []
    
    for symbol, stock_data in stocks.items():
        prediction = stock_data['prediction']
        confidence = stock_data['confidence']
        score = stock_data['score']
        
        # Prediction label colors
        pred_colors = {
            'STRONG_BUY': 'success',
            'BUY': 'info', 
            'WEAK_BUY': 'primary',
            'HOLD': 'warning',
            'WEAK_SELL': 'danger',
            'SELL': 'danger',
            'STRONG_SELL': 'dark',
            # English version compatibility
            'STRONG_BUY': 'success',
            'BUY': 'primary',
            'HOLD': 'warning', 
            'SELL': 'danger',
            'STRONG_SELL': 'dark'
        }
        
        pred_icons = {
            'STRONG_BUY': '🚀',
            'BUY': '📈',
            'WEAK_BUY': '🔼',
            'HOLD': '➡️',
            'WEAK_SELL': '🔽',
            'SELL': '📉',
            'STRONG_SELL': '💥',
            # English version compatibility
            'STRONG_BUY': '🚀',
            'BUY': '📈',
            'HOLD': '➡️',
            'SELL': '📉', 
            'STRONG_SELL': '💥'
        }
        
        color = pred_colors.get(prediction, 'secondary')
        icon = pred_icons.get(prediction, '❓')
        
        pred_card = dbc.Card([
            dbc.CardBody([
                html.H6(f"{icon} {symbol}", className="card-title mb-1"),
                dbc.Badge(prediction, color=color, className="mb-1"),
                html.P(f"Confidence: {confidence:.0%}", className="small mb-1"),
                html.P(f"Score: {score:+d}", className="small text-muted mb-0")
            ], className="text-center py-2")
        ], className="mb-2")
        
        predictions.append(pred_card)
    
    # Add summary statistics
    total_stocks = len(stocks)
    buy_signals = sum(1 for stock in stocks.values() 
                     if stock['prediction'] in ['BUY', 'STRONG_BUY', 'WEAK_BUY'])
    sell_signals = sum(1 for stock in stocks.values() 
                      if stock['prediction'] in ['SELL', 'STRONG_SELL', 'WEAK_SELL'])
    
    summary_card = dbc.Card([
        dbc.CardBody([
            html.H6("📊 Summary", className="card-title"),
            html.P(f"Bullish: {buy_signals}/{total_stocks}", className="small mb-1 text-success"),
            html.P(f"Bearish: {sell_signals}/{total_stocks}", className="small mb-1 text-danger"),
            html.P(f"观望: {total_stocks-buy_signals-sell_signals}/{total_stocks}", 
                  className="small mb-0 text-warning")
        ], className="text-center py-2")
    ], color="dark", outline=True)
    
    return html.Div([summary_card, html.Hr()] + predictions)

@app.callback(
    Output('detailed-analysis', 'children'),
    [Input('multi-stock-data-store', 'data'),
     Input('indicator-selector', 'value')]
)
def update_detailed_analysis(data, indicators):
    """Update detailed technical analysis"""
    if not data or 'stocks' not in data or len(data['stocks']) == 0:
        return html.Div()
    
    stocks = data['stocks']
    colors = px.colors.qualitative.Set1
    
    analysis_cards = []
    
    # RSI analysis
    if 'RSI' in indicators:
        rsi_fig = go.Figure()
        for i, (symbol, stock_data) in enumerate(stocks.items()):
            df = pd.DataFrame(stock_data['data'])
            if df.empty or 'RSI' not in df.columns:
                continue
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            color = colors[i % len(colors)]
            
            rsi_fig.add_trace(go.Scatter(
                x=df['Datetime'], y=df['RSI'],
                name=f"{symbol} RSI",
                line=dict(color=color, width=2)
            ))
        
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought(70)")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold(30)")
        rsi_fig.update_layout(template="plotly_dark", height=300, yaxis=dict(range=[0, 100]))
        
        analysis_cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 RSI Relative Strength Index"),
                    dbc.CardBody([dcc.Graph(figure=rsi_fig)])
                ])
            ], width=6)
        )
    
    # MACD analysis
    if 'MACD' in indicators:
        macd_fig = go.Figure()
        for i, (symbol, stock_data) in enumerate(stocks.items()):
            df = pd.DataFrame(stock_data['data'])
            if df.empty or 'MACD' not in df.columns:
                continue
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            color = colors[i % len(colors)]
            
            macd_fig.add_trace(go.Scatter(
                x=df['Datetime'], y=df['MACD'],
                name=f"{symbol} MACD",
                line=dict(color=color, width=2)
            ))
            macd_fig.add_trace(go.Scatter(
                x=df['Datetime'], y=df['MACD_Signal'],
                name=f"{symbol} Signal Line",
                line=dict(color=color, width=1, dash='dash'),
                opacity=0.7
            ))
        
        macd_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        macd_fig.update_layout(template="plotly_dark", height=300)
        
        analysis_cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📈 MACD Moving Average Convergence Divergence"),
                    dbc.CardBody([dcc.Graph(figure=macd_fig)])
                ])
            ], width=6)
        )
    
    # Connors RSI analysis
    if 'CRSI' in indicators:
        crsi_fig = go.Figure()
        for i, (symbol, stock_data) in enumerate(stocks.items()):
            df = pd.DataFrame(stock_data['data'])
            if df.empty or 'CRSI' not in df.columns:
                continue
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            color = colors[i % len(colors)]
            
            crsi_fig.add_trace(go.Scatter(
                x=df['Datetime'], y=df['CRSI'],
                name=f"{symbol} Connors RSI",
                line=dict(color=color, width=2)
            ))
        
        crsi_fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought(80)")
        crsi_fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold(20)")
        crsi_fig.update_layout(template="plotly_dark", height=300, yaxis=dict(range=[0, 100]))
        
        analysis_cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🎯 Connors RSI"),
                    dbc.CardBody([dcc.Graph(figure=crsi_fig)])
                ])
            ], width=6)
        )
    
    # Kalman filter trend analysis
    if 'KALMAN' in indicators:
        kalman_fig = go.Figure()
        for i, (symbol, stock_data) in enumerate(stocks.items()):
            df = pd.DataFrame(stock_data['data'])
            if df.empty or 'Kalman_Trend' not in df.columns:
                continue
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            color = colors[i % len(colors)]
            
            kalman_fig.add_trace(go.Scatter(
                x=df['Datetime'], y=df['Kalman_Trend'],
                name=f"{symbol} Kalman Trend",
                line=dict(color=color, width=2)
            ))
        
        kalman_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        kalman_fig.update_layout(template="plotly_dark", height=300)
        
        analysis_cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🔬 Kalman Filter Trend Analysis"),
                    dbc.CardBody([dcc.Graph(figure=kalman_fig)])
                ])
            ], width=6)
        )
    
    # If no technical indicators are selected, display default RSI and MACD
    if not analysis_cards:
        return html.Div([
            dbc.Alert(
                "Please select technical indicators above to view detailed analysis charts.",
                color="info",
                className="text-center"
            )
        ])
    
    # Arrange cards in rows, maximum 2 per row
    rows = []
    for i in range(0, len(analysis_cards), 2):
        row_cards = analysis_cards[i:i+2]
        rows.append(dbc.Row(row_cards, className="mb-3"))
    
    return html.Div(rows)

# ========================================
# Launch Application
# ========================================

if __name__ == '__main__':
    print("🚀 Launching ATLAS Multi-Stock Financial Dashboard...")
    print("📊 Access at: http://localhost:8050")
    print("⚠️  Note: YFinance has access limits, data updates every 2 minutes")
    print("💡 Supports analysis of up to 8 stocks simultaneously")
    print("🔧 Technical indicators: MA, Bollinger Bands, RSI, MACD, Connors RSI, Kalman Filter, FFT Filter")
    print("🤖 AI predictions based on multi-indicator comprehensive analysis algorithm")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=8050
    )