import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# --- 1. UI SETUP ---
st.set_page_config(page_title="Supply & Demand Backtester", layout="wide")
st.title("XAUT/USD Supply & Demand - Backtesting App")

st.sidebar.header("Strategy Parameters")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=1000)
lot_size = st.sidebar.number_input("Lot Size (oz)", value=1.0) 
momentum_mult = st.sidebar.slider("Momentum Multiplier (ATR)", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
atr_len = st.sidebar.number_input("ATR Length", min_value=5, value=14)
rr_ratio = st.sidebar.slider("Risk Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
days_history = st.sidebar.slider("Days of Data (15m TF)", min_value=5, max_value=59, value=30)

# --- 2. DATA FETCHING ---
@st.cache_data(ttl=900)
def get_data(days):
    df = yf.download("GC=F", period=f"{days}d", interval="15m")
    
    # Strip hidden ticker layers from yfinance columns (The MultiIndex Fix)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df.dropna(inplace=True)
    return df

with st.spinner("Fetching data and mapping zones..."):
    df = get_data(days_history)

# --- 3. CALCULATIONS & ENGINE ---
if not df.empty:
    # Calculate ATR manually
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/atr_len, adjust=False).mean()
    
    # Calculate Candle Body Size
    df['Body'] = abs(df['Close'] - df['Open'])
    
    balance = initial_capital
    equity_curve = []
    trade_log = []
    
    in_pos = False
    pos_type = None
    entry_price, sl, tp = 0, 0, 0
    
    active_demand_zones = []
    active_supply_zones = []
    
    # Plotly shape storage for UI
    zone_shapes = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        date = df.index[i]
        prev_row = df.iloc[i-1]
        
        # 1. Manage Active Positions (Exits)
        if in_pos:
            if pos_type == 'LONG':
                if row['Low'] <= sl: 
                    loss = (sl - entry_price) * lot_size
                    balance += loss
                    trade_log.append({'Date': date, 'Type': 'LONG', 'Entry': entry_price, 'Exit': sl, 'Result': 'Loss', 'PnL': loss})
                    in_pos = False
                elif row['High'] >= tp: 
                    profit = (tp - entry_price) * lot_size
                    balance += profit
                    trade_log.append({'Date': date, 'Type': 'LONG', 'Entry': entry_price, 'Exit': tp, 'Result': 'Win', 'PnL': profit})
                    in_pos = False
            
            elif pos_type == 'SHORT':
                if row['High'] >= sl: 
                    loss = (entry_price - sl) * lot_size
                    balance += loss
                    trade_log.append({'Date': date, 'Type': 'SHORT', 'Entry': entry_price, 'Exit': sl, 'Result': 'Loss', 'PnL': loss})
                    in_pos = False
                elif row['Low'] <= tp: 
                    profit = (entry_price - tp) * lot_size
                    balance += profit
                    trade_log.append({'Date': date, 'Type': 'SHORT', 'Entry': entry_price, 'Exit': tp, 'Result': 'Win', 'PnL': profit})
                    in_pos = False

        equity_curve.append(balance)

        # 2. Check for Zone Mitigation & Entries
        if not in_pos:
            # Check Demand Zones (Buy)
            for z in active_demand_zones[:]:
                if row['Low'] <= z['top']:
                    # Price tapped demand zone -> Go Long
                    entry_price = row['Low'] if row['Open'] > z['top'] else row['Open']
                    sl = z['bottom'] - (row['ATR'] * 0.2)
                    risk = entry_price - sl
                    if risk > 0:
                        tp = entry_price + (risk * rr_ratio)
                        pos_type = 'LONG'
                        in_pos = True
                        active_demand_zones.remove(z) # Mitigated
                        break
            
            # Check Supply Zones (Sell)
            if not in_pos:
                for z in active_supply_zones[:]:
                    if row['High'] >= z['bottom']:
                        # Price tapped supply zone -> Go Short
                        entry_price = row['High'] if row['Open'] < z['bottom'] else row['Open']
                        sl = z['top'] + (row['ATR'] * 0.2)
                        risk = sl - entry_price
                        if risk > 0:
                            tp = entry_price - (risk * rr_ratio)
                            pos_type = 'SHORT'
                            in_pos = True
                            active_supply_zones.remove(z) # Mitigated
                            break

        # 3. Clean up broken zones (Price closed through them)
        active_demand_zones = [z for z in active_demand_zones if row['Close'] > z['bottom']]
        active_supply_zones = [z for z in active_supply_zones if row['Close'] < z['top']]

        # 4. Identify New Zones
        is_expansion = row['Body'] > (row['ATR'] * momentum_mult)
        
        if is_expansion:
            if row['Close'] > row['Open']:
                # Bullish Expansion -> Demand Zone formed by previous candle
                zone = {'top': prev_row['High'], 'bottom': prev_row['Low'], 'start': df.index[i-1]}
                active_demand_zones.append(zone)
                zone_shapes.append(dict(type="rect", x0=zone['start'], y0=zone['bottom'], x1=df.index[-1], y1=zone['top'], fillcolor="rgba(0, 255, 0, 0.1)", line=dict(width=0)))
            
            elif row['Close'] < row['Open']:
                # Bearish Expansion -> Supply Zone formed by previous candle
                zone = {'top': prev_row['High'], 'bottom': prev_row['Low'], 'start': df.index[i-1]}
                active_supply_zones.append(zone)
                zone_shapes.append(dict(type="rect", x0=zone['start'], y0=zone['bottom'], x1=df.index[-1], y1=zone['top'], fillcolor="rgba(255, 0, 0, 0.1)", line=dict(width=0)))

    df['Equity'] = equity_curve
    trade_df = pd.DataFrame(trade_log)

    # --- 4. UI DISPLAY ---
    tab1, tab2 = st.tabs(["Dashboard & Chart", "Trade Log"])

    with tab1:
        total_trades = len(trade_df)
        win_rate = (len(trade_df[trade_df['Result'] == 'Win']) / total_trades * 100) if total_trades > 0 else 0
        net_profit = balance - initial_capital

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Balance", f"${balance:,.2f}", f"{net_profit:,.2f}")
        col2.metric("Total Trades", total_trades)
        col3.metric("Win Rate", f"{win_rate:.1f}%")
        col4.metric("Active Zones", len(active_demand_zones) + len(active_supply_zones))

        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price')])
        
        # Add Trade Markers
        if not trade_df.empty:
            longs = trade_df[trade_df['Type'] == 'LONG']
            shorts = trade_df[trade_df['Type'] == 'SHORT']
            fig.add_trace(go.Scatter(x=longs['Date'], y=longs['Entry'] - 2, mode='markers', marker=dict(color='blue', size=12, symbol='triangle-up'), name='Long Entry'))
            fig.add_trace(go.Scatter(x=shorts['Date'], y=shorts['Entry'] + 2, mode='markers', marker=dict(color='magenta', size=12, symbol='triangle-down'), name='Short Entry'))

        fig.update_layout(height=700, template='plotly_dark', title="Price Action & Supply/Demand Zones", xaxis_rangeslider_visible=False, shapes=zone_shapes[-50:]) # Only draw last 50 shapes to keep mobile fast
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if not trade_df.empty:
            st.dataframe(trade_df.style.map(lambda x: 'color: green' if x == 'Win' else ('color: red' if x == 'Loss' else ''), subset=['Result']))
        else:
            st.info("No trades executed. Try lowering the Momentum Multiplier to find more zones.")

else:
    st.error("Failed to fetch data.")
