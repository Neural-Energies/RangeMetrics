import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go

st.title('Stock Analysis App')

symbol = st.text_input('Enter stock symbol', 'AAPL')
start_date = st.date_input('Start date', value=pd.to_datetime('2000-01-01'))
end_date = st.date_input('End date', value=pd.to_datetime('2023-07-07'))

option_analysis = st.selectbox('Which analysis do you want to display?', ['Gap Analysis', 'Range Metrics'])

def analyze_stock_gap(df):
    df['Prev Close'] = df['Close'].shift()
    df['Prev Low'] = df['Low'].shift()
    df['Prev High'] = df['High'].shift()

    df['Reg Gap Up'] = df['Open'] > df['Prev Close']
    df['Reg Gap Down'] = df['Open'] < df['Prev Close']
    df['Alt Gap Up'] = df['Open'] > df['Prev High']
    df['Alt Gap Down'] = df['Open'] < df['Prev Low']

    df['Reg Gap Fill'] = df['Reg Gap Up'] & (df['Low'] <= df['Prev Close']) | df['Reg Gap Down'] & (df['High'] >= df['Prev Close'])
    df['Alt Gap Fill'] = df['Alt Gap Up'] & (df['Low'] <= df['Prev High']) | df['Alt Gap Down'] & (df['High'] >= df['Prev Low'])

    df['Reg Gap Up Size'] = df['Open'] - df['Prev Close']
    df['Reg Gap Down Size'] = abs(df['Prev Close'] - df['Open'])

    df['Alt Gap Up Size'] = df['Open'] - df['Prev High']
    df['Alt Gap Down Size'] = abs(df['Prev Low'] - df['Open'])

    df = df.dropna()

    return df

if option_analysis == 'Gap Analysis':
    gap_types = ['Regular Gap Up', 'Regular Gap Down', 'Alternative Gap Up', 'Alternative Gap Down']
    selected_gap_type = st.selectbox('Select a gap type', gap_types)

    df = yf.download(symbol, start=start_date, end=end_date, interval='1d')
    df = analyze_stock_gap(df)

    if df is not None:
        gap_sizes = None
        if selected_gap_type == 'Regular Gap Up':
            gap_sizes = df[df['Reg Gap Up']]['Reg Gap Up Size']
        elif selected_gap_type == 'Regular Gap Down':
            gap_sizes = df[df['Reg Gap Down']]['Reg Gap Down Size']
        elif selected_gap_type == 'Alternative Gap Up':
            gap_sizes = df[df['Alt Gap Up']]['Alt Gap Up Size']
        elif selected_gap_type == 'Alternative Gap Down':
            gap_sizes = df[df['Alt Gap Down']]['Alt Gap Down Size']

        selected_gap_size = st.number_input('Select a gap size', min_value=int(gap_sizes.min()), max_value=100, step=1, value=int(gap_sizes.min()), format="%i")

        submit_button = st.button('Submit Gap Size')

        if submit_button:
            gap_df = gap_sizes[(gap_sizes > selected_gap_size - 1) & (gap_sizes <= selected_gap_size)]
            
            if gap_df.empty:
                st.write(f"No gaps between {selected_gap_size - 1} and {selected_gap_size} were found.")
            else:
                gap_occurrence_prob = len(gap_df) / len(df)
                st.subheader(selected_gap_type)
                st.write(f"Probability that a gap between {selected_gap_size - 1} and {selected_gap_size} points occurs: {gap_occurrence_prob:.2%}")
                if selected_gap_type.startswith('Regular'):
                    gap_fill_prob = df.loc[gap_df.index, 'Reg Gap Fill'].mean()
                    st.write(f"Gap Fill Probability: {gap_fill_prob:.2%}")
                else:
                    gap_fill_prob = df.loc[gap_df.index, 'Alt Gap Fill'].mean()
                    st.write(f"Gap Fill Probability: {gap_fill_prob:.2%}")


# Rest of the code for 'Range Metrics' option...

elif option_analysis == 'Range Metrics':
    # Allow the user to select the frequency
    frequency = st.selectbox('Select Frequency', ['Daily', 'Weekly', 'Monthly', 'Quarterly'])

    # Map the selection to the corresponding interval
    interval_map = {
        'Daily': '1d',
        'Weekly': '1wk',
        'Monthly': '1mo',
        'Quarterly': '3mo'
    }
    interval = interval_map[frequency]

    # Download historical market data at the selected interval
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval)

    # Continue with your analysis using df...

    # List of metrics
    metrics = ['Day High - Day Open', 'Open - Low', 'Open - Close', 'Close - Mid',
               'Open - Open.shift(1)', 'Close - Close.shift(1)', 'Mid - Mid.shift(1)',
               'High - High.shift(1)', 'Low - Low.shift(1)', 'High - Low',
               'Close - Open.shift(1)', 'Open - Close.shift(1)', 'Average Volume',
               'Volume Change', 'Relative Volume', 'ATR', 'Historical Volatility',
               'ROC', 'RSI', 'Percent Change', 'Range']

    # Drop Down list for the user to select which metrics to display
    option = st.selectbox('Which metric do you want to display?', metrics)
   

    def analyze_stock_metrics(df):
        df['Day High - Day Open'] = df['High'] - df['Open']
        df['Open - Low'] = df['Open'] - df['Low']
        df['Open - Close'] = np.abs(df['Open'] - df['Close'])
        df['Close - Mid'] = np.abs(df['Close'] - ((df['High'] + df['Low']) / 2))
        df['Open - Open.shift(1)'] = np.abs(df['Open'] - df['Open'].shift(1))
        df['Close - Close.shift(1)'] = np.abs(df['Close'] - df['Close'].shift(1))
        df['Mid - Mid.shift(1)'] = np.abs(((df['High'] + df['Low']) / 2) - ((df['High'].shift(1) + df['Low'].shift(1)) / 2))
        df['High - High.shift(1)'] = np.abs(df['High'] - df['High'].shift(1))
        df['Low - Low.shift(1)'] = np.abs(df['Low'] - df['Low'].shift(1))
        df['High - Low'] = df['High'] - df['Low']
        df['Close - Open.shift(1)'] = np.abs(df['Close'] - df['Open'].shift(1))
        df['Open - Close.shift(1)'] = np.abs(df['Open'] - df['Close'].shift(1))
        df['Average Volume'] = df['Volume'].rolling(window=20).mean()  # 20-day average
        df['Volume Change'] = df['Volume'].diff()
        df['Relative Volume'] = df['Volume'] / df['Average Volume']

        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()  # ATR over 14 days

        df['Log Return'] = np.log(df['Close'] / df['Close'].shift())
        df['Historical Volatility'] = df['Log Return'].rolling(window=21).std() * np.sqrt(252)  # 21-day HV

        df['ROC'] = df['Close'].pct_change(periods=10)  # 10-day ROC

        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        df['RSI'] = 100 - (100 / (1 + rs))

        df['Percent Change'] = df['Close'].pct_change() * 100
        df['Range'] = df['High'] - df['Low']
        df = df.dropna()

        return df

    if st.button('Analyze Metrics'):
        # Compute metrics...
        df = analyze_stock_metrics(df)

        # Calculate statistics for the selected metric
        mean = df[option].mean()
        median = df[option].median()
        mode_series = df[option].mode()
        mode = mode_series[0] if not mode_series.empty else None
        max_val = df[option].max()

        # Print out the mean, mode, median, and max values
        st.write(f"Mean: {mean:.2f}")
        st.write(f"Median: {median:.2f}")
        if mode is not None:
            st.write(f"Mode: {mode:.2f}")
        else:
            st.write("Mode: No mode found")
        st.write(f"Max: {max_val:.2f}")

        # Draw an interactive plot for the selected metric
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[option], mode='lines+markers', name=option))

        fig.add_shape(
            type='line',
            x0=df.index.min(), y0=mean, x1=df.index.max(), y1=mean,
            line=dict(color='Red', width=4, dash='dash'),
            name='Mean'
        )

        fig.add_shape(
            type='line',
            x0=df.index.min(), y0=median, x1=df.index.max(), y1=median,
            line=dict(color='Green', width=4, dash='dot'),
            name='Median'
        )

        fig.add_shape(
            type='line',
            x0=df.index.min(), y0=mode, x1=df.index.max(), y1=mode,
            line=dict(color='Blue', width=4, dash='longdash'),
            name='Mode'
        )

        fig.add_shape(
            type='line',
            x0=df.index.min(), y0=max_val, x1=df.index.max(), y1=max_val,
            line=dict(color='Purple', width=4, dash='dashdot'),
            name='Max'
        )

        fig.update_layout(title=f"{option} over time for {symbol}", xaxis_title='Date', yaxis_title=option)
        st.plotly_chart(fig)
