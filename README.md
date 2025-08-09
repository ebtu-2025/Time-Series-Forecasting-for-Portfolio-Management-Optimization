# Time-Series-Forecasting-for-Portfolio-Management-Optimization
Time Series Forecasting for Portfolio Management Optimization
________________________________________
Task 1: Preprocess and Explore the Data
1.1 Load Historical Data (TSLA, BND, SPY) Using yfinance
python
CopyEdit
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# Define assets and date range
assets = ['TSLA', 'BND', 'SPY']
start_date = '2015-07-01'
end_date = '2025-07-31'

# Download data
data = {}
for asset in assets:
    ticker = yf.Ticker(asset)
    df = ticker.history(start=start_date, end=end_date)
    data[asset] = df

# Example: Check TSLA data
data['TSLA'].head()
1.2 Data Cleaning and Preprocessing
•	Check for missing values and data types.
•	Handle missing values with forward fill or interpolation if any.
•	Convert dates to datetime and set as index.
•	Keep only relevant columns (e.g., ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Adj Close']).
python
CopyEdit
for asset in assets:
    df = data[asset]
    # Check missing
    print(f"{asset} missing values:\n", df.isnull().sum())
    
    # Forward fill missing values if any
    df.fillna(method='ffill', inplace=True)
    data[asset] = df
1.3 Basic Statistics
python
CopyEdit
for asset in assets:
    print(f"\n{asset} Summary Statistics:")
    print(data[asset]['Close'].describe())
1.4 Exploratory Data Analysis (EDA)
1.4.1 Closing Price Trends
python
CopyEdit
plt.figure(figsize=(14, 7))
for asset in assets:
    plt.plot(data[asset]['Close'], label=asset)
plt.title('Closing Price Trends (2015-2025)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
1.4.2 Daily Returns and Volatility
Calculate daily percentage change (returns) and rolling volatility (standard deviation):
python
CopyEdit
for asset in assets:
    df = data[asset]
    df['Daily_Return'] = df['Adj Close'].pct_change()
    df['Rolling_Volatility_30d'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(252)  # Annualized vol
    data[asset] = df

plt.figure(figsize=(14, 7))
plt.plot(data['TSLA']['Daily_Return'], label='TSLA Daily Return', alpha=0.5)
plt.title('TSLA Daily Returns')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(data['TSLA']['Rolling_Volatility_30d'], label='TSLA 30-Day Rolling Volatility')
plt.title('TSLA Rolling Volatility (Annualized)')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.show()
1.4.3 Outlier Detection on Returns
python
CopyEdit
# Plot histogram and boxplot for TSLA daily returns
plt.figure(figsize=(12, 5))
sns.histplot(data['TSLA']['Daily_Return'].dropna(), bins=100, kde=True)
plt.title('TSLA Daily Return Distribution')
plt.show()

plt.figure(figsize=(12, 3))
sns.boxplot(x=data['TSLA']['Daily_Return'].dropna())
plt.title('TSLA Daily Return Boxplot')
plt.show()
Look for days with unusually high or low returns:
python
CopyEdit
outliers = data['TSLA'][data['TSLA']['Daily_Return'].abs() > 0.1]
print("Days with >10% daily returns:\n", outliers[['Daily_Return']])
1.5 Stationarity Test (Augmented Dickey-Fuller)
Check stationarity of closing prices and returns:
python
CopyEdit
def adf_test(series, title=''):
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(), autolag='AIC')
    labels = ['ADF Statistic', 'p-value', '# Lags Used', '# Observations']
    for value, label in zip(result[:4], labels):
        print(f'{label} : {value}')
    if result[1] <= 0.05:
        print("=> Strong evidence against the null hypothesis, data is stationary.")
    else:
        print("=> Weak evidence against null hypothesis, data is non-stationary.")

# Test on TSLA Adj Close
adf_test(data['TSLA']['Adj Close'], 'TSLA Adjusted Close Price')

# Test on TSLA Daily Returns
adf_test(data['TSLA']['Daily_Return'], 'TSLA Daily Returns')
Interpretation:
Typically, stock prices are non-stationary (require differencing), daily returns are usually stationary, making them better for modeling.
1.6 Risk Metrics: Value at Risk (VaR) and Sharpe Ratio (Annualized)
Calculate for TSLA daily returns:
python
CopyEdit
# VaR at 5% confidence level (historical method)
VaR_5 = np.percentile(data['TSLA']['Daily_Return'].dropna(), 5)
print(f"TSLA 5% Historical VaR: {VaR_5:.4f}")

# Sharpe Ratio (Assuming risk-free rate ~0 for simplicity)
mean_return = data['TSLA']['Daily_Return'].mean() * 252  # Annualized
std_return = data['TSLA']['Daily_Return'].std() * np.sqrt(252)  # Annualized volatility
sharpe_ratio = mean_return / std_return
print(f"TSLA Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
________________________________________
Summary of Task 1
•	Loaded and cleaned 10 years of data for TSLA, BND, SPY.
•	Explored price trends, volatility, and daily returns.
•	Tested stationarity showing that prices are non-stationary but returns are stationary.
•	Calculated risk metrics providing baseline insights on Tesla’s risk and returns.
Task 2: Develop Time Series Forecasting Models
We will build and compare:
•	ARIMA (AutoRegressive Integrated Moving Average) — classical statistical model
•	LSTM (Long Short-Term Memory) — deep learning model for sequence data
________________________________________
2.1 Data Preparation for Modeling
•	We will forecast Tesla's Adjusted Close price.
•	Split data chronologically:
o	Train: July 1, 2015 – Dec 31, 2023
o	Test: Jan 1, 2024 – July 31, 2025
python
CopyEdit
tsla = data['TSLA'][['Adj Close']].copy()

train_end = '2023-12-31'
train = tsla.loc[:train_end]
test = tsla.loc[train_end:]

print(f"Train size: {train.shape[0]}, Test size: {test.shape[0]}")
________________________________________
2.2 ARIMA Model
2.2.1 Check Stationarity and Differencing
•	Earlier we saw prices are non-stationary → we difference the series once (d=1).
2.2.2 Use pmdarima's auto_arima for Parameter Tuning
python
CopyEdit
import pmdarima as pm

# Auto ARIMA on training data
arima_model = pm.auto_arima(train, seasonal=False, stepwise=True,
                            suppress_warnings=True, max_p=5, max_q=5, d=1,
                            trace=True, error_action='ignore')

print(arima_model.summary())
2.2.3 Fit ARIMA and Forecast
python
CopyEdit
# Fit model
arima_model.fit(train)

# Forecast length = length of test set
n_forecast = len(test)
arima_forecast, conf_int = arima_model.predict(n_periods=n_forecast, return_conf_int=True)

# Convert to DataFrame for convenience
forecast_index = test.index
arima_pred = pd.Series(arima_forecast, index=forecast_index)
2.2.4 Visualize Forecast vs Actual
python
CopyEdit
plt.figure(figsize=(14,7))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(arima_pred.index, arima_pred, label='ARIMA Forecast')
plt.fill_between(arima_pred.index, conf_int[:,0], conf_int[:,1], color='pink', alpha=0.3)
plt.title('ARIMA Model Forecast vs Actual (TSLA Adj Close)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
2.2.5 Evaluate ARIMA Performance
python
CopyEdit
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae_arima = mean_absolute_error(test, arima_pred)
rmse_arima = mean_squared_error(test, arima_pred, squared=False)
mape_arima = np.mean(np.abs((test.values - arima_pred.values) / test.values)) * 100

print(f"ARIMA MAE: {mae_arima:.2f}")
print(f"ARIMA RMSE: {rmse_arima:.2f}")
print(f"ARIMA MAPE: {mape_arima:.2f}%")
________________________________________
2.3 LSTM Model
2.3.1 Data Scaling & Sequence Preparation
LSTMs require scaled input and sequences.
python
CopyEdit
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
tsla_scaled = scaler.fit_transform(tsla)

# Create sequences: use past 60 days to predict next day
def create_sequences(data, seq_length=60):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 60
X, y = create_sequences(tsla_scaled, seq_length)

# Split to train/test
train_size = train.shape[0]
X_train, y_train = X[:train_size - seq_length], y[:train_size - seq_length]
X_test, y_test = X[train_size - seq_length:], y[train_size - seq_length:]
2.3.2 Build LSTM Model
python
CopyEdit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model.summary()
2.3.3 Train the Model
python
CopyEdit
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
2.3.4 Make Predictions and Inverse Scale
python
CopyEdit
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test)

# Prepare index for test set
test_dates = tsla.index[train_size:]
y_pred_series = pd.Series(y_pred.flatten(), index=test_dates)
y_test_series = pd.Series(y_test_actual.flatten(), index=test_dates)
2.3.5 Visualize LSTM Predictions vs Actual
python
CopyEdit
plt.figure(figsize=(14,7))
plt.plot(tsla.index[:train_size], tsla.iloc[:train_size], label='Train')
plt.plot(test_dates, y_test_series, label='Test Actual')
plt.plot(test_dates, y_pred_series, label='LSTM Forecast')
plt.title('LSTM Model Forecast vs Actual (TSLA Adj Close)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
2.3.6 Evaluate LSTM Performance
python
CopyEdit
mae_lstm = mean_absolute_error(y_test_series, y_pred_series)
rmse_lstm = mean_squared_error(y_test_series, y_pred_series, squared=False)
mape_lstm = np.mean(np.abs((y_test_series.values - y_pred_series.values) / y_test_series.values)) * 100

print(f"LSTM MAE: {mae_lstm:.2f}")
print(f"LSTM RMSE: {rmse_lstm:.2f}")
print(f"LSTM MAPE: {mape_lstm:.2f}%")
________________________________________
2.4 Model Comparison and Discussion
Metric	ARIMA	LSTM
MAE	(from above)	(from above)
RMSE	(from above)	(from above)
MAPE (%)	(from above)	(from above)
Discussion:
•	ARIMA is simpler and interpretable but might struggle with complex nonlinear patterns.
•	LSTM can capture nonlinear temporal dependencies but requires more data preprocessing and training time.
•	Choose the model with better test performance (lower MAE, RMSE, MAPE) and reasonable interpretability for next tasks.
Task 3: Forecast Future Market Trends (TSLA Stock Price)
________________________________________
3.1 Generate Future Forecast (6–12 months)
We'll forecast Tesla's adjusted closing price for the next 12 months beyond July 31, 2025.
________________________________________
3.1.1 Prepare Input Data for Forecasting
•	Since LSTM needs sequences, we start from the last 60 days available (July 31, 2025).
•	We'll iteratively predict day-by-day, appending predictions to input to forecast next day.
python
CopyEdit
import datetime

forecast_horizon = 252  # Approx 1 trading year (12 months)

# Last 60 days of scaled data for starting point
last_60_days = tsla_scaled[-seq_length:].reshape(1, seq_length, 1)

future_predictions_scaled = []
current_input = last_60_days.copy()

for _ in range(forecast_horizon):
    pred = model.predict(current_input)[0, 0]
    future_predictions_scaled.append(pred)
    
    # Append prediction and remove oldest to keep sequence length constant
    current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

# Inverse scale predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
________________________________________
3.1.2 Create Future Date Index (Trading Days Approximation)
python
CopyEdit
last_date = tsla.index[-1]
future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)

forecast_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Forecasted_Adj_Close'])
________________________________________
3.2 Visualize Forecast with Historical Data and Confidence Interval Proxy
Since LSTM does not directly give confidence intervals, we can visualize a shaded area using rolling volatility as an uncertainty proxy.
python
CopyEdit
plt.figure(figsize=(14, 7))
plt.plot(tsla['Adj Close'], label='Historical TSLA Price')
plt.plot(forecast_df.index, forecast_df['Forecasted_Adj_Close'], label='LSTM Forecast', color='orange')

# Approximate confidence interval: +/- 1 std deviation of recent returns scaled to price level
last_volatility = data['TSLA']['Rolling_Volatility_30d'][-1] / np.sqrt(252)  # daily vol
conf_upper = forecast_df['Forecasted_Adj_Close'] * (1 + last_volatility)
conf_lower = forecast_df['Forecasted_Adj_Close'] * (1 - last_volatility)

plt.fill_between(forecast_df.index, conf_lower, conf_upper, color='orange', alpha=0.2, label='Approx. Confidence Interval')

plt.title('TSLA Adjusted Close Price Forecast (12 Months)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
________________________________________
3.3 Interpret Results
Trend Analysis
•	Long-term trend: Look at the slope of forecasted prices — upward, downward, or flat.
•	Patterns: Check for sudden jumps or dips.
•	Anomalies: Large deviations compared to historical volatility.
Volatility and Risk
•	The confidence interval width remains constant here (proxy based on recent volatility).
•	In reality, uncertainty usually grows over longer forecast horizons.
•	A narrow confidence interval implies more certainty; wider intervals imply higher risk.
Market Opportunities and Risks
•	Opportunities: If the forecast shows sustained price growth, it might signal buy/hold positions.
•	Risks: High volatility or predicted declines warn of potential losses or need for risk management.
Task 4: Portfolio Optimization Using Modern Portfolio Theory (MPT)
________________________________________
4.1 Calculate Expected Returns and Covariance Matrix
4.1.1 TSLA Expected Return from Forecast
•	Calculate expected return from the LSTM forecasted price series:
python
CopyEdit
# Calculate daily returns from forecasted prices
forecast_df['Daily_Return'] = forecast_df['Forecasted_Adj_Close'].pct_change()

# Annualize expected return (mean daily return * 252 trading days)
tsla_expected_return = forecast_df['Daily_Return'].mean() * 252
print(f"TSLA Expected Annual Return (Forecasted): {tsla_expected_return:.4f}")
________________________________________
4.1.2 BND and SPY Historical Returns
Calculate historical daily returns and annualize mean returns:
python
CopyEdit
returns_df = pd.DataFrame()

for asset in ['BND', 'SPY']:
    df = data[asset]
    df['Daily_Return'] = df['Adj Close'].pct_change()
    returns_df[asset] = df['Daily_Return']

# Annualized mean returns
bnd_expected_return = returns_df['BND'].mean() * 252
spy_expected_return = returns_df['SPY'].mean() * 252

print(f"BND Expected Annual Return (Historical): {bnd_expected_return:.4f}")
print(f"SPY Expected Annual Return (Historical): {spy_expected_return:.4f}")
________________________________________
4.1.3 Covariance Matrix of Daily Returns
Use combined daily returns for all three assets:
python
CopyEdit
# Combine all returns including TSLA historical returns (for covariance matrix)
returns_df['TSLA'] = data['TSLA']['Adj Close'].pct_change()

# Drop NA rows
returns_df = returns_df.dropna()

# Covariance matrix (annualized)
cov_matrix = returns_df.cov() * 252

print("Annualized Covariance Matrix:")
print(cov_matrix)
________________________________________
4.2 Portfolio Optimization Using PyPortfolioOpt
python
CopyEdit
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Expected returns vector (TSLA forecast + historical BND, SPY)
mu = np.array([tsla_expected_return, bnd_expected_return, spy_expected_return])

# Align order with returns_df columns
assets_order = ['TSLA', 'BND', 'SPY']
cov_matrix = cov_matrix.loc[assets_order, assets_order]

# Initialize Efficient Frontier optimizer
ef = EfficientFrontier(mu, cov_matrix)

# Find max Sharpe ratio portfolio
max_sharpe_weights = ef.max_sharpe()
ef.portfolio_performance(verbose=True)

# Find minimum volatility portfolio
ef_minvol = EfficientFrontier(mu, cov_matrix)
min_vol_weights = ef_minvol.min_volatility()
ef_minvol.portfolio_performance(verbose=True)
________________________________________
4.3 Plot Efficient Frontier
python
CopyEdit
import matplotlib.pyplot as plt

ef = EfficientFrontier(mu, cov_matrix)
fig, ax = plt.subplots(figsize=(10,6))
ef.plot_efficient_frontier(ax=ax, show_assets=True)

# Mark Max Sharpe ratio portfolio
ret_sharpe, vol_sharpe, _ = ef.portfolio_performance()
ax.scatter(vol_sharpe, ret_sharpe, marker="*", s=200, c='r', label='Max Sharpe')

# Mark Min Volatility portfolio
ret_minvol, vol_minvol, _ = ef_minvol.portfolio_performance()
ax.scatter(vol_minvol, ret_minvol, marker="X", s=200, c='g', label='Min Volatility')

plt.title('Efficient Frontier with Key Portfolios')
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Return')
plt.legend()
plt.show()
________________________________________
4.4 Final Recommended Portfolio Summary
Example output (replace with your actual numbers):
Asset	Max Sharpe Weights	Min Volatility Weights
TSLA	35%	10%
BND	15%	50%
SPY	50%	40%
•	Max Sharpe portfolio prioritizes highest risk-adjusted return. Good for growth-seeking investors comfortable with volatility.
•	Min Volatility portfolio prioritizes lower risk, suitable for conservative investors.
Task 5: Strategy Backtesting (Aug 1, 2024 – July 31, 2025)
________________________________________
5.1 Define Backtesting Period and Benchmark
python
CopyEdit
backtest_start = '2024-08-01'
backtest_end = '2025-07-31'

# Extract price data for backtesting period
tsla_bt = data['TSLA'].loc[backtest_start:backtest_end]['Adj Close']
bnd_bt = data['BND'].loc[backtest_start:backtest_end]['Adj Close']
spy_bt = data['SPY'].loc[backtest_start:backtest_end]['Adj Close']

# Benchmark: Static 60% SPY, 40% BND portfolio
benchmark_weights = np.array([0, 0.4, 0.6])  # TSLA=0, BND=0.4, SPY=0.6
________________________________________
5.2 Compute Daily Returns for Backtesting
python
CopyEdit
bt_prices = pd.DataFrame({'TSLA': tsla_bt, 'BND': bnd_bt, 'SPY': spy_bt})
bt_returns = bt_prices.pct_change().dropna()
________________________________________
5.3 Simulate Strategy Portfolio Performance
Use weights from Max Sharpe portfolio from Task 4 (replace with your actual weights):
python
CopyEdit
# Example weights, replace with your optimized portfolio weights
strategy_weights = np.array([
    max_sharpe_weights.get('TSLA', 0),
    max_sharpe_weights.get('BND', 0),
    max_sharpe_weights.get('SPY', 0)
])

# Calculate daily portfolio returns
strategy_returns = bt_returns.dot(strategy_weights)

# Calculate benchmark returns
benchmark_returns = bt_returns.dot(benchmark_weights)
________________________________________
5.4 Calculate Cumulative Returns
python
CopyEdit
strategy_cum_returns = (1 + strategy_returns).cumprod() - 1
benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
________________________________________
5.5 Plot Cumulative Returns Comparison
python
CopyEdit
plt.figure(figsize=(14, 7))
plt.plot(strategy_cum_returns, label='Strategy Portfolio')
plt.plot(benchmark_cum_returns, label='Benchmark (60% SPY / 40% BND)')
plt.title('Backtest: Cumulative Returns Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()
________________________________________
5.6 Calculate Performance Metrics: Sharpe Ratio & Total Return
Assuming risk-free rate = 0 for simplicity:
python
CopyEdit
def sharpe_ratio(returns):
    return returns.mean() / returns.std() * np.sqrt(252)

strategy_sharpe = sharpe_ratio(strategy_returns)
benchmark_sharpe = sharpe_ratio(benchmark_returns)

strategy_total_return = strategy_cum_returns[-1]
benchmark_total_return = benchmark_cum_returns[-1]

print(f"Strategy Total Return: {strategy_total_return:.2%}")
print(f"Benchmark Total Return: {benchmark_total_return:.2%}")
print(f"Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
print(f"Benchmark Sharpe Ratio: {benchmark_sharpe:.2f}")
________________________________________
5.7 Summary
•	If strategy total return and Sharpe ratio exceed the benchmark’s, your forecast-driven portfolio optimization has added value.
•	Otherwise, consider revisiting model assumptions, rebalancing frequency, or incorporating additional data.

