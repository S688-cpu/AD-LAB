import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the stock price data
stock_prices = pd.read_csv(r"C:\Users\KIIT0001\OneDrive\Desktop\AD LAB 3 AAPL.csv")

# Print columns to ensure correct column names
print(stock_prices.columns)

# Update the column name for 'Date' correctly
stock_prices['Date'] = pd.to_datetime(stock_prices['Date'])

# Since there is no 'ticker' column, use all the data for now
stock_data = stock_prices[['Date', 'Close']]  # Just use the Date and Close columns

# Sort by date
stock_data = stock_data.sort_values(by='Date')

# Normalize the 'Close' prices for both models (LSTM and Linear Regression)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[['Close']])

# Create sequences for LSTM input
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60  # Use the past 60 days to predict the next day's price
X, y = create_dataset(scaled_data, time_step)

# Reshape X to be 3D as required by LSTM [samples, time steps, features]
X_lstm = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

# --- Linear Regression Model ---
# Prepare the data for Linear Regression (use last "time_step" days' prices)
X_lr = np.array([scaled_data[i:i + time_step].flatten() for i in range(len(scaled_data) - time_step)])
y_lr = scaled_data[time_step:]

# Split Linear Regression data into training and testing
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train_lr, y_train_lr)

# Make predictions using the Linear Regression model
y_pred_lr = model_lr.predict(X_test_lr)

# Evaluate the Linear Regression model
mse_lr = mean_squared_error(y_test_lr, y_pred_lr)
r2_lr = r2_score(y_test_lr, y_pred_lr)

# --- LSTM Model ---
# Build and train the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(LSTM(units=50, return_sequences=False))
model_lstm.add(Dense(units=1))  # Output layer

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Predicting using LSTM model
predicted_prices_lstm = model_lstm.predict(X_test)
predicted_prices_lstm = scaler.inverse_transform(predicted_prices_lstm)  # Inverse scaling

# --- Evaluation of LSTM Model ---
mse_lstm = mean_squared_error(y_test, predicted_prices_lstm)
r2_lstm = r2_score(y_test, predicted_prices_lstm)

# Output the comparison of both models
print(f"Linear Regression Model MSE: {mse_lr}, R2: {r2_lr}")
print(f"LSTM Model MSE: {mse_lstm}, R2: {r2_lstm}")

# Visualizing the predicted vs actual prices for both models
plt.figure(figsize=(10, 6))

# Actual vs Predicted for Linear Regression
plt.subplot(2, 1, 1)
plt.plot(y_test_lr, color='blue', label='Actual Price (LR)')
plt.plot(y_pred_lr, color='red', linestyle='dashed', label='Predicted Price (LR)')
plt.title('Linear Regression Model: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

# Actual vs Predicted for LSTM
plt.subplot(2, 1, 2)
plt.plot(y_test, color='blue', label='Actual Price (LSTM)')
plt.plot(predicted_prices_lstm, color='red', linestyle='dashed', label='Predicted Price (LSTM)')
plt.title('LSTM Model: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

plt.tight_layout()
plt.show()
