from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)

# Load the datasets and prepare the model
def train_model():
    stock_prices = pd.read_csv(r"C:\Users\KIIT\Desktop\lab03\AAPL.csv")
    
    # Process the data
    stock_prices['Date'] = pd.to_datetime(stock_prices['Date'])
    stock_prices = stock_prices.sort_values(by='Date')
    stock_prices['Target'] = stock_prices['Close'].shift(-1)
    stock_prices = stock_prices.dropna()

    features = ['Close', 'Volume']
    X = stock_prices[features]
    y = stock_prices['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'linear_regression_model.pkl')
    return model

# Load or train the model
if os.path.exists('linear_regression_model.pkl'):
    model = joblib.load('linear_regression_model.pkl')
else:
    model = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        close = float(data['close'])
        volume = float(data['volume'])
        prediction = model.predict([[close, volume]])[0]
        return jsonify({'predicted_price': prediction})
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'Prediction failed. Please check your input and try again.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
