# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import joblib # Keep joblib in case you want to save/load models separately

# --- Streamlit App Title ---
st.title("Stock Prediction App")

# --- Configuration ---
DATA_FOLDER = "/content/sample_data/data"
FEATURES = ['previous', 'high', 'low', 'volume', 'foreign_buy', 'foreign_sell']

# --- Helper function to get list of stocks ---
@st.cache_data
def get_stock_list(folder):
    """Gets a list of available stock CSV files."""
    if not os.path.exists(folder):
        st.error(f"Data folder not found at: {folder}")
        return []
    stocks = [f for f in os.listdir(folder) if f.endswith(".csv")]
    return stocks

# --- Data Loading Function ---
@st.cache_data
def load_stock_data(stock_file_path, features):
    """Loads and preprocesses data for a single stock."""
    try:
        df = pd.read_csv(stock_file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.dropna(subset=features + ['close'])
        return df
    except Exception as e:
        st.error(f"Error loading data for {stock_file_path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# --- Model Training Function ---
@st.cache_resource # Use st.cache_resource for models
def train_stock_model(X_train, y_train, ticker):
    """Trains a RandomForestRegressor model."""
    # You could add logic here to load a pre-trained model if it exists
    # For now, we retrain for the selected stock
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    # Optional: Save the model if you want to reuse it later without retraining
    # joblib.dump(model, f'{ticker}_model.pkl')
    return model

# --- Main App Logic ---

stocks = get_stock_list(DATA_FOLDER)

if not stocks:
    st.warning("No stock data files found in the specified folder.")
else:
    # --- Stock Selection ---
    selected_stock = st.selectbox("Select a Stock", options=stocks)

    if selected_stock:
        stock_file_path = os.path.join(DATA_FOLDER, selected_stock)
        df = load_stock_data(stock_file_path, FEATURES)

        if not df.empty:
            X = df[FEATURES]
            y = df['close']

            # Ensure enough data points for splitting and prediction
            if len(df) < 2 or len(X) == 0 or len(y) == 0:
                st.warning(f"Not enough data points or missing 'close' column for {selected_stock} to perform prediction.")
            else:
                # Use a fixed test size, ensure at least 1 data point for testing
                test_size = max(0.2, 1 / len(df)) # Ensure test_size is at least large enough for 1 sample

                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=test_size)

                    if len(X_train) > 0 and len(X_test) > 0:
                        ticker = selected_stock.replace(".csv", "")
                        model = train_stock_model(X_train, y_train, ticker)

                        # --- Make Predictions ---
                        y_pred = model.predict(X_test)

                        # Predict the next day's price based on the last available features
                        next_day_prediction = None
                        if not X.empty:
                            next_day_features = X.iloc[-1].values.reshape(1, -1)
                            next_day_prediction = model.predict(next_day_features)[0]

                        # --- Display Results ---
                        st.subheader(f"Prediction Results for {ticker}")

                        # Calculate RMSE only if test set has more than one sample
                        if len(y_test) > 1:
                             rmse = root_mean_squared_error(y_test, y_pred)
                             st.write(f"**RMSE:** {rmse:.2f}")
                        else:
                            st.write("**RMSE:** Cannot calculate RMSE with only one test sample.")

                        st.write(f"**Last Available Close Price:** {y.iloc[-1]:.2f}")

                        if next_day_prediction is not None:
                            st.write(f"**Predicted Next Day Close Price:** {next_day_prediction:.2f}")
                        else:
                             st.write("**Next Day Prediction:** Could not make a prediction.")


                        # Optional: Display actual vs predicted plot for the test set
                        st.subheader("Actual vs Predicted (Test Set)")
                        fig, ax = plt.subplots()
                        ax.plot(y_test.index, y_test.values, label='Actual')
                        ax.plot(y_test.index, y_pred, label='Predicted')
                        ax.set_title(f'{ticker} - Actual vs Predicted Prices')
                        ax.set_xlabel('Data Index (Time)') # Note: Index is not necessarily date here
                        ax.set_ylabel('Close Price')
                        ax.legend()
                        st.pyplot(fig)

                    else:
                         st.warning(f"Data split resulted in empty training or testing set for {selected_stock}.")

                except Exception as e:
                    st.error(f"An error occurred during model training or prediction for {selected_stock}: {e}")
        else:
            st.warning(f"Could not load or preprocess data for {selected_stock}. Please check the file.")