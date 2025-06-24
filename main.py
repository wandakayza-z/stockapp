# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
# import matplotlib.pyplot as plt # Comment out matplotlib if not actively using plots in the basic app
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import joblib

# --- Streamlit App Title ---
st.title("Stock Prediction App")

# --- Configuration ---
# Updated path to look in the current directory (repository root)
DATA_FOLDER = "." # "." represents the current directory
FEATURES = ['previous', 'high', 'low', 'volume', 'foreign_buy', 'foreign_sell']

# --- Helper function to get list of stocks ---
# Modify to look in the current directory
@st.cache_data
def get_stock_list(folder):
    """Gets a list of available stock CSV files in the specified folder (or current directory)."""
    # In this case, folder is "."
    if not os.path.exists(folder):
         # This case is unlikely if folder is "." as the current dir always exists
        print(f"Error: Specified folder not found at: {folder}")
        st.error(f"Specified folder not found at: {folder}")
        return []
    try:
        # List files directly in the specified folder (which is the root ".")
        # Filter for files ending with .csv
        items = os.listdir(folder)
        stocks = [f for f in items if os.path.isfile(os.path.join(folder, f)) and f.endswith(".csv")]
        print(f"Found {len(stocks)} stock files in {folder}") # Log how many files found
        if not stocks:
             print(f"No .csv files found in the directory: {folder}")
        return stocks
    except Exception as e:
        print(f"Error listing files in directory {folder}: {e}") # Log any listing errors
        st.error(f"Error listing files in directory {folder}: {e}")
        return []

# --- Data Loading Function ---
# No change needed here, it takes the full path
@st.cache_data
def load_stock_data(stock_file_path, features):
    """Loads and preprocesses data for a single stock."""
    print(f"Attempting to load data from: {stock_file_path}") # Log file path being loaded
    try:
        df = pd.read_csv(stock_file_path)
        print(f"Successfully loaded data with {len(df)} rows from {stock_file_path}") # Log successful load
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.dropna(subset=features + ['close'])
        print(f"Data after preprocessing: {len(df)} rows") # Log rows after dropping NA
        return df
    except Exception as e:
        print(f"Error loading data for {stock_file_path}: {e}") # Log loading error
        st.error(f"Error loading data for {stock_file_path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# --- Model Training Function ---
@st.cache_resource # Use st.cache_resource for models
def train_stock_model(X_train, y_train, ticker):
    """Trains a RandomForestRegressor model."""
    print(f"Training model for {ticker} with {len(X_train)} samples") # Log training start
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    print(f"Model trained for {ticker}") # Log training completion
    return model

# --- Main App Logic ---

# Pass "." to get_stock_list to look in the current directory
stocks = get_stock_list(DATA_FOLDER)

if not stocks:
    st.warning(f"No stock data files found directly in the repository root ({DATA_FOLDER}). Please ensure your CSVs are there.")
else:
    # --- Stock Selection ---
    selected_stock = st.selectbox("Select a Stock", options=stocks)

    if selected_stock:
        # Construct the full path using "."
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
                    # Ensure test set size is not larger than the dataset after dropping NAs
                    test_size = min(test_size, len(df) -1) if len(df) > 1 else 0

                    if test_size > 0:
                         X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=test_size)
                    else:
                         # Handle case where only one data point is left or no data
                         X_train = X
                         y_train = y
                         X_test = pd.DataFrame() # Empty test set
                         y_test = pd.Series() # Empty test set


                    if len(X_train) > 0: # Ensure training set is not empty
                        ticker = selected_stock.replace(".csv", "")
                        model = train_stock_model(X_train, y_train, ticker)

                        # --- Make Predictions ---
                        # Only predict on test set if it's not empty
                        y_pred = model.predict(X_test) if not X_test.empty else np.array([])


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
                        elif len(y_test) == 1:
                            # Can calculate RMSE for a single point, but it's just the absolute error
                            rmse = root_mean_squared_error(y_test, y_pred)
                            st.write(f"**Error on single test point:** {rmse:.2f}")
                        else:
                            st.write("**RMSE:** Cannot calculate RMSE with an empty test set.")

                        st.write(f"**Last Available Close Price:** {y.iloc[-1]:.2f}" if not y.empty else "**Last Available Close Price:** N/A")


                        if next_day_prediction is not None:
                            st.write(f"**Predicted Next Day Close Price:** {next_day_prediction:.2f}")
                        else:
                             st.write("**Next Day Prediction:** Could not make a prediction.")


                        # Optional: Display actual vs predicted plot for the test set
                        # Consider commenting this out initially to simplify debugging
                        # if not y_test.empty:
                        #     st.subheader("Actual vs Predicted (Test Set)")
                        #     fig, ax = plt.subplots()
                        #     ax.plot(y_test.index, y_test.values, label='Actual')
                        #     ax.plot(y_test.index, y_pred, label='Predicted')
                        #     ax.set_title(f'{ticker} - Actual vs Predicted Prices')
                        #     ax.set_xlabel('Data Index (Time)') # Note: Index is not necessarily date here
                        #     ax.set_ylabel('Close Price')
                        #     ax.legend()
                        #     st.pyplot(fig)

                    else:
                         st.warning(f"Data split resulted in an empty training set for {selected_stock}. Cannot train model.")

                except Exception as e:
                    print(f"An error occurred during model training or prediction for {selected_stock}: {e}") # Log exception details
                    st.error(f"An error occurred during model training or prediction for {selected_stock}: {e}")
        else:
            st.warning(f"Could not load or preprocess data for {selected_stock}. Please check the file.")