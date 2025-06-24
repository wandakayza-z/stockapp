import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt # Keep matplotlib for plotting
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import joblib

# --- Konfigurasi Tampilan Streamlit ---

st.set_page_config(layout="wide", page_title="Stock Prediction App") 

st.title("Aplikasi Prediksi Saham") 

# --- Konfigurasi Data ---

DATA_FOLDER = "."
FEATURES = ['previous', 'high', 'low', 'volume', 'foreign_buy', 'foreign_sell']

@st.cache_data
def get_stock_list(folder):
    """Mendapatkan daftar file CSV saham yang tersedia."""
    try:
        items = os.listdir(folder)
        stocks = [f for f in items if os.path.isfile(os.path.join(folder, f)) and f.endswith(".csv")]
        print(f"Found {len(stocks)} stock files in {folder}")
        if not stocks:
             print(f"No .csv files found in the directory: {folder}")
        return stocks
    except Exception as e:
        print(f"Error listing files in directory {folder}: {e}")
        return []

# --- Fungsi Memuat Data ---
@st.cache_data
def load_stock_data(stock_file_path, features):
    """Memuat dan melakukan pra-pemrosesan data untuk satu saham."""
    print(f"Attempting to load data from: {stock_file_path}")
    try:
        df = pd.read_csv(stock_file_path)
        print(f"Successfully loaded data with {len(df)} rows from {stock_file_path}")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.dropna(subset=features + ['close'])
        print(f"Data after preprocessing: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading data for {stock_file_path}: {e}")
        # Tidak perlu menampilkan error Streamlit di sini, akan ditangani nanti
        return pd.DataFrame()

@st.cache_resource # Gunakan st.cache_resource untuk model
def train_stock_model(X_train, y_train, ticker):
    """Melatih model RandomForestRegressor."""
    print(f"Training model for {ticker} with {len(X_train)} samples")
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    print(f"Model trained for {ticker}")
    return model

stocks = get_stock_list(DATA_FOLDER)

with st.sidebar:
    st.header("Pengaturan")
    if not stocks:
         st.warning(f"Tidak ada file data saham (.csv) yang ditemukan di root repository ('{DATA_FOLDER}').")
         selected_stock = None # Set None jika tidak ada saham
    else:
        selected_stock = st.selectbox("Pilih Saham", options=stocks, help="Pilih file CSV saham dari daftar yang tersedia di repository Anda.")
        st.info("Data saham dimuat dari file CSV di repository GitHub Anda.")


# Konten utama
if selected_stock:
    stock_file_path = os.path.join(DATA_FOLDER, selected_stock)
    df = load_stock_data(stock_file_path, FEATURES)

    if not df.empty:
        X = df[FEATURES]
        y = df['close']

        # Menggunakan container untuk mengelompokkan info
        with st.container():
             st.subheader(f"Analisis dan Prediksi untuk {selected_stock.replace('.csv', '')}")

             # Menampilkan info data terakhir
             if not y.empty:
                 st.write(f"**Harga Penutupan Terakhir:** {y.iloc[-1]:.2f}")
             else:
                 st.write("**Harga Penutupan Terakhir:** Tidak Tersedia")

             
             st.write("Data Terbaru:")
             st.dataframe(df.tail().style.format(subset=['previous', 'high', 'low', 'close', 'foreign_buy', 'foreign_sell', 'volume'], formatter='{:.2f}'))


        if len(df) < 2 or len(X) == 0 or len(y) == 0:
            st.warning(f"Tidak cukup data (minimal 2 baris) atau kolom 'close' hilang untuk {selected_stock} guna melakukan prediksi dan analisis.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                 st.subheader("Metrik Model")

                 # Memastikan test_size valid
                 test_size = 0.2 
                 if len(df) * test_size < 1: 
                      test_size = 1 if len(df) > 1 else 0,

                 if test_size > 0:
                     try:
                         X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=test_size)

                         if len(X_train) > 0 and len(X_test) > 0:
                            ticker = selected_stock.replace(".csv", "")
                            model = train_stock_model(X_train, y_train, ticker)

                            # --- Membuat Prediksi ---
                            y_pred = model.predict(X_test)

                            
                            rmse = root_mean_squared_error(y_test, y_pred)
                            st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")

                            
                            st.subheader("Prediksi Hari Berikutnya")
                            next_day_prediction = None
                            if not X.empty:
                                next_day_features = X.iloc[-1].values.reshape(1, -1)
                                next_day_prediction = model.predict(next_day_features)[0]

                                st.metric("Prediksi Harga Tutup Hari Berikutnya", f"{next_day_prediction:.2f}")
                            else:
                                 st.write("Tidak dapat memprediksi hari berikutnya: Data fitur terakhir tidak tersedia.")


                         elif len(X_train) == 0:
                             st.warning("Dataset pelatihan kosong setelah split data.")
                         else: # len(X_test) == 0
                             st.warning("Dataset pengujian kosong setelah split data. Tidak dapat menghitung RMSE.")

                     except Exception as e:
                        st.error(f"Terjadi kesalahan saat melatih model atau membuat prediksi: {e}")

                 else:
                      st.warning("Tidak cukup data untuk split pelatihan/pengujian.")


            with col2:
                # --- Visualisasi Data dan Prediksi ---
                st.subheader("Grafik Harga Saham")

                if not y.empty:
                    fig, ax = plt.subplots(figsize=(10, 6)) 
                    ax.plot(df['date'], y, label='Harga Tutup Aktual')

                    
                    if len(y_test) > 0 and len(y_pred) > 0:
                         
                         ax.plot(df.loc[y_test.index, 'date'], y_pred, label='Prediksi (Data Uji)', linestyle='--')

                    ax.set_title(f'Harga Tutup Saham {selected_stock.replace(".csv", "")}')
                    ax.set_xlabel('Tanggal')
                    ax.set_ylabel('Harga Tutup')
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.info("Tidak ada data harga tutup untuk ditampilkan.")


    else:
        st.error(f"Gagal memuat atau memproses data untuk {selected_stock}. Mohon periksa file CSV.")

st.markdown("---")
st.markdown("Aplikasi Prediksi Saham sederhana menggunakan RandomForestRegressor.")