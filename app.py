import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, GRU, Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Multiply

st.set_page_config(page_title="Universal ML Predictor", layout="wide")

def calc_nse(obs, sim):
    return 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2))

def create_features(df, date_col, input_cols):
    df = df.copy()
    # 1. Original Seasonal Date Math
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # 2. Lag and Rolling Features for Inputs
    for col in input_cols:
        for lag in range(1, 4):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        df[f'{col}_sum3'] = df[col].rolling(3).sum()
        df[f'{col}_sum7'] = df[col].rolling(7).sum()
    
    return df.drop(columns=['day_of_year'])

st.title("🧠 Universal Model Predictor")

train_file = st.file_uploader("Upload Training Data (CSV/Excel)", type=['csv', 'xlsx'])

if train_file:
    df = pd.read_csv(train_file) if train_file.name.endswith('.csv') else pd.read_excel(train_file)
    cols = df.columns.tolist()
    
    st.write("**Dataset Configuration**")
    c1, c2, c3 = st.columns(3)
    date_col = c1.selectbox("Date Column", cols, index=0)
    target_col = c2.selectbox("Target Column", cols, index=len(cols)-1)
    input_cols = c3.multiselect("Input Features", [c for c in cols if c not in [date_col, target_col]])

    st.write("**Architecture & Hyperparameters**")
    mc1, mc2 = st.columns([1, 2])
    with mc1:
        model_type = st.selectbox("Select Model", [
            "HistGradientBoosting", "Random Forest", "XGBoost", "LightGBM",
            "SVR", "KNN", "ANN (MLP)", 
            "LSTM", "CNN-LSTM", "GRU", "Transformer", "HydroAttentionNet"
        ])
    
    with mc2:
        with st.expander(f"⚙️ {model_type} Settings", expanded=True):
            if model_type in ["HistGradientBoosting", "LightGBM"]:
                hgb_iter = st.slider("Max Iterations/Trees", 50, 1000, 500, 50)
                hgb_lr = st.number_input("Learning Rate", 0.001, 0.5, 0.05, format="%.3f")
            elif model_type in ["Random Forest", "XGBoost"]:
                tree_est = st.slider("n_estimators", 50, 1000, 100, 50)
                tree_depth = st.slider("max_depth", 2, 50, 10)
            elif model_type == "SVR":
                svr_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
                svr_c = st.number_input("C", 0.1, 100.0, 1.0)
            elif model_type == "KNN":
                knn_n = st.slider("n_neighbors", 1, 50, 5)
            elif model_type == "ANN (MLP)":
                ann_hidden = st.text_input("Hidden Layers (comma separated)", "100, 50")
                ann_iter = st.slider("Max Iterations", 100, 2000, 500)
            elif model_type in ["LSTM", "CNN-LSTM", "GRU", "Transformer", "HydroAttentionNet"]:
                dl_epochs = st.slider("Epochs", 10, 200, 50, 10)
                dl_batch = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
                dl_units = st.slider("Main Layer Units", 16, 256, 64, 16)

    if st.button("🚀 Train Model", type="primary") and input_cols:
        with st.spinner(f'Training {model_type}...'):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            df = df.dropna(subset=[date_col, target_col]).sort_values(date_col)
            
            df_feat = create_features(df, date_col, input_cols)
            df_clean = df_feat.dropna().reset_index(drop=True)
            
            X = df_clean.drop(columns=[date_col, target_col], errors='ignore')
            y = df_clean[target_col].values
            
            split = int(len(df_clean) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y[:split], y[split:]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # --- 2D Tree Models (Unscaled) ---
            if model_type == "HistGradientBoosting":
                model = HistGradientBoostingRegressor(max_iter=hgb_iter, learning_rate=hgb_lr)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            elif model_type == "LightGBM":
                model = lgb.LGBMRegressor(n_estimators=hgb_iter, learning_rate=hgb_lr)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            elif model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=tree_est, max_depth=tree_depth)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            elif model_type == "XGBoost":
                model = xgb.XGBRegressor(n_estimators=tree_est, max_depth=tree_depth)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            # --- 2D Distance Models (Scaled) ---
            elif model_type == "SVR":
                model = SVR(kernel=svr_kernel, C=svr_c)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            elif model_type == "KNN":
                model = KNeighborsRegressor(n_neighbors=knn_n)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            elif model_type == "ANN (MLP)":
                layers = tuple(int(x.strip()) for x in ann_hidden.split(','))
                model = MLPRegressor(hidden_layer_sizes=layers, max_iter=ann_iter)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
            # --- 3D Deep Learning Sequence Models (Scaled & Reshaped) ---
            elif model_type in ["LSTM", "CNN-LSTM", "GRU", "Transformer", "HydroAttentionNet"]:
                X_tr_3d = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
                X_te_3d = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
                
                if model_type == "LSTM":
                    model = Sequential([LSTM(dl_units, activation='relu', input_shape=(1, X_train_scaled.shape[1])), Dense(1)])
                elif model_type == "GRU":
                    model = Sequential([GRU(dl_units, activation='relu', input_shape=(1, X_train_scaled.shape[1])), Dense(1)])
                elif model_type == "CNN-LSTM":
                    model = Sequential([Conv1D(64, 1, activation='relu', input_shape=(1, X_train_scaled.shape[1])), LSTM(dl_units, activation='relu'), Dense(1)])
                elif model_type == "Transformer":
                    inputs = Input(shape=(1, X_train_scaled.shape[1]))
                    attn = MultiHeadAttention(num_heads=2, key_dim=32)(inputs, inputs)
                    norm = LayerNormalization()(attn + inputs)
                    flat = GlobalAveragePooling1D()(norm)
                    outputs = Dense(1)(flat)
                    model = Model(inputs, outputs)
                elif model_type == "HydroAttentionNet":
                    inputs = Input(shape=(1, X_train_scaled.shape[1]))
                    attn_probs = Dense(X_train_scaled.shape[1], activation='softmax')(inputs)
                    attn_mul = Multiply()([inputs, attn_probs])
                    flat = Flatten()(attn_mul)
                    dense = Dense(dl_units, activation='relu')(flat)
                    outputs = Dense(1)(dense)
                    model = Model(inputs, outputs)
                
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_tr_3d, y_train, epochs=dl_epochs, batch_size=dl_batch, verbose=0)
                y_pred = model.predict(X_te_3d).flatten()
                
            st.success("Training Complete!")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R²", round(r2_score(y_test, y_pred), 3))
            m2.metric("NSE", round(calc_nse(y_test, y_pred), 3))
            m3.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
            m4.metric("MAE", round(mean_absolute_error(y_test, y_pred), 2))
            
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(df_clean[date_col].iloc[split:], y_test, label="Observed", color='#00b4d8')
            ax.plot(df_clean[date_col].iloc[split:], y_pred, label="Predicted", color='#ff4b4b', linestyle='--')
            ax.legend()
            st.pyplot(fig)
