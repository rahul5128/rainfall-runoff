import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, Conv1D,
    MaxPooling1D, BatchNormalization, GRU,
    Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)

import torch
import torch.nn as nn
import torch.nn.functional as F

st.set_page_config(page_title="Universal Hydrological Predictor", layout="wide")

# =========================
# HELPER FUNCTIONS & METRICS
# =========================
def calc_nse(obs, sim):
    return 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2))

def create_features(df, date_col, input_cols):
    df = df.copy()
    
    # 1. Cyclical seasonal encoding on Date
    df['day_sin'] = np.sin(2 * np.pi * df[date_col].dt.dayofyear / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df[date_col].dt.dayofyear / 365.25)
    
    # 2. Lag (1-3) and Rolling sums (3, 7) for ALL selected input features
    for col in input_cols:
        for lag in range(1, 4):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        df[f'{col}_sum3'] = df[col].rolling(3).sum()
        df[f'{col}_sum7'] = df[col].rolling(7).sum()
    
    return df

def create_sequences(x, y, window=7):
    X_seq, y_seq = [], []
    for i in range(len(x) - window):
        X_seq.append(x[i : i + window])
        y_seq.append(y[i + window])
    return np.array(X_seq), np.array(y_seq)

def create_pred_sequences(x, window=7):
    X_seq = []
    for i in range(len(x) - window):
        X_seq.append(x[i : i + window])
    return np.array(X_seq)

# =========================
# PYTORCH HYDROATTENTIONNET
# =========================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, outputs):
        weights = F.softmax(self.attn(outputs), dim=1)
        context = torch.sum(weights * outputs, dim=1)
        return context

class HydroAttentionNet(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(HydroAttentionNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        return self.fc(context)

def physics_informed_loss(output, target):
    mse = nn.MSELoss()(output, target)
    peak_mask = (target > 1.0).float()
    peak_penalty = torch.mean(peak_mask * (output - target) ** 2)
    return mse + (0.5 * peak_penalty)

# =========================
# APPLICATION INTERFACE
# =========================
st.title("🌊 Universal Hydrological & Time-Series Model Predictor")
st.caption("End-to-end framework for hydrological sequence & tabular modeling")

st.subheader("1. Training & Validation")
train_file = st.file_uploader("Upload Training Data (CSV or Excel)", type=['csv', 'xlsx'], key="train")

if train_file:
    df_raw = pd.read_csv(train_file) if train_file.name.endswith('.csv') else pd.read_excel(train_file)
    cols = df_raw.columns.tolist()

    st.write("---")
    st.write("### 📌 Configuration & Parameter Tuning")
    c_m1, c_m2 = st.columns([1, 2])
    
    with c_m1:
        model_type = st.selectbox(
            "1. Select Model",
            [
                "HistGradientBoosting", "HydroAttentionNet", "CNN-LSTM", 
                "Random Forest", "XGBoost", "LightGBM", 
                "LSTM", "GRU", "Transformer", 
                "SVR", "KNN", "ANN (MLP)"
            ]
        )
    
    with c_m2:
        with st.expander(f"⚙️ 2. Modify Hyperparameters for {model_type}", expanded=True):
            if model_type == "HistGradientBoosting":
                hgb_iter = st.number_input("max_iter", value=500, step=50)
                hgb_lr = st.number_input("learning_rate", value=0.05, step=0.01, format="%.3f")
                hgb_leaves = st.number_input("max_leaf_nodes", value=63, step=1)
                hgb_min_leaf = st.number_input("min_samples_leaf", value=10, step=1)
                hgb_l2 = st.number_input("l2_regularization", value=0.1, step=0.05, format="%.2f")
            
            elif model_type == "HydroAttentionNet":
                hydro_hidden = st.number_input("Hidden Size", value=128, step=16)
                hydro_lr = st.number_input("Learning Rate", value=0.0005, step=0.0001, format="%.4f")
                hydro_epochs = st.number_input("Epochs", value=150, step=10)
                window_size = st.number_input("Window Size (Days)", value=7, min_value=3, max_value=30)

            elif model_type == "CNN-LSTM":
                cnn_epochs = st.number_input("Epochs", value=100, step=10)
                cnn_batch = st.number_input("Batch Size", value=32, step=16)
                cnn_lr = st.number_input("Learning Rate", value=0.001, step=0.0005, format="%.4f")
                window_size = st.number_input("Window Size (Days)", value=7, min_value=3, max_value=30)

            elif model_type == "Random Forest":
                rf_trees = st.number_input("n_estimators", value=100, step=10)
                rf_depth = st.number_input("max_depth", value=15, step=1)

            elif model_type in ["XGBoost", "LightGBM"]:
                gb_trees = st.number_input("n_estimators", value=100, step=50)
                gb_lr = st.number_input("learning_rate", value=0.05, step=0.01, format="%.3f")
                gb_depth = st.number_input("max_depth", value=6, step=1)

            elif model_type in ["LSTM", "GRU"]:
                rec_units = st.number_input("Units", value=64, step=16)
                rec_epochs = st.number_input("Epochs", value=50, step=10)
                rec_batch = st.number_input("Batch Size", value=32, step=16)
                window_size = st.number_input("Window Size (Days)", value=7, min_value=3, max_value=30)

            elif model_type == "Transformer":
                trans_heads = st.number_input("Attention Heads", value=2, step=1)
                trans_epochs = st.number_input("Epochs", value=50, step=10)
                trans_batch = st.number_input("Batch Size", value=32, step=16)
                window_size = st.number_input("Window Size (Days)", value=7, min_value=3, max_value=30)

            elif model_type == "SVR":
                svr_kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
                svr_c = st.number_input("C", value=1.0, step=0.5)

            elif model_type == "KNN":
                knn_n = st.number_input("n_neighbors", value=5, step=1)

            elif model_type == "ANN (MLP)":
                mlp_layers = st.text_input("Hidden Layers", "100, 50")
                mlp_iter = st.number_input("Max Iterations", value=500, step=50)

    st.write("---")
    st.write("### 📌 Column Mapping")
    c3, c4, c5 = st.columns(3)
    
    # 3. Date Dropdown
    date_col = c3.selectbox("3. Select Date Column", cols, index=0)
    
    # 5. Target Dropdown
    target_matches = [i for i, c in enumerate(cols) if 'discharge' in c.lower() or 'target' in c.lower()]
    target_default = target_matches[0] if target_matches else len(cols) - 1
    target_col = c5.selectbox("5. Select Target Column", cols, index=target_default)

    # 4. Input Features Dropdown
    default_inputs = [c for c in cols if c not in [date_col, target_col, 'Water Level(m)']]
    input_cols = c4.multiselect(
        "4. Select Input Features",
        [c for c in cols if c not in [date_col, target_col]],
        default=default_inputs
    )

    # =========================
    # TRAINING WORKFLOW
    # =========================
    if st.button("🚀 Train & Evaluate Model", type="primary"):
        if not input_cols:
            st.error("Please select at least one Input Feature.")
        else:
            with st.spinner(f"Processing features and training {model_type}..."):
                # Clean dataframe: sort chronologically and drop missing targets first
                df = df_raw.copy()
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.sort_values(date_col)
                df = df.dropna(subset=[target_col]).reset_index(drop=True)
                
                # Create lag and rolling sum features for ALL selected input features
                df_feat = create_features(df, date_col, input_cols)
                df_clean = df_feat.dropna().reset_index(drop=True)

                # Collect all generated feature names
                engineered_cols = ['day_sin', 'day_cos']
                for col in input_cols:
                    engineered_cols.append(col)
                    for lag in range(1, 4):
                        engineered_cols.append(f'{col}_lag{lag}')
                    engineered_cols.append(f'{col}_sum3')
                    engineered_cols.append(f'{col}_sum7')

                is_sequence_model = model_type in ["HydroAttentionNet", "CNN-LSTM", "LSTM", "GRU", "Transformer"]

                if not is_sequence_model:
                    # Tabular Pipeline (HistGradientBoosting, RF, XGBoost, LightGBM, SVR, KNN, ANN)
                    X = df_clean[engineered_cols]
                    y = df_clean[target_col].values

                    split = int(len(df_clean) * 0.8)
                    X_train, X_test = X.iloc[:split], X.iloc[split:]
                    y_train, y_test = y[:split], y[split:]
                    eval_dates = df_clean[date_col].iloc[split:]

                    scaler_x = StandardScaler()
                    X_train_scaled = scaler_x.fit_transform(X_train)
                    X_test_scaled = scaler_x.transform(X_test)

                    if model_type == "HistGradientBoosting":
                        model = HistGradientBoostingRegressor(
                            max_iter=hgb_iter,
                            learning_rate=hgb_lr,
                            max_leaf_nodes=hgb_leaves,
                            min_samples_leaf=hgb_min_leaf,
                            l2_regularization=hgb_l2,
                            random_state=42
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    elif model_type == "Random Forest":
                        model = RandomForestRegressor(n_estimators=rf_trees, max_depth=rf_depth, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    elif model_type == "XGBoost":
                        model = xgb.XGBRegressor(n_estimators=gb_trees, learning_rate=gb_lr, max_depth=gb_depth, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    elif model_type == "LightGBM":
                        model = lgb.LGBMRegressor(n_estimators=gb_trees, learning_rate=gb_lr, max_depth=gb_depth, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    elif model_type == "SVR":
                        model = SVR(kernel=svr_kernel, C=svr_c)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)

                    elif model_type == "KNN":
                        model = KNeighborsRegressor(n_neighbors=knn_n)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)

                    elif model_type == "ANN (MLP)":
                        layers = tuple(int(x.strip()) for x in mlp_layers.split(','))
                        model = MLPRegressor(hidden_layer_sizes=layers, max_iter=mlp_iter, random_state=42)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)

                    st.session_state['trained_model'] = model
                    st.session_state['scaler_x'] = scaler_x
                    st.session_state['scaler_y'] = None
                    st.session_state['feature_cols'] = engineered_cols

                else:
                    # Deep Learning Sequence Pipeline
                    scaler_x = StandardScaler()
                    scaler_y = StandardScaler()

                    raw_features = list(input_cols) + ['day_sin', 'day_cos']
                    X_scaled = scaler_x.fit_transform(df_clean[raw_features])
                    y_scaled = scaler_y.fit_transform(df_clean[[target_col]])

                    win = int(window_size)
                    X_seq, y_seq = create_sequences(X_scaled, y_scaled, win)

                    split_seq = int(len(X_seq) * 0.8)
                    X_train, X_test = X_seq[:split_seq], X_seq[split_seq:]
                    y_train, y_test_sc = y_seq[:split_seq], y_seq[split_seq:]
                    eval_dates = df_clean[date_col].iloc[win + split_seq:].reset_index(drop=True)

                    if model_type == "HydroAttentionNet":
                        X_train_torch = torch.FloatTensor(X_train)
                        X_test_torch = torch.FloatTensor(X_test)
                        y_train_torch = torch.FloatTensor(y_train)

                        model = HydroAttentionNet(input_size=len(raw_features), hidden_size=int(hydro_hidden))
                        optimizer = torch.optim.Adam(model.parameters(), lr=hydro_lr)

                        for epoch in range(int(hydro_epochs)):
                            model.train()
                            optimizer.zero_grad()
                            out = model(X_train_torch)
                            loss = physics_informed_loss(out, y_train_torch)
                            loss.backward()
                            optimizer.step()

                        model.eval()
                        with torch.no_grad():
                            pred_scaled = model(X_test_torch).numpy()
                        y_pred = scaler_y.inverse_transform(pred_scaled).flatten()
                        y_test = scaler_y.inverse_transform(y_test_sc).flatten()

                    elif model_type == "CNN-LSTM":
                        model = models.Sequential([
                            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(win, len(raw_features))),
                            BatchNormalization(),
                            MaxPooling1D(pool_size=2),
                            LSTM(100, return_sequences=True),
                            Dropout(0.2),
                            LSTM(50),
                            Dropout(0.2),
                            Dense(32, activation='relu'),
                            Dense(1)
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cnn_lr), loss='mse')
                        model.fit(
                            X_train, y_train,
                            epochs=int(cnn_epochs),
                            batch_size=int(cnn_batch),
                            validation_data=(X_test, y_test_sc),
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                            verbose=0
                        )
                        pred_scaled = model.predict(X_test, verbose=0)
                        y_pred = scaler_y.inverse_transform(pred_scaled).flatten()
                        y_test = scaler_y.inverse_transform(y_test_sc).flatten()

                    elif model_type == "LSTM":
                        model = models.Sequential([
                            LSTM(int(rec_units), input_shape=(win, len(raw_features))),
                            Dropout(0.2),
                            Dense(32, activation='relu'),
                            Dense(1)
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
                        model.fit(X_train, y_train, epochs=int(rec_epochs), batch_size=int(rec_batch), verbose=0)
                        pred_scaled = model.predict(X_test, verbose=0)
                        y_pred = scaler_y.inverse_transform(pred_scaled).flatten()
                        y_test = scaler_y.inverse_transform(y_test_sc).flatten()

                    elif model_type == "GRU":
                        model = models.Sequential([
                            GRU(int(rec_units), input_shape=(win, len(raw_features))),
                            Dropout(0.2),
                            Dense(32, activation='relu'),
                            Dense(1)
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
                        model.fit(X_train, y_train, epochs=int(rec_epochs), batch_size=int(rec_batch), verbose=0)
                        pred_scaled = model.predict(X_test, verbose=0)
                        y_pred = scaler_y.inverse_transform(pred_scaled).flatten()
                        y_test = scaler_y.inverse_transform(y_test_sc).flatten()

                    elif model_type == "Transformer":
                        inp = Input(shape=(win, len(raw_features)))
                        attn = MultiHeadAttention(num_heads=int(trans_heads), key_dim=16)(inp, inp)
                        norm = LayerNormalization()(attn + inp)
                        pool = GlobalAveragePooling1D()(norm)
                        dense = Dense(32, activation='relu')(pool)
                        out = Dense(1)(dense)
                        model = models.Model(inp, out)
                        model.compile(optimizer='adam', loss='mse')
                        model.fit(X_train, y_train, epochs=int(trans_epochs), batch_size=int(trans_batch), verbose=0)
                        pred_scaled = model.predict(X_test, verbose=0)
                        y_pred = scaler_y.inverse_transform(pred_scaled).flatten()
                        y_test = scaler_y.inverse_transform(y_test_sc).flatten()

                    st.session_state['trained_model'] = model
                    st.session_state['scaler_x'] = scaler_x
                    st.session_state['scaler_y'] = scaler_y
                    st.session_state['window_size'] = win
                    st.session_state['raw_features'] = raw_features

                st.session_state['model_type'] = model_type
                st.session_state['date_col'] = date_col
                st.session_state['target_col'] = target_col
                st.session_state['input_cols'] = input_cols
                st.session_state['is_sequence_model'] = is_sequence_model

                # Metrics calculation on the 20% validation split
                r2 = r2_score(y_test, y_pred)
                nse = calc_nse(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)

                st.success(f"Training Complete! Selected Model: {model_type}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R² Score", round(r2, 4))
                m2.metric("NSE", round(nse, 4))
                m3.metric("RMSE", round(rmse, 2))
                m4.metric("MAE", round(mae, 2))

                fig, ax = plt.subplots(figsize=(14, 4))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                ax.plot(eval_dates.values, y_test, label="Observed", color='#00b4d8', linewidth=1.5)
                ax.plot(eval_dates.values, y_pred, label=f"Predicted ({model_type})", color='#ff4b4b', linestyle='--', linewidth=1.2)
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.2)
                ax.legend(facecolor='#0e1117', labelcolor='white')
                st.pyplot(fig)

# =========================
# PHASE 2: GENERATE TARGET
# =========================
st.write("---")
st.subheader("2. Generate Target for New Data")

if 'trained_model' not in st.session_state:
    st.info("Train a model above first to unlock target generation.")
else:
    st.write(f"Active Model: **{st.session_state['model_type']}**")
    new_file = st.file_uploader("Upload New Unseen Dataset (CSV/Excel)", type=['csv', 'xlsx'], key="pred")
    
    if new_file:
        pred_df = pd.read_csv(new_file) if new_file.name.endswith('.csv') else pd.read_excel(new_file)
        date_c = st.session_state['date_col']
        in_cols = st.session_state['input_cols']
        target_name = st.session_state['target_col']
        model = st.session_state['trained_model']
        model_type = st.session_state['model_type']
        
        missing = [c for c in in_cols if c not in pred_df.columns]
        
        if date_c not in pred_df.columns:
            st.error(f"Missing date column '{date_c}' in file.")
        elif missing:
            st.error(f"Missing input columns: {missing}")
        else:
            if st.button("🔮 Generate Target Predictions", type="primary"):
                pred_df[date_c] = pd.to_datetime(pred_df[date_c], errors='coerce')
                pred_df = pred_df.sort_values(date_c).reset_index(drop=True)
                
                if not st.session_state['is_sequence_model']:
                    pred_feat = create_features(pred_df, date_c, in_cols)
                    
                    X_cols = st.session_state['feature_cols']
                    X_new = pred_feat[X_cols]
                    
                    if model_type in ["SVR", "KNN", "ANN (MLP)"]:
                        X_new_sc = st.session_state['scaler_x'].transform(X_new)
                        preds = model.predict(X_new_sc)
                    else:
                        preds = model.predict(X_new)
                        
                    res = pd.DataFrame({
                        date_c: pred_df[date_c],
                        f"Predicted_{target_name}": preds
                    })
                else:
                    win = st.session_state['window_size']
                    seq_features = st.session_state['raw_features']
                    pred_df['day_sin'] = np.sin(2 * np.pi * pred_df[date_c].dt.dayofyear / 365.25)
                    pred_df['day_cos'] = np.cos(2 * np.pi * pred_df[date_c].dt.dayofyear / 365.25)
                    
                    X_sc = st.session_state['scaler_x'].transform(pred_df[seq_features])
                    X_seq_new = create_pred_sequences(X_sc, win)
                    
                    if len(X_seq_new) == 0:
                        st.error(f"Uploaded file has fewer rows than the required window size ({win} days).")
                        res = None
                    else:
                        if model_type == "HydroAttentionNet":
                            model.eval()
                            with torch.no_grad():
                                sc_preds = model(torch.FloatTensor(X_seq_new)).numpy()
                            preds = st.session_state['scaler_y'].inverse_transform(sc_preds).flatten()
                        else:
                            sc_preds = model.predict(X_seq_new, verbose=0)
                            preds = st.session_state['scaler_y'].inverse_transform(sc_preds).flatten()
                            
                        res = pd.DataFrame({
                            date_c: pred_df[date_c].iloc[win:].reset_index(drop=True),
                            f"Predicted_{target_name}": preds
                        })
                    
                if 'res' in locals() and res is not None:
                    st.success("Predictions generated!")
                    st.dataframe(res.head(15), use_container_width=True)
                    csv = res.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name=f"predicted_{target_name}.csv",
                        mime="text/csv"
                    )
