import io
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
def calc_pbias(obs, sim):
    return 100.0 * np.sum(sim - obs) / np.sum(obs)

def create_features(df, date_col, use_date_encoding, feature_configs):
    df = df.copy()
    engineered_cols = []
    
    # 1. Optional Cyclical seasonal encoding on Date
    if use_date_encoding and date_col in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df[date_col].dt.dayofyear / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df[date_col].dt.dayofyear / 365.25)
        engineered_cols.extend(['day_sin', 'day_cos'])
    
    # 2. Apply Custom Granular Feature Configurations
    for i, cfg in enumerate(feature_configs):
        col = cfg['col']
        trans = cfg['trans']
        val = cfg['val']
        
        if col not in df.columns:
            continue
            
        if trans == "Raw":
            new_col = f"{col}_raw_{i}"
            df[new_col] = df[col]
            engineered_cols.append(new_col)
        elif trans == "Lag":
            new_col = f"{col}_lag{val}_{i}"
            df[new_col] = df[col].shift(val)
            engineered_cols.append(new_col)
        elif trans == "Rolling Sum":
            new_col = f"{col}_sum{val}_{i}"
            df[new_col] = df[col].rolling(val).sum()
            engineered_cols.append(new_col)
            
    return df, engineered_cols

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

def to_excel_bytes(dataframe):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        dataframe.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

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
col_title, col_manual = st.columns([5, 1])

with col_title:
    st.title("🌊 Universal Model Predictor")
    st.caption("End-to-end framework for hydrological sequence & tabular modeling")

with col_manual:
    st.write("<br>", unsafe_allow_html=True)
    with st.popover("📖 Manual"):
        st.markdown("""
        <div style='font-size: 14px;'>
        <b>How to Use This App:</b><br><br>
        <b>1. Training & Validation</b>
        <ul>
            <li><b>Upload Data:</b> Provide your historical dataset (CSV or Excel).</li>
            <li><b>Select Model & Hyperparameters:</b> Choose your architecture and tune the settings.</li>
            <li><b>Date Encoding:</b> Toggle the checkbox to include or exclude cyclical date features (day_sin, day_cos).</li>
            <li><b>Granular Input Features:</b> Click <b>➕ Add Feature</b> to build your feature set one by one. Select the base column, then choose if it should be used as Raw, a Lag (e.g., Lag 1), or a Rolling Sum (e.g., Rolling 3).</li>
            <li><b>Target Lags:</b> You can select your Target column in the feature list and apply Lags/Rolling Sums to use historical target values as predictors!</li>
            <li>Click <b>Train & Evaluate</b>.</li>
        </ul>
        <b>2. Generate Target (Inference)</b>
        <ul>
            <li>Upload a new <b>raw</b> dataset. You do NOT need to calculate lags or rolling sums yourself.</li>
            <li>As long as your uploaded file contains the original base columns, the app will automatically engineer all the specific lags and rolling windows you configured during training.</li>
            <li>Click <b>Generate Target Predictions</b> to download your future forecast.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

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
    st.write("### 📌 Column Mapping & Granular Feature Builder")
    c3, c4 = st.columns(2)
    
    # Date & Target Base Selection
    date_col = c3.selectbox("Date Column", cols, index=0)
    use_date = c3.checkbox("Include Date Encoding (day_sin, day_cos)", value=True)
    
    target_matches = [i for i, c in enumerate(cols) if 'discharge' in c.lower() or 'target' in c.lower()]
    target_default = target_matches[0] if target_matches else len(cols) - 1
    target_col = c4.selectbox("Target Column", cols, index=target_default)

    st.write("**Construct Input Features**")
    
    if 'num_features' not in st.session_state:
        st.session_state.num_features = 1
        
    def add_feature():
        st.session_state.num_features += 1
    def remove_feature():
        if st.session_state.num_features > 1:
            st.session_state.num_features -= 1

    feature_configs = []
    
    for i in range(st.session_state.num_features):
        fc1, fc2, fc3 = st.columns([2, 2, 1])
        col_sel = fc1.selectbox(f"Feature {i+1} Base Column", cols, key=f"fcol_{i}")
        trans_sel = fc2.selectbox(f"Transform {i+1}", ["Raw", "Lag", "Rolling Sum"], key=f"ftrans_{i}", label_visibility="collapsed")
        
        val_sel = 0
        if trans_sel != "Raw":
            val_sel = fc3.number_input(f"Value {i+1}", min_value=1, value=1, step=1, key=f"fval_{i}", label_visibility="collapsed")
            
        feature_configs.append({
            'col': col_sel,
            'trans': trans_sel,
            'val': val_sel
        })

    bc1, bc2 = st.columns([1, 8])
    bc1.button("➕ Add Feature", on_click=add_feature)
    bc2.button("➖ Remove Last", on_click=remove_feature)

    # =========================
    # TRAINING WORKFLOW
    # =========================
    st.write("<br>", unsafe_allow_html=True)
    if st.button("🚀 Train & Evaluate Model", type="primary"):
        if len(feature_configs) == 0:
            st.error("Please add at least one Input Feature.")
        else:
            with st.spinner(f"Processing granular features and training {model_type}..."):
                
                # Clean dataframe: sort chronologically and drop missing targets first
                df = df_raw.copy()
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.sort_values(date_col)
                df = df.dropna(subset=[target_col]).reset_index(drop=True)
                
                # Dynamically build features based on user configuration
                df_feat, engineered_cols = create_features(df, date_col, use_date, feature_configs)
                df_clean = df_feat.dropna().reset_index(drop=True)

                is_sequence_model = model_type in ["HydroAttentionNet", "CNN-LSTM", "LSTM", "GRU", "Transformer"]

                if not is_sequence_model:
                    # Tabular Pipeline
                    X = df_clean[engineered_cols]
                    y = df_clean[target_col].values

                    split = int(len(df_clean) * 0.8)
                    X_train, X_test = X.iloc[:split], X.iloc[split:]
                    y_train, y_test = y[:split], y[split:]
                    train_dates = df_clean[date_col].iloc[:split]
                    eval_dates = df_clean[date_col].iloc[split:]

                    scaler_x = StandardScaler()
                    X_train_scaled = scaler_x.fit_transform(X_train)
                    X_test_scaled = scaler_x.transform(X_test)

                    if model_type == "HistGradientBoosting":
                        model = HistGradientBoostingRegressor(
                            max_iter=hgb_iter, learning_rate=hgb_lr, max_leaf_nodes=hgb_leaves,
                            min_samples_leaf=hgb_min_leaf, l2_regularization=hgb_l2, random_state=42
                        )
                        model.fit(X_train, y_train)
                        y_train_pred = model.predict(X_train)
                        y_pred = model.predict(X_test)

                    elif model_type == "Random Forest":
                        model = RandomForestRegressor(n_estimators=rf_trees, max_depth=rf_depth, random_state=42)
                        model.fit(X_train, y_train)
                        y_train_pred = model.predict(X_train)
                        y_pred = model.predict(X_test)

                    elif model_type == "XGBoost":
                        model = xgb.XGBRegressor(n_estimators=gb_trees, learning_rate=gb_lr, max_depth=gb_depth, random_state=42)
                        model.fit(X_train, y_train)
                        y_train_pred = model.predict(X_train)
                        y_pred = model.predict(X_test)

                    elif model_type == "LightGBM":
                        model = lgb.LGBMRegressor(n_estimators=gb_trees, learning_rate=gb_lr, max_depth=gb_depth, random_state=42)
                        model.fit(X_train, y_train)
                        y_train_pred = model.predict(X_train)
                        y_pred = model.predict(X_test)

                    elif model_type == "SVR":
                        model = SVR(kernel=svr_kernel, C=svr_c)
                        model.fit(X_train_scaled, y_train)
                        y_train_pred = model.predict(X_train_scaled)
                        y_pred = model.predict(X_test_scaled)

                    elif model_type == "KNN":
                        model = KNeighborsRegressor(n_neighbors=knn_n)
                        model.fit(X_train_scaled, y_train)
                        y_train_pred = model.predict(X_train_scaled)
                        y_pred = model.predict(X_test_scaled)

                    elif model_type == "ANN (MLP)":
                        layers = tuple(int(x.strip()) for x in mlp_layers.split(','))
                        model = MLPRegressor(hidden_layer_sizes=layers, max_iter=mlp_iter, random_state=42)
                        model.fit(X_train_scaled, y_train)
                        y_train_pred = model.predict(X_train_scaled)
                        y_pred = model.predict(X_test_scaled)

                    st.session_state['trained_model'] = model
                    st.session_state['scaler_x'] = scaler_x
                    st.session_state['scaler_y'] = None

                else:
                    # Sequence Pipeline
                    scaler_x = StandardScaler()
                    scaler_y = StandardScaler()

                    X_scaled = scaler_x.fit_transform(df_clean[engineered_cols])
                    y_scaled = scaler_y.fit_transform(df_clean[[target_col]])

                    win = int(window_size)
                    X_seq, y_seq = create_sequences(X_scaled, y_scaled, win)

                    split_seq = int(len(X_seq) * 0.8)
                    X_train, X_test = X_seq[:split_seq], X_seq[split_seq:]
                    y_train_sc, y_test_sc = y_seq[:split_seq], y_seq[split_seq:]
                    
                    train_dates = df_clean[date_col].iloc[win : win + split_seq].reset_index(drop=True)
                    eval_dates = df_clean[date_col].iloc[win + split_seq :].reset_index(drop=True)

                    if model_type == "HydroAttentionNet":
                        X_train_torch, X_test_torch = torch.FloatTensor(X_train), torch.FloatTensor(X_test)
                        y_train_torch = torch.FloatTensor(y_train_sc)

                        model = HydroAttentionNet(input_size=len(engineered_cols), hidden_size=int(hydro_hidden))
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
                            train_pred_sc = model(X_train_torch).numpy()
                            test_pred_sc = model(X_test_torch).numpy()
                            
                        y_train_pred = scaler_y.inverse_transform(train_pred_sc).flatten()
                        y_train = scaler_y.inverse_transform(y_train_sc).flatten()
                        y_pred = scaler_y.inverse_transform(test_pred_sc).flatten()
                        y_test = scaler_y.inverse_transform(y_test_sc).flatten()

                    elif model_type == "CNN-LSTM":
                        model = models.Sequential([
                            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(win, len(engineered_cols))),
                            BatchNormalization(), MaxPooling1D(pool_size=2),
                            LSTM(100, return_sequences=True), Dropout(0.2),
                            LSTM(50), Dropout(0.2), Dense(32, activation='relu'), Dense(1)
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cnn_lr), loss='mse')
                        model.fit(X_train, y_train_sc, epochs=int(cnn_epochs), batch_size=int(cnn_batch), validation_data=(X_test, y_test_sc), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=0)
                        
                        y_train_pred = scaler_y.inverse_transform(model.predict(X_train, verbose=0)).flatten()
                        y_train = scaler_y.inverse_transform(y_train_sc).flatten()
                        y_pred = scaler_y.inverse_transform(model.predict(X_test, verbose=0)).flatten()
                        y_test = scaler_y.inverse_transform(y_test_sc).flatten()

                    elif model_type in ["LSTM", "GRU"]:
                        Layer = LSTM if model_type == "LSTM" else GRU
                        model = models.Sequential([
                            Layer(int(rec_units), input_shape=(win, len(engineered_cols))),
                            Dropout(0.2), Dense(32, activation='relu'), Dense(1)
                        ])
                        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
                        model.fit(X_train, y_train_sc, epochs=int(rec_epochs), batch_size=int(rec_batch), verbose=0)
                        
                        y_train_pred = scaler_y.inverse_transform(model.predict(X_train, verbose=0)).flatten()
                        y_train = scaler_y.inverse_transform(y_train_sc).flatten()
                        y_pred = scaler_y.inverse_transform(model.predict(X_test, verbose=0)).flatten()
                        y_test = scaler_y.inverse_transform(y_test_sc).flatten()

                    elif model_type == "Transformer":
                        inp = Input(shape=(win, len(engineered_cols)))
                        attn = MultiHeadAttention(num_heads=int(trans_heads), key_dim=16)(inp, inp)
                        norm = LayerNormalization()(attn + inp)
                        pool = GlobalAveragePooling1D()(norm)
                        dense = Dense(32, activation='relu')(pool)
                        out = Dense(1)(dense)
                        model = models.Model(inp, out)
                        model.compile(optimizer='adam', loss='mse')
                        model.fit(X_train, y_train_sc, epochs=int(trans_epochs), batch_size=int(trans_batch), verbose=0)
                        
                        y_train_pred = scaler_y.inverse_transform(model.predict(X_train, verbose=0)).flatten()
                        y_train = scaler_y.inverse_transform(y_train_sc).flatten()
                        y_pred = scaler_y.inverse_transform(model.predict(X_test, verbose=0)).flatten()
                        y_test = scaler_y.inverse_transform(y_test_sc).flatten()

                    st.session_state['trained_model'] = model
                    st.session_state['scaler_x'] = scaler_x
                    st.session_state['scaler_y'] = scaler_y
                    st.session_state['window_size'] = win

                # Save global state for Inference
                st.session_state['model_type'] = model_type
                st.session_state['date_col'] = date_col
                st.session_state['target_col'] = target_col
                st.session_state['feature_configs'] = feature_configs
                st.session_state['engineered_cols'] = engineered_cols
                st.session_state['use_date_encoding'] = use_date
                st.session_state['is_sequence_model'] = is_sequence_model

                # Testing Metrics calculation
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                pbias = calc_pbias(y_test, y_pred)

                st.success(f"Training Complete! Selected Model: {model_type}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R² Score", round(r2, 4))
                m2.metric("RMSE", round(rmse, 2))
                m3.metric("MAE", round(mae, 2))
                m4.metric("PBIAS (%)", f"{round(pbias, 2)}%")

                # Hydrograph plot
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
                # EXCEL EXPORT (TRAIN & TEST)
                # =========================
                clean_name = model_type.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                pred_col_name = f"{clean_name}_predicted"

                df_train_export = pd.DataFrame({
                    'Date': pd.Series(train_dates).values,
                    'Observed': np.asarray(y_train).flatten(),
                    pred_col_name: np.asarray(y_train_pred).flatten()
                })

                df_test_export = pd.DataFrame({
                    'Date': pd.Series(eval_dates).values,
                    'Observed': np.asarray(y_test).flatten(),
                    pred_col_name: np.asarray(y_pred).flatten()
                })

                st.write("---")
                st.write("### 📥 Download Prediction Files (.xlsx)")
                d_c1, d_c2 = st.columns(2)

                train_excel_data = to_excel_bytes(df_train_export)
                d_c1.download_button(
                    label="📊 Download Training Predictions",
                    data=train_excel_data,
                    file_name=f"{clean_name}_training_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                test_excel_data = to_excel_bytes(df_test_export)
                d_c2.download_button(
                    label="📈 Download Testing Predictions",
                    data=test_excel_data,
                    file_name=f"{clean_name}_testing_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# =========================
# PHASE 2: GENERATE TARGET
# =========================
st.write("---")
st.subheader("2. Generate Target for New Data")

if 'trained_model' not in st.session_state:
    st.info("Train a model above first to unlock target generation.")
else:
    st.write(f"Active Model: **{st.session_state['model_type']}**")
    new_file = st.file_uploader("Upload New Unseen Raw Dataset (CSV/Excel)", type=['csv', 'xlsx'], key="pred")
    
    if new_file:
        pred_df = pd.read_csv(new_file) if new_file.name.endswith('.csv') else pd.read_excel(new_file)
        
        date_c = st.session_state['date_col']
        target_name = st.session_state['target_col']
        feature_configs = st.session_state['feature_configs']
        use_date = st.session_state['use_date_encoding']
        model = st.session_state['trained_model']
        model_type = st.session_state['model_type']
        
        # Check required base columns based on what the user built
        req_base_cols = list(set([cfg['col'] for cfg in feature_configs]))
        missing = [c for c in req_base_cols if c not in pred_df.columns]
        
        if date_c not in pred_df.columns:
            st.error(f"Missing date column '{date_c}' in file.")
        elif missing:
            st.error(f"Missing base columns in file required to engineer your features: {missing}")
        else:
            if st.button("🔮 Generate Target Predictions", type="primary"):
                pred_df[date_c] = pd.to_datetime(pred_df[date_c], errors='coerce')
                pred_df = pred_df.sort_values(date_c).reset_index(drop=True)
                
                # Apply granular engineering seamlessly on raw input data
                pred_feat, engineered_cols = create_features(pred_df, date_c, use_date, feature_configs)
                pred_feat = pred_feat.dropna(subset=engineered_cols).reset_index(drop=True)
                
                if not st.session_state['is_sequence_model']:
                    X_new = pred_feat[engineered_cols]
                    
                    if model_type in ["SVR", "KNN", "ANN (MLP)"]:
                        X_new_sc = st.session_state['scaler_x'].transform(X_new)
                        preds = model.predict(X_new_sc)
                    else:
                        preds = model.predict(X_new)
                        
                    res = pd.DataFrame({
                        date_c: pred_feat[date_c],
                        f"Predicted_{target_name}": preds
                    })
                else:
                    win = st.session_state['window_size']
                    X_sc = st.session_state['scaler_x'].transform(pred_feat[engineered_cols])
                    X_seq_new = create_pred_sequences(X_sc, win)
                    
                    if len(X_seq_new) == 0:
                        st.error(f"Uploaded file has fewer valid rows than the required window size ({win} days) after dropping history padding.")
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
                            date_c: pred_feat[date_c].iloc[win:].reset_index(drop=True),
                            f"Predicted_{target_name}": preds
                        })
                    
                if 'res' in locals() and res is not None:
                    st.success("Predictions generated!")
                    st.dataframe(res.head(15), use_container_width=True)
                    excel_pred_data = to_excel_bytes(res)
                    st.download_button(
                        label="📥 Download Predictions (Excel)",
                        data=excel_pred_data,
                        file_name=f"predicted_{target_name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
