import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Rainfall-Runoff App", layout="wide")

def calc_nse(obs, sim):
    return 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2))
def calc_kge(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def create_features(df, rainfall_cols):
    df = df.copy()
    for col in rainfall_cols:
        for lag in range(1, 4):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        df[f'{col}_sum3'] = df[col].rolling(3).sum()
        df[f'{col}_sum7'] = df[col].rolling(7).sum()
    return df

st.title("🌧️ Rainfall-Runoff Model")
st.caption("Powered by HistGradientBoosting | Optimized for Large Datasets")

# --- PHASE 1: TRAINING ---
st.subheader("1. Training & Validation")
st.write("Upload your complete dataset. The model will automatically ignore rows where Discharge is missing.")

train_file = st.file_uploader("Upload Training Data (CSV or Excel)", type=['csv', 'xlsx'], key="train")

if train_file:
    df = pd.read_csv(train_file) if train_file.name.endswith('.csv') else pd.read_excel(train_file)
    cols = df.columns.tolist()
    
    st.write("**Map your columns:**")
    c1, c2, c3 = st.columns(3)
    date_col = c1.selectbox("Select Date Column", cols, index=cols.index('Date') if 'Date' in cols else 0)
    target_col = c2.selectbox("Select Discharge Column", cols, index=cols.index('Discharge(cusec)') if 'Discharge(cusec)' in cols else 0)
    rain_cols = c3.multiselect("Select Rainfall Columns (Select 1 or more)", [c for c in cols if c not in [date_col, target_col]], default=[c for c in cols if 'rainfall' in c.lower()])

    if st.button("🚀 Train Model", type="primary") and rain_cols:
        with st.spinner('Training model...'):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            
            # Clean missing target data
            df = df.dropna(subset=[date_col, target_col]).sort_values(date_col)
            
            df_feat = create_features(df, rain_cols)
            df_clean = df_feat.dropna().reset_index(drop=True) # Drops rows with NaNs from lags
            
            if len(df_clean) < 10:
                st.error("Not enough continuous data to train after removing missing values.")
            else:
                X = df_clean.drop(columns=[date_col, target_col, 'Water Level(m)'], errors='ignore')
                y = df_clean[target_col].values
                
                split = int(len(df_clean) * 0.8)
                X_train, X_test = X.iloc[:split], X.iloc[split:]
                y_train, y_test = y[:split], y[split:]
                
                model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, random_state=42)
                model.fit(X_train, y_train)
                
                st.session_state['model'] = model
                st.session_state['rain_cols'] = rain_cols
                st.session_state['date_col'] = date_col
                
                y_pred = model.predict(X_test)
                
                st.success(f"Model trained successfully on {len(X_train)} rows!")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("R²", round(r2_score(y_test, y_pred), 3))
                m2.metric("NSE", round(calc_nse(y_test, y_pred), 3))
                m3.metric("KGE", round(calc_kge(y_test, y_pred), 3))
                m4.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))
                m5.metric("MAE", round(mean_absolute_error(y_test, y_pred), 2))
                
                fig, ax = plt.subplots(figsize=(12, 3))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                ax.plot(df_clean[date_col].iloc[split:], y_test, label="Observed", color='#00b4d8')
                ax.plot(df_clean[date_col].iloc[split:], y_pred, label="Predicted", color='#ff4b4b', linestyle='--')
                ax.tick_params(colors='white')
                ax.legend(facecolor='#0e1117', labelcolor='white')
                st.pyplot(fig)

st.divider()

# --- PHASE 2: PREDICTION ---
st.subheader("2. Generate Missing Discharge")
if 'model' in st.session_state:
    st.write("Upload a file containing Dates and Rainfall to generate missing discharge values.")
    pred_file = st.file_uploader("Upload Prediction Data", type=['csv', 'xlsx'], key="pred")
    
    if pred_file:
        new_df = pd.read_csv(pred_file) if pred_file.name.endswith('.csv') else pd.read_excel(pred_file)
        
        if st.button("🔮 Generate Discharge", type="primary"):
            new_df[st.session_state['date_col']] = pd.to_datetime(new_df[st.session_state['date_col']], errors='coerce')
            new_df = new_df.dropna(subset=[st.session_state['date_col']]).sort_values(st.session_state['date_col'])
            
            new_df_feat = create_features(new_df, st.session_state['rain_cols']).dropna(subset=st.session_state['rain_cols']).reset_index(drop=True)
            
            X_new = new_df_feat[st.session_state['model'].feature_names_in_]
            
            preds = st.session_state['model'].predict(X_new)
            res_df = pd.DataFrame({'Date': new_df_feat[st.session_state['date_col']], 'Predicted_Discharge': preds})
            
            st.dataframe(res_df.head(10), use_container_width=True)
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Predictions (CSV)", data=csv, file_name="predicted_discharge.csv", mime="text/csv")
