import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Streamlit backward-compatible rerun helper
def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .fraud-alert {
        background-color: #b91c1c;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
    .safe-alert {
        background-color: #059669;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Feature engineering function (same as training)
def feature_engineering(df):
    """Apply the same feature engineering as training"""
    df_fe = df.copy()
    
    # Time-based features
    df_fe['Hour'] = (df_fe['Time'] / 3600) % 24
    df_fe['Day'] = df_fe['Time'] // (3600 * 24)
    
    # Amount-based features
    df_fe['Amount_log'] = np.log1p(df_fe['Amount'])
    df_fe['Amount_sqrt'] = np.sqrt(df_fe['Amount'])
    df_fe['Amount_squared'] = df_fe['Amount'] ** 2
    
    # Binning Amount
    df_fe['Amount_bin'] = pd.cut(df_fe['Amount'], 
                                   bins=[-0.01, 10, 50, 100, 500, np.inf],
                                   labels=[0, 1, 2, 3, 4])
    df_fe['Amount_bin'] = df_fe['Amount_bin'].astype(int)
    
    # Time of day categories
    df_fe['Time_of_day'] = pd.cut(df_fe['Hour'], 
                                    bins=[-0.01, 6, 12, 18, 24],
                                    labels=[0, 1, 2, 3])
    df_fe['Time_of_day'] = df_fe['Time_of_day'].astype(int)
    
    # Statistical features from V columns
    v_cols = [col for col in df.columns if col.startswith('V')]
    df_fe['V_mean'] = df_fe[v_cols].mean(axis=1)
    df_fe['V_std'] = df_fe[v_cols].std(axis=1)
    df_fe['V_min'] = df_fe[v_cols].min(axis=1)
    df_fe['V_max'] = df_fe[v_cols].max(axis=1)
    df_fe['V_range'] = df_fe['V_max'] - df_fe['V_min']
    df_fe['V_median'] = df_fe[v_cols].median(axis=1)
    df_fe['V_skew'] = df_fe[v_cols].skew(axis=1)
    df_fe['V_kurtosis'] = df_fe[v_cols].kurtosis(axis=1)
    
    # Interaction features
    important_v = ['V1', 'V3', 'V4', 'V10', 'V12', 'V14', 'V17']
    for i in range(len(important_v)):
        for j in range(i+1, len(important_v)):
            df_fe[f'{important_v[i]}_{important_v[j]}_interaction'] = (
                df_fe[important_v[i]] * df_fe[important_v[j]]
            )
    
    # Polynomial features
    df_fe['V1_squared'] = df_fe['V1'] ** 2
    df_fe['V3_squared'] = df_fe['V3'] ** 2
    df_fe['V4_squared'] = df_fe['V4'] ** 2
    df_fe['V14_squared'] = df_fe['V14'] ** 2
    
    return df_fe

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model_files = [
            'fraud_detection_model_XGBoost.pkl',
            'fraud_detection_model_LightGBM.pkl',
            'fraud_detection_model_CatBoost.pkl',
            'fraud_detection_model_Random_Forest.pkl',
            'model.pkl'
        ]
        
        model = None
        model_file_used = None
        for model_file in model_files:
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                model_file_used = model_file
                st.sidebar.success(f"✅ Loaded model: {model_file}")
                break
        
        if model is None:
            st.error("❌ No model file found. Please upload a trained model.")
            return None, None, None
        
        scaler = joblib.load('scaler.pkl') if os.path.exists('scaler.pkl') else None

        # Determine expected feature names/order
        feature_names = None
        if os.path.exists('feature_names.pkl'):
            feature_names = joblib.load('feature_names.pkl')
        elif scaler is not None and hasattr(scaler, 'feature_names_in_'):
            try:
                feature_names = list(scaler.feature_names_in_)
            except Exception:
                feature_names = None

        if feature_names is None:
            base_sample = {
                'Time': 0.0,
                'Amount': 0.0,
                **{f'V{i}': 0.0 for i in range(1, 29)}
            }
            df_sample = pd.DataFrame([base_sample])
            df_fe_sample = feature_engineering(df_sample)
            if 'Class' in df_fe_sample.columns:
                df_fe_sample = df_fe_sample.drop('Class', axis=1)
            feature_names = df_fe_sample.columns.tolist()

            try:
                joblib.dump(feature_names, 'feature_names.pkl')
            except Exception:
                pass

        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def predict_transaction(model, scaler, transaction_data, expected_features=None):
    """Predict if transaction is fraudulent - FIXED VERSION"""
    try:
        # Create DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Apply feature engineering
        df_fe = feature_engineering(df)
        
        # Remove Class if present
        if 'Class' in df_fe.columns:
            df_fe = df_fe.drop('Class', axis=1)
        
        # Align features with expected features from training
        if expected_features is not None:
            missing_features = set(expected_features) - set(df_fe.columns)
            if missing_features:
                for feat in missing_features:
                    df_fe[feat] = 0
            df_fe = df_fe[expected_features]
        
        # Scale features
        if scaler is not None:
            X_scaled = scaler.transform(df_fe)
        else:
            X_scaled = df_fe.values
        
        # FIXED: Simplified probability extraction
        y_pred_proba = model.predict_proba(X_scaled)[0]
        
        # Get class labels if available
        if hasattr(model, 'classes_'):
            classes = model.classes_
            # Find index of fraud class (1)
            if 1 in classes:
                fraud_idx = list(classes).index(1)
                fraud_prob = float(y_pred_proba[fraud_idx])
            else:
                # If no class 1, assume second column is fraud
                fraud_prob = float(y_pred_proba[1]) if len(y_pred_proba) > 1 else float(y_pred_proba[0])
        else:
            # Assume binary: [legit_prob, fraud_prob]
            fraud_prob = float(y_pred_proba[1]) if len(y_pred_proba) > 1 else float(y_pred_proba[0])
        
        # Ensure probability is valid
        fraud_prob = max(0.0, min(1.0, fraud_prob))
        legit_prob = 1.0 - fraud_prob
        
        # Return as [legit, fraud] format
        probability = np.array([legit_prob, fraud_prob])
        
        # Make prediction based on fraud probability
        prediction = 1 if fraud_prob >= 0.5 else 0
        
        # Debug output
        st.sidebar.write("**Debug Info:**")
        st.sidebar.write(f"Raw proba shape: {y_pred_proba.shape}")
        st.sidebar.write(f"Raw proba: {y_pred_proba}")
        st.sidebar.write(f"Fraud prob: {fraud_prob:.4f}")
        st.sidebar.write(f"Prediction: {prediction}")
        
        return prediction, probability
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def create_gauge_chart(probability, threshold=0.5):
    """Create a gauge chart for fraud probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Probability (%)", 'font': {'size': 24}},
        delta = {'reference': threshold * 100},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#00cc66'},
                {'range': [30, 70], 'color': '#ffcc00'},
                {'range': [70, 100], 'color': '#ff4b4b'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100}}))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# Main app
def main():
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("### AI-Powered Real-Time Transaction Analysis")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # FIXED: Lower default threshold for imbalanced data
    threshold = st.sidebar.slider(
        "Decision threshold for FRAUD",
        0.05, 0.90, 0.30, 0.05,  # Changed default from 0.50 to 0.30
        help="Label as FRAUD when predicted probability ≥ threshold. Lower threshold catches more fraud but increases false positives."
    )
    
    st.sidebar.info(f"⚠️ Recommended threshold for imbalanced fraud detection: 0.20-0.35")
    
    # Load model
    model, scaler, expected_features = load_model_and_scaler()
    
    if model is None:
        st.warning("⚠️ Please upload model files to proceed")
        return
    
    st.markdown("---")
    st.subheader("🔍 Enter Transaction Details")
    
    # Initialize defaults
    if 'time_val' not in st.session_state:
        st.session_state['time_val'] = 0.0
    if 'amount_val' not in st.session_state:
        st.session_state['amount_val'] = 100.0
    for _i in range(1, 29):
        st.session_state.setdefault(f'v{_i}', 0.0)

    # Apply presets
    if st.session_state.get('apply_random', False):
        st.session_state['time_val'] = float(np.random.uniform(0, 172800))
        st.session_state['amount_val'] = float(np.random.lognormal(mean=4.0, sigma=1.0))
        for _i in range(1, 29):
            st.session_state[f'v{_i}'] = float(np.random.randn())
        st.session_state['apply_random'] = False
    if st.session_state.get('apply_fraud', False):
        # Typical fraud patterns based on research
        st.session_state['time_val'] = float(np.random.uniform(60000, 80000))  # Often at night
        st.session_state['amount_val'] = float(np.random.uniform(200, 2000))
        # Set suspicious V features (these typically indicate fraud)
        st.session_state['v1'] = -2.5
        st.session_state['v3'] = -3.5
        st.session_state['v4'] = 2.8
        st.session_state['v10'] = -2.0
        st.session_state['v12'] = -3.0
        st.session_state['v14'] = -4.5
        st.session_state['v17'] = -3.5
        for _i in [2, 5, 6, 7, 8, 9, 11, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]:
            st.session_state[f'v{_i}'] = float(np.random.randn())
        st.session_state['apply_fraud'] = False
    if st.session_state.get('reset_inputs', False):
        st.session_state['time_val'] = 0.0
        st.session_state['amount_val'] = 100.0
        for _i in range(1, 29):
            st.session_state[f'v{_i}'] = 0.0
        st.session_state['reset_inputs'] = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_time = st.number_input("Time (seconds from first transaction)", 
                               min_value=0.0, value=float(st.session_state.get('time_val', 0.0)), 
                               help="Time elapsed since first transaction", key='time_val')
        amount = st.number_input("Transaction Amount ($)", 
                                 min_value=0.0, value=float(st.session_state.get('amount_val', 100.0)),
                                 help="Transaction amount in dollars", key='amount_val')
    
    with col2:
        st.markdown("#### Quick Presets")
        q1, q2, q3 = st.columns(3)
        with q1:
            if st.button("🎲 Random"):
                st.session_state['apply_random'] = True
                _safe_rerun()
        with q2:
            if st.button("⚠️ Fraud Pattern"):
                st.session_state['apply_fraud'] = True
                _safe_rerun()
        with q3:
            if st.button("↺ Reset"):
                st.session_state['reset_inputs'] = True
                _safe_rerun()
    
    # V features
    st.markdown("#### PCA Features (V1-V28)")
    with st.expander("V Features (Click to expand)", expanded=False):
        v_cols = st.columns(4)
        v_features = {}
        
        for i in range(1, 29):
            col_idx = (i - 1) % 4
            with v_cols[col_idx]:
                default_val = st.session_state.get(f'v{i}', 0.0)
                v_features[f'V{i}'] = st.number_input(
                    f'V{i}', 
                    value=float(default_val),
                    format="%.6f",
                    key=f'v{i}'
                )
    
    st.markdown("---")
    
    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🔮 Analyze Transaction", 
                                use_container_width=True,
                                type="primary")
    
    if predict_btn:
        transaction_data = {
            'Time': transaction_time,
            'Amount': amount,
            **v_features
        }
        
        with st.spinner('🔄 Analyzing transaction...'):
            prediction, probability = predict_transaction(model, scaler, transaction_data, expected_features)
        
        if prediction is not None:
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                is_fraud = 1 if probability[1] >= threshold else 0
                if is_fraud == 1:
                    st.markdown(
                        '<div class="fraud-alert">⚠️ FRAUDULENT TRANSACTION</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="safe-alert">✅ LEGITIMATE TRANSACTION</div>',
                        unsafe_allow_html=True
                    )
            
            with col2:
                st.metric(
                    "Fraud Probability",
                    f"{probability[1]*100:.2f}%",
                    delta=f"{(probability[1]-threshold)*100:.2f}% from threshold"
                )
            
            with col3:
                st.metric(
                    "Confidence Score",
                    f"{max(probability)*100:.2f}%",
                    delta="High" if max(probability) > 0.8 else "Medium"
                )
            
            # Gauge chart
            st.plotly_chart(create_gauge_chart(probability[1], threshold), 
                           use_container_width=True)
            
            # Detailed probabilities
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Legitimate Probability:** {probability[0]*100:.2f}%")
            with col2:
                st.info(f"**Fraud Probability:** {probability[1]*100:.2f}%")
            
            # Risk level
            st.markdown("#### 🎯 Risk Assessment")
            risk_level = "LOW" if probability[1] < 0.3 else "MEDIUM" if probability[1] < 0.7 else "HIGH"
            risk_color = "green" if risk_level == "LOW" else "orange" if risk_level == "MEDIUM" else "red"
            
            st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")
            
            if is_fraud == 1:
                st.error("""
                ### ⚠️ Recommended Actions:
                - 🚫 Block the transaction immediately
                - 📞 Contact the cardholder for verification
                - 🔍 Review recent account activity
                - 📝 File a fraud report
                """)
            else:
                st.success("""
                ### ✅ Transaction Approved
                - Transaction appears legitimate
                - No immediate action required
                """)

if __name__ == "__main__":
    main()