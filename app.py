import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time as time_module
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
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #0f172a; /* slate-900 */
        color: #e5e7eb; /* gray-200 */
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #1f2937; /* gray-800 */
    }
    h1 {
        color: #60a5fa; /* blue-400 */
    }
    .fraud-alert {
        background-color: #b91c1c; /* red-700 */
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
    .safe-alert {
        background-color: #059669; /* emerald-600 */
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #ffffff;
        border: 0;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        filter: brightness(1.05);
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
def _map_probabilities(model, raw_proba):
    """Map model's predict_proba output so index 1 is always FRAUD probability.
    Returns (probability_array, fraud_index, classes_array)."""
    try:
        # Ensure ndarray
        raw = np.array(raw_proba).reshape(-1)
        classes = np.array(getattr(model, 'classes_', [0, 1]))
        fraud_index = None
        if raw.size == 2 and 1 in classes:
            fraud_index = int(np.where(classes == 1)[0][0])
            fraud_prob = float(raw[fraud_index])
            probability = np.array([max(0.0, 1.0 - fraud_prob), max(0.0, min(1.0, fraud_prob))])
        elif raw.size == 1:
            fraud_prob = float(raw[0])
            probability = np.array([max(0.0, 1.0 - fraud_prob), max(0.0, min(1.0, fraud_prob))])
            fraud_index = 0
        else:
            # Fallback assume second column
            fraud_prob = float(raw[1]) if raw.size >= 2 else 0.0
            probability = np.array([max(0.0, 1.0 - fraud_prob), max(0.0, min(1.0, fraud_prob))])
            fraud_index = 1 if raw.size >= 2 else 0
        return probability, fraud_index, classes
    except Exception:
        return np.array([0.0, 0.0]), None, np.array([])

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        # Try to load from different possible locations
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

        # 1) Prefer an explicit saved feature list
        if os.path.exists('feature_names.pkl'):
            feature_names = joblib.load('feature_names.pkl')

        # 2) If not present, use feature names stored inside the fitted scaler
        if feature_names is None and scaler is not None and hasattr(scaler, 'feature_names_in_'):
            try:
                feature_names = list(scaler.feature_names_in_)
            except Exception:
                feature_names = None

        # 3) As a final fallback, synthesize the names using the same
        #    feature engineering pipeline to ensure deterministic order
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

            # Persist for subsequent runs (best-effort)
            try:
                joblib.dump(feature_names, 'feature_names.pkl')
            except Exception:
                pass

        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def predict_transaction(model, scaler, transaction_data, expected_features=None):
    """Predict if transaction is fraudulent"""
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
            # Ensure all expected features are present
            missing_features = set(expected_features) - set(df_fe.columns)
            if missing_features:
                for feat in missing_features:
                    df_fe[feat] = 0
            
            # Reorder columns to match training order
            df_fe = df_fe[expected_features]
        
        # Scale features
        if scaler is not None:
            X_scaled = scaler.transform(df_fe)
        else:
            X_scaled = df_fe.values
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        raw_proba = model.predict_proba(X_scaled)[0]

        # Normalize probabilities so index 1 corresponds to FRAUD (class label 1)
        probability, fraud_index, classes = _map_probabilities(model, raw_proba)

        return prediction, probability, {"classes": classes, "fraud_index": fraud_index, "raw_proba": raw_proba}
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.error(f"Feature shape: {df_fe.shape if 'df_fe' in locals() else 'N/A'}")
        if 'df_fe' in locals():
            st.error(f"Features: {list(df_fe.columns)[:10]}...")
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

def create_feature_importance_chart(model, feature_names, top_n=15):
    """Create feature importance chart"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig = go.Figure(go.Bar(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            orientation='h',
            marker=dict(
                color=importances[indices],
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Most Important Features',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    return None

# Main app
def main():
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("### AI-Powered Real-Time Transaction Analysis")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    threshold = st.sidebar.slider(
        "Decision threshold for FRAUD",
        0.10, 0.90, 0.50, 0.05,
        help="Label a transaction as FRAUD when predicted probability ≥ threshold"
    )
    # Cost-sensitive controls
    st.sidebar.markdown("#### Cost & Prevalence (optional)")
    prevalence = st.sidebar.number_input(
        "Assumed fraud prevalence (0-1)", min_value=0.0, max_value=1.0, value=0.001, step=0.0005,
        help="Estimated base rate of fraud in your traffic"
    )
    cost_fp = st.sidebar.number_input(
        "Cost of false positive (legit flagged)", min_value=0.0, value=1.0, step=0.5,
        help="Relative business cost of blocking a legitimate transaction"
    )
    cost_fn = st.sidebar.number_input(
        "Cost of false negative (fraud missed)", min_value=0.0, value=50.0, step=1.0,
        help="Relative business cost of missing a fraudulent transaction"
    )
    # Bayes-optimal threshold for probabilistic classifier
    # t* = (cost_fn * (1 - prevalence)) / (cost_fp * prevalence + cost_fn * (1 - prevalence))
    try:
        denom = (cost_fp * prevalence) + (cost_fn * (1.0 - prevalence))
        recommended_threshold = (cost_fn * (1.0 - prevalence)) / denom if denom > 0 else threshold
        st.sidebar.info(f"Recommended threshold: {recommended_threshold:.3f}")
    except Exception:
        recommended_threshold = threshold
    
    # Load model
    model, scaler, expected_features = load_model_and_scaler()
    
    if model is None:
        st.warning("⚠️ Please upload model files to proceed")
        uploaded_model = st.file_uploader("Upload Model (.pkl)", type=['pkl'])
        uploaded_scaler = st.file_uploader("Upload Scaler (.pkl)", type=['pkl'])
        
        if uploaded_model and uploaded_scaler:
            with open('model.pkl', 'wb') as f:
                f.write(uploaded_model.read())
            with open('scaler.pkl', 'wb') as f:
                f.write(uploaded_scaler.read())
            st.success("Files uploaded! Please refresh the page.")
        return
    
    # Sidebar options
    st.sidebar.markdown("---")
    prediction_mode = st.sidebar.radio(
        "Select Mode:",
        ["Single Transaction", "Batch Prediction", "Model Information"]
    )
    
    if prediction_mode == "Single Transaction":
        st.markdown("---")
        st.subheader("🔍 Enter Transaction Details")
        
        # Initialize defaults once
        if 'time_val' not in st.session_state:
            st.session_state['time_val'] = 0.0
        if 'amount_val' not in st.session_state:
            st.session_state['amount_val'] = 100.0
        for _i in range(1, 29):
            st.session_state.setdefault(f'v{_i}', 0.0)

        # Apply preset flags BEFORE widgets instantiate
        if st.session_state.get('apply_random', False):
            st.session_state['time_val'] = float(np.random.uniform(0, 172800))
            st.session_state['amount_val'] = float(np.random.lognormal(mean=4.0, sigma=1.0))
            for _i in range(1, 29):
                st.session_state[f'v{_i}'] = float(np.random.randn())
            st.session_state['apply_random'] = False
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
            q1, q2 = st.columns(2)
            with q1:
                if st.button("🎲 Generate Random Transaction"):
                    st.session_state['apply_random'] = True
                    _safe_rerun()
            with q2:
                if st.button("↺ Reset Inputs"):
                    st.session_state['reset_inputs'] = True
                    _safe_rerun()
        
        # Create input fields for V1-V28
        st.markdown("#### PCA Features (V1-V28)")
        st.info("💡 These are PCA-transformed features. Use sample data or leave as default.")
        
        # Create expandable section for V features
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
            # Prepare transaction data
            transaction_data = {
                'Time': transaction_time,
                'Amount': amount,
                **v_features
            }
            
            # Show loading animation
            with st.spinner('🔄 Analyzing transaction...'):
                time_module.sleep(1)
                prediction, probability, debug_info = predict_transaction(model, scaler, transaction_data, expected_features)
            
            if prediction is not None:
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Result columns
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
                
                # Additional decision metrics
                try:
                    odds = probability[1] / max(1e-12, (1 - probability[1]))
                    log_odds = np.log(max(1e-12, odds))
                    # Expected cost for both decisions
                    # If we approve (legit): cost = cost_fn * p(fraud)
                    expected_cost_approve = cost_fn * probability[1]
                    # If we block (fraud): cost = cost_fp * p(legit)
                    expected_cost_block = cost_fp * (1 - probability[1])
                    cm1, cm2, cm3 = st.columns(3)
                    with cm1:
                        st.metric("Likelihood Ratio (odds)", f"{odds:.3f}")
                    with cm2:
                        st.metric("Log-Odds", f"{log_odds:.3f}")
                    with cm3:
                        st.metric("Recommended Threshold", f"{recommended_threshold:.3f}")
                    cex1, cex2 = st.columns(2)
                    with cex1:
                        st.info(f"Expected Cost if APPROVE: {expected_cost_approve:.3f}")
                    with cex2:
                        st.info(f"Expected Cost if BLOCK: {expected_cost_block:.3f}")
                except Exception:
                    pass

                # Debug details for probability mapping
                with st.expander("🔧 Debug: Model Probability Mapping", expanded=False):
                    try:
                        st.write({
                            "classes_": list(debug_info.get("classes", [])) if isinstance(debug_info.get("classes", []), (list, np.ndarray)) else debug_info.get("classes", []),
                            "fraud_index": debug_info.get("fraud_index"),
                            "raw_proba": [float(x) for x in (debug_info.get("raw_proba") or [])] if isinstance(debug_info.get("raw_proba"), (list, np.ndarray)) else debug_info.get("raw_proba")
                        })
                    except Exception:
                        st.write("No debug info available")

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
                risk_level = "LOW" if probability[1] < max(0.3, threshold-0.2) else "MEDIUM" if probability[1] < max(0.7, threshold+0.2) else "HIGH"
                risk_color = "green" if risk_level == "LOW" else "orange" if risk_level == "MEDIUM" else "red"
                
                st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")
                
                if is_fraud == 1:
                    st.error("""
                    ### ⚠️ Recommended Actions:
                    - 🚫 Block the transaction immediately
                    - 📞 Contact the cardholder for verification
                    - 🔍 Review recent account activity
                    - 📝 File a fraud report
                    - 🔒 Consider temporary card suspension
                    """)
                else:
                    st.success("""
                    ### ✅ Transaction Approved
                    - Transaction appears legitimate
                    - No immediate action required
                    - Continue monitoring account activity
                    """)
                
                # Transaction summary
                with st.expander("📋 Transaction Summary"):
                    summary_data = {
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Amount': f"${amount:.2f}",
                        'Time Elapsed': f"{transaction_time:.0f} seconds",
                        'Prediction': "FRAUD" if is_fraud == 1 else "LEGITIMATE",
                        'Fraud Probability': f"{probability[1]*100:.2f}%",
                        'Confidence': f"{max(probability)*100:.2f}%"
                    }
                    st.json(summary_data)
    
    elif prediction_mode == "Batch Prediction":
        st.markdown("---")
        st.subheader("📁 Batch Transaction Analysis")
        
        st.info("""
        Upload a CSV file with the following columns:
        - Time, V1-V28, Amount
        - Optional: Class (for validation)
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} transactions")
                
                # Show sample data
                with st.expander("📊 Data Preview"):
                    st.dataframe(df.head(10))
                
                if st.button("🔮 Analyze All Transactions", type="primary"):
                    with st.spinner('Processing transactions...'):
                        # Apply feature engineering
                        df_fe = feature_engineering(df)
                        
                        # Remove Class if present
                        has_labels = 'Class' in df_fe.columns
                        if has_labels:
                            y_true = df_fe['Class'].copy()
                            df_fe = df_fe.drop('Class', axis=1)
                        
                        # Align features with expected features
                        if expected_features is not None:
                            missing_features = set(expected_features) - set(df_fe.columns)
                            if missing_features:
                                for feat in missing_features:
                                    df_fe[feat] = 0
                            df_fe = df_fe[expected_features]
                        
                        # Scale and predict
                        if scaler is not None:
                            X_scaled = scaler.transform(df_fe)
                        else:
                            X_scaled = df_fe.values

                        # Robust fraud probability mapping using classes_
                        try:
                            classes = np.array(getattr(model, 'classes_', [0, 1]))
                            if 1 in classes and len(classes) >= 2:
                                fraud_index = int(np.where(classes == 1)[0][0])
                                probabilities = model.predict_proba(X_scaled)[:, fraud_index]
                            else:
                                # Fallback assume second column is fraud
                                probs = model.predict_proba(X_scaled)
                                probabilities = probs[:, 1] if probs.shape[1] >= 2 else probs[:, 0]
                        except Exception:
                            probs = model.predict_proba(X_scaled)
                            probabilities = probs[:, 1] if probs.shape[1] >= 2 else probs[:, 0]
                        predictions = (probabilities >= threshold).astype(int)
                        
                        # Add results to dataframe
                        df['Prediction'] = predictions
                        df['Fraud_Probability'] = probabilities
                        df['Risk_Level'] = pd.cut(probabilities, 
                                                   bins=[0, max(0.3, threshold-0.2), max(0.7, threshold+0.2), 1],
                                                   labels=['LOW', 'MEDIUM', 'HIGH'], include_lowest=True)
                    
                    # Summary metrics
                    st.markdown("---")
                    st.subheader("📊 Analysis Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Transactions", len(df))
                    with col2:
                        fraud_count = predictions.sum()
                        st.metric("Fraudulent", int(fraud_count), 
                                 delta=f"{fraud_count/len(df)*100:.1f}%")
                    with col3:
                        legitimate_count = len(df) - fraud_count
                        st.metric("Legitimate", int(legitimate_count),
                                 delta=f"{legitimate_count/len(df)*100:.1f}%")
                    with col4:
                        avg_prob = probabilities.mean()
                        st.metric("Avg Fraud Prob", f"{avg_prob*100:.2f}%")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Prediction distribution
                        fig = px.pie(
                            values=[legitimate_count, fraud_count],
                            names=['Legitimate', 'Fraudulent'],
                            title='Transaction Distribution',
                            color_discrete_sequence=['#00cc66', '#ff4b4b']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Risk level distribution
                        risk_counts = df['Risk_Level'].value_counts()
                        fig = px.bar(
                            x=risk_counts.index,
                            y=risk_counts.values,
                            title='Risk Level Distribution',
                            labels={'x': 'Risk Level', 'y': 'Count'},
                            color=risk_counts.index,
                            color_discrete_map={
                                'LOW': '#00cc66',
                                'MEDIUM': '#ffcc00',
                                'HIGH': '#ff4b4b'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Probability distribution
                    fig = px.histogram(
                        df,
                        x='Fraud_Probability',
                        nbins=50,
                        title='Fraud Probability Distribution',
                        labels={'Fraud_Probability': 'Probability'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # If true labels available, show performance
                    if has_labels:
                        from sklearn.metrics import (
                            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                            average_precision_score, confusion_matrix, matthews_corrcoef,
                            balanced_accuracy_score, precision_recall_curve, roc_curve
                        )
                        
                        st.markdown("---")
                        st.subheader("🎯 Model Performance")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            acc = accuracy_score(y_true, predictions)
                            st.metric("Accuracy", f"{acc*100:.2f}%")
                        with col2:
                            prec = precision_score(y_true, predictions)
                            st.metric("Precision", f"{prec*100:.2f}%")
                        with col3:
                            rec = recall_score(y_true, predictions)
                            st.metric("Recall", f"{rec*100:.2f}%")
                        with col4:
                            f1 = f1_score(y_true, predictions)
                            st.metric("F1-Score", f"{f1*100:.2f}%")
                        with col5:
                            auc = roc_auc_score(y_true, probabilities)
                            st.metric("ROC-AUC", f"{auc*100:.2f}%")

                        # Additional metrics
                        ap = average_precision_score(y_true, probabilities)
                        bal_acc = balanced_accuracy_score(y_true, predictions)
                        mcc = matthews_corrcoef(y_true, predictions)
                        st.info(f"Average Precision (PR-AUC): {ap:.3f} | Balanced Acc: {bal_acc:.3f} | MCC: {mcc:.3f}")

                        # Confusion matrix
                        cm = confusion_matrix(y_true, predictions)
                        cm_fig = px.imshow(
                            cm,
                            text_auto=True,
                            color_continuous_scale='Blues',
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=["Legit","Fraud"], y=["Legit","Fraud"]
                        )
                        cm_fig.update_layout(title="Confusion Matrix", height=400)
                        st.plotly_chart(cm_fig, use_container_width=True)

                        # ROC and PR Curves
                        fpr, tpr, _ = roc_curve(y_true, probabilities)
                        prec_curve, rec_curve, _ = precision_recall_curve(y_true, probabilities)

                        roc_fig = go.Figure()
                        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
                        roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
                        roc_fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=400)
                        st.plotly_chart(roc_fig, use_container_width=True)

                        pr_fig = go.Figure()
                        pr_fig.add_trace(go.Scatter(x=rec_curve, y=prec_curve, mode='lines', name='PR'))
                        pr_fig.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision', height=400)
                        st.plotly_chart(pr_fig, use_container_width=True)

                        # Top-K fraud capture
                        try:
                            k_pct = 0.05
                            k = max(1, int(len(df) * k_pct))
                            order = np.argsort(-probabilities)
                            topk_idx = order[:k]
                            y_topk = (y_true.iloc[topk_idx] if hasattr(y_true, 'iloc') else np.array(y_true)[topk_idx])
                            capture_rate = (y_topk.sum() / max(1, y_true.sum())) if y_true.sum() > 0 else 0.0
                            st.metric(f"Top {int(k_pct*100)}% Capture", f"{capture_rate*100:.1f}% of frauds")
                        except Exception:
                            pass
                    
                    # Show flagged transactions
                    st.markdown("---")
                    st.subheader("🚨 Flagged Transactions (High Risk)")
                    
                    flagged = df[df['Prediction'] == 1].sort_values('Fraud_Probability', ascending=False)
                    
                    if len(flagged) > 0:
                        st.dataframe(
                            flagged[['Time', 'Amount', 'Fraud_Probability', 'Risk_Level']].head(20),
                            use_container_width=True
                        )
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Results",
                            data=csv,
                            file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv'
                        )
                    else:
                        st.success("✅ No fraudulent transactions detected!")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    else:  # Model Information
        st.markdown("---")
        st.subheader("🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Model Details")
            model_info = {
                "Model Type": type(model).__name__,
                "Status": "✅ Loaded and Ready",
                "Scaler": "✅ Available" if scaler is not None else "❌ Not Available",
                "Feature Names": "✅ Available" if expected_features is not None else "❌ Not Available"
            }
            
            for key, value in model_info.items():
                st.info(f"**{key}:** {value}")
        
        with col2:
            st.markdown("#### ⚙️ Model Parameters")
            if hasattr(model, 'get_params'):
                params = model.get_params()
                st.json({k: str(v) for k, v in list(params.items())[:10]})
        
        # Feature importance
        st.markdown("---")
        st.subheader("📊 Feature Importance Analysis")
        
        # Get feature names
        if expected_features is not None:
            feature_names = expected_features
        else:
            # Generate sample data to get feature names
            sample_data = {
                'Time': 0.0,
                'Amount': 100.0,
                **{f'V{i}': 0.0 for i in range(1, 29)}
            }
            df_sample = pd.DataFrame([sample_data])
            df_fe = feature_engineering(df_sample)
            feature_names = df_fe.columns.tolist()
        
        fig = create_feature_importance_chart(model, feature_names)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
        
        # Statistics
        st.markdown("---")
        st.subheader("📈 Usage Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Features", len(feature_names))
        with col2:
            st.metric("Original Features", 30)
        with col3:
            st.metric("Engineered Features", len(feature_names) - 30)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>💳 Credit Card Fraud Detection System | Powered by Machine Learning</p>
        <p>⚠️ For demonstration purposes only. Always verify with additional security measures.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()