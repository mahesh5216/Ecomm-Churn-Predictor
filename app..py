import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap

# Load Model
model = joblib.load('xgb_model.pkl')
feature_names = [f for f in model.get_booster().feature_names if f.lower() != 'customerid']

st.title("🚀 Real-Time E-Comm Churn Predictor")

tenure = st.sidebar.slider("Tenure (Months)", 0, 60, 12)
complain = st.sidebar.selectbox("Has Complained?", [0, 1])
day_since_last = st.sidebar.slider("Days Since Last Order", 0, 30, 5)

if st.button("Calculate Risk & Explain"):
    # Create input with ONLY valid features
    full_features = model.get_booster().feature_names
    input_df = pd.DataFrame(np.zeros((1, len(full_features))), columns=full_features)
    
    # Map inputs (Ensuring we don't touch customerid)
    if 'tenure' in input_df.columns: input_df['tenure'] = tenure
    if 'complain' in input_df.columns: input_df['complain'] = complain
    if 'daysincelastorder' in input_df.columns: input_df['daysincelastorder'] = day_since_last
    
    # Prediction
    prob = model.predict_proba(input_df)[0][1]
    st.subheader(f"Churn Probability: {prob:.1%}")
    
    # SHAP Explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # Find top reason EXCLUDING customerid
    feature_impacts = pd.Series(shap_values[0], index=full_features)
    if 'customerid' in feature_impacts:
        feature_impacts = feature_impacts.drop('customerid')
    
    top_reason = feature_impacts.abs().idxmax()
    
    st.write(f"### 🔍 Analysis")
    st.info(f"The most significant factor is: **{top_reason.upper()}**")
