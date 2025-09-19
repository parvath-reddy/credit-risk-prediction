import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model and encoders
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load("models/extra_trees_credit_model.pkl")
    encoders = {
        'Sex': joblib.load("models/Sex_encoder.pkl"),
        'Housing': joblib.load("models/Housing_encoder.pkl"),
        'Saving accounts': joblib.load("models/Saving_accounts_encoder.pkl"),
        'Checking account': joblib.load("models/Checking_account_encoder.pkl")
    }
    return model, encoders

# App title
st.title("🏦 Credit Risk Prediction App")
st.write("Enter applicant information to predict if the credit risk is good or bad.")

# Load model and encoders
try:
    model, encoders = load_model_and_encoders()
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.number_input("Job (0–3)", min_value=0, max_value=3, value=1)
        housing = st.selectbox("Housing", ["own", "rent", "free"])
    
    with col2:
        saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich"])
        checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
        credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
        duration = st.number_input("Duration (months)", min_value=1, value=12)
    
    # Prediction button
    if st.button("Predict Risk", type="primary"):
        try:
            # Convert inputs into a DataFrame
            input_df = pd.DataFrame({
                "Age": [age],
                "Sex": [encoders["Sex"].transform([sex])[0]],
                "Job": [job],
                "Housing": [encoders["Housing"].transform([housing])[0]],
                "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
                "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
                "Credit amount": [credit_amount],
                "Duration": [duration]
            })
            
            # Make prediction
            pred = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0]
            
            # Display results
            if pred == 1:
                st.success(f"✅ The predicted credit risk is: **GOOD** (Confidence: {pred_proba[1]:.2%})")
            else:
                st.error(f"❌ The predicted credit risk is: **BAD** (Confidence: {pred_proba[0]:.2%})")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.info("Please ensure all model files are in the 'models/' directory.")