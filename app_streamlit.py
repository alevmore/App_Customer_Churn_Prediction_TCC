import streamlit as st
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier


# Load model
try:
    with open('model_xgb.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'model_xgb.pkl' not found. Please ensure the file is available.")

#  Layout for the app
text_col1, text_col2 = st.columns([9, 1])
col1, col2, col3 = st.columns(3)

# Header and input section
text_col1.title('Customer Churn Prediction for a Telecommunications Company')
text_col1.header('Enter new customer details')

with col1:
    
    subscription_age = st.number_input('Subscripted for how many years?', min_value=0)
    bill_avg = st.number_input('Average bill: ', min_value=0.0)
    service_failure_count = st.number_input('Count of service failure: ', min_value=0)


with col2:
    download_over_limit = st.number_input('Overlimit download time: ', min_value=0)
    download_avg = st.number_input('Average download time: ', min_value=0.0)
    upload_avg = st.number_input('Average upload time: ', min_value=0.0)
    

with col3:  
    is_tv_subscriber = st.selectbox('Subscription on TV?', ['No', 'Yes'])
    is_movie_package_subscriber = st.selectbox('Subscription on movie package?', ['No', 'Yes'])
    

# Creating a DataFrame from the input
new_data = pd.DataFrame({
    'is_tv_subscriber': [1 if is_tv_subscriber == 'Yes' else 0],
    'is_movie_package_subscriber': [1 if is_movie_package_subscriber == 'Yes' else 0],
    'subscription_age': [subscription_age],
    'bill_avg': [bill_avg],
    'service_failure_count': [service_failure_count],
    'download_avg': [download_avg],
    'upload_avg': [upload_avg],
    'download_over_limit': [download_over_limit]
})

# Check if the model was loaded successfully before making predictions

if st.button('Predict Churn Probability'):
    if 'loaded_model' in locals():
        try:
            # Predicting the probability of churn
            churn_prob = loaded_model.predict_proba(new_data)[:, 1][0]

            # Displaying the result
            st.header('Result of prediction')
            if churn_prob > 0.5:
                st.write(f"The customer has a high probability of churn ({churn_prob:.2f})")
            else:
                st.write(f"The customer has a low probability of churn ({churn_prob:.2f})")
        except Exception as e:
            st.error(f"An error occurred while predicting: {e}")
    else:
        st.warning("Model not loaded. Predictions cannot be made.")