import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the trained model
model = load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('onehotencoder.pkl', 'rb') as file:
    onehot_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler_encoder = pickle.load(file)

## Streamlit App
st.title('Customer Churn Prediction')

# User Inputs
geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = int(st.selectbox('Has Credit Card', [1, 0]))
is_active_member = int(st.selectbox('Is Active Member', [0, 1]))

# Encode categorical features
geo_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

gender_encoded = label_encoder.transform([gender])[0]  # Encode Gender

# Create DataFrame with the correct feature order
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],  
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],  
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Merge with one-hot encoded geography
input_df = pd.concat([input_data, geo_encoded_df], axis=1)

# Ensure the feature order matches the training phase
feature_order = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain'
]
input_df = input_df[feature_order]

# Scale the input data
input_scaled = scaler_encoder.transform(input_df)

# Make the prediction
prediction = model.predict(input_scaled)

# Get probability of churn
prediction_proba = prediction[0][0]
st.write(f'Churn probability is : {prediction_proba:.2f}')
# Display results
if prediction_proba > 0.5:
    st.write('The customer is **likely** to churn.')
else:
    st.write('The customer is **not likely** to churn.')
