import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Page title and description
st.title("House Price Predictor")
st.write("Enter the details of a house to get an instant price estimate.")

# Input boxes
size = st.number_input("House size (sq ft)", min_value=100, max_value=10000, value=1500)
bedrooms = st.number_input("Number of bedrooms", min_value=1, max_value=10, value=3)
age = st.number_input("Age of house (years)", min_value=0, max_value=100, value=5)

# Predict button
if st.button("Predict Price"):
    input_data = np.array([[size, bedrooms, age]])
    prediction = model.predict(input_data)
    
    st.success(f"Predicted House Price: Rs {prediction[0]:,.0f}")
    st.balloons()