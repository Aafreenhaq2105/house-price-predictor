import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Train the model directly
df = pd.read_csv('data.csv')
X = df[['size', 'bedrooms', 'age']]
y = df['price']
model = LinearRegression()
model.fit(X, y)

# Page title
st.title("House Price Predictor")
st.write("Enter the details of a house to get an instant price estimate.")

# Input boxes
size = st.number_input("House size (sq ft)", min_value=100, max_value=10000, value=1500)
bedrooms = st.number_input("Number of bedrooms", min_value=1, max_value=10, value=3)
age = st.number_input("Age of house (years)", min_value=0, max_value=100, value=5)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[size, bedrooms, age]], columns=['size', 'bedrooms', 'age'])
    prediction = model.predict(input_data)
    st.success(f"Predicted House Price: Rs {prediction[0]:,.0f}")
    st.balloons()