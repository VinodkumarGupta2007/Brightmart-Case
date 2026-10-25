import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('model.sav')

# Set up the page configuration
st.set_page_config(
    page_title="Advertising Sales Prediction",
    page_icon="📈",
    layout="centered"
)

# App title and description
st.title("📈 Advertising Sales Predictor")
st.write("""
This app predicts **Sales** based on the advertising budgets allocated to **TV**, **Radio**, and **Newspaper** media channels.
""")

# Sidebar for user inputs
st.sidebar.header("Configuration Panel")
st.sidebar.write("Adjust the budgets (in thousands of dollars) below:")

tv_budget = st.sidebar.slider("TV Budget ($k)", min_value=0.0, max_value=350.0, value=150.0, step=1.0)
radio_budget = st.sidebar.slider("Radio Budget ($k)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
newspaper_budget = st.sidebar.slider("Newspaper Budget ($k)", min_value=0.0, max_value=150.0, value=20.0, step=1.0)

# Display the selected inputs on the main panel
st.subheader("Selected Advertising Budgets")
input_data = pd.DataFrame({
    'TV': [tv_budget],
    'Radio': [radio_budget],
    'Newspaper': [newspaper_budget]
})
st.dataframe(input_data)

# Make prediction
if st.button("Predict Sales"):
    prediction = model.predict(input_data)
    
    # Display the result
    st.success(f"### Predicted Sales: **{prediction[0]:.2f}** units")
    
    # Display a visual breakdown / contribution indicator
    st.info(
        f"Contribution Details (based on coefficients):\n" 
        f"- TV Contribution: {tv_budget * 0.0447:.2f}\n"
        f"- Radio Contribution: {radio_budget * 0.1892:.2f}\n"
        f"- Newspaper Contribution: {newspaper_budget * 0.0028:.2f}"
    )
