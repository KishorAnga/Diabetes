import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import numpy as np
import joblib

model = joblib.load("diabetes_logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Diabetes Prediction App")

pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 100)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin,
                             insulin, bmi, dpf, age]])
    
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.error("Diabetic")
    else:
        st.success("Not Diabetic")
