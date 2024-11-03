import streamlit as st
import pickle
import numpy as np

# Load the model
with open('postech-iadev/rf-model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Prediction Form")

# Input form
with st.form("input_form"):
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", options=["male", "female"])
    bmi = st.number_input("BMI", min_value=0.0)
    children = st.number_input("Number of Children", min_value=0)
    smoker = st.selectbox("Smoker", options=["yes", "no"])
    # region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

    # Submit button
    submitted = st.form_submit_button("Submit")

    if submitted:
        # Convert inputs to appropriate format
        sex = 1 if sex == "male" else 0
        smoker = 1 if smoker == "yes" else 0

        input_data = np.array([[age, sex, bmi, children, smoker]])
        
        # # Apply one-hot encoding for 'region'
        # region_northeast = 1 if region == "northeast" else 0
        # region_northwest = 1 if region == "northwest" else 0
        # region_southeast = 1 if region == "southeast" else 0
        # region_southwest = 1 if region == "southwest" else 0

        # # Prepare data as a numpy array
        # input_data = np.array([[age, sex, bmi, children, smoker,
        #                         region_northeast, region_northwest, region_southeast, region_southwest]])

        # Make prediction 
        prediction = model.predict(input_data)

        st.write("Prediction:", prediction)