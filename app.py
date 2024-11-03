import streamlit as st
import pickle
import numpy as np

# Load the model
with open('rf-model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set page configuration
st.set_page_config(page_title="Insurance Cost Prediction", page_icon="💰")

# App title and description
st.title("💼 Insurance Cost Prediction Form")
st.markdown("""
This application estimates health insurance costs based on personal details. 
Fill out the form below and click **Submit** to receive an estimate.
""")

# Input form with two-column layout
with st.form("input_form"):
    st.subheader("Enter Your Details")

    # Divide into two columns
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, help="Enter your age in years")
        sex = st.selectbox("Sex", options=["male", "female"], help="Select your gender")
        bmi = st.number_input("BMI", min_value=0.0, help="Enter your Body Mass Index (BMI)")

    with col2:
        children = st.number_input("Number of Children", min_value=0, help="Enter the number of children/dependents")
        smoker = st.radio("Smoker", options=["yes", "no"], help="Indicate if you smoke")  # Radio button for Smoker

    # Submit button
    submitted = st.form_submit_button("Submit")

    if submitted:
        # Convert inputs to appropriate format
        sex = 1 if sex == "male" else 0
        smoker = 1 if smoker == "yes" else 0

        # Prepare data as a numpy array
        input_data = np.array([[age, sex, bmi, children, smoker]])

        # Make prediction
        prediction = model.predict(input_data)

        # Display result with styling
        st.markdown("### Estimated Insurance Cost")
        st.write("Based on your inputs, your estimated insurance cost is:")
        st.success(f"R$ {prediction[0]:,.2f}")
