import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open('XGBClassifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the main function for Streamlit
def main():
    st.title("Insurance Claim Prediction App")
    
    # Input fields for user data
    age = st.number_input("Enter Age", min_value=0, max_value=100, value=25)
    sex = st.selectbox("Select Sex", options=[('Male', 1), ('Female', 0)])
    bmi = st.number_input("Enter BMI", min_value=0.0, max_value=60.0, value=25.0)
    children = st.number_input("Enter Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Are you a smoker?", options=[('Yes', 1), ('No', 0)])
    region = st.selectbox("Select Region", options=[('Northeast', 0), ('Northwest', 1), ('Southeast', 2), ('Southwest', 3)])
    charges = st.number_input("Enter Insurance Charges", min_value=0.0, max_value=100000.0)
    
    # When the 'Predict' button is clicked
    if st.button("Predict"):
        # Arrange the data in an input array for the model (same order as training)
        input_data = np.array([[age, sex[1], bmi, children, smoker[1], region[1], charges]])
        
        # Use the model to make a prediction
        prediction = model.predict(input_data)
        
        # Display the result message based on the prediction
        if prediction[0] == 1:
            st.success("Congrats! Your insurance is claimed.")
        else:
            st.error("Sorry! Your insurance is not claimed.")
    
# Run the main function (Streamlit handles server startup)
if __name__ == '__main__':
    main()
