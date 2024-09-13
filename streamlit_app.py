import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open('XGBClassifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the main function for Streamlit
def main():
    # Display a single logo with adjusted size
    st.image("logo2.jpeg", width=150)  # Adjust width as needed
    
    st.title("Claim Denial Prediction App")
    
    # Input fields for user data
    age = st.number_input("Enter Age", min_value=0, max_value=100, value=25)
    
    # Sex selection with only label displayed
    sex_option = st.selectbox("Select Gender", options=["Male", "Female"])
    sex = 1 if sex_option == "Male" else 0
    
    bmi = st.number_input("Enter BMI", min_value=0.0, max_value=60.0, value=25.0)
    children = st.number_input("Enter Number of Policy dependents", min_value=0, max_value=10, value=0)
    
    # Smoker selection with only label displayed
    smoker_option = st.selectbox("Are you a smoker?", options=["Yes", "No"])
    smoker = 1 if smoker_option == "Yes" else 0
    
    region = st.selectbox("Select Region", options=["Northeast", "Northwest", "Southeast", "Southwest"])
    region_map = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
    region_value = region_map[region]
    
    charges = st.number_input("Enter Insurance Charges", min_value=0.0, max_value=100000.0)
    
    # When the 'Predict' button is clicked
    if st.button("Predict"):
        # Arrange the data in an input array for the model (same order as training)
        input_data = np.array([[age, sex, bmi, children, smoker, region_value, charges]])
        
        # Use the model to make a prediction
        prediction = model.predict(input_data)
        
        # Display the result message based on the prediction
        if prediction[0] == 1:
            st.success("Congratulations, Claim accepted")
        else:
            st.error("Sorry, Claim denied.")
    
# Run the main function (Streamlit handles server startup)
if __name__ == '__main__':
    main()
