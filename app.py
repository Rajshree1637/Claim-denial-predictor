from flask import Flask, request # type: ignore
import pickle
import numpy as np # type: ignore

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
with open('XGBClassifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the prediction route
@app.route('/predict', methods=['GET'])
def predict():
    # Get input data from the query parameters
    age = int(request.args.get('age'))
    sex = int(request.args.get('sex'))
    bmi = float(request.args.get('bmi'))
    children = int(request.args.get('children'))
    smoker = int(request.args.get('smoker'))
    region = int(request.args.get('region'))
    charges = float(request.args.get('charges'))

    # Arrange the data as an input array for the model (same order as training)
    input_data = np.array([[age, sex, bmi, children, smoker, region, charges]])

    # Use the model to make a prediction
    prediction = model.predict(input_data)

    # Determine the result message based on the prediction
    if prediction[0] == 1:
        result_message = 'Congrats! Your insurance is claimed.'
    else:
        result_message = 'Sorry! Your insurance is not claimed.'

    # Return the result message as plain text
    return result_message

@app.route('/hello',methods=['GET'])
def say_hello():
    return "hello"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
