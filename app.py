from flask import Flask, request, jsonify
import pickle
import numpy as np
from predict import predict_habitability

app = Flask(__name__)

# Load the model and scaler
with open("habitability_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/predict', methods=['POST'])
def predict_habitability_route():
    try:
        # Get data from request
        data = request.json['factors']

        # Log received data
        print("Received data:", data)

        # Validate input data length
        if len(data) != 10:
            return jsonify({"error": "Invalid number of factors. Expected 10."}), 400

        # Create a dictionary for the factors
        new_factors = {
            'distance_from_sun': data[0],
            'mass': data[1],
            'radius': data[2],
            'orbital_period': data[3],
            'temperature': data[4],
            'atmospheric_thickness': data[5],
            'magnetic_field': data[6],
            'oxygen_percentage': data[7],
            'co2_percentage': data[8],
            'nitrogen_percentage': data[9],
            'Temp_Distance_Interaction': data[10],
            'Temp_Mass_Interaction': data[11]
        }

        # Call the prediction function
        prediction = predict_habitability(new_factors)

        # Return prediction as JSON
        return jsonify({"habitability_percentage": prediction}), 200

    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({"error": str(e)}), 500

def get_user_input():
    try:
        distance_from_sun = float(input("Enter the distance from the sun (in AU): "))
        mass = float(input("Enter the mass (relative to Earth's mass): "))
        radius = float(input("Enter the radius (relative to Earth's radius): "))
        orbital_period = float(input("Enter the orbital period (in days): "))
        temperature = float(input("Enter the average temperature (in Kelvin): "))
        atmospheric_thickness = float(input("Enter the atmospheric thickness (in % relative to Earth): "))
        magnetic_field = int(input("Is there a magnetic field? (1 for Yes, 0 for No): "))
        oxygen_percentage = float(input("Enter the oxygen percentage (in %): "))
        co2_percentage = float(input("Enter the CO2 percentage (in %): "))
        nitrogen_percentage = float(input("Enter the nitrogen percentage (in %): "))

        # Return as dictionary
        return {
            'distance_from_sun': distance_from_sun,
            'mass': mass,
            'radius': radius,
            'orbital_period': orbital_period,
            'temperature': temperature,
            'atmospheric_thickness': atmospheric_thickness,
            'magnetic_field': magnetic_field,
            'oxygen_percentage': oxygen_percentage,
            'co2_percentage': co2_percentage,
            'nitrogen_percentage': nitrogen_percentage                        
        }
    except ValueError as e:
        print(f"Invalid input: {e}")
        return None
if __name__ == '__main__':
    app.run(debug=True)
