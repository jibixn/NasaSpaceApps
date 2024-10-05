import pickle
import numpy as np
from flask import Flask, request, jsonify
import os

from flask_cors import CORS


# Load the model and scaler
with open("habitability_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)
CORS(app)


@app.route('/receiveData', methods=['POST'])
def receive_data():

    data = request.json
    distance_from_sun = float(data['distance'])
    mass = float(data['mass'])
    radius = float(data['radius'])
    orbital_period = float(data['orbitalPeriod'])
    stellar_mass = float(data['stellarMass'])
    stellar_radius = float(data['stellarRadius'])
    temperature = float(data['temp'])
    Etemperature = float(data['ETemp'])
    system_age = float(data['systemAge'])
    atmospheric_thickness = float(data['atmosphericthickness'])
    magnetic_field = int(data['magneticfield'])
    oxygen_percentage = float(data['oxygen'])
    co2_percentage = float(data['carbon'])
    nitrogen_percentage = float(data['nitrogen'])

    user_factors= {
    'Distance From Sun (AU)': distance_from_sun,
    'Planetary Mass (Earth Mass)': mass,
    'Planetary Radius (Earth Radius)': radius,
    'Planetary Orbital Period (Days)': orbital_period,
    'Stellar Mass (Solar Mass)': stellar_mass,
    'Stellar Radius (Solar Radius)': stellar_radius,
    'Stellar Effective Temperature (K)': temperature,
    'Planet Temperature (K)': Etemperature,
    'Planetary System Age (Billion Years)': system_age,
    'Atmospheric Thickness (%)': atmospheric_thickness,
    'Magnetic Field': magnetic_field,
    'Oxygen Percentage (%)': oxygen_percentage,
    'CO2 Percentage (%)': co2_percentage,
    'Nitrogen Percentage (%)': nitrogen_percentage                        
    }
    # Calculate habitability score based on the incoming parameters
    if(user_factors):
        score = predict_habitability(user_factors)
    return jsonify({'habitability_score': score})


# Function to get user input
# def get_user_input():
#     try:
#         distance_from_sun = float(input("Enter the distance from the sun (in AU): "))
#         mass = float(input("Enter the mass (relative to Earth's mass): "))
#         radius = float(input("Enter the radius (relative to Earth's radius): "))
#         orbital_period = float(input("Enter the orbital period (in days): "))
#         stellar_mass = float(input("Enter the stellar mass (relative to Solar mass): "))
#         stellar_radius = float(input("Enter the stellar radius (relative to Solar radius): "))
#         temperature = float(input("Enter the stellar effective temperature (in Kelvin): "))
#         Etemperature = float(input("Enter the surface temperature (in Kelvin): "))
#         system_age = float(input("Enter the planetary system age (in billion years): "))
#         atmospheric_thickness = float(input("Enter the atmospheric thickness (0-1 relative to Earth): "))
#         magnetic_field = int(input("Is there a magnetic field? (1 for Yes, 0 for No): "))
#         oxygen_percentage = float(input("Enter the oxygen percentage (in %): "))
#         co2_percentage = float(input("Enter the CO2 percentage (in %): "))
#         nitrogen_percentage = float(input("Enter the nitrogen percentage (in %): "))

#         # Return as dictionary
#         return {
#             'Distance From Sun (AU)': distance_from_sun,
#             'Planetary Mass (Earth Mass)': mass,
#             'Planetary Radius (Earth Radius)': radius,
#             'Planetary Orbital Period (Days)': orbital_period,
#             'Stellar Mass (Solar Mass)': stellar_mass,
#             'Stellar Radius (Solar Radius)': stellar_radius,
#             'Stellar Effective Temperature (K)': temperature,
#             'Planet Temperature (K)': Etemperature,
#             'Planetary System Age (Billion Years)': system_age,
#             'Atmospheric Thickness (%)': atmospheric_thickness,
#             'Magnetic Field': magnetic_field,
#             'Oxygen Percentage (%)': oxygen_percentage,
#             'CO2 Percentage (%)': co2_percentage,
#             'Nitrogen Percentage (%)': nitrogen_percentage                        
#         }
#     except ValueError as e:
#         print(f"Invalid input: {e}")
#         return None

# Function to calculate temperature suitability
def calculate_temperature_factor(temperature, co2_percentage, optimal_temp=288, temp_range=100, co2_optimal=0.04):
    co2_effect = 1 + (min(co2_percentage / 100, 0.5) - co2_optimal) * 0.05
    adjusted_temperature = temperature * co2_effect
    temp_suitability = max(0, 1 - abs(adjusted_temperature - optimal_temp) / temp_range)
    return temp_suitability

# Function to calculate the water body percentage
def calculate_water_body_percentage(distance_from_sun, temperature, atmospheric_thickness, magnetic_field):
    water_factor_distance = max(0.1, 1 / distance_from_sun)
    water_factor_temperature = max(0, 1 - abs(temperature - 288) / 100)
    water_factor_atmosphere = min(atmospheric_thickness, 1.5)  # Assuming atmospheric thickness is already scaled
    water_factor_magnetic = 1 if magnetic_field == 1 else 0.5
    water_body_percentage = water_factor_distance * water_factor_temperature * water_factor_atmosphere * water_factor_magnetic * 71
    return min(water_body_percentage, 100)

# Function to calculate vegetation percentage
def calculate_vegetation_percentage(water_body_percentage, oxygen_percentage, temperature_factor, co2_percentage):
    co2_optimal = 0.04
    co2_sensitivity = max(0, 1 - abs(co2_percentage / 100 - co2_optimal) * 5)
    vegetation_percentage = water_body_percentage * (oxygen_percentage / 21) * temperature_factor * co2_sensitivity
    return min(vegetation_percentage, 100)

# Function to calculate cloud cover percentage
def calculate_cloud_percentage(atmospheric_thickness, water_body_percentage, co2_percentage):
    co2_cloud_effect = min(max(0.7 + (co2_percentage / 100 - 0.04), 0), 1.1)  # Adjusted base and range
    cloud_percentage = min(atmospheric_thickness * max(water_body_percentage / 71, 0.1) * 1.2 * co2_cloud_effect, 100)  # Adjusted multiplier
    return cloud_percentage*100





# Function to handle predictions
def predict_habitability(new_factors):
    try:
        # Extracting factors
        distance_from_sun = new_factors['Distance From Sun (AU)']
        mass = new_factors['Planetary Mass (Earth Mass)']
        radius = new_factors['Planetary Radius (Earth Radius)']
        orbital_period = new_factors['Planetary Orbital Period (Days)']
        stellar_mass = new_factors['Stellar Mass (Solar Mass)']
        stellar_radius = new_factors['Stellar Radius (Solar Radius)']
        temperature = new_factors['Stellar Effective Temperature (K)']
        Ptemperature = new_factors['Planet Temperature (K)']
        system_age = new_factors['Planetary System Age (Billion Years)']
        atmospheric_thickness = new_factors['Atmospheric Thickness (%)']
        magnetic_field = new_factors['Magnetic Field']
        oxygen_percentage = new_factors['Oxygen Percentage (%)']
        co2_percentage = new_factors['CO2 Percentage (%)']
        nitrogen_percentage = new_factors['Nitrogen Percentage (%)']

        # Calculate additional features
        temp_factor = calculate_temperature_factor(Ptemperature, co2_percentage)
        water_body_percentage = calculate_water_body_percentage(distance_from_sun, Ptemperature, atmospheric_thickness, magnetic_field)
        vegetation_percentage = calculate_vegetation_percentage(water_body_percentage, oxygen_percentage, temp_factor, co2_percentage)
        cloud_percentage = calculate_cloud_percentage(atmospheric_thickness, water_body_percentage, co2_percentage)

        # Create input array for model
        factors_array = np.array([
            distance_from_sun,
            mass,
            radius,
            orbital_period,
            stellar_mass,
            stellar_radius,
            temperature,
            system_age,
            # Include calculated values if needed
        ])

        # Reshape for scaler
        factors_array_scaled = scaler.transform([factors_array])

        # Predict using the model
        prediction = model.predict(factors_array_scaled)

        # Print results
        print(f"\nResults:")
        print(f"Water Body Percentage (calculated): {water_body_percentage:.2f}%")
        print(f"Vegetation Percentage: {vegetation_percentage:.2f}%")
        print(f"Temperature Suitability Factor: {temp_factor:.2f}")
        print(f"Cloud Cover Percentage: {cloud_percentage:.2f}%")
        if(prediction[0]>100):
            prediction[0]=100
        

        print(f"Predicted Habitability Score: {prediction[0]:.2f}")

        return prediction[0]
    except Exception as e:
        print(f"Error in predict_habitability: {e}")
        return None

# Main logic to predict habitability
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    receive_data()

