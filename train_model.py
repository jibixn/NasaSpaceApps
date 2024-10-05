import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer, RobustScaler
import pickle

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Define features and target variable
    feature_cols = [
        'Distance From Sun (AU)', 
        'Planetary Mass (Earth Mass)', 
        'Planetary Radius (Earth Radius)', 
        'Planetary Orbital Period (Days)', 
        'Stellar Mass (Solar Mass)', 
        'Stellar Radius (Solar Radius)', 
        'Stellar Effective Temperature (K)', 
        'Planetary System Age (Billion Years)'
    ]
    
    # Separate features and target
    X = df[feature_cols]
    y = df['Habitability Score']

    # Scale features using RobustScaler or any preferred method
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# Specify your CSV file path here
file_path = 'planet_data.csv'

# Load and preprocess the data
X_scaled, y, scaler = load_and_preprocess_data(file_path)

# Transform the target variable
pt = PowerTransformer(method='yeo-johnson')
y_transformed = pt.fit_transform(y.values.reshape(-1, 1)).ravel()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_transformed, test_size=0.2, random_state=42)

# Define the model
gb = GradientBoostingRegressor(random_state=42)

# Set the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=gb, param_distributions=param_dist, 
                                   n_iter=20, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1, random_state=42)

# Fit the model
random_search.fit(X_train, y_train)

# Best model
best_gb = random_search.best_estimator_

# Predictions (transform back to original scale)
y_pred_transformed = best_gb.predict(X_test)
y_pred = pt.inverse_transform(y_pred_transformed.reshape(-1, 1)).ravel()
y_test_original = pt.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Evaluate the model
mse = mean_squared_error(y_test_original, y_pred)
r_squared = r2_score(y_test_original, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r_squared)
print("Best Parameters:", random_search.best_params_)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': [
        'Distance From Sun (AU)', 
        'Planetary Mass (Earth Mass)', 
        'Planetary Radius (Earth Radius)', 
        'Planetary Orbital Period (Days)', 
        'Stellar Mass (Solar Mass)', 
        'Stellar Radius (Solar Radius)', 
        'Stellar Effective Temperature (K)', 
        'Planetary System Age (Billion Years)'
    ],
    'importance': best_gb.feature_importances_
})

feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Save the trained GradientBoostingRegressor model
with open('habitability_model.pkl', 'wb') as model_file:
    pickle.dump(best_gb, model_file)

# Save the scaler for prediction later
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")
