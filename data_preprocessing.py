import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset for training a model.

    Parameters:
    - file_path: str - Path to the CSV file containing the data.

    Returns:
    - X_scaled: numpy.ndarray - Scaled feature values.
    - y: pandas.Series - Target variable values.
    - scaler: MinMaxScaler - Fitted scaler object for future use.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Print the columns to debug
    print("Columns in DataFrame:", data.columns.tolist())

    # Check for missing values
    if data.isnull().any().any():
        print("Missing values found. Filling missing values.")
        data.fillna(data.mean(), inplace=True)  # Example strategy

    # Check data types
    print("Data types in DataFrame:")
    print(data.dtypes)

    # Column names to drop
    columns_to_drop = ["Habitability_Score", "Planet"]
    for col in columns_to_drop:
        if col not in data.columns:
            print(f"Warning: {col} not found in DataFrame")

    # Drop 'Planet' and target column
    X = data.drop(columns=columns_to_drop, errors='ignore')  # Use errors='ignore' for safety
    y = data["Habitability_Score"]  # This is your target variable

    # Create interaction terms
    X['Temp_Distance_Interaction'] = X['Temperature (K)'] * X['Distance_From_Sun (AU)']
    X['Temp_Mass_Interaction'] = X['Temperature (K)'] * X['Mass (Earth Mass)']

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
