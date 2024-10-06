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
    
    data = pd.read_csv(file_path)

    
    print("Columns in DataFrame:", data.columns.tolist())

    
    if data.isnull().any().any():
        print("Missing values found. Filling missing values.")
        data.fillna(data.mean(), inplace=True)  

    
    print("Data types in DataFrame:")
    print(data.dtypes)

    
    columns_to_drop = ["Habitability_Score", "Planet"]
    for col in columns_to_drop:
        if col not in data.columns:
            print(f"Warning: {col} not found in DataFrame")

    
    X = data.drop(columns=columns_to_drop, errors='ignore')  
    y = data["Habitability_Score"]  

    
    X['Temp_Distance_Interaction'] = X['Temperature (K)'] * X['Distance_From_Sun (AU)']
    X['Temp_Mass_Interaction'] = X['Temperature (K)'] * X['Mass (Earth Mass)']

    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
