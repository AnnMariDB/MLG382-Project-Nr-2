# 4.  predict.py

import pandas as pd
import joblib

def predict_new(model, scaler, new_df, output_path="predicted_clusters.csv"):
    """
    Predict clusters for new RFM data using a trained model and scaler.

    Parameters:
        model: Trained clustering model (e.g., KMeans).
        scaler: Fitted scaler object (e.g., StandardScaler).
        new_df (pd.DataFrame): DataFrame containing 'Recency', 'Frequency', 'Monetary'.
        output_path (str): File path to save predictions. Default is 'predicted_clusters.csv'.

    Returns:
        pd.DataFrame: The original DataFrame with an added 'Cluster' column.
    """
    required_features = ['Recency', 'Frequency', 'Monetary']
    missing = [col for col in required_features if col not in new_df.columns]
    
    if missing:
        raise ValueError(f"[ERROR] Missing required features: {missing}")

    # Scale the features
    new_scaled = scaler.transform(new_df[required_features])

    # Predict and assign clusters
    new_df['Cluster'] = model.predict(new_scaled)

    # Save predictions
    new_df.to_csv(output_path, index=False)
    print(f"[INFO] Predictions saved to: {output_path}")

    return new_df


