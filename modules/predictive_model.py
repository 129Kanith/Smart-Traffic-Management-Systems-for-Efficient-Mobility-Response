import numpy as np
import pandas as pd

def predict_future_traffic(session_data_df):
    """
    Predicts future traffic congestion based on recent historical data.
    This is a foundational framework where an LSTM or Random Forest model 
    can be integrated for advanced forecasting.
    
    Args:
        session_data_df (pd.DataFrame): DataFrame containing recent session history.
        
    Returns:
        dict: Prediction results including vehicle count, density, and trend.
    """
    if session_data_df is None or len(session_data_df) < 15:
        return {
            "predicted_vehicles": "Gathering...", 
            "predicted_density": "N/A", 
            "trend": "N/A"
        }
        
    # Get the last 15 records
    recent_counts = session_data_df["Vehicles"].tail(15).values
    
    # Calculate simple moving average and trend
    avg_count = np.mean(recent_counts)
    trend_val = recent_counts[-1] - recent_counts[0]
    
    # Linear projection + small variance for realism
    predicted_count = int(avg_count + (trend_val * 0.5) + np.random.randint(-1, 2))
    predicted_count = max(0, predicted_count)
    
    # Determine predicted density
    if predicted_count <= 5:
        density = "LOW"
    elif predicted_count <= 15:
        density = "MEDIUM"
    else:
        density = "HIGH"
        
    # Determine trend label
    if trend_val > 2:
        trend_str = "⬆️ Increasing"
    elif trend_val < -2:
        trend_str = "⬇️ Decreasing"
    else:
        trend_str = "➡️ Stable"
        
    return {
        "predicted_vehicles": predicted_count,
        "predicted_density": density,
        "trend": trend_str
    }
