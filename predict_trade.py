import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def predict_trade(trade_features):
    """
    Predict if a trade will be profitable using the trained ML model
    
    Args:
        trade_features: Dictionary with trade features at entry time
        
    Returns:
        Dictionary with predictions
    """
    
    # Load models and scaler
    try:
        classifier = joblib.load('best_classifier_model.pkl')
        regressor = joblib.load('best_regressor_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        feature_cols = pd.read_csv('feature_columns.csv', header=None)[0].tolist()
    except FileNotFoundError as e:
        return {'error': f'Model files not found: {e}'}
    
    # Prepare feature vector in exact order expected by model
    # Note: First feature '0' seems to be a placeholder, skip it
    feature_vector = []
    for col in feature_cols:
        if col == '0' or col == 0:
            # Skip placeholder feature
            continue
        if col in trade_features:
            val = trade_features[col]
            # Handle NaN
            if pd.isna(val):
                val = 0
            feature_vector.append(float(val))
        else:
            # Use default value if feature missing
            feature_vector.append(0.0)
    
    feature_array = np.array(feature_vector).reshape(1, -1)
    
    # Make predictions
    win_probability = classifier.predict_proba(feature_array)[0][1]
    predicted_win = classifier.predict(feature_array)[0]
    predicted_pnl = regressor.predict(feature_array)[0]
    
    return {
        'predicted_win': bool(predicted_win),
        'win_probability': float(win_probability),
        'predicted_pnl': float(predicted_pnl),
        'confidence': 'High' if win_probability > 0.7 else 'Medium' if win_probability > 0.5 else 'Low',
        'recommendation': 'TAKE TRADE' if predicted_win and win_probability > 0.6 else 'AVOID TRADE'
    }

def create_trade_features_from_entry(entry_data):
    """
    Create feature vector from entry data
    
    Args:
        entry_data: Dictionary with entry information
        
    Returns:
        Dictionary with features ready for prediction
    """
    
    features = {}
    
    # Basic trade info
    features['Risk'] = entry_data.get('Risk', 0)
    # WindowType encoding: 300=0, 1000=1, 1630=2
    window_type = str(entry_data.get('WindowType', '0'))
    if window_type in ['300', '0300']:
        features['WindowType'] = 0
    elif window_type == '1000':
        features['WindowType'] = 1
    elif window_type == '1630':
        features['WindowType'] = 2
    else:
        features['WindowType'] = 0
    
    # Type encoding (BUY=1, SELL=0)
    features['Type'] = 1 if entry_data.get('Type') == 'BUY' else 0
    
    # EMA values
    features['EMA_9_5M'] = entry_data.get('EMA_9_5M', 0)
    features['EMA_21_5M'] = entry_data.get('EMA_21_5M', 0)
    features['EMA_50_5M'] = entry_data.get('EMA_50_5M', 0)
    features['EMA_200_1H'] = entry_data.get('EMA_200_1H', 0)
    
    # ATR indicators
    features['ATR'] = entry_data.get('ATR', 0)
    features['ATR_Pct'] = entry_data.get('ATR_Pct', 0)
    features['ATR_Ratio'] = entry_data.get('ATR_Ratio', 0)
    
    # Consolidation
    features['Is_Consolidating'] = entry_data.get('Is_Consolidating', 0)
    features['Is_Tight_Range'] = entry_data.get('Is_Tight_Range', 0)
    features['Consolidation_Score'] = entry_data.get('Consolidation_Score', 0)
    
    # Trend indicators
    features['Trend_Score'] = entry_data.get('Trend_Score', 0)
    features['EMA_9_Above_21'] = entry_data.get('EMA_9_Above_21', 0)
    features['EMA_21_Above_50'] = entry_data.get('EMA_21_Above_50', 0)
    features['Price_Above_EMA200_1H'] = entry_data.get('Price_Above_EMA200_1H', 0)
    
    # Range info
    features['RangeSize'] = entry_data.get('RangeSize', 0)
    features['RangeSizePct'] = entry_data.get('RangeSizePct', 0)
    
    # Time features
    entry_time = entry_data.get('EntryTime', datetime.now())
    if isinstance(entry_time, str):
        entry_time = pd.to_datetime(entry_time)
    elif isinstance(entry_time, datetime):
        entry_time = pd.Timestamp(entry_time)
    features['EntryHour'] = entry_time.hour
    features['EntryDayOfWeek'] = entry_time.weekday()
    
    return features

if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("TRADE PREDICTION EXAMPLE")
    print("="*60)
    
    # Example trade entry data
    example_entry = {
        'Type': 'BUY',
        'Risk': 5.0,
        'WindowType': '1630',
        'EMA_9_5M': 3200.0,
        'EMA_21_5M': 3195.0,
        'EMA_50_5M': 3190.0,
        'EMA_200_1H': 3180.0,
        'ATR': 3.0,
        'ATR_Pct': 0.09,
        'ATR_Ratio': 1.1,
        'Is_Consolidating': 0,
        'Is_Tight_Range': 0,
        'Consolidation_Score': 0.2,
        'Trend_Score': 0.8,
        'EMA_9_Above_21': 1,
        'EMA_21_Above_50': 1,
        'Price_Above_EMA200_1H': 1,
        'RangeSize': 5.0,
        'RangeSizePct': 0.15,
        'EntryTime': datetime.now()
    }
    
    # Create features
    features = create_trade_features_from_entry(example_entry)
    
    # Make prediction
    prediction = predict_trade(features)
    
    if 'error' not in prediction:
        print(f"\nPrediction Results:")
        print(f"  Predicted Win: {prediction['predicted_win']}")
        print(f"  Win Probability: {prediction['win_probability']:.2%}")
        print(f"  Predicted P&L: ${prediction['predicted_pnl']:.2f}")
        print(f"  Confidence: {prediction['confidence']}")
        print(f"  Recommendation: {prediction['recommendation']}")
    else:
        print(f"Error: {prediction['error']}")

