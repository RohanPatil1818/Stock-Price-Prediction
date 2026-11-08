
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

def prepare_modeling_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Create the Target Variable (n+1 Direction)
    # Shift(-1) pulls the NEXT hour's close price up to the current row
    df['next_close'] = df['Close'].shift(-1)
    
    # Target is 1 if the price goes UP next hour, 0 if it goes DOWN
    df['target_direction'] = (df['next_close'] > df['Close']).astype(int)
    
    # 2. Drop the last row because its 'next_close' will be NaN (the future hasn't happened yet)
    df = df.dropna()
    return df

def train_and_evaluate(df: pd.DataFrame, ticker: str):
    # The features we engineered in Day 3
    features = ['return_1h', 'rolling_sentiment_3h', 'vol_spike', 'hourly_sentiment_score']
    
    X = df[features]
    y = df['target_direction']
    
    # Walk-Forward Validation: Trains on chronological blocks, testing on the next block
    tscv = TimeSeriesSplit(n_splits=3) 
    
    fold = 1
    accuracies = []
    
    print("\nCross-Validation Results:")
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Initialize LightGBM Classifier
        model = lgb.LGBMClassifier(
            n_estimators=50, 
            max_depth=3, 
            learning_rate=0.05, 
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate out-of-sample
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        print(f"  Fold {fold} Accuracy: {acc:.2f}")
        fold += 1
        
    avg_accuracy = np.mean(accuracies)
    print(f"Average Out-of-Sample Accuracy: {avg_accuracy:.2%}")
    
    # Train the final production model on ALL historical data
    final_model = lgb.LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbose=-1)
    final_model.fit(X, y)
    
    return final_model, features