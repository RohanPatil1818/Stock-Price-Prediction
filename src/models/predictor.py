
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_absolute_error

FEATURES = ["return_1h", "rolling_sentiment_3h", "vol_spike", "hourly_sentiment_score"]

def prepare_modeling_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["next_close"] = df["Close"].shift(-1)
    df["next_return"] = (df["next_close"] / df["Close"]) - 1.0
    df["target_direction"] = (df["next_close"] > df["Close"]).astype(int)
    return df.dropna()

def train_and_evaluate(df: pd.DataFrame, ticker: str = ""):
    X = df[FEATURES]
    y_dir = df["target_direction"]
    y_ret = df["next_return"]

    tscv = TimeSeriesSplit(n_splits=3)
    accuracies = []
    maes = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_dir_train, y_dir_test = y_dir.iloc[train_idx], y_dir.iloc[test_idx]
        y_ret_train, y_ret_test = y_ret.iloc[train_idx], y_ret.iloc[test_idx]

        clf = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1)
        clf.fit(X_train, y_dir_train)
        accuracies.append(accuracy_score(y_dir_test, clf.predict(X_test)))

        reg = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1)
        reg.fit(X_train, y_ret_train)
        maes.append(mean_absolute_error(y_ret_test, reg.predict(X_test)))

    avg_acc = float(np.mean(accuracies))
    avg_mae = float(np.mean(maes))

    final_clf = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1)
    final_clf.fit(X, y_dir)

    final_reg = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1)
    final_reg.fit(X, y_ret)

    return final_clf, final_reg, FEATURES, avg_acc, avg_mae