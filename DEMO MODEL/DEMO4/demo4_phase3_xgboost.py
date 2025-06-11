import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

import os
import glob

os.makedirs("data", exist_ok=True)

all_files = glob.glob("/Users/kondousatoshishi/Downloads/金融モデル/DEMO4/data4/day_feature/*_demo4.csv")
for file in all_files:
    df =pd.read_csv(file, parse_dates=["Date"], index_col="Date")

    # 目的変数 target を作成: 3 クラス分類 (±1% の5日リターン)
    future_ret_pct = (df["Close"].shift(-5) - df["Close"]) / df["Close"]
    threshold = 0.01  # ±1% を閾値とする
    df["target"] = np.select(
        [future_ret_pct > threshold,   # 上昇 (+1%以上)
         future_ret_pct < -threshold], # 下落 (-1%以下)
        [0, 2],
        default=1                      # 横ばい (±1%未満)
    )
    df = df.dropna(subset=["target"])
        
    # 説明変数　X, 目的変数 y　を準備
    df_feat = df[["momentum_3d", "RSI_daily", "Weekly_volatility", "range_ratio", "target"]].dropna()
    X = df_feat[["momentum_3d", "RSI_daily", "Weekly_volatility", "range_ratio"]]
    y = df_feat["target"]
    
    print(len(df))
    tscv = TimeSeriesSplit(n_splits=5)
    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="mlogloss")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"=== Fold {fold} Evaluation ===")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
    # 結果をCSVで保存
    result_df = pd.DataFrame({
        'Date': X_test.index,
        'momentum_3d': X_test['momentum_3d'],
        'RSI_daily': X_test['RSI_daily'],
        'Weekly_volatility': X_test['Weekly_volatility'],
        'range_ratio': X_test['range_ratio'],
        'actual': y_test.values,
        'predicted': y_pred
    })
    ticker = os.path.basename(file).split("_")[0]
    out_file_path = f"/Users/kondousatoshishi/Downloads/金融モデル/DEMO4/data4/day_result/{ticker}_results.csv"
    result_df.to_csv(out_file_path, index=False)
    print(f"Results saved to {out_file_path}")
