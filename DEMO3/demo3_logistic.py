import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import os
import glob

os.makedirs("data", exist_ok=True)
all_files = glob.glob("/Users/kondousatoshishi/Downloads/金融モデル/data_demo3/*_demo3.csv")
for file in all_files :
    df = pd.read_csv(file, parse_dates=["Date"], index_col="Date")
    print(df.head())
    
    # 目的変数　target を作成: 翌月終値が上がるか
    df["target"] = (df["Close"].shift(-1) > df["Close"].astype(int))
    
    # 最後の行は target が Nan になるので削除
    df = df.dropna(subset=["target"])
    
    # 説明変数 X, 目的変数 y を準備
    # 欠損値(NaN)のある行を除外
    df_feat = df[["momentum_3m", "RSI_month_end", "target"]].dropna()
    X = df_feat[["momentum_3m", "RSI_month_end"]]
    y = df_feat["target"]
    
    X_train, X_test = X.iloc[:-10], X.iloc[-10:]
    y_train, y_test = y.iloc[:-10], y.iloc[-10:]
    
    # ロジスティック回帰モデルを定義して学習
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    
    # テストセットで予測
    y_pred = model.predict(X_test)
    
    # 精度評価を出力
    
    print("=== モデル評価（最終10か月テスト） ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 結果をCSVで保存
    result_df = pd.DataFrame({
        'Date': X_test.index,
        'momentum_3m': X_test['momentum_3m'],
        'RSI_month_end': X_test['RSI_month_end'],
        'actual': y_test.values,
        'predicted': y_pred
    })
    out_file = file.replace('_demo3.csv', '_results.csv')
    result_df.to_csv(out_file, index=False)
    print(f"Results saved to {out_file}")
    
 