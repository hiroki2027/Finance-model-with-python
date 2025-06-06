import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

# 1. combined CSV (RSI_month_end + momentum_3m が含まれている)を読み込む
file_path = "/Users/kondousatoshishi/Downloads/金融モデル/data/4063.T_month_end_with_RSI_momentum.csv"
df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

print(df.head())

# 2. 目的変数　taget を作成: 翌月の終値が上がるかどうか
df["target"] = (df["Close"].shift(-1) > df["Close"].astype(int))

# 3. 最後の行は target がNaNになるので削除
df = df.dropna(subset=["target"])

# 4. 説明変数 X, 目的変数 y　を準備
# 欠損値（NaN）のある行を除外
df_feat = df[["momentum_3m", "RSI_month_end", "target"]].dropna()
X = df_feat[["momentum_3m", "RSI_month_end"]]
y = df_feat["target"]

# 5. 学習用とテスト用に分割（最後の 10件をテスト用として残す)
# train_test_spilt はランダムシャッフルで分割するため不適
X_train, X_test = X.iloc[:-10], X.iloc[-10:]
y_train, y_test = y.iloc[:-10], y.iloc[-10:]

# 6. ロジスティック回帰モデルを定義して学習
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# 7.テストセットで予測
y_pred = model.predict(X_test)

# 8. 精度評価を出力
print("=== モデル評価（最終10か月テスト） ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


