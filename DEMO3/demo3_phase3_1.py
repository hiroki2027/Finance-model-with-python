## phase2　データ収集と前処理
##### 修正済み #####

import yfinance as yf

import numpy as np
import pandas as pd
from datetime import datetime 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import os
import glob

os.makedirs("data", exist_ok=True)

tickers = {
    "7203.T", "9984.T", "8306.T", "6758.T", "9432.T",
    "4063.T", "6981.T", "7733.T", "8058.T", "6861.T"
}

# データ収集
start_date = "2013-06-01"
end_date = datetime.today().strftime("%Y-%m-%d")

# 日足データを取得して保存
for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    df.dropna(inplace=True)
    df.to_csv(f"data/{ticker}_day.csv")
    print(f"Saved: {ticker}")

###### これはいらない ######
all_files = glob.glob("data/*_day.csv")
df_list = []
for file in all_files:
    temp_df = pd.read_csv(file, header=0, index_col=0)
    temp_df.index = pd.to_datetime(temp_df.index, format="%Y-%m-%d", errors="coerce")
    df_list.append(temp_df)
df = pd.concat(df_list, axis=0).sort_index()
###### これはいらない ######


###### 修正済み ######
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

