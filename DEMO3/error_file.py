## phase2　データ収集と前処理
### このファイルでは、10このファイルを全て処理しているため、異なるファイルとの境目がわからなくなっている

import yfinance as yf

import numpy as np
import pandas as pd
from datetime import datetime 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import os
import glob

os.makedirs("data", exist_ok=True)

# ファイルは取得済
all_files = glob.glob("data/*_day.csv")
df_list = []
for file in all_files:
    temp_df = pd.read_csv(file, header=0, index_col=0)
    temp_df.index = pd.to_datetime(temp_df.index, format="%Y-%m-%d", errors="coerce")
    temp_df.index.name = "Date"
    df_list.append(temp_df)
df = pd.concat(df_list, axis=0)
print(df.head())

# Close列を float に変換
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# RSIを計算する関数
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

# 日足RSI(14)計算
df["RSI_daily"] = compute_rsi(df["Close"], period=14)
print(df.iloc[:15])

 # 月末RSIと対応する実際の営業日を取得する
rsi_series = df["RSI_daily"].resample("M").last()
 # 期間ごとに最後のインデックス（営業日）を取得
month_last_trading = df["RSI_daily"].resample("ME").apply(lambda x: x.index[-1])

 # RSI_month_end 列を NaN で初期化
df["RSI_month_end"] = np.nan
 # 各月の最後の営業日に対応する RSI を代入
for dt, val in zip(month_last_trading.values, rsi_series.values):
     df.at[dt, "RSI_month_end"] = val

# 必要に応じて月末データだけ抽出する場合
df_month_end = df[df["RSI_month_end"].notna()]
df_month_end = df_month_end.drop(columns=["RSI_daily"])

print(df_month_end.iloc[:32])

# momentum
df_month_end["momentum_3m"] = (df_month_end["Close"].pct_change(periods=3))
data = df_month_end[["momentum_3m", "Close"]].dropna()
data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

X = data["momentum_3m"]
y = data["target"]

print(df_month_end)

# ④ 新しいCSVファイルとして保存（元ファイルを上書きしないよう別名で保存）
output_path = "/Users/kondousatoshishi/Downloads/金融モデル/data/4063.T_Demo3.csv"
df_month_end.to_csv(output_path)