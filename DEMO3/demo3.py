## phase2　データ収集と前処理

import yfinance as yf

import numpy as np
import pandas as pd
from datetime import datetime 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import os
import glob

os.makedirs("data", exist_ok=True)

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

# ファイルは取得済
all_files = glob.glob("data/*_day.csv")
for file in all_files:
    temp_df = pd.read_csv(file, header=0, index_col=0)
    print(temp_df.head())
    temp_df.index = pd.to_datetime(temp_df.index, format="%Y-%m-%d", errors="coerce")
    temp_df.index.name = "Date"
    print(temp_df)
    
    # Close列sを float　に変換
    temp_df["Close"] = pd.to_numeric(temp_df["Close"], errors="coerce")
    
    # 日足RSI(14)を計算
    temp_df["RSI_daily"] = compute_rsi(temp_df["Close"], period=14)
    #?? 月足RSIと対応する実際の営業日を取得
    rsi_series = temp_df["RSI_daily"].resample("ME").last()
    # 期間ごとに最後のインデックス（営業日）を取得
    month_last_trading = temp_df["RSI_daily"].resample("ME").apply(lambda x: x.index[-1])
    
    # RSI_month_end 列をNaNで初期化
    temp_df["RSI_month_end"] = np.nan
    # 各月の最後の営業日に対応する RSI を代入
    for dt, val in zip(month_last_trading.values, rsi_series.values):
        temp_df.at[dt, "RSI_month_end"] = val
        
    # 必要に応じて月末データだけ抽出する場合
    df_month_end = temp_df[temp_df["RSI_month_end"].notna()]
    df_month_end = df_month_end.drop(columns=["RSI_daily"])
    print(df_month_end)
 
    # momentum
    df_month_end["momentum_3m"] = (df_month_end["Close"].pct_change(periods=3))
    data = df_month_end[["momentum_3m", "Close"]].dropna()
    data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    
    X = data["momentum_3m"]
    # data["target"]はdf_month_endに追加する必要はない
    y = data["target"]
    
    print(df_month_end)
    # ④ 新しいCSVファイルとして保存（元ファイルを上書きしないよう別名で保存）
    ticker = os.path.basename(file).split("_")[0]
    output_path = f"/Users/kondousatoshishi/Downloads/金融モデル/data_demo3/{ticker}_demo3.csv"
    df_month_end.to_csv(output_path)
    
    print(f"Saved: {ticker}_Demo3.csv")