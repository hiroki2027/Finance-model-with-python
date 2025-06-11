### 特徴量エンジニアリング ###
#### 日足からモメンタム、RSIを考える ###

import numpy as np
import pandas as pd
from datetime import datetime

import os
import glob

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

os.makedirs('data', exist_ok=True)

all_files = glob.glob("DEMO4/data4/day_data/*_day.csv")

    # RSIの計算関数
def compute_rsi(series, period=14):
     delta = series.diff()
     gain = delta.clip(lower=0)
     loss = -delta.clip(upper=0)

     avg_gain = gain.rolling(window=period, min_periods=period).mean()
     avg_loss = loss.rolling(window=period, min_periods=period).mean()

     RS = avg_gain / avg_loss
     RSI = 100 - (100 / (1 + RS))
     return RSI
 
for file in all_files:
    temp_df = pd.read_csv(file, header=0, index_col=0)
    print((temp_df.head()))
    temp_df.index = pd.to_datetime(temp_df.index, format="%Y-%m-%d", errors="coerce")
    temp_df.index.name = "Date"
    print(temp_df)
    
    # Close列を float に変換
    temp_df["Close"] = pd.to_numeric(temp_df["Close"], errors="coerce")
    
    # Open, High, Low 列も float に変換（計算エラー防止）
    for col in ["Open", "High", "Low"]:
        temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce")
    
    # 日足RSI を計算
    temp_df["RSI_daily"] = compute_rsi(temp_df["Close"], period=14)
    
    # momentum_3d
    temp_df["momentum_3d"] = (temp_df["Close"].pct_change(periods=3))
    
    # 週次ボラティリティ
    ## 対数リターンを計算
    temp_df["log_ret"] = np.log(temp_df["Close"] / temp_df["Close"].shift(1))
    ## 週次にリサンプリングして、週内の標準偏差をとる
    Weekly_vol = temp_df["log_ret"].resample("W").std()
    temp_df["Weekly_volatility"] = Weekly_vol.reindex(temp_df.index, method='ffill')
    
    # 週次高安レンジ比率
    Weekly = temp_df[["Open", "High", "Low", "Close"]].resample("W").agg({
        "Open": "first",
        "Close": "last",
        "Low": "min",
        "High": "max"
    })
    print(Weekly.dtypes)
    Weekly["range_ratio"] = (Weekly["High"] - Weekly["Low"]) / Weekly["Open"]

    # 日次に週次特徴量をマージ（forward fill）
    Weekly = Weekly[["range_ratio"]].reindex(temp_df.index, method='ffill')
    temp_df["range_ratio"] = Weekly["range_ratio"]

    # Weekly_volatiltiy → Weekly_volatility に修正
    temp_df.rename(columns={"Weekly_volatiltiy": "Weekly_volatility"}, inplace=True)
    print(temp_df.head())
    
    # CSVファイルとして保存
    temp_df = temp_df.iloc[7:]
    ticker = os.path.basename(file).split("_")[0]
    output_paht = f"/Users/kondousatoshishi/Downloads/金融モデル/DEMO4/data4/day_feature/{ticker}_demo4.csv"
    temp_df.to_csv(output_paht)
    
    print(f"Saved: {ticker}_demo4.csv")
    
    