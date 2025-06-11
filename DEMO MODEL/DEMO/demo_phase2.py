## phase2 データ収集と前処理

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

os.makedirs("data", exist_ok=True)

tickers = [
    "7203.T", "9984.T", "8306.T", "6758.T", "9432.T",
    "4063.T", "6981.T", "7733.T", "8058.T", "6861.T"
]

# データ収集
start_date = "2013-06-01"
end_date = datetime.today().strftime("%Y-%m-%d")

# 月足データを取得して保存
for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date, interval="1mo")
    df.dropna(inplace=True) # 欠損値の削除
    df.to_csv(f"data/{ticker}_monthly.csv")
    print(f"Saved: {ticker}")
