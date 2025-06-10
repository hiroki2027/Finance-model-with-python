## データ収集
import yfinance as yf

import numpy as np
import pandas as pd
from datetime import datetime

import os
import glob

os.makedirs("data4/day_data", exist_ok=True)

tickers = {
    "7203.T", "9984.T", "8306.T", "6758.T", "9432.T",
    "4063.T", "6981.T", "7733.T", "8058.T", "6861.T"
}

# データ収集
start_data = "2013-06-01"
end_data = datetime.today().strftime("%Y-%m-%d")

# 日足データを取得して保存
for ticker in tickers:
    df = yf.download(ticker, start=start_data, end=end_data)
    df.dropna(inplace=True)
    df.to_csv(f"data4/day_data/{ticker}_day.csv")
    print(f"Saves: {ticker}")
    
all_files = glob.glob("data4/day_data/*_day.csv")
