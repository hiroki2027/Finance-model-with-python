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

# ファイルは取得済
all_files = glob.glob("data/*_day.csv")
df_list = []
for file in all_files:
    temp_df = pd.read_csv(file, header=0, index_col=0)
    temp_df.index = pd.to_datetime(temp_df.index, format="%Y-%m-%d", errors="coerce")
    df_list.append(temp_df)
df = pd.concat(df_list, axis=0).sort_index()

# DataFrame の行・列数
print("Shape:", df.shape)

# インデックスの範囲と型
print("Index:", df.index.dtype, "| From", df.index.min(), "to", df.index.max())

# カラム一覧と型
print("\nColumns and dtypes:")
print(df.dtypes)

# 先頭 5 行
print("\nHead:")
print(df.head())


# (必要に応じて) 統計要約
print("\nDescribe:")
print(df.describe())

