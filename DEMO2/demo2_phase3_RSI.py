import numpy as np
import pandas as pd

# 一度日足でデータを読み込む
df = pd.read_csv(
    "/Users/kondousatoshishi/Downloads/金融モデル/data/4063.T_day.csv",
    skiprows=2,
    names=["Date", "Close", "High", "Low", "Open", "Volume"],
    parse_dates=["Date"],
    dayfirst=False,
    date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d", errors="coerce")
)

# Date列を datetime に変換し、欠損行を削除
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
df = df.dropna(subset=["Date"])

# インデックスをDateに設定
df = df.set_index("Date").sort_index()

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

# 月末RSI付きのデータを CSV に保存
df_month_end.to_csv("/Users/kondousatoshishi/Downloads/金融モデル/data/4063.T_month_end_with_RSI.csv")