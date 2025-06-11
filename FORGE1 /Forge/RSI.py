import numpy as np
import pandas as pd

def compute_rsi(series, period=14):
    delta = series.diff()

    # .clip: pandas/Numpy 共通の範囲切り取り関数-指定境界より小さいor大きい値をその境界値に痴漢
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing using EMA (指数荷重移動) (alpha = 1/period)
    # RSiの定義的には"Wilderの指数平均"が妥当
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename(f"RSI{period}")

def rsi_signal(rsi: pd.Series,
               lower: float = 30,
               upper: float = 70,
               neutral: int = 0) :
    """
    +1  : RSI < lower  → 買い（売られ過ぎ）
    -1  : RSI > upper  → 売り（買われ過ぎ）
     0  : 中立
    """
    sig = pd.Series(neutral, index=rsi.index, dtype=np.int8)
    sig[rsi < lower]  = 1
    sig[rsi > upper]  = -1
    return sig.rename("RSI_sig")

daily = pd.read_parquet("data/raw/7203.T.parquet").set_index("Date")["Adj Close"]

# 週次 RSI(14) → 週末値
rsi14_daily = compute_rsi(daily, 14)
weekly_rsi  = rsi14_daily.resample("W-FRI").last()

# シグナル化
weekly_sig = rsi_signal(weekly_rsi, lower=30, upper=70)

# 月次保有ロジックへ統合
signals = pd.concat([weekly_sig], axis=1)   # 他シグナルも横に連結