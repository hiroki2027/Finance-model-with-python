## 特徴量エンジニアリング: モメンタム
### ３ヶ月モメンタムを計算:
import numpy as np
import pandas as pd

df = pd.read_csv("/Users/kondousatoshishi/Downloads/金融モデル/data/4063.T_monthly.csv")
# Close列をfloatに変換する
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df["momentum_3m"] = df["Close"].pct_change(periods=3)

### 欠損値を除去してモデル入力用データを整形:
data = df[["momentum_3m", "Close"]].dropna()
data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

X = data[["momentum_3m"]]
y = data["target"]


# ① 既存のCSVファイルを読み込む
file_path = "/Users/kondousatoshishi/Downloads/金融モデル/data/4063.T_month_end_with_RSI.csv"
df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

# ② モメンタム（3か月前比）を計算して新しい列に追加
df["momentum_3m"] = df["Close"].pct_change(periods=3)

# ③ 追加後のDataFrameを確認（必要に応じて）
print(df.head(10))

# ④ 新しいCSVファイルとして保存（元ファイルを上書きしないよう別名で保存）
output_path = "/Users/kondousatoshishi/Downloads/金融モデル/data/4063.T_month_end_with_RSI_momentum.csv"
df.to_csv(output_path)