import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


file_path = "/Users/kondousatoshishi/Downloads/金融モデル/data/date_RSI_momentum/4063.T_month_end_with_RSI_momentum.csv"
df = pd.read_csv(file_path, index_col=0, parse_dates=True)
print(df.head())
print(df.info())

df.plot(figsize=(10, 12), subplots=True)
plt.show()

print(df.diff().head())

df_2 = df.drop(columns=["momentum_3m"])
df_2 = df_2.pct_change().mean()
df_2.plot(kind='bar')
plt.show()

# 対数収益率を使う: ある種の正規化
rets = np.log(df / df.shift(1))
print(rets.head().round(3))

rets = rets.cumsum().apply(np.exp).plot()
plt.show()