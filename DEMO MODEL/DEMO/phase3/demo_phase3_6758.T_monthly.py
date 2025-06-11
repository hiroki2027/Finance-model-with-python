## 特徴量エンジニアリング: モメンタム
### ３ヶ月モメンタムを計算:
import numpy as np
import pandas as pd

df = pd.read_csv("/Users/kondousatoshishi/Downloads/金融モデル/data/6758.T_monthly.csv")
# Close列をfloatに変換する
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df["momentum_3m"] = df["Close"].pct_change(periods=3)

### 欠損値を除去してモデル入力用データを整形:
data = df[["momentum_3m", "Close"]].dropna()
data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

X = data[["momentum_3m"]]
y = data["target"]


### ロジスティック回帰モデルの学習・予測・評価:from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 時系列を保ったまま学習・テスト分割（最後の10件をテスト用に）
X_train, X_test = X[:-10], X[-10:]
y_train, y_test = y[:-10], y[-10:]

# モデルの作成と学習
model = LogisticRegression()
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 制度評価
print("Accuracy", accuracy_score(y_test, y_pred))
print("Classification Report")
print(classification_report(y_test, y_pred))



