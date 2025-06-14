# FORGE1 
#### 本格的に株式市場で運用するモデルのための作業

---

## 🎯 目標指標
| 指標 | クリアライン |
|------|--------------|
| Balanced Accuracy | **0.45 以上** |
| Macro‑F1 | **0.42 以上** |
| 最大ドローダウン（簡易BT） | **–15 % 以内** |


---

## 1️⃣  ラベルを “動的” に  
- 固定 ±1 % ルールから脱却し、**ATR ベース**で閾値を自動調整。  
- 銘柄ごとにボラを反映した「上・横・下」に振り分け → クラス偏りを解消。  

> *例：リターン > 0.75×ATR → “上昇”、< –0.75×ATR → “下落”*

---

## 2️⃣  ハイパラは Optuna に一任  
- learning_rate、max_depth、subsample…を**50〜100 トライアル**。  
- 目的関数は **`multi:softprob`** で確率出力を取得。  
- **クラス重み**を設定し、少数派クラスを平等に学習。  

*Expected: Accuracy +5〜8pt、Macro‑F1 +0.10前後*

---

## 3️⃣  少数派クラスの“救済策”  
- 各 fold でクラス出現頻度を算出 → **`scale_pos_weight`** に反映。  
- 上昇/下落シグナルの **Recall** を重視して改善効果をモニタリング。

---

## 4️⃣  検証は Walk‑Forward で完結  
1. 時系列に沿い、過去で学習→直後区間を検証をスライド。  
2. 各スライスで **Balanced Accuracy / Macro‑F1** を記録。  
3. 結果を `TICKER_foldX.csv` に自動出力。  

未来情報漏れゼロ、信頼性の高いスコアを確保。

---

## 5️⃣  コスト込みの迅速バックテスト  
- 売買コスト：往復 **±0.5 %** を仮定。  
- シグナル：【上→買い】【下→売り】【横→様子見】。  
- 累積 P&L と最大 DD を算出し、目標指標との整合性をチェック。

---

## 6️⃣  衛生管理と高速化  
- 乱数シードは全て `42` で固定。  
- `logging` でパラメータやスコアを一元記録。  
- データ読み書きは **Parquet** 形式で高速化。

---

### 🔥 優先度チャート  
| 緊急度 | 重要度 | タスク |
|:------:|:------:|--------|
| 🔥     | ⭐⭐⭐   | **1. 動的ラベル切り替え** |
| 🔥     | ⭐⭐⭐   | **2. Optuna チューニング** |
| ⚡️     | ⭐⭐☆   | **3. クラス重み設定** |
| ⚡️     | ⭐⭐☆   | **4. Walk‑Forward 導入** |
| 🌤     | ⭐⭐☆   | **5. コスト込みミニBT** |
| 🌱     | ⭐☆☆   | **6. Parquet & ロギング整備** |

---

> **Feature & Model unchanged.**  
> *動的ラベル → ハイパラ最適化 → クラス重み* の順で精度を底上げ。
  
_最終更新: 2025‑06‑11_