import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_excel("analyze_winner_snapshots.xlsx")
df = df.fillna(0)

df['花色分散度 (d項)'] = 1 - df['最多單一花色丟棄比例']

feature_cols = [
    '當下巡數',
    '當下副露數',
    '當下3至7比例', 
    '花色分散度 (d項)',
    '當下丟字比例'  
]

X = df[feature_cols]
Y = df['是否已聽牌 (Y)']

model = LinearRegression()
model.fit(X, Y)

Y_pred_raw = model.predict(X)
Y_pred_prob = np.clip(Y_pred_raw, 0, 1) 
Y_pred_class = (Y_pred_prob >= 0.5).astype(int)

print(f"Accuracy: {accuracy_score(Y, Y_pred_class):.4f}\n")

print(classification_report(Y, Y_pred_class))

print(f"截距 (w0): {model.intercept_:.4f}")

print(f"當下巡數(a) 的權重: {model.coef_[0]:.4f}")
print(f"當下副露數(b) 的權重: {model.coef_[1]:.4f}")
print(f"當下3至7比例(c) 的權重: {model.coef_[2]:.4f}")
print(f"花色分散度(d) 的權重: {model.coef_[3]:.4f}")
print(f"當下丟字比例(e) 的權重: {model.coef_[4]:.4f}")