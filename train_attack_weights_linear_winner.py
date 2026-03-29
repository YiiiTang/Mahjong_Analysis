import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_excel("analyze_attack_winner_only.xlsx")
df = df.fillna(0)

feature_cols = [
    '當下巡數',      
    '當下副露數',    
    '向聽數 (Shanten)',   
    '進張數 (Ukeire)'   
]

X = df[feature_cols]
Y = df['Target_剩餘巡數']

model = LinearRegression()
model.fit(X, Y)

Y_pred_raw = model.predict(X)
Y_pred_clipped = np.maximum(Y_pred_raw, 0) 

print(f"MAE (平均誤差巡數): {mean_absolute_error(Y, Y_pred_clipped):.4f}")
print(f"MSE (均方誤差): {mean_squared_error(Y, Y_pred_clipped):.4f}")
print(f"R-squared: {r2_score(Y, Y_pred_clipped):.4f}\n")

print(f"截距 (w0): {model.intercept_:.4f}")

print(f"當下巡數 的權重: {model.coef_[0]:.4f}")
print(f"當下副露數 的權重(w1): {model.coef_[1]:.4f}")
print(f"向聽數的權重(w2): {model.coef_[2]:.4f}")
print(f"進張數的權重(w3): {model.coef_[3]:.4f}")