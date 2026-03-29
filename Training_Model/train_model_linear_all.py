import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_excel("analyze_all_players_snapshots.xlsx")
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

Y_pred = model.predict(X)

mae = mean_absolute_error(Y, Y_pred)
mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)

print(f"MAE (平均絕對誤差): {mae:.4f}") # 數值越小代表預測的機率越貼近真實的 0 或 1
print(f"MSE (均方誤差): {mse:.4f}")
print(f"R-squared (決定係數): {r2:.4f}\n")

print(f"截距 (w0): {model.intercept_:.4f}")

print(f"當下巡數(a) 的權重: {model.coef_[0]:.4f}")
print(f"當下副露數(b) 的權重: {model.coef_[1]:.4f}")
print(f"當下3至7比例(c) 的權重: {model.coef_[2]:.4f}")
print(f"花色分散度(d) 的權重: {model.coef_[3]:.4f}")
print(f"當下丟字比例(e) 的權重: {model.coef_[4]:.4f}")