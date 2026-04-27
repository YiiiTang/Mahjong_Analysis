import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_excel("analyze_all_players_snapshots.xlsx")
df = df.fillna(0)

# [修改] 加入新特徵
feature_cols = [
    'feat_a_巡數',
    'feat_b_吃碰數',
    'feat_c_中張比例',
    'feat_d_花色集中度',
    'feat_e_字牌比例',
    'feat_f_摸切比例',
    'feat_g_連續摸切強度',
    'feat_h_摸切轉手切',
    'feat_i_第一張被打出',
    'feat_j_第二張被打出',
    'feat_k_第三張被打出',
    'feat_l_第四張被打出'
]

X = df[feature_cols]
Y = df['Target_是否已聽牌']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "="*45)
print("請將以下參數複製到 track_tenpai_state_linear.py 中")
print("="*45)
print(f"MEANS = {{\n    'a': {scaler.mean_[0]:.4f}, 'b': {scaler.mean_[1]:.4f}, 'c': {scaler.mean_[2]:.4f}, 'd': {scaler.mean_[3]:.4f},\n    'e': {scaler.mean_[4]:.4f}, 'f': {scaler.mean_[5]:.4f}, 'g': {scaler.mean_[6]:.4f}, 'h': {scaler.mean_[7]:.4f},\n    'i': {scaler.mean_[8]:.4f}, 'j': {scaler.mean_[9]:.4f}, 'k': {scaler.mean_[10]:.4f}, 'l': {scaler.mean_[11]:.4f}\n}}")
print(f"SCALES = {{\n    'a': {scaler.scale_[0]:.4f}, 'b': {scaler.scale_[1]:.4f}, 'c': {scaler.scale_[2]:.4f}, 'd': {scaler.scale_[3]:.4f},\n    'e': {scaler.scale_[4]:.4f}, 'f': {scaler.scale_[5]:.4f}, 'g': {scaler.scale_[6]:.4f}, 'h': {scaler.scale_[7]:.4f},\n    'i': {scaler.scale_[8]:.4f}, 'j': {scaler.scale_[9]:.4f}, 'k': {scaler.scale_[10]:.4f}, 'l': {scaler.scale_[11]:.4f}\n}}")
print("="*45 + "\n")


model = LinearRegression()
model.fit(X_scaled, Y)

Y_pred = model.predict(X_scaled)

mae = mean_absolute_error(Y, Y_pred)
mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)

print(f"MAE (平均絕對誤差): {mae:.4f}") 
print(f"MSE (均方誤差): {mse:.4f}")
print(f"R-squared (決定係數): {r2:.4f}\n")

print(f"截距 (w0): {model.intercept_:.4f}")

print(f"巡數(a) 的權重: {model.coef_[0]:.4f}")
print(f"吃碰數(b) 的權重: {model.coef_[1]:.4f}")
print(f"中張比例(c) 的權重: {model.coef_[2]:.4f}")
print(f"花色集中度(d) 的權重: {model.coef_[3]:.4f}")
print(f"字牌比例(e) 的權重: {model.coef_[4]:.4f}")
print(f"摸切比例(f) 的權重: {model.coef_[5]:.4f}")
print(f"連續摸切強度(g) 的權重: {model.coef_[6]:.4f}")
print(f"摸切轉手切(h) 的權重: {model.coef_[7]:.4f}")
print(f"第一張被打出(i) 的權重: {model.coef_[8]:.4f}")
print(f"第二張被打出(j) 的權重: {model.coef_[9]:.4f}")
print(f"第三張被打出(k) 的權重: {model.coef_[10]:.4f}")
print(f"第四張被打出(l) 的權重: {model.coef_[11]:.4f}")