import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_excel("analyze_all_players_snapshots.xlsx")
df = df.fillna(0)

feature_cols = [
    'feat_a_巡數',
    'feat_b_吃碰數',
    'feat_c_中張比例',
    'feat_d_花色集中度',
    'feat_e_字牌比例',
    'feat_f_摸切比例',
    'feat_g_連續摸切強度',
    'feat_h_摸切轉手切'
]

X = df[feature_cols]
Y = df['Target_是否已聽牌']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)       

"""
print("\n" + "="*45)
print("請將以下參數複製到 track_tenpai_state_logistic.py 中")
print("="*45)
print(f"MEANS = {{\n    'a': {scaler.mean_[0]:.4f}, 'b': {scaler.mean_[1]:.4f}, 'c': {scaler.mean_[2]:.4f}, 'd': {scaler.mean_[3]:.4f},\n    'e': {scaler.mean_[4]:.4f}, 'f': {scaler.mean_[5]:.4f}, 'g': {scaler.mean_[6]:.4f}, 'h': {scaler.mean_[7]:.4f}\n}}")
print(f"SCALES = {{\n    'a': {scaler.scale_[0]:.4f}, 'b': {scaler.scale_[1]:.4f}, 'c': {scaler.scale_[2]:.4f}, 'd': {scaler.scale_[3]:.4f},\n    'e': {scaler.scale_[4]:.4f}, 'f': {scaler.scale_[5]:.4f}, 'g': {scaler.scale_[6]:.4f}, 'h': {scaler.scale_[7]:.4f}\n}}")
print("="*45 + "\n")
"""

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, Y_train)

Y_pred = model.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}\n")

print(f"截距 (w0): {model.intercept_[0]:.4f}")

print(f"巡數(a) 的權重: {model.coef_[0][0]:.4f}")
print(f"吃碰數(b) 的權重: {model.coef_[0][1]:.4f}")
print(f"中張比例(c) 的權重: {model.coef_[0][2]:.4f}")
print(f"花色集中度(d) 的權重: {model.coef_[0][3]:.4f}")
print(f"字牌比例(e) 的權重: {model.coef_[0][4]:.4f}")
print(f"摸切比例(f) 的權重: {model.coef_[0][5]:.4f}")
print(f"連續摸切強度(g) 的權重: {model.coef_[0][6]:.4f}")
print(f"摸切轉手切(h) 的權重: {model.coef_[0][7]:.4f}")