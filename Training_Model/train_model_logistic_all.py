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
    'feat_c_花色集中度',
    'feat_d_摸切比例',
    'feat_e_連續摸切強度',
    'feat_f_摸切轉手切',
    'feat_g_中張第一張被打出',
    'feat_h_中張第二張被打出',
    'feat_i_中張第三張被打出',
    'feat_j_中張第四張被打出',
    'feat_k_字牌第一張被打出',
    'feat_l_字牌第二張被打出',
    'feat_m_字牌第三張被打出',
    'feat_n_字牌第四張被打出',
    'feat_o_邊張第一張被打出',
    'feat_p_邊張第二張被打出',
    'feat_q_邊張第三張被打出',
    'feat_r_邊張第四張被打出'
]

X = df[feature_cols]
Y = df['Target_是否已聽牌']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)       

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, Y_train)

Y_pred = model.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}\n")
print(f"截距 (w0): {model.intercept_[0]:.4f}")

print(f"巡數(a) 的權重: {model.coef_[0][0]:.4f}")
print(f"吃碰數(b) 的權重: {model.coef_[0][1]:.4f}")
print(f"花色集中度(c) 的權重: {model.coef_[0][2]:.4f}")
print(f"摸切比例(d) 的權重: {model.coef_[0][3]:.4f}")
print(f"連續摸切強度(e) 的權重: {model.coef_[0][4]:.4f}")
print(f"摸切轉手切(f) 的權重: {model.coef_[0][5]:.4f}")
print(f"中張第一張被打出(g) 的權重: {model.coef_[0][6]:.4f}")
print(f"中張第二張被打出(h) 的權重: {model.coef_[0][7]:.4f}")
print(f"中張第三張被打出(i) 的權重: {model.coef_[0][8]:.4f}")
print(f"中張第四張被打出(j) 的權重: {model.coef_[0][9]:.4f}")
print(f"字牌第一張被打出(k) 的權重: {model.coef_[0][10]:.4f}")
print(f"字牌第二張被打出(l) 的權重: {model.coef_[0][11]:.4f}")
print(f"字牌第三張被打出(m) 的權重: {model.coef_[0][12]:.4f}")
print(f"字牌第四張被打出(n) 的權重: {model.coef_[0][13]:.4f}")
print(f"邊張第一張被打出(o) 的權重: {model.coef_[0][14]:.4f}")
print(f"邊張第二張被打出(p) 的權重: {model.coef_[0][15]:.4f}")
print(f"邊張第三張被打出(q) 的權重: {model.coef_[0][16]:.4f}")
print(f"邊張第四張被打出(r) 的權重: {model.coef_[0][17]:.4f}\n")

w = model.coef_[0]

print("="*50)
print("請將以下參數複製到 track_tenpai_state_logistic.py 中")
print("="*50)

print(f"WEIGHTS = {{\n    'a': {w[0]:.4f}, 'b': {w[1]:.4f}, 'c': {w[2]:.4f}, 'd': {w[3]:.4f},\n    'e': {w[4]:.4f}, 'f': {w[5]:.4f}, 'g': {w[6]:.4f}, 'h': {w[7]:.4f},\n    'i': {w[8]:.4f}, 'j': {w[9]:.4f}, 'k': {w[10]:.4f}, 'l': {w[11]:.4f},\n    'm': {w[12]:.4f}, 'n': {w[13]:.4f}, 'o': {w[14]:.4f}, 'p': {w[15]:.4f},\n    'q': {w[16]:.4f}, 'r': {w[17]:.4f}\n}}")

print(f"\nMEANS = {{\n    'a': {scaler.mean_[0]:.4f}, 'b': {scaler.mean_[1]:.4f}, 'c': {scaler.mean_[2]:.4f}, 'd': {scaler.mean_[3]:.4f},\n    'e': {scaler.mean_[4]:.4f}, 'f': {scaler.mean_[5]:.4f}, 'g': {scaler.mean_[6]:.4f}, 'h': {scaler.mean_[7]:.4f},\n    'i': {scaler.mean_[8]:.4f}, 'j': {scaler.mean_[9]:.4f}, 'k': {scaler.mean_[10]:.4f}, 'l': {scaler.mean_[11]:.4f},\n    'm': {scaler.mean_[12]:.4f}, 'n': {scaler.mean_[13]:.4f}, 'o': {scaler.mean_[14]:.4f}, 'p': {scaler.mean_[15]:.4f},\n    'q': {scaler.mean_[16]:.4f}, 'r': {scaler.mean_[17]:.4f}\n}}")

print(f"\nSCALES = {{\n    'a': {scaler.scale_[0]:.4f}, 'b': {scaler.scale_[1]:.4f}, 'c': {scaler.scale_[2]:.4f}, 'd': {scaler.scale_[3]:.4f},\n    'e': {scaler.scale_[4]:.4f}, 'f': {scaler.scale_[5]:.4f}, 'g': {scaler.scale_[6]:.4f}, 'h': {scaler.scale_[7]:.4f},\n    'i': {scaler.scale_[8]:.4f}, 'j': {scaler.scale_[9]:.4f}, 'k': {scaler.scale_[10]:.4f}, 'l': {scaler.scale_[11]:.4f},\n    'm': {scaler.scale_[12]:.4f}, 'n': {scaler.scale_[13]:.4f}, 'o': {scaler.scale_[14]:.4f}, 'p': {scaler.scale_[15]:.4f},\n    'q': {scaler.scale_[16]:.4f}, 'r': {scaler.scale_[17]:.4f}\n}}")

print("="*50 + "\n")