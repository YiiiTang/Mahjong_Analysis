import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}\n")

# print(classification_report(Y_test, Y_pred))

print(f"截距 (w0): {model.intercept_[0]:.4f}")

print(f"巡數(a) 的權重: {model.coef_[0][0]:.4f}")
print(f"吃碰數(b) 的權重: {model.coef_[0][1]:.4f}")
print(f"中張比例(c) 的權重: {model.coef_[0][2]:.4f}")
print(f"花色集中度(d) 的權重: {model.coef_[0][3]:.4f}")
print(f"字牌比例(e) 的權重: {model.coef_[0][4]:.4f}")
print(f"摸切比例(f) 的權重: {model.coef_[0][5]:.4f}")
print(f"連續摸切強度(g) 的權重: {model.coef_[0][6]:.4f}")
print(f"摸切轉手切(h) 的權重: {model.coef_[0][7]:.4f}")