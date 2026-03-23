import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_excel("analyze_all_players_snapshots.xlsx")

df = df.fillna(0)

feature_cols = [
    '當下巡數', 
    '當下副露數', 
    '當下3至7比例', 
    '當下丟字比例', 
    '最多單一花色丟棄比例'
]
X = df[feature_cols]
Y = df['是否已聽牌 (Y)']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")

print(classification_report(Y_test, Y_pred))

print(f"截距 (w0): {model.intercept_[0]:.4f}")

print(f"當下巡數(a) 的權重: {model.coef_[0]:.4f}")
print(f"當下副露數(b) 的權重: {model.coef_[1]:.4f}")
print(f"當下3至7比例(c) 的權重: {model.coef_[2]:.4f}")
print(f"花色分散度(d) 的權重: {model.coef_[3]:.4f}")
print(f"當下丟字比例(e) 的權重: {model.coef_[4]:.4f}")