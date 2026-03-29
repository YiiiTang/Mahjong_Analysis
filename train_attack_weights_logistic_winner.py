import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_excel("analyze_attack_winner_only.xlsx")
df = df.fillna(0)

df.columns = df.columns.str.strip()

df = df[df['向聽數 (Shanten)'] >= -1]

MAX_TURN = 18        
MAX_UKEIRE = 40        

df['巡數比例'] = df['當下巡數'] / MAX_TURN
df['進張比例'] = df['進張數 (Ukeire)'] / MAX_UKEIRE
df['副露x向聽'] = df['當下副露數'] * df['向聽數 (Shanten)']

feature_cols = [
    '巡數比例',        
    '當下副露數',      
    '向聽數 (Shanten)',
    '進張比例',       
    '副露x向聽'      
]

X = df[feature_cols]

target_cols = [c for c in df.columns if '胡牌' in c]
if not target_cols:
    raise ValueError("找不到含 '胡牌' 的欄位，請確認 Excel")
target_col = target_cols[0]
Y = df[target_col]

model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model.fit(X, Y)

Y_pred_class = model.predict(X)
Y_pred_prob = model.predict_proba(X)[:, 1]

print(f"Accuracy: {accuracy_score(Y, Y_pred_class):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(Y, Y_pred_prob):.4f}\n")

"""
print("分類報告 (Classification Report):")
print(classification_report(Y, Y_pred_class))
"""

print(f"截距 (w0): {model.intercept_[0]:.4f}")

print(f"當下巡數 的權重(w1): {model.coef_[0][0]:.4f}")
print(f"當下副露數 的權重(w2): {model.coef_[0][1]:.4f}")
print(f"向聽數的權重(w3): {model.coef_[0][2]:.4f}")
print(f"進張數的權重(w4): {model.coef_[0][3]:.4f}")
print(f"副露x向聽的權重(w5): {model.coef_[0][4]:.4f}")