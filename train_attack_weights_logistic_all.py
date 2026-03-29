import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

df = pd.read_excel("analyze_attack_snapshots_all.xlsx")
df = df.fillna(0)

df = df[df['向聽數 (Shanten)'] >= -1]

feature_cols = [
    '當下巡數',      
    '當下副露數',    
    '向聽數 (Shanten)',   
    '進張數 (Ukeire)'   
]

X = df[feature_cols]
Y = df['Target_是否胡牌(Win)']

model = LogisticRegression(class_weight='balanced', random_state=42)
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