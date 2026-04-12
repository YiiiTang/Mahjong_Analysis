import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.font_manager as fm

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

df_train = pd.read_excel("analyze_all_players_snapshots.xlsx").fillna(0)
df_test = pd.read_excel("analyze_test_snapshots.xlsx").fillna(0) 

feature_cols = [
    'feat_a_巡數', 'feat_b_吃碰數', 'feat_c_中張比例', 'feat_d_花色集中度',
    'feat_e_字牌比例', 'feat_f_摸切比例', 'feat_g_連續摸切強度', 'feat_h_摸切轉手切'
]
target_col = 'Target_是否已聽牌'

X_train_full = df_train[feature_cols]
Y_train = df_train[target_col]
X_test_full = df_test[feature_cols]
Y_test = df_test[target_col]

def get_accuracy(model, X, Y):
    Y_pred = model.predict(X)
    Y_pred_binary = (Y_pred >= 0.5).astype(int)
    return accuracy_score(Y, Y_pred_binary)

scaler_full = StandardScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full)
X_test_full_scaled = scaler_full.transform(X_test_full)

base_model = LinearRegression()
base_model.fit(X_train_full_scaled, Y_train)
base_acc = get_accuracy(base_model, X_test_full_scaled, Y_test)
print(f"[基準模型] 測試集準確率 (全特徵): {base_acc:.4%}\n")
print("-" * 50)

importance_results = {}

for feature in feature_cols:
    X_train_ablated = X_train_full.drop(columns=[feature])
    X_test_ablated = X_test_full.drop(columns=[feature])
    
    scaler_ablated = StandardScaler()
    X_train_ablated_scaled = scaler_ablated.fit_transform(X_train_ablated)
    X_test_ablated_scaled = scaler_ablated.transform(X_test_ablated)
    
    ablated_model = LinearRegression()
    ablated_model.fit(X_train_ablated_scaled, Y_train)
    
    ablated_acc = get_accuracy(ablated_model, X_test_ablated_scaled, Y_test)
    
    acc_drop = base_acc - ablated_acc
    importance_results[feature] = acc_drop
    
    change_str = f"{-acc_drop:+.4%}" 
    print(f"移除特徵: {feature}")
    print(f"   -> 新準確率: {ablated_acc:.4%} (變化: {change_str})")

print("\n" + "="*50)
print("特徵重要性排名")
print("="*50)

sorted_importance = sorted(importance_results.items(), key=lambda x: x[1], reverse=True)

for rank, (feat, drop) in enumerate(sorted_importance, 1):
    if drop > 0:
        print(f"{rank}. {feat}: 準確率下降了 {drop:.4%}")
    else:
        print(f"{rank}. {feat}: 準確率反升了 {-drop:.4%} (可能是干擾特徵)")

features_for_plot = [f.replace('feat_', '') for f in feature_cols]
drops_percent_for_plot = [importance_results[f] * 100 for f in feature_cols]

features_plot_rev = features_for_plot[::-1]
drops_plot_rev = drops_percent_for_plot[::-1]

plt.figure(figsize=(10, 6))
colors = ['#4C72B0' if val > 0 else '#C44E52' for val in drops_plot_rev]
bars = plt.barh(features_plot_rev, drops_plot_rev, color=colors)

plt.xlabel('準確率下降幅度 (%)', fontsize=12)
plt.title('移除單一特徵對預測聽牌準確率的影響', fontsize=14, fontweight='bold')
plt.axvline(0, color='black', linewidth=1.2, linestyle='--') 

max_positive = max([val for val in drops_percent_for_plot if val > 0] + [0.1]) 
left_limit = min(drops_percent_for_plot) - (max_positive * 0.3) - 0.5
right_limit = max(drops_percent_for_plot) + (max_positive * 0.15)
plt.xlim(left_limit, right_limit)

for bar in bars:
    width = bar.get_width()
    offset = max_positive * 0.02
    
    if width > 0:
        plt.text(width + offset, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}%', va='center', ha='left', fontsize=10)
    else:
        plt.text(width - offset, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}%', va='center', ha='right', fontsize=10)

plt.tight_layout()
save_path = "feature_importance_ablation_linear.png"
plt.savefig(save_path, dpi=300)