import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.font_manager as fm
import re
import os

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

df_train = pd.read_excel("analyze_all_players_snapshots.xlsx").fillna(0)
df_test = pd.read_excel("analyze_test_snapshots.xlsx").fillna(0) 

def get_meld_count(row):
    if row['feat_b4_四副露'] == 1: return 4
    if row['feat_b3_三副露'] == 1: return 3
    if row['feat_b2_二副露'] == 1: return 2
    if row['feat_b1_一副露'] == 1: return 1
    return 0

df_train['meld_count'] = df_train.apply(get_meld_count, axis=1)
df_test['meld_count'] = df_test.apply(get_meld_count, axis=1)

feature_cols = [
    'feat_a_巡數', 'feat_c_花色集中度', 'feat_d_摸切比例', 'feat_e_連續摸切強度', 'feat_f_摸切轉手切',
    'feat_g_中張第一張被打出', 'feat_h_中張第二張被打出', 'feat_i_中張第三張被打出', 'feat_j_中張第四張被打出',
    'feat_k_字牌第一張被打出', 'feat_l_字牌第二張被打出', 'feat_m_字牌第三張被打出', 'feat_n_字牌第四張被打出',
    'feat_o_邊張第一張被打出', 'feat_p_邊張第二張被打出', 'feat_q_邊張第三張被打出', 'feat_r_邊張第四張被打出'
]
target_col = 'Target_是否已聽牌'

def get_accuracy(model, X, Y):
    Y_pred = model.predict(X)
    return accuracy_score(Y, Y_pred)

for meld in range(5):
    print(f"\n" + "="*50)
    print(f"正在執行 {meld} 副露的消融實驗 (Logistic)")
    print("="*50)

    subset_train = df_train[df_train['meld_count'] == meld]
    subset_test = df_test[df_test['meld_count'] == meld]
    
    if len(subset_train) < 10 or len(subset_test) < 5:
        print(f"警告: {meld} 副露的資料量太少 (Train: {len(subset_train)}, Test: {len(subset_test)})，跳過此組。\n")
        continue

    X_train_full = subset_train[feature_cols]
    Y_train = subset_train[target_col]
    X_test_full = subset_test[feature_cols]
    Y_test = subset_test[target_col]

    scaler_full = StandardScaler()
    X_train_full_scaled = scaler_full.fit_transform(X_train_full)
    X_test_full_scaled = scaler_full.transform(X_test_full)

    base_model = LogisticRegression(max_iter=1000)
    base_model.fit(X_train_full_scaled, Y_train)
    base_acc = get_accuracy(base_model, X_test_full_scaled, Y_test)
    print(f"[{meld} 副露 基準模型] 測試集準確率 (全特徵): {base_acc:.4%}\n")

    importance_results = {}

    for feature in feature_cols:
        X_train_ablated = X_train_full.drop(columns=[feature])
        X_test_ablated = X_test_full.drop(columns=[feature])
        
        scaler_ablated = StandardScaler()
        X_train_ablated_scaled = scaler_ablated.fit_transform(X_train_ablated)
        X_test_ablated_scaled = scaler_ablated.transform(X_test_ablated)
        
        ablated_model = LogisticRegression(max_iter=1000)
        ablated_model.fit(X_train_ablated_scaled, Y_train)
        
        ablated_acc = get_accuracy(ablated_model, X_test_ablated_scaled, Y_test)
        acc_drop = base_acc - ablated_acc
        importance_results[feature] = acc_drop

    features_for_plot = [re.sub(r'^feat_[a-z]_', '', f) for f in feature_cols]
    drops_percent_for_plot = [importance_results[f] * 100 for f in feature_cols]

    features_plot_rev = features_for_plot[::-1]
    drops_plot_rev = drops_percent_for_plot[::-1]

    plt.figure(figsize=(10, 8))
    colors = ['#4C72B0' if val > 0 else '#C44E52' for val in drops_plot_rev]
    bars = plt.barh(features_plot_rev, drops_plot_rev, color=colors)

    plt.xlabel('準確率下降幅度 (%)', fontsize=12)
    plt.title(f'[{meld} 副露] 移除單一特徵對預測聽牌準確率的影響 (Logistic)', fontsize=14, fontweight='bold')
    plt.axvline(0, color='black', linewidth=1.2, linestyle='--') 

    max_positive = max([val for val in drops_percent_for_plot if val > 0] + [0.1]) 
    min_val = min(drops_percent_for_plot)
    
    left_limit = min_val - (max_positive * 0.5) - 0.5 if min_val < 0 else -1.0
    right_limit = max(drops_percent_for_plot) + (max_positive * 0.3) + 0.5
    plt.xlim(left_limit, right_limit)

    for bar in bars:
        width = bar.get_width()
        offset = max_positive * 0.03 + 0.1
        
        if width > 0:
            plt.text(width + offset, bar.get_y() + bar.get_height()/2, 
                     f'{width:.2f}%', va='center', ha='left', fontsize=10)
        else:
            plt.text(width - offset, bar.get_y() + bar.get_height()/2, 
                     f'{width:.2f}%', va='center', ha='right', fontsize=10)

    plt.tight_layout()
    save_path = f"feature_importance_ablation_logistic_meld_{meld}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 圖表已儲存至: {save_path}")