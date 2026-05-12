import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import re
import os

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

df_train = pd.read_excel("analyze_all_players_snapshots.xlsx").fillna(0)
df_test = pd.read_excel("analyze_test_snapshots.xlsx").fillna(0) 

df_train['meld_count'] = df_train['feat_b_吃碰數']
df_test['meld_count'] = df_test['feat_b_吃碰數']

df_train['phase'] = df_train['feat_a_巡數'].apply(lambda x: 'early' if x <= 8 else 'late')
df_test['phase'] = df_test['feat_a_巡數'].apply(lambda x: 'early' if x <= 8 else 'late')

feature_cols = [
    'feat_a_巡數', 'feat_c_花色集中度', 'feat_d_中張比例(3 ~ 7)', 'feat_e_邊張比例(1、2、8、9)',
    'feat_f_字牌比例', 'feat_g_摸切比例', 'feat_h_連續摸切強度', 'feat_i_摸切轉手切',
    'feat_j_中張第一張被打出', 'feat_k_中張第二張被打出', 'feat_l_中張第三張被打出', 'feat_m_中張第四張被打出',
    'feat_n_字牌第一張被打出', 'feat_o_字牌第二張被打出', 'feat_p_字牌第三張被打出', 'feat_q_字牌第四張被打出',
    'feat_r_邊張(1、9)第一張被打出', 'feat_s_邊張(1、9)第二張被打出', 'feat_t_邊張(1、9)第三張被打出', 'feat_u_邊張(1、9)第四張被打出',
    'feat_v_邊張(2、8)第一張被打出', 'feat_w_邊張(2、8)第二張被打出', 'feat_x_邊張(2、8)第三張被打出', 'feat_y_邊張(2、8)第四張被打出'
]
target_col = 'Target_是否已聽牌'

for meld in range(5):
    for phase in ['early', 'late']:
        phase_label = "前期 (1~8巡)" if phase == 'early' else "後期 (9巡~結束)"
        print(f"\n" + "="*60)
        print(f"正在執行 {meld} 副露 | {phase_label} 的 Drop-Column 特徵實驗 (Linear)")
        print("="*60)

        subset_train = df_train[(df_train['meld_count'] == meld) & (df_train['phase'] == phase)]
        subset_test = df_test[(df_test['meld_count'] == meld) & (df_test['phase'] == phase)]
        
        if len(subset_train) < 10 or len(subset_test) < 5:
            print(f"警告: {meld} 副露 ({phase_label}) 的資料量太少，跳過此組。\n")
            continue

        X_train_full = subset_train[feature_cols]
        Y_train = subset_train[target_col]
        X_test_full = subset_test[feature_cols]
        Y_test = subset_test[target_col]
        
        scaler_full = StandardScaler()
        X_train_full_scaled = scaler_full.fit_transform(X_train_full)
        X_test_full_scaled = scaler_full.transform(X_test_full)

        base_model = LinearRegression()
        base_model.fit(X_train_full_scaled, Y_train)
        
        Y_pred_base = base_model.predict(X_test_full_scaled)
        base_mse = mean_squared_error(Y_test, Y_pred_base)
        print(f"[{meld} 副露 | {phase_label} 基準模型] 測試集 MSE: {base_mse:.6f}\n")

        importance_results = {}
        for feature_to_remove in feature_cols:
            reduced_features = [f for f in feature_cols if f != feature_to_remove]
            
            X_train_reduced = subset_train[reduced_features]
            X_test_reduced = subset_test[reduced_features]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_reduced)
            X_test_scaled = scaler.transform(X_test_reduced)
            
            model = LinearRegression()
            model.fit(X_train_scaled, Y_train)
            
            pred = model.predict(X_test_scaled)
            mse_removed = mean_squared_error(Y_test, pred)
            
            delta_mse = mse_removed - base_mse
            importance_results[feature_to_remove] = delta_mse

        features_for_plot = [re.sub(r'^feat_[a-z]_', '', f) for f in feature_cols]
        drops_for_plot = [importance_results[f] * 1000 for f in feature_cols]

        features_plot_rev = features_for_plot[::-1]
        drops_plot_rev = drops_for_plot[::-1]

        plt.figure(figsize=(10, 8))
        colors = ['#4C72B0' if val > 0 else '#C44E52' for val in drops_plot_rev]
        bars = plt.barh(features_plot_rev, drops_plot_rev, color=colors)

        plt.xlabel('MSE 誤差增加量 (x1000)', fontsize=12)
        plt.title(f'[{meld} 副露 | {phase_label}] 移除特徵對預測誤差的影響 (Linear)', fontsize=14, fontweight='bold')
        plt.axvline(0, color='black', linewidth=1.2, linestyle='--') 

        max_positive = max([val for val in drops_for_plot if val > 0] + [0.001]) 
        min_val = min(drops_for_plot)
        
        left_limit = min_val - (max_positive * 0.5) - 0.1 if min_val < 0 else -0.5
        right_limit = max(drops_for_plot) + (max_positive * 0.3) + 0.5
        plt.xlim(left_limit, right_limit)

        for bar in bars:
            width = bar.get_width()
            offset = max_positive * 0.03 + 0.05
            if width > 0:
                plt.text(width + offset, bar.get_y() + bar.get_height()/2, 
                         f'{width:.3f}', va='center', ha='left', fontsize=10)
            else:
                plt.text(width - offset, bar.get_y() + bar.get_height()/2, 
                         f'{width:.3f}', va='center', ha='right', fontsize=10)

        plt.tight_layout()
        save_path = f"drop_column_importance_linear_meld_{meld}_{phase}.png"
        plt.savefig(save_path, dpi=300)
        plt.close() 
        print(f"✅ 圖表已儲存至: {save_path}")