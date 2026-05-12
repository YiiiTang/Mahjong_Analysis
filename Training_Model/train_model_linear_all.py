import pandas as pd
import numpy as np
import re
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === 設定絕對路徑 ===
DATA_PATH = r"D:\Project\Mahjong\Analyze_Mahjong\Training_Model\analyze_all_players_snapshots.xlsx"
TARGET_SCRIPT = r"D:\Project\Mahjong\Analyze_Mahjong\track_tenpai_state_linear.py"
# ===================

if not os.path.exists(DATA_PATH):
    print(f"❌ 找不到訓練資料：{DATA_PATH}")
    exit()

df = pd.read_excel(DATA_PATH)
df = df.fillna(0)

df['meld_count'] = df['feat_b_吃碰數']

df['phase'] = df['feat_a_巡數'].apply(lambda x: 'early' if x <= 8 else 'late')

feature_cols = [
    'feat_a_巡數', 'feat_c_花色集中度', 'feat_d_中張比例(3 ~ 7)', 'feat_e_邊張比例(1、2、8、9)',
    'feat_f_字牌比例', 'feat_g_摸切比例', 'feat_h_連續摸切強度', 'feat_i_摸切轉手切',
    'feat_j_中張第一張被打出', 'feat_k_中張第二張被打出', 'feat_l_中張第三張被打出', 'feat_m_中張第四張被打出',
    'feat_n_字牌第一張被打出', 'feat_o_字牌第二張被打出', 'feat_p_字牌第三張被打出', 'feat_q_字牌第四張被打出',
    'feat_r_邊張(1、9)第一張被打出', 'feat_s_邊張(1、9)第二張被打出', 'feat_t_邊張(1、9)第三張被打出', 'feat_u_邊張(1、9)第四張被打出',
    'feat_v_邊張(2、8)第一張被打出', 'feat_w_邊張(2、8)第二張被打出', 'feat_x_邊張(2、8)第三張被打出', 'feat_y_邊張(2、8)第四張被打出'
]

keys = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

all_weights = {i: {} for i in range(5)}
all_means = {i: {} for i in range(5)}
all_scales = {i: {} for i in range(5)}
all_intercepts = {i: {} for i in range(5)}

for i in range(5):
    for phase in ['early', 'late']:
        print(f"========== 訓練 {i} 副露 | {phase.upper()} 階段 (Linear) ==========")
        subset_df = df[(df['meld_count'] == i) & (df['phase'] == phase)]
        
        if len(subset_df) < 10:
            print(f"警告: {i} 副露 ({phase}) 的資料量太少 ({len(subset_df)} 筆)，跳過訓練，補 0。\n")
            all_weights[i][phase] = {k: 0.0 for k in keys}
            all_means[i][phase] = {k: 0.0 for k in keys}
            all_scales[i][phase] = {k: 1.0 for k in keys}
            all_intercepts[i][phase] = 0.0
            continue
            
        X = subset_df[feature_cols]
        Y = subset_df['Target_是否已聽牌']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_scaled, Y)
        
        Y_pred = model.predict(X_scaled)
        
        mae = mean_absolute_error(Y, Y_pred)
        mse = mean_squared_error(Y, Y_pred)
        r2 = r2_score(Y, Y_pred)
        w = model.coef_
        
        print(f"資料筆數: {len(subset_df)} 筆")
        print(f"MAE (平均絕對誤差): {mae:.4f}")
        print(f"MSE (均方誤差): {mse:.4f}")
        print(f"R-squared (決定係數): {r2:.4f}\n")
        
        print(f"截距 (w0): {model.intercept_:.4f}")
        for idx, col in enumerate(feature_cols):
            print(f"{col} 的權重: {w[idx]:.4f}")
        print("\n")
        
        all_weights[i][phase] = {k: w[idx] for idx, k in enumerate(keys)}
        all_means[i][phase] = {k: scaler.mean_[idx] for idx, k in enumerate(keys)}
        all_scales[i][phase] = {k: scaler.scale_[idx] for idx, k in enumerate(keys)}
        all_intercepts[i][phase] = model.intercept_

def dict_to_string(name, d):
    s = f"{name} = {{\n"
    for meld in range(5):
        s += f"    {meld}: {{\n"
        for phase in ['early', 'late']:
            features = d[meld][phase]
            s += f"        '{phase}': {{\n"
            s += f"            'a': {features['a']:.4f}, 'c': {features['c']:.4f}, 'd': {features['d']:.4f}, 'e': {features['e']:.4f}, 'f': {features['f']:.4f},\n"
            s += f"            'g': {features['g']:.4f}, 'h': {features['h']:.4f}, 'i': {features['i']:.4f}, 'j': {features['j']:.4f}, 'k': {features['k']:.4f},\n"
            s += f"            'l': {features['l']:.4f}, 'm': {features['m']:.4f}, 'n': {features['n']:.4f}, 'o': {features['o']:.4f}, 'p': {features['p']:.4f},\n"
            s += f"            'q': {features['q']:.4f}, 'r': {features['r']:.4f}, 's': {features['s']:.4f}, 't': {features['t']:.4f}, 'u': {features['u']:.4f},\n"
            s += f"            'v': {features['v']:.4f}, 'w': {features['w']:.4f}, 'x': {features['x']:.4f}, 'y': {features['y']:.4f}\n"
            s += "        },\n"
        s += "    },\n"
    s += "}\n"
    return s

new_params_code = ""
new_params_code += dict_to_string("WEIGHTS_BY_MELD", all_weights)
new_params_code += dict_to_string("MEANS_BY_MELD", all_means)
new_params_code += dict_to_string("SCALES_BY_MELD", all_scales)

new_params_code += "\nINTERCEPTS_BY_MELD = {\n"
for meld in range(5):
    new_params_code += f"    {meld}: {{\n"
    new_params_code += f"        'early': {all_intercepts[meld]['early']:.4f},\n"
    new_params_code += f"        'late': {all_intercepts[meld]['late']:.4f}\n"
    new_params_code += "    },\n"
new_params_code += "}\n"

def update_tracking_script(target_file, new_code):
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = re.compile(r'(# AUTO-UPDATE-START.*?)(.*?)(\n# AUTO-UPDATE-END)', re.DOTALL)
        if not pattern.search(content):
            print(f"❌ 錯誤：找不到 AUTO-UPDATE-START 標籤。")
            return

        new_content = pattern.sub(r'\1\n' + new_code + r'\3', content)

        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        print(f"✅ 成功更新參數至：{target_file}")
    except FileNotFoundError:
        print(f"❌ 找不到目標檔案 {target_file}")

update_tracking_script(TARGET_SCRIPT, new_params_code)