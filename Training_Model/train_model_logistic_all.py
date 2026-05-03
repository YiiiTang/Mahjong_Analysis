import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# === 設定絕對路徑 ===
DATA_PATH = r"D:\Project\Mahjong\Analyze_Mahjong\Training_Model\analyze_all_players_snapshots.xlsx"
TARGET_SCRIPT = r"D:\Project\Mahjong\Analyze_Mahjong\track_tenpai_state_logistic.py"
# ===================

if not os.path.exists(DATA_PATH):
    print(f"❌ 找不到訓練資料：{DATA_PATH}")
    exit()

df = pd.read_excel(DATA_PATH)
df = df.fillna(0)

def get_meld_count(row):
    if row['feat_b4_四副露'] == 1: return 4
    if row['feat_b3_三副露'] == 1: return 3
    if row['feat_b2_二副露'] == 1: return 2
    if row['feat_b1_一副露'] == 1: return 1
    return 0

df['meld_count'] = df.apply(get_meld_count, axis=1)

feature_cols = [
    'feat_a_巡數', 'feat_c_花色集中度', 'feat_d_摸切比例', 'feat_e_連續摸切強度', 'feat_f_摸切轉手切',
    'feat_g_中張第一張被打出', 'feat_h_中張第二張被打出', 'feat_i_中張第三張被打出', 'feat_j_中張第四張被打出',
    'feat_k_字牌第一張被打出', 'feat_l_字牌第二張被打出', 'feat_m_字牌第三張被打出', 'feat_n_字牌第四張被打出',
    'feat_o_邊張第一張被打出', 'feat_p_邊張第二張被打出', 'feat_q_邊張第三張被打出', 'feat_r_邊張第四張被打出'
]

all_weights = {}
all_means = {}
all_scales = {}
all_intercepts = {}
keys = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']

for i in range(5):
    print(f"========== 訓練 {i} 副露的模型 (Logistic) ==========")
    subset_df = df[df['meld_count'] == i]
    
    if len(subset_df) < 10:
        print(f"警告: {i} 副露的資料量太少 ({len(subset_df)} 筆)，跳過訓練。\n")
        all_weights[i] = {k: 0.0 for k in keys}
        all_means[i] = {k: 0.0 for k in keys}
        all_scales[i] = {k: 1.0 for k in keys}
        all_intercepts[i] = 0.0
        continue
        
    X = subset_df[feature_cols]
    Y = subset_df['Target_是否已聽牌']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, Y_train)
    
    Y_pred = model.predict(X_test_scaled)
    w = model.coef_[0]
    
    print(f"資料筆數: {len(subset_df)} 筆")
    print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}\n")
    
    print(f"截距 (w0): {model.intercept_[0]:.4f}")
    print(f"巡數(a) 的權重: {w[0]:.4f}")
    print(f"花色集中度(c) 的權重: {w[1]:.4f}")
    print(f"摸切比例(d) 的權重: {w[2]:.4f}")
    print(f"連續摸切強度(e) 的權重: {w[3]:.4f}")
    print(f"摸切轉手切(f) 的權重: {w[4]:.4f}")
    print(f"中張第一張被打出(g) 的權重: {w[5]:.4f}")
    print(f"中張第二張被打出(h) 的權重: {w[6]:.4f}")
    print(f"中張第三張被打出(i) 的權重: {w[7]:.4f}")
    print(f"中張第四張被打出(j) 的權重: {w[8]:.4f}")
    print(f"字牌第一張被打出(k) 的權重: {w[9]:.4f}")
    print(f"字牌第二張被打出(l) 的權重: {w[10]:.4f}")
    print(f"字牌第三張被打出(m) 的權重: {w[11]:.4f}")
    print(f"字牌第四張被打出(n) 的權重: {w[12]:.4f}")
    print(f"邊張第一張被打出(o) 的權重: {w[13]:.4f}")
    print(f"邊張第二張被打出(p) 的權重: {w[14]:.4f}")
    print(f"邊張第三張被打出(q) 的權重: {w[15]:.4f}")
    print(f"邊張第四張被打出(r) 的權重: {w[16]:.4f}\n")
    
    all_weights[i] = {k: w[idx] for idx, k in enumerate(keys)}
    all_means[i] = {k: scaler.mean_[idx] for idx, k in enumerate(keys)}
    all_scales[i] = {k: scaler.scale_[idx] for idx, k in enumerate(keys)}
    all_intercepts[i] = model.intercept_[0]

def dict_to_string(name, d):
    s = f"{name} = {{\n"
    for meld, features in d.items():
        s += f"    {meld}: {{\n"
        s += f"        'a': {features['a']:.4f}, 'c': {features['c']:.4f}, 'd': {features['d']:.4f}, 'e': {features['e']:.4f}, 'f': {features['f']:.4f},\n"
        s += f"        'g': {features['g']:.4f}, 'h': {features['h']:.4f}, 'i': {features['i']:.4f}, 'j': {features['j']:.4f},\n"
        s += f"        'k': {features['k']:.4f}, 'l': {features['l']:.4f}, 'm': {features['m']:.4f}, 'n': {features['n']:.4f},\n"
        s += f"        'o': {features['o']:.4f}, 'p': {features['p']:.4f}, 'q': {features['q']:.4f}, 'r': {features['r']:.4f}\n"
        s += "    },\n"
    s += "}\n"
    return s

new_params_code = ""
new_params_code += dict_to_string("WEIGHTS_BY_MELD", all_weights)
new_params_code += dict_to_string("MEANS_BY_MELD", all_means)
new_params_code += dict_to_string("SCALES_BY_MELD", all_scales)
new_params_code += "\nINTERCEPTS_BY_MELD = {\n"
for k, v in all_intercepts.items():
    new_params_code += f"    {k}: {v:.4f},\n"
new_params_code += "}\n"

def update_tracking_script(target_file, new_code):
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = re.compile(r'(# AUTO-UPDATE-START.*?)(.*?)(\n# AUTO-UPDATE-END)', re.DOTALL)
        if not pattern.search(content):
            print(f"❌ 錯誤：在 {target_file} 中找不到 AUTO-UPDATE-START 或 AUTO-UPDATE-END 標籤，無法自動更新。")
            return

        new_content = pattern.sub(r'\1\n' + new_code + r'\3', content)

        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        print(f"✅ 成功跨資料夾更新參數至：\n   {target_file}")
    except FileNotFoundError:
        print(f"❌ 找不到目標檔案 {target_file}，請確認 trace 腳本的路徑是否正確。")

update_tracking_script(TARGET_SCRIPT, new_params_code)