import pandas as pd
import os

# ==========================================
# 設定區
# ==========================================
LINEAR_EXCEL = "Flickering_Cases_Linear.xlsx"
LOGISTIC_EXCEL = "Flickering_Cases_Logistic.xlsx"
OUTPUT_EXCEL = "Common_Flickering_Cases.xlsx"

# 特徵欄位清單 (已移除 feat_g_摸切比例)
feature_cols = [
    'feat_a_巡數', 'feat_b_吃碰數', 'feat_c_花色集中度', 'feat_d_中張比例(3 ~ 7)', 
    'feat_e_邊張比例(1、2、8、9)', 'feat_f_字牌比例', 
    'feat_h_目前連續摸切', 'feat_i_摸切轉手切', 'feat_j_摸切轉手切次數',
    'feat_z1_第9巡起最近連續摸切次數', 'feat_z2_第9巡起前兩巡連續摸切',
    'feat_k_中張第一張被打出', 'feat_l_中張第二張被打出', 'feat_m_中張第三張被打出', 
    'feat_n_中張第四張被打出', 'feat_o_字牌第一張被打出', 'feat_p_字牌第二張被打出', 
    'feat_q_字牌第三張被打出', 'feat_r_字牌第四張被打出', 'feat_s_邊張(1、9)第一張被打出', 
    'feat_t_邊張(1、9)第二張被打出', 'feat_u_邊張(1、9)第三張被打出', 
    'feat_v_邊張(1、9)第四張被打出', 'feat_w_邊張(2、8)第一張被打出', 
    'feat_x_邊張(2、8)第二張被打出', 'feat_y_邊張(2、8)第三張被打出', 
    'feat_z_邊張(2、8)第四張被打出'
]
# ==========================================

def find_common_flickering():
    print(f"📂 正在讀取資料...")

    if not os.path.exists(LINEAR_EXCEL):
        print(f"❌ 找不到檔案: {LINEAR_EXCEL}")
        return
    if not os.path.exists(LOGISTIC_EXCEL):
        print(f"❌ 找不到檔案: {LOGISTIC_EXCEL}")
        return

    df_linear = pd.read_excel(LINEAR_EXCEL)
    df_logistic = pd.read_excel(LOGISTIC_EXCEL)
    
    if df_linear.empty:
        print("⚠️ 線性迴歸的檔案沒有記錄到任何倒退案例。")
        return
    if df_logistic.empty:
        print("⚠️ 邏輯迴歸的檔案沒有記錄到任何倒退案例。")
        return

    df_linear = df_linear.rename(columns={'發生掉分的 Step_ID': 'Target_Step_ID', '上一巡 (Step_ID)': 'Prev_Step_ID'})
    df_logistic = df_logistic.rename(columns={'發生掉機率的 Step_ID': 'Target_Step_ID', '上一巡 (Step_ID)': 'Prev_Step_ID'})

    print("🔍 正在尋找兩個模型「共同發生判斷倒退」的交集案例...")

    merge_on_cols = [
        '檔案名稱', '玩家', 'Target_Step_ID', 'Prev_Step_ID', 
        '上一巡捨牌', '當下捨牌', '上一巡實際向聽', '當下實際向聽'
    ]
    for feat in feature_cols:
        merge_on_cols.extend([f'上一巡_{feat}', f'當下_{feat}'])

    common_df = pd.merge(
        df_linear, 
        df_logistic, 
        on=merge_on_cols,
        suffixes=('_Linear', '_Logistic') 
    )

    if common_df.empty:
        print("✅ 找完了！沒有發現任何「兩個模型同時發生判斷倒退」的盤面。")
        return
        
    common_df['綜合跌幅'] = common_df['分數跌幅'] + common_df['機率跌幅']
    common_df = common_df.sort_values(by='綜合跌幅', ascending=False)

    cols_order = [
        '檔案名稱', '玩家', 'Prev_Step_ID', 'Target_Step_ID', 
        '上一巡捨牌', '當下捨牌', 
        '上一巡實際向聽', '當下實際向聽',
        '分數跌幅', '機率跌幅', '綜合跌幅',
        '上一巡預測分數', '當下預測分數', 
        '上一巡預測機率', '當下預測機率'
    ]

    for feat in feature_cols:
        cols_order.extend([f'上一巡_{feat}', f'當下_{feat}'])

    common_df = common_df[[col for col in cols_order if col in common_df.columns]]

    common_df.to_excel(OUTPUT_EXCEL, index=False)
    
    print(f"⚠️ 總共找到 {len(common_df)} 筆「雙模型共同判斷倒退」的案例！")
    print(f"📁 已儲存詳細結果至: {OUTPUT_EXCEL}\n")

    print("--- 📉 綜合跌幅最嚴重的 Top 5 案例預覽 ---")
    for i, row in common_df.head(5).iterrows():
        print(f"檔案: {row['檔案名稱']} | 玩家: {row['玩家']}")
        print(f"  捨牌: {row['上一巡捨牌']} -> {row['當下捨牌']}")
        print(f"  [Linear] 分數變化: {row['上一巡預測分數']} -> {row['當下預測分數']} (跌幅: {row['分數跌幅']})")
        print(f"  [Logistic] 機率變化: {row['上一巡預測機率']} -> {row['當下預測機率']} (跌幅: {row['機率跌幅']})")
        print(f"  綜合跌幅: {row['綜合跌幅']:.4f}\n")

if __name__ == "__main__":
    find_common_flickering()