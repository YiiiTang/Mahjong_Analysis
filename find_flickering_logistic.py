import pandas as pd
import os

# ==========================================
# 邏輯迴歸 (Logistic) 設定區
# ==========================================
INPUT_EXCEL = "Batch_Logistic_Tracking_Result.xlsx"  
OUTPUT_EXCEL = "Flickering_Cases_Logistic.xlsx"
THRESHOLD = 0.5
SCORE_COL = '預測聽牌機率'

# 特徵欄位清單
feature_cols = [
    'feat_a_巡數', 'feat_b_吃碰數', 'feat_c_花色集中度', 'feat_d_中張比例(3 ~ 7)', 
    'feat_e_邊張比例(1、2、8、9)', 'feat_f_字牌比例', 'feat_g_摸切比例', 
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

def find_flickering_cases():
    print(f"📂 [Logistic] 正在讀取資料: {INPUT_EXCEL}...")
    try:
        df = pd.read_excel(INPUT_EXCEL)
    except FileNotFoundError:
        print(f"❌ 找不到檔案 {INPUT_EXCEL}，請確認是否已經產生過追蹤結果。")
        return

    df = df.sort_values(by=['檔案名稱', '玩家', 'Step_ID']).reset_index(drop=True)

    print("🔍 [Logistic] 正在尋找「上一巡判斷聽牌，下一巡判斷未聽牌」的案例...")
    
    flickering_results = []
    grouped = df.groupby(['檔案名稱', '玩家'])

    for (file_name, player), group_df in grouped:
        group_df = group_df.reset_index(drop=True)
        
        group_df['is_predicted_tenpai'] = (group_df[SCORE_COL] >= THRESHOLD).astype(int)
        group_df['prev_predicted_tenpai'] = group_df['is_predicted_tenpai'].shift(1)
        
        flickering_mask = (group_df['prev_predicted_tenpai'] == 1) & (group_df['is_predicted_tenpai'] == 0)
        flicker_indices = group_df.index[flickering_mask]
        
        for idx in flicker_indices:
            current_row = group_df.iloc[idx]
            prev_row = group_df.iloc[idx - 1]

            case_data = {
                '檔案名稱': file_name,
                '玩家': player,
                '發生掉機率的 Step_ID': current_row['Step_ID'],
                '上一巡 (Step_ID)': prev_row['Step_ID'],
                
                '上一巡捨牌': f"{prev_row.get('丟棄牌', '')} ({prev_row['動作']})",
                '當下捨牌': f"{current_row.get('丟棄牌', '')} ({current_row['動作']})",
                
                '上一巡預測機率': round(prev_row[SCORE_COL], 4),
                '當下預測機率': round(current_row[SCORE_COL], 4),
                '機率跌幅': round(prev_row[SCORE_COL] - current_row[SCORE_COL], 4),
                
                '上一巡實際向聽': prev_row['實際向聽數'],
                '當下實際向聽': current_row['實際向聽數']
            }

            for feat in feature_cols:
                case_data[f'上一巡_{feat}'] = prev_row.get(feat, None)
                case_data[f'當下_{feat}'] = current_row.get(feat, None)

            flickering_results.append(case_data)

    if not flickering_results:
        print("✅ [Logistic] 沒有找到任何「判斷倒退」的案例。模型表現穩定。")
        return

    result_df = pd.DataFrame(flickering_results)
    result_df = result_df.sort_values(by='機率跌幅', ascending=False)
    result_df.to_excel(OUTPUT_EXCEL, index=False)
    
    print(f"⚠️ [Logistic] 總共找到 {len(flickering_results)} 筆判斷倒退的案例。")
    print(f"📁 已儲存詳細結果至: {OUTPUT_EXCEL}\n")
    
    print("--- 📉 [Logistic] 跌幅最嚴重的 Top 3 案例預覽 ---")
    for i, row in result_df.head(3).iterrows():
        print(f"檔案: {row['檔案名稱']} | 玩家: {row['玩家']}")
        print(f"  捨牌: {row['上一巡捨牌']} -> {row['當下捨牌']}")
        print(f"  機率: {row['上一巡預測機率']} -> {row['當下預測機率']} (跌幅: {row['機率跌幅']})\n")

if __name__ == "__main__":
    find_flickering_cases()