import pandas as pd
import os

# ==========================================
# 設定區
# ==========================================
LINEAR_EXCEL = "Flickering_Cases_Linear.xlsx"
LOGISTIC_EXCEL = "Flickering_Cases_Logistic.xlsx"
OUTPUT_EXCEL = "Common_Flickering_Cases.xlsx"
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

    common_df = pd.merge(
        df_linear, 
        df_logistic, 
        on=['檔案名稱', '玩家', 'Target_Step_ID', 'Prev_Step_ID', '上一巡捨牌', '當下捨牌', '上一巡實際向聽', '當下實際向聽', '上一巡巡數', '當下巡數', '上一巡吃碰數', '當下吃碰數'],
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
        '上一巡預測機率', '當下預測機率',
        '上一巡巡數', '當下巡數',
        '上一巡吃碰數', '當下吃碰數'
    ]

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