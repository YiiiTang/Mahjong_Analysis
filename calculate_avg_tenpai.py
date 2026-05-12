import pandas as pd
import os
from pathlib import Path

def calculate_tenpai_stats():

    file_path = Path("Training_Model/analyze_all_players_snapshots.xlsx")
    
    if not file_path.exists():
        print(f"❌ 找不到檔案：{file_path.resolve()}")
        print("請先確認您已經執行過 find_rule_all.py 產生資料。")
        return

    print("📂 正在讀取資料，請稍候...")
    df = pd.read_excel(file_path)

    required_cols = ['檔案名稱', '玩家位置', 'feat_a_巡數', 'Target_是否已聽牌']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ 檔案中缺少必要欄位：{col}")
            return

    tenpai_df = df[df['Target_是否已聽牌'] == 1]

    if tenpai_df.empty:
        print("⚠️ 資料中沒有任何聽牌紀錄！")
        return

    first_tenpai_turns = tenpai_df.groupby(['檔案名稱', '玩家位置'])['feat_a_巡數'].min()

    total_cases = len(first_tenpai_turns)
    avg_turn = first_tenpai_turns.mean()
    median_turn = first_tenpai_turns.median()
    min_turn = first_tenpai_turns.min()
    max_turn = first_tenpai_turns.max()

    print("\n" + "="*40)
    print("📊 聽牌巡數統計報告")
    print("="*40)
    print(f"總計有效聽牌人次 : {total_cases} 次")
    print(f"平均聽牌巡數     : {avg_turn:.2f} 巡")
    print(f"中位數聽牌巡數   : {median_turn:.0f} 巡")
    print(f"最快聽牌巡數     : {min_turn} 巡")
    print(f"最慢聽牌巡數     : {max_turn} 巡")
    print("="*40)

    print("\n📈 最常見的聽牌巡數分佈 (Top 5):")
    turn_counts = first_tenpai_turns.value_counts().head(5)
    for turn, count in turn_counts.items():
        percentage = (count / total_cases) * 100
        print(f"第 {turn:2d} 巡聽牌 : {count} 次 ({percentage:.1f}%)")

if __name__ == "__main__":
    calculate_tenpai_stats()