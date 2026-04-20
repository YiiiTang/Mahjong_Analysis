import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import glob
import tkinter as tk
from tkinter import filedialog
from analyze import getRound

matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'simhei']
matplotlib.rcParams['axes.unicode_minus'] = False

def batch_investigate_games():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("🔍 請選擇包含「牌譜檔案 (.txt)」的資料夾...")
    folder_path = filedialog.askdirectory(title="請選擇牌譜檔案所在的資料夾")
    
    if not folder_path:
        print("⚠️ 沒有選擇任何資料夾")
        return
        
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not txt_files:
        print("⚠️ 該資料夾中沒有找到 .txt 檔案")
        return
        
    print(f"📁 找到 {len(txt_files)} 個牌譜檔案，準備載入 Excel 資料...")

    excel_file = "Batch_Linear_Tracking_Result.xlsx"
    try:
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        print(f"❌ 找不到 {excel_file}，請確認檔案與程式在同一目錄下。")
        return

    output_dir = "Batch_Game_Investigation"
    os.makedirs(output_dir, exist_ok=True)

    plot_styles = {
        '預測聽牌分數': {'color': 'black', 'marker': 'x', 'lw': 3, 'ls': '--'},
        '中張比例': {'color': 'tab:blue', 'marker': 'o', 'lw': 1.5, 'ls': '-'},
        '花色集中度': {'color': 'tab:orange', 'marker': 's', 'lw': 1.5, 'ls': '-'},
        '字牌比例': {'color': 'tab:green', 'marker': '^', 'lw': 1.5, 'ls': '-'},
        '摸切比例': {'color': 'tab:red', 'marker': 'D', 'lw': 1.5, 'ls': '-'},
        '連續摸切強度': {'color': 'tab:purple', 'marker': 'v', 'lw': 1.5, 'ls': '-'},
        '摸切轉手切': {'color': 'tab:brown', 'marker': 'p', 'lw': 1.5, 'ls': '-'},
        '當下副露': {'color': 'tab:pink', 'marker': '*', 'lw': 1.5, 'ls': '-'}
    }

    players = ['E', 'S', 'W', 'N']

    for i, txt_file_path in enumerate(txt_files, 1):
        target_game = os.path.basename(txt_file_path)
        print(f"[{i}/{len(txt_files)}] ⏳ 正在處理: {target_game}...")

        try:
            states = getRound(txt_file_path)
            winner_loc = states.winnerLoc
        except Exception as e:
            print(f"  ❌ 解析牌譜失敗，已跳過: {e}")
            continue

        df_game = df[df['檔案名稱'] == target_game].copy()
        if df_game.empty:
            print(f"  ⚠️ Excel 中找不到該局資料，已跳過。")
            continue

        for player in players:
            df_player = df_game[df_game['玩家'] == player].sort_values(by='當下巡數')
            if df_player.empty: 
                continue

            plt.figure(figsize=(12, 6))
            ax = plt.gca()
            
            for feature, style in plot_styles.items():
                if feature in df_player.columns:
                    ax.plot(df_player['當下巡數'], df_player[feature], 
                             label=feature, color=style['color'], 
                             marker=style['marker'], linewidth=style['lw'], 
                             linestyle=style['ls'], alpha=0.85)

            if '實際向聽數' in df_player.columns:
                ax2 = ax.twinx()
                ax2.plot(df_player['當下巡數'], df_player['實際向聽數'], color='gray', linestyle=':', marker='.', alpha=0.6)
                ax2.set_ylabel('實際向聽數', color='gray')
                ax2.invert_yaxis()

            is_winner = (player == winner_loc)
            winner_text = " [WINNER]" if is_winner else ""
            
            plt.title(f'單局調查：全特徵與預測紀錄{winner_text}\n檔案: {target_game} | 玩家: {player}', 
                      fontsize=16, color='red' if is_winner else 'black', pad=15)
            
            ax.set_xlabel('當下巡數', fontsize=12)
            ax.set_ylabel('特徵值 / 預測分數', fontsize=12)
            
            ax.set_xticks(df_player['當下巡數'].astype(int))
            
            ax.legend(loc='center left', bbox_to_anchor=(1.10, 0.5), fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout(rect=[0, 0, 0.9, 1])

            win_suffix = "_WIN" if is_winner else ""
            safe_name = target_game.replace(".txt", "")
            save_path = os.path.join(output_dir, f"{safe_name}_Player_{player}{win_suffix}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            
    print(f"\n✅ 批次處理完成！所有圖片已儲存至 {output_dir} 資料夾")
    
    try:
        os.startfile(os.path.abspath(output_dir)) if os.name == 'nt' else os.system(f'open "{os.path.abspath(output_dir)}"')
    except: 
        pass

if __name__ == "__main__":
    batch_investigate_games()