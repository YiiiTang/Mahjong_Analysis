import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import tkinter as tk
from tkinter import filedialog
from analyze import getRound

matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'simhei']
matplotlib.rcParams['axes.unicode_minus'] = False

def investigate_game_with_winner():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("🔍 請選擇想要調查的「牌譜檔案 (.txt)」...")
    txt_file_path = filedialog.askopenfilename(
        title="請選擇要調查的牌譜檔案 (.txt)",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    
    if not txt_file_path:
        print("⚠️ 沒有選擇任何檔案")
        return
        
    target_game = os.path.basename(txt_file_path)

    try:
        states = getRound(txt_file_path)
        winner_loc = states.winnerLoc
        print(f"📁 已選擇牌譜: {target_game}")
        print(f"🏆 本局贏家位置: {winner_loc}")
    except Exception as e:
        print(f"❌ 解析牌譜失敗: {e}")
        return

    excel_file = "Batch_Linear_Tracking_Result.xlsx"
    try:
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        print(f"❌ 找不到 {excel_file}")
        return

    df_game = df[df['檔案名稱'] == target_game].copy()
    if df_game.empty:
        print(f"❌ Excel 中找不到該局資料。")
        return

    output_dir = "Game_Investigation"
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
    for player in players:
        df_player = df_game[df_game['玩家'] == player].sort_values(by='當下巡數')
        if df_player.empty: continue

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
        ax.set_xticks(df_player['當下巡數'])
        ax.legend(loc='center left', bbox_to_anchor=(1.10, 0.5), fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.9, 1])

        win_suffix = "_WIN" if is_winner else ""
        safe_name = target_game.replace(".txt", "")
        save_path = os.path.join(output_dir, f"{safe_name}_Player_{player}{win_suffix}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    print(f"\n✅ 調查完成！圖表已儲存至 {output_dir}")
    
    try:
        os.startfile(os.path.abspath(output_dir)) if os.name == 'nt' else os.system(f'open "{os.path.abspath(output_dir)}"')
    except: pass

if __name__ == "__main__":
    investigate_game_with_winner()