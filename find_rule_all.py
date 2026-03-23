from analyze import getRound
from fileProcess import RoundState, States, playerState
from typing import List
from datetime import datetime
import pandas as pd
from pathlib import Path

def list_files(directory_path):
    p = Path(directory_path)
    files = [entry for entry in p.iterdir() if entry.is_file()]
    return files

def save_file(results):
    # 更改了輸出的檔名，以符合新的資料內容
    output_path = Path("analyze_all_players_snapshots.xlsx")
    if not results:
        print("沒有符合條件的資料可供儲存。")
        return
        
    output_data = pd.concat(results, ignore_index=True)
    try:
        output_data.to_excel(output_path, index=False)
        print(f"資料已成功匯出至 {output_path.resolve()}")
    except PermissionError:
        fallback_path = output_path.with_name(
            f"{output_path.stem}_{datetime.now():%Y%m%d_%H%M%S}{output_path.suffix}"
        )
        output_data.to_excel(fallback_path, index=False)
        print(f"檔案 {output_path} 正在使用中，已改為匯出至 {fallback_path.resolve()}")

if __name__ == "__main__":
    results = []
    
    # 請將此處替換為您實際的牌譜資料夾路徑
    folder_path = "D:\\Project\\Mahjong\\Board\\TAAI_MJ_2025\\3"

    for file in list_files(folder_path):
        file_path_str = str(file) 
        print(f"Processing: {file.name}")
        try:
            states = getRound(file_path_str)
        except Exception as e:
            print(f"讀取 {file.name} 時發生錯誤: {e}")
            continue
        
        # 取得贏家位置，如果沒有贏家（荒牌），winner_loc 會是 None 或空字串
        winner_loc = states.winnerLoc
        
        # 建立一個字典，用來分別追蹤 4 個玩家 (E, S, W, N) 的狀態
        players = ['E', 'S', 'W', 'N']
        player_stats = {
            loc: {
                'turn_count': 0,
                'meld_count': 0,
                'total_discard': 0,
                'discard_3_to_7': 0,
                'discard_wan': 0,
                'discard_tong': 0,
                'discard_tiao': 0,
                'discard_zi': 0
            } for loc in players
        }
        
        game_snapshots = []
        
        # 沿著時間線逐一檢視每一個動作
        for j in range(1, len(states.state)):
            step_data = states.state[j].stepData
            if len(step_data) < 3:
                continue
                
            actor_loc = step_data[1]
            action = step_data[2]
            
            # 確保動作是由四個玩家之一發出的
            if actor_loc not in players:
                continue
            
            # 取得當前動作玩家的專屬統計資料
            stats = player_stats[actor_loc]
                
            # 計算巡數 (摸牌次數)
            if action == 'M':
                stats['turn_count'] += 1
                
            # 計算副露數
            elif action in ('P', 'E', 'EM', 'EL', 'ER', 'UG', 'HG', 'G'):
                stats['meld_count'] += 1
                
            # 擷取快照：只有在「打牌」或「摸切」時才生成一筆資料
            elif action in ('HD', 'MD'):
                stats['total_discard'] += 1
                tile_str = step_data[3] if len(step_data) > 3 else ""
                
                # 統計當下這張棄牌的屬性
                if tile_str.isdigit():
                    card_num = int(tile_str)
                    suit = card_num // 100
                    face_value = (card_num // 10) % 10
                    
                    # 判斷是否為 3~7 的中張
                    if suit in (1, 2, 3) and 3 <= face_value <= 7:
                        stats['discard_3_to_7'] += 1
                        
                    # 分類花色與字牌
                    if suit == 1:
                        stats['discard_wan'] += 1
                    elif suit == 2:
                        stats['discard_tong'] += 1
                    elif suit == 3:
                        stats['discard_tiao'] += 1
                    elif suit == 4:
                        stats['discard_zi'] += 1
                
                # 取得該玩家打完這張牌之後，當下的聽牌狀態 (Y)
                # shantenCount: 0 為聽牌，-1 為胡牌
                player_state = states.get_player(states.state[j], actor_loc)
                is_tenpai = 1 if player_state.shantenCount <= 0 else 0
                
                # 計算當下累積的各項特徵比例 (X)
                td = stats['total_discard']
                mid_tile_ratio = stats['discard_3_to_7'] / td if td > 0 else 0.0
                zi_ratio = stats['discard_zi'] / td if td > 0 else 0.0
                
                # 計算「最多被丟的花色比例 (排除字牌)」
                max_suit_count = max(stats['discard_wan'], stats['discard_tong'], stats['discard_tiao'])
                total_suit_discard = stats['discard_wan'] + stats['discard_tong'] + stats['discard_tiao']
                max_suit_ratio = max_suit_count / total_suit_discard if total_suit_discard > 0 else 0.0
                
                # 判斷這個玩家是否為該局最終贏家
                is_winner_eventually = 1 if actor_loc == winner_loc else 0
                
                # 將這一切打包成一筆快照資料
                snapshot = {
                    '檔案名稱': file.name,
                    '玩家位置': actor_loc,
                    '最終是否獲勝': is_winner_eventually,
                    'Step_ID': j,
                    '動作類型': action,
                    '丟棄的牌': tile_str,
                    '當下巡數': stats['turn_count'],
                    '當下副露數': stats['meld_count'],
                    '累積丟牌數': td,
                    '當下3至7比例': round(mid_tile_ratio, 4),
                    '當下丟字比例': round(zi_ratio, 4),
                    '最多單一花色丟棄比例': round(max_suit_ratio, 4),
                    '是否已聽牌 (Y)': is_tenpai
                }
                game_snapshots.append(snapshot)
                
        # 將這局收集到的所有快照加入總表中
        if game_snapshots:
            results.append(pd.DataFrame(game_snapshots))
            
    # 執行存檔
    if results:
        save_file(results)