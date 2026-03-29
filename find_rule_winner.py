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
    output_path = Path("analyze_winner_snapshots.xlsx")
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
    
    folder_path = "D:\\Project\\Mahjong\\Board\\TAAI_MJ_2025\\3"

    for file in list_files(folder_path):
        file_path_str = str(file) 
        print(f"Processing: {file.name}")
        try:
            states = getRound(file_path_str)
        except Exception as e:
            print(f"讀取 {file.name} 時發生錯誤: {e}")
            continue
        
        winner_loc = states.winnerLoc

        if not winner_loc:
            continue
            
        turn_count = 0
        meld_count = 0
        total_discard = 0
        total_exposed = 0
        
        discard_3_to_7 = 0
        discard_wan = 0
        discard_tong = 0
        discard_tiao = 0
        discard_zi = 0
        
        game_snapshots = []
        
        for j in range(1, len(states.state)):
            step_data = states.state[j].stepData
            if len(step_data) < 3:
                continue
                
            actor_loc = step_data[1]
            action = step_data[2]
            
            if actor_loc != winner_loc:
                continue
                
            if action == 'M':
                turn_count += 1
                
            elif action in ('P', 'E', 'EM', 'EL', 'ER', 'UG', 'HG', 'G', 'HD', 'MD'):
                is_discard = action in ('HD', 'MD')

                if not is_discard:
                    meld_count += 1
                    tiles_to_process = step_data[3:]
                else:
                    total_discard += 1
                    tiles_to_process = [step_data[3]] if len(step_data) > 3 else []
                
                for tile_str in tiles_to_process:
                    if tile_str.isdigit():
                        total_exposed += 1
                        card_num = int(tile_str)
                        suit = card_num // 100
                        face_value = (card_num // 10) % 10
                        
                        if suit in (1, 2, 3) and 3 <= face_value <= 7:
                            discard_3_to_7 += 1
                            
                        if suit == 1:
                            discard_wan += 1
                        elif suit == 2:
                            discard_tong += 1
                        elif suit == 3:
                            discard_tiao += 1
                        elif suit == 4:
                            discard_zi += 1
                
                if is_discard:
                    tile_str = step_data[3] if len(step_data) > 3 else ""
                    player_state = states.get_player(states.state[j], winner_loc)
                    is_tenpai = 1 if player_state.shantenCount <= 0 else 0
                
                    mid_tile_ratio = discard_3_to_7 / total_exposed if total_exposed > 0 else 0.0
                    wan_ratio = discard_wan / total_exposed if total_exposed > 0 else 0.0
                    tong_ratio = discard_tong / total_exposed if total_exposed > 0 else 0.0
                    tiao_ratio = discard_tiao / total_exposed if total_exposed > 0 else 0.0
                    zi_ratio = discard_zi / total_exposed if total_exposed > 0 else 0.0
                    
                    max_suit_count = max(discard_wan, discard_tong, discard_tiao)
                    total_suit_discard = discard_wan + discard_tong + discard_tiao
                    max_suit_ratio = max_suit_count / total_suit_discard if total_suit_discard > 0 else 0.0
                    
                    snapshot = {
                        '檔案名稱': file.name,
                        '贏家位置': winner_loc,
                        'Step_ID': j,
                        '動作類型': action,
                        '丟棄的牌': tile_str,
                        '當下巡數': turn_count,
                        '當下副露數': meld_count,
                        '累積丟牌數': total_discard,
                        '累積曝光牌數': total_exposed,
                        '當下3至7比例': round(mid_tile_ratio, 4),
                        '當下丟字比例': round(zi_ratio, 4),
                        '最多單一花色丟棄比例': round(max_suit_ratio, 4),
                        '是否已聽牌 (Y)': is_tenpai
                    }
                    game_snapshots.append(snapshot)
                
        if game_snapshots:
            results.append(pd.DataFrame(game_snapshots))
            
    if results:
        save_file(results)