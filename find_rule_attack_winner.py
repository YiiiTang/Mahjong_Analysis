from analyze import getRound
from fileProcess import RoundState, States, playerState
from mjanalyzer_local import state_cards_to_tile_ids, analyze_tiles
from collections import Counter
from typing import List
from datetime import datetime
import pandas as pd
from pathlib import Path

def list_files(directory_path):
    p = Path(directory_path)
    return [entry for entry in p.iterdir() if entry.is_file()]

def save_file(results):
    output_path = Path("analyze_attack_winner_only.xlsx")
    if not results:
        print("沒有符合條件的資料可供儲存。")
        return
        
    output_data = pd.concat(results, ignore_index=True)
    try:
        output_data.to_excel(output_path, index=False)
        print(f"贏家進攻資料已成功匯出至 {output_path.resolve()}")
    except PermissionError:
        fallback_path = output_path.with_name(
            f"{output_path.stem}_{datetime.now():%Y%m%d_%H%M%S}{output_path.suffix}"
        )
        output_data.to_excel(fallback_path, index=False)
        print(f"檔案正在使用中，已改為匯出至 {fallback_path.resolve()}")

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
        
        winner_total_turns = 0
        for state in states.state:
            if len(state.stepData) >= 3 and state.stepData[1] == winner_loc and state.stepData[2] == 'M':
                winner_total_turns += 1
        
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
            elif action in ('P', 'E', 'EM', 'EL', 'ER', 'UG', 'HG', 'G'):
                meld_count += 1
            
            if action in ('HD', 'MD'):
                player_state = states.get_player(states.state[j-1], winner_loc) 
                
                try:
                    hand_tiles_ids = state_cards_to_tile_ids(player_state.tiles)
                    dead_tiles_ids = state_cards_to_tile_ids(states.state[j].abandonTiles)
                    dead_counts = Counter(dead_tiles_ids)
                    
                    analysis_result = analyze_tiles(hand_tiles_ids, dead_counts, validate=False)
                    shanten = analysis_result.shanten
                    ukeire = analysis_result.summaryStats.totalEffective
                except Exception as e:
                    continue

                remaining_turns = winner_total_turns - turn_count
                
                snapshot = {
                    '檔案名稱': file.name,
                    '贏家位置': winner_loc,
                    'Step_ID': j,
                    '當下巡數': turn_count,
                    '當下副露數': meld_count,
                    '向聽數 (Shanten)': shanten,
                    '進張數 (Ukeire)': ukeire,
                    '剩餘牌山': states.state[j].mountainCount,
                    'Target_是否胡牌(Win)': 1 if remaining_turns == 0 else 0,
                    'Target_剩餘巡數': remaining_turns
                }
                game_snapshots.append(snapshot)
                
        if game_snapshots:
            results.append(pd.DataFrame(game_snapshots))
            
    if results:
        save_file(results)