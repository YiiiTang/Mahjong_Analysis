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

def is_tenpai(ref, states, loc):
    p = states.get_player(ref, loc)
    sh = p.shantenCount
    if sh == -1:
        return True
    return sh == 0 and (len(p.tiles) % 3 == 1)

def first_tenpai_step(states, loc_char):
    idx = states.player_index_from_loc(loc_char)
    for ref in states.state:
        p = ref.player[idx]
        if p.shantenCount == 0 and len(p.tiles) in (16, 13, 10, 7):
            return ref.stepId
    return -2

def not_winner(winner:str) -> List[str]:
      return [loc for loc in ['W', 'S', 'N', 'E'] if loc != winner]

def save_file(results):
      output_path = Path("analyze_mid_3_to_7.xlsx")
      output_data = pd.concat(results, ignore_index=True)
      try:
            output_data.to_excel(output_path, index=False)
            print(f"Exported to {output_path.resolve()}")
      except PermissionError:
            fallback_path = output_path.with_name(
                  f"{output_path.stem}_{datetime.now():%Y%m%d_%H%M%S}{output_path.suffix}"
            )
            output_data.to_excel(fallback_path, index=False)
            print(f"{output_path} is in use. Exported to {fallback_path.resolve()}")

if __name__ == "__main__":
    results = []

    for file in list_files("D:\\Project\\Mahjong\\Board\\TAAI_MJ_2025\\3"):
        file_path_str = str(file) 
        print(f"\nProcessing: {file_path_str}")
        try:
            states = getRound(file_path_str)
        except Exception as e:
            print(f"Error reading {file_path_str}: {e}")
            continue
        
        count_win_mountain = states.state[states.steps_count()].mountainCount
        
        for player_loc in ['E', 'S', 'W', 'N']:
            step_table = {"M":0, "HD":0, "MD":0, "P":0, "E":0, "EM":0, "EL":0, "ER":0, "SM":0, "UG":0, "HG":0, "G":0, "H":0}
            
            ting_pai_step = first_tenpai_step(states, player_loc)
            is_ting_pai = (ting_pai_step != -2)
            is_winner = (player_loc == states.winnerLoc)
            
            for j in range(1, len(states.state)):
                if states.state[j].stepData[1] == player_loc:
                    action = states.state[j].stepData[2]
                    if action in step_table:
                        step_table[action] += 1
                    else:
                        step_table[action] = 1
            
            count_ting_mountain = None
            diff_mountain = None
            
            # 聽前各式統計變數
            mo_before_ting = None
            hd_before_ting = None
            md_before_ting = None
            calls_before_ting = None
            count_discard_3_to_7 = None  
            count_discard_total_before_ting = None 
            ratio_discard_3_to_7 = None 
            
            if is_ting_pai:
                mo_before_ting = 0
                hd_before_ting = 0
                md_before_ting = 0
                calls_before_ting = 0
                
                count_discard_3_to_7 = 0
                count_discard_total_before_ting = 0
                
                for i in range(1, ting_pai_step):
                    step_data = states.state[i].stepData
                    if step_data[1] == player_loc:
                        action = step_data[2]
                        
                        # 紀錄聽前所有行為次數
                        if action == 'M':
                            mo_before_ting += 1
                        elif action == 'HD':
                            hd_before_ting += 1
                            count_discard_total_before_ting += 1 
                        elif action == 'MD':
                            md_before_ting += 1
                            count_discard_total_before_ting += 1 
                        elif action in ('P', 'E', 'EM', 'EL', 'ER', 'UG', 'HG', 'G'):
                            calls_before_ting += 1
                            
                        # 檢查 3~7 的棄牌
                        if action in ('HD', 'MD'):
                            tile_str = step_data[3]
                            if tile_str.isdigit():
                                card_num = int(tile_str)
                                suit = card_num // 100
                                face_value = (card_num // 10) % 10
                                
                                if suit in (1, 2, 3) and 3 <= face_value <= 7:
                                    count_discard_3_to_7 += 1
                                    
                count_ting_mountain = states.state[ting_pai_step].mountainCount
                
                if is_winner:
                    diff_mountain = count_ting_mountain - count_win_mountain
                else:
                    diff_mountain = None
                
                if count_discard_total_before_ting > 0:
                    ratio_discard_3_to_7 = count_discard_3_to_7 / count_discard_total_before_ting
                else:
                    ratio_discard_3_to_7 = 0.0
                    
            else:
                mo_before_ting = None
                hd_before_ting = None
                md_before_ting = None
                calls_before_ting = None
                
                count_discard_3_to_7 = None
                count_discard_total_before_ting = None
                ratio_discard_3_to_7 = None
                diff_mountain = None
            
            pd_data = pd.DataFrame([{
                'filename': file_path_str,
                'winner': states.winnerLoc,
                'player': player_loc,
                'is_winner': is_winner,      
                '是否聽牌': is_ting_pai,       
                '總摸牌次數': step_table["M"],
                '總打牌次數': step_table["HD"],
                '總摸切次數': step_table["MD"],
                '總吃碰槓次數': step_table.get("P", 0) + step_table.get("E", 0) + step_table.get("EM", 0) + step_table.get("EL", 0) + step_table.get("ER", 0) + step_table.get("UG", 0) + step_table.get("HG", 0) + step_table.get("G", 0),
                '聽前摸牌次數': mo_before_ting,
                '聽前打牌次數': hd_before_ting,
                '聽前摸切次數': md_before_ting,
                '聽前吃碰槓次數': calls_before_ting,
                '聽前丟棄3至7': count_discard_3_to_7,
                '聽前總丟牌數': count_discard_total_before_ting,
                '聽前丟棄3至7比例': ratio_discard_3_to_7,
                '聽牌時排山剩餘張': count_ting_mountain,
                '胡牌時排山剩餘張': count_win_mountain if is_winner else None,
                '胡牌-聽牌剩餘張': diff_mountain
            }])
            
            results.append(pd_data)
    
    if results:
        save_file(results)