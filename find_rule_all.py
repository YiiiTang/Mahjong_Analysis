from analyze import getRound
from fileProcess import RoundState, States, playerState
from typing import List
from datetime import datetime
import pandas as pd
from pathlib import Path

def list_files(directory_path):
    p = Path(directory_path)
    return [entry for entry in p.iterdir() if entry.is_file()]

def save_file(results):
    target_dir = Path("Training_Model")
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "analyze_all_players_snapshots.xlsx"
    
    if not results:
        print("沒有符合條件的資料可供儲存。")
        return
        
    output_data = pd.concat(results, ignore_index=True)
    try:
        output_data.to_excel(output_path, index=False)
        print(f"進攻資料已成功匯出至 {output_path.resolve()}")
    except PermissionError:
        fallback_path = target_dir / f"{output_path.stem}_{datetime.now():%Y%m%d_%H%M%S}{output_path.suffix}"
        output_data.to_excel(fallback_path, index=False)
        print(f"檔案正在使用中，已改為匯出至 {fallback_path.resolve()}")

if __name__ == "__main__":
    results = []
    folder_path = "D:\\Project\\Mahjong\\Board\\Train"

    for file in list_files(folder_path):
        print(f"Processing: {file.name}")
        try:
            states = getRound(str(file))
        except Exception as e:
            print(f"讀取 {file.name} 時發生錯誤: {e}")
            continue
        
        winner_loc = states.winnerLoc
        players = ['E', 'S', 'W', 'N']

        player_stats = {
            loc: {
                'turn_count': 0,
                'meld_count': 0,
                'total_discard': 0,

                'discard_3_to_7': 0,
                'discard_1289': 0,
                'discard_wan': 0,
                'discard_tong': 0,
                'discard_tiao': 0,
                'discard_zi': 0,

                'moqie_count': 0,
                'current_continuous_moqie': 0,
                'max_continuous_moqie': 0,
                'moqie_to_shouqie_count': 0,

                'last_discard_type': None
            } for loc in players
        }

        game_snapshots = []
        
        global_discard_count = {}

        for j in range(1, len(states.state)):
            step_data = states.state[j].stepData
            if len(step_data) < 3:
                continue

            actor_loc = step_data[1]
            action = step_data[2]

            if actor_loc not in players:
                continue

            stats = player_stats[actor_loc]

            if action in ('P', 'E', 'EM', 'EL', 'ER', 'UG', 'HG', 'G', 'HD', 'MD'):
                is_discard = action in ('HD', 'MD')

                if not is_discard:
                    stats['meld_count'] += 1

                else:
                    stats['turn_count'] += 1
                    stats['total_discard'] += 1

                    tile_str = step_data[3] if len(step_data) > 3 else ""
                    
                    discard_nth = 0
                    tile_name = "" 

                    if tile_str and tile_str.isdigit():
                        card_num = int(tile_str)
                        suit = card_num // 100
                        face_value = (card_num // 10) % 10
                        
                        if suit == 4:
                            honor_dict = {1: '東', 2: '南', 3: '西', 4: '北', 5: '白', 6: '發', 7: '中'}
                            tile_name = honor_dict.get(face_value, f"{face_value}字")
                        else:
                            type_dict = {0: '花', 1: '萬', 2: '筒', 3: '條'}
                            if suit in type_dict:
                                tile_name = f"{face_value}{type_dict[suit]}"
                        
                        if tile_name:
                            global_discard_count[tile_name] = global_discard_count.get(tile_name, 0) + 1
                            discard_nth = global_discard_count[tile_name]

                    if action == 'MD':
                        stats['moqie_count'] += 1
                        stats['current_continuous_moqie'] += 1

                        if stats['current_continuous_moqie'] > stats['max_continuous_moqie']:
                            stats['max_continuous_moqie'] = stats['current_continuous_moqie']

                    elif action == 'HD':
                        if stats['current_continuous_moqie'] >= 2:
                            stats['moqie_to_shouqie_count'] += 1

                        stats['current_continuous_moqie'] = 0

                    stats['last_discard_type'] = action
                    
                    is_zhong = False
                    is_zi = False
                    is_bian_19 = False
                    is_bian_28 = False

                    if tile_str.isdigit():
                        card_num = int(tile_str)
                        suit = card_num // 100
                        face_value = (card_num // 10) % 10

                        if suit in (1, 2, 3):
                            if 3 <= face_value <= 7:
                                stats['discard_3_to_7'] += 1
                                is_zhong = True
                            elif face_value in (1, 9):
                                stats['discard_1289'] += 1
                                is_bian_19 = True
                            elif face_value in (2, 8):
                                stats['discard_1289'] += 1
                                is_bian_28 = True
                        elif suit == 4:
                            stats['discard_zi'] += 1
                            is_zi = True

                        if suit == 1:
                            stats['discard_wan'] += 1
                        elif suit == 2:
                            stats['discard_tong'] += 1
                        elif suit == 3:
                            stats['discard_tiao'] += 1

                    td = stats['total_discard']
                    turn = stats['turn_count']

                    def safe_div(a, b):
                        return a / b if b > 0 else 0.0

                    number_tiles = stats['discard_wan'] + stats['discard_tong'] + stats['discard_tiao']
                    max_suit = max(stats['discard_wan'], stats['discard_tong'], stats['discard_tiao'])
                    
                    feat_val_concentration = round(1.0 - safe_div(max_suit, number_tiles), 4)
                    
                    feat_val_prop_zhong = round(safe_div(stats['discard_3_to_7'], td), 4)
                    feat_val_prop_bian_1289 = round(safe_div(stats['discard_1289'], td), 4)
                    feat_val_prop_zi = round(safe_div(stats['discard_zi'], td), 4)

                    feat_val_moqie_rate = round(safe_div(stats['moqie_count'], td), 4)
                    feat_val_moqie_strength = round(safe_div(max(stats['max_continuous_moqie'] - 2, 0), turn), 4)
                    feat_val_mo_to_shou = round(safe_div(stats['moqie_to_shouqie_count'], turn), 4)

                    player_state = states.get_player(states.state[j], actor_loc)
                    is_tenpai = 1 if player_state.shantenCount <= 0 else 0

                    snapshot = {
                        '檔案名稱': file.name,
                        '玩家位置': actor_loc,
                        'Step_ID': j,
                        '動作類型': action,
                        
                        '丟棄的牌(代碼)': tile_str,
                        '丟棄的牌(名稱)': tile_name, 
                        '累積丟牌數': td,

                        # 對應使用者要求的 Feature Cols
                        'feat_a_巡數': turn,
                        'feat_b_吃碰數': stats['meld_count'],
                        'feat_c_花色集中度': feat_val_concentration,
                        'feat_d_中張比例(3 ~ 7)': feat_val_prop_zhong,
                        'feat_e_邊張比例(1、2、8、9)': feat_val_prop_bian_1289,
                        'feat_f_字牌比例': feat_val_prop_zi,

                        'feat_g_摸切比例': feat_val_moqie_rate,
                        'feat_h_連續摸切強度': feat_val_moqie_strength,
                        'feat_i_摸切轉手切': feat_val_mo_to_shou,

                        'feat_j_中張第一張被打出': 1 if (is_zhong and discard_nth == 1) else 0,
                        'feat_k_中張第二張被打出': 1 if (is_zhong and discard_nth == 2) else 0,
                        'feat_l_中張第三張被打出': 1 if (is_zhong and discard_nth == 3) else 0,
                        'feat_m_中張第四張被打出': 1 if (is_zhong and discard_nth == 4) else 0,

                        'feat_n_字牌第一張被打出': 1 if (is_zi and discard_nth == 1) else 0,
                        'feat_o_字牌第二張被打出': 1 if (is_zi and discard_nth == 2) else 0,
                        'feat_p_字牌第三張被打出': 1 if (is_zi and discard_nth == 3) else 0,
                        'feat_q_字牌第四張被打出': 1 if (is_zi and discard_nth == 4) else 0,

                        'feat_r_邊張(1、9)第一張被打出': 1 if (is_bian_19 and discard_nth == 1) else 0,
                        'feat_s_邊張(1、9)第二張被打出': 1 if (is_bian_19 and discard_nth == 2) else 0,
                        'feat_t_邊張(1、9)第三張被打出': 1 if (is_bian_19 and discard_nth == 3) else 0,
                        'feat_u_邊張(1、9)第四張被打出': 1 if (is_bian_19 and discard_nth == 4) else 0,

                        'feat_v_邊張(2、8)第一張被打出': 1 if (is_bian_28 and discard_nth == 1) else 0,
                        'feat_w_邊張(2、8)第二張被打出': 1 if (is_bian_28 and discard_nth == 2) else 0,
                        'feat_x_邊張(2、8)第三張被打出': 1 if (is_bian_28 and discard_nth == 3) else 0,
                        'feat_y_邊張(2、8)第四張被打出': 1 if (is_bian_28 and discard_nth == 4) else 0,

                        'Target_是否已聽牌': is_tenpai
                    }

                    game_snapshots.append(snapshot)

        if game_snapshots:
            results.append(pd.DataFrame(game_snapshots))

    if results:
        save_file(results)