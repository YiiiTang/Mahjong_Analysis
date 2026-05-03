import pandas as pd
import json
from pathlib import Path
from analyze import getRound
from fileProcess import RoundState, States, playerState

# ==========================================
# AUTO-UPDATE-START
WEIGHTS_BY_MELD = {
    0: {
        'a': 0.0225, 'c': 0.0097, 'd': 0.0129, 'e': 0.0144, 'f': 0.0074,
        'g': 0.0142, 'h': 0.0039, 'i': 0.0010, 'j': 0.0003,
        'k': 0.0029, 'l': 0.0003, 'm': -0.0008, 'n': 0.0003,
        'o': 0.0096, 'p': 0.0044, 'q': 0.0032, 'r': 0.0013
    },
    1: {
        'a': 0.1398, 'c': 0.0302, 'd': 0.0149, 'e': 0.0065, 'f': 0.0023,
        'g': 0.2031, 'h': 0.1147, 'i': 0.0511, 'j': 0.0133,
        'k': 0.1350, 'l': 0.1153, 'm': 0.0899, 'n': 0.0510,
        'o': 0.1902, 'p': 0.1331, 'q': 0.0777, 'r': 0.0339
    },
    2: {
        'a': 0.1723, 'c': 0.0404, 'd': 0.0313, 'e': 0.0006, 'f': -0.0170,
        'g': 0.1465, 'h': 0.0773, 'i': 0.0376, 'j': 0.0081,
        'k': 0.0309, 'l': 0.0393, 'm': 0.0352, 'n': 0.0204,
        'o': 0.0943, 'p': 0.0663, 'q': 0.0372, 'r': 0.0139
    },
    3: {
        'a': 0.1492, 'c': 0.0519, 'd': 0.0348, 'e': -0.0189, 'f': -0.0079,
        'g': -0.0560, 'h': -0.0527, 'i': -0.0407, 'j': -0.0160,
        'k': -0.0377, 'l': -0.0567, 'm': -0.0559, 'n': -0.0427,
        'o': -0.0899, 'p': -0.0826, 'q': -0.0601, 'r': -0.0319
    },
    4: {
        'a': 0.0443, 'c': 0.0554, 'd': 0.0138, 'e': 0.0155, 'f': -0.0080,
        'g': -0.0345, 'h': -0.0458, 'i': -0.0226, 'j': -0.0137,
        'k': -0.0146, 'l': -0.0263, 'm': -0.0352, 'n': -0.0198,
        'o': -0.0379, 'p': -0.0538, 'q': -0.0554, 'r': -0.0175
    },
}
MEANS_BY_MELD = {
    0: {
        'a': 4.6082, 'c': 0.5427, 'd': 0.1050, 'e': 0.0037, 'f': 0.0063,
        'g': 0.1644, 'h': 0.0479, 'i': 0.0101, 'j': 0.0012,
        'k': 0.2312, 'l': 0.1502, 'm': 0.0663, 'n': 0.0158,
        'o': 0.2034, 'p': 0.0809, 'q': 0.0242, 'r': 0.0040
    },
    1: {
        'a': 6.2907, 'c': 0.4428, 'd': 0.1884, 'e': 0.0137, 'f': 0.0185,
        'g': 0.2063, 'h': 0.0608, 'i': 0.0125, 'j': 0.0012,
        'k': 0.1165, 'l': 0.0949, 'm': 0.0586, 'n': 0.0206,
        'o': 0.2534, 'p': 0.1265, 'q': 0.0416, 'r': 0.0072
    },
    2: {
        'a': 8.2204, 'c': 0.4313, 'd': 0.2360, 'e': 0.0233, 'f': 0.0257,
        'g': 0.2688, 'h': 0.1004, 'i': 0.0208, 'j': 0.0022,
        'k': 0.0416, 'l': 0.0577, 'm': 0.0514, 'n': 0.0252,
        'o': 0.2216, 'p': 0.1433, 'q': 0.0557, 'r': 0.0111
    },
    3: {
        'a': 9.5429, 'c': 0.4578, 'd': 0.2686, 'e': 0.0327, 'f': 0.0302,
        'g': 0.2931, 'h': 0.1254, 'i': 0.0312, 'j': 0.0042,
        'k': 0.0223, 'l': 0.0493, 'm': 0.0517, 'n': 0.0273,
        'o': 0.1727, 'p': 0.1422, 'q': 0.0652, 'r': 0.0152
    },
    4: {
        'a': 10.6546, 'c': 0.4791, 'd': 0.2865, 'e': 0.0384, 'f': 0.0353,
        'g': 0.3104, 'h': 0.1358, 'i': 0.0446, 'j': 0.0057,
        'k': 0.0134, 'l': 0.0504, 'm': 0.0554, 'n': 0.0268,
        'o': 0.1460, 'p': 0.1338, 'q': 0.0580, 'r': 0.0185
    },
}
SCALES_BY_MELD = {
    0: {
        'a': 3.2897, 'c': 0.3765, 'd': 0.1741, 'e': 0.0270, 'f': 0.0303,
        'g': 0.3706, 'h': 0.2136, 'i': 0.0998, 'j': 0.0353,
        'k': 0.4216, 'l': 0.3573, 'm': 0.2488, 'n': 0.1249,
        'o': 0.4025, 'p': 0.2727, 'q': 0.1536, 'r': 0.0633
    },
    1: {
        'a': 3.0097, 'c': 0.3106, 'd': 0.1734, 'e': 0.0504, 'f': 0.0460,
        'g': 0.4046, 'h': 0.2389, 'i': 0.1111, 'j': 0.0344,
        'k': 0.3209, 'l': 0.2930, 'm': 0.2348, 'n': 0.1421,
        'o': 0.4349, 'p': 0.3324, 'q': 0.1997, 'r': 0.0845
    },
    2: {
        'a': 2.8396, 'c': 0.2051, 'd': 0.1617, 'e': 0.0628, 'f': 0.0488,
        'g': 0.4433, 'h': 0.3005, 'i': 0.1429, 'j': 0.0472,
        'k': 0.1997, 'l': 0.2331, 'm': 0.2209, 'n': 0.1566,
        'o': 0.4153, 'p': 0.3504, 'q': 0.2294, 'r': 0.1049
    },
    3: {
        'a': 2.7006, 'c': 0.1532, 'd': 0.1554, 'e': 0.0729, 'f': 0.0491,
        'g': 0.4552, 'h': 0.3312, 'i': 0.1738, 'j': 0.0647,
        'k': 0.1477, 'l': 0.2165, 'm': 0.2214, 'n': 0.1629,
        'o': 0.3780, 'p': 0.3493, 'q': 0.2469, 'r': 0.1225
    },
    4: {
        'a': 2.5660, 'c': 0.1266, 'd': 0.1427, 'e': 0.0731, 'f': 0.0497,
        'g': 0.4627, 'h': 0.3425, 'i': 0.2065, 'j': 0.0755,
        'k': 0.1149, 'l': 0.2187, 'm': 0.2289, 'n': 0.1614,
        'o': 0.3531, 'p': 0.3405, 'q': 0.2337, 'r': 0.1347
    },
}

INTERCEPTS_BY_MELD = {
    0: 0.0239,
    1: 0.1534,
    2: 0.3666,
    3: 0.6061,
    4: 0.8496,
}

# AUTO-UPDATE-END (請勿刪除此行)
# ==========================================

def calculate_linear_score(features, current_meld_count):
    meld = min(current_meld_count, 4)
    
    W = WEIGHTS_BY_MELD[meld]
    M = MEANS_BY_MELD[meld]
    S = SCALES_BY_MELD[meld]
    intercept = INTERCEPTS_BY_MELD[meld]
    
    keys = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']
    
    score = intercept
    for k in keys:
        scaled_val = (features[k] - M[k]) / S[k] if S[k] != 0 else 0
        score += W[k] * scaled_val
        
    return score

def track_game_state_linear(file_path):
    states = getRound(file_path)
    players = ['E', 'S', 'W', 'N']
    file_name = Path(file_path).name
    
    player_stats = {
        loc: {
            'turn_count': 0, 'meld_count': 0, 'total_discard': 0,
            'discard_wan': 0, 'discard_tong': 0,
            'discard_tiao': 0, 'discard_zi': 0, 'moqie_count': 0,
            'current_continuous_moqie': 0, 'max_continuous_moqie': 0,
            'moqie_to_shouqie_count': 0, 'last_discard_type': None
        } for loc in players
    }

    tracker_log = []
    global_discard_count = {}

    for j in range(1, len(states.state)):
        step_data = states.state[j].stepData
        if len(step_data) < 3: continue

        actor_loc = step_data[1]
        action = step_data[2]

        if actor_loc not in players: continue
        stats = player_stats[actor_loc]

        if action in ('P', 'E', 'EM', 'EL', 'ER', 'UG', 'HG', 'G', 'HD', 'MD'):
            is_discard = action in ('HD', 'MD')
            if not is_discard:
                stats['meld_count'] += 1
            else:
                stats['turn_count'] += 1
                stats['total_discard'] += 1
                tile_str = step_data[3] if len(step_data) > 3 else ""

                if action == 'MD':
                    stats['moqie_count'] += 1
                    stats['current_continuous_moqie'] += 1
                    if stats['current_continuous_moqie'] > stats['max_continuous_moqie']:
                        stats['max_continuous_moqie'] = stats['current_continuous_moqie']
                elif action == 'HD':
                    if stats['current_continuous_moqie'] >= 2:
                        stats['moqie_to_shouqie_count'] += 1
                    stats['current_continuous_moqie'] = 0

                discard_nth = 0
                tile_name = "" 
                
                is_zhong = False
                is_zi = False
                is_bian = False
                
                if tile_str and tile_str.isdigit():
                    card_num = int(tile_str)
                    suit, val = card_num // 100, (card_num // 10) % 10
                    
                    if suit in (1, 2, 3) and 3 <= val <= 7:
                        is_zhong = True
                    elif suit == 4:
                        is_zi = True
                    elif suit in (1, 2, 3) and val in (1, 2, 8, 9):
                        is_bian = True
                    
                    if suit == 4:
                        honor_dict = {1: '東', 2: '南', 3: '西', 4: '北', 5: '白', 6: '發', 7: '中'}
                        tile_name = honor_dict.get(val, f"{val}字")
                    else:
                        type_dict = {0: '花', 1: '萬', 2: '筒', 3: '條'}
                        if suit in type_dict:
                            tile_name = f"{val}{type_dict[suit]}"
                    
                    if tile_name:
                        global_discard_count[tile_name] = global_discard_count.get(tile_name, 0) + 1
                        discard_nth = global_discard_count[tile_name]

                    if suit == 1: stats['discard_wan'] += 1
                    elif suit == 2: stats['discard_tong'] += 1
                    elif suit == 3: stats['discard_tiao'] += 1
                    elif suit == 4: stats['discard_zi'] += 1

                td, turn = stats['total_discard'], stats['turn_count']
                def safe_div(a, b): return a / b if b > 0 else 0.0

                num_tiles = stats['discard_wan'] + stats['discard_tong'] + stats['discard_tiao']
                max_s = max(stats['discard_wan'], stats['discard_tong'], stats['discard_tiao'])
                
                feat_c_concentration = 1.0 - safe_div(max_s, num_tiles)
                feat_d_moqie_rate = safe_div(stats['moqie_count'], td)
                feat_e_moqie_strength = safe_div(max(stats['max_continuous_moqie'] - 2, 0), turn)
                feat_f_mo_to_shou = safe_div(stats['moqie_to_shouqie_count'], turn)

                features = {
                    'a': turn, 
                    'b1': 1 if stats['meld_count'] == 1 else 0, 
                    'b2': 1 if stats['meld_count'] == 2 else 0, 
                    'b3': 1 if stats['meld_count'] == 3 else 0, 
                    'b4': 1 if stats['meld_count'] >= 4 else 0, 
                    'c': feat_c_concentration, 
                    'd': feat_d_moqie_rate, 
                    'e': feat_e_moqie_strength, 
                    'f': feat_f_mo_to_shou,
                    'g': 1 if (is_zhong and discard_nth == 1) else 0,
                    'h': 1 if (is_zhong and discard_nth == 2) else 0,
                    'i': 1 if (is_zhong and discard_nth == 3) else 0,
                    'j': 1 if (is_zhong and discard_nth == 4) else 0,
                    'k': 1 if (is_zi and discard_nth == 1) else 0,
                    'l': 1 if (is_zi and discard_nth == 2) else 0,
                    'm': 1 if (is_zi and discard_nth == 3) else 0,
                    'n': 1 if (is_zi and discard_nth == 4) else 0,
                    'o': 1 if (is_bian and discard_nth == 1) else 0,
                    'p': 1 if (is_bian and discard_nth == 2) else 0,
                    'q': 1 if (is_bian and discard_nth == 3) else 0,
                    'r': 1 if (is_bian and discard_nth == 4) else 0
                }
                
                pred_score = calculate_linear_score(features, stats['meld_count'])

                actual_shanten = states.get_player(states.state[j], actor_loc).shantenCount
                action_name = "手切" if action == "HD" else "摸切"

                # 這裡已經移除了終端機的洗版列印

                tracker_log.append({
                    '檔案名稱': file_name, 'Step_ID': j, '玩家': actor_loc, '動作': action_name, '丟棄牌': tile_str,
                    '實際向聽數': actual_shanten, '預測聽牌分數': round(pred_score, 4),
                    '特徵值': {
                        'feat_a_巡數': turn, 
                        'feat_b1_一副露': features['b1'], 
                        'feat_b2_二副露': features['b2'], 
                        'feat_b3_三副露': features['b3'], 
                        'feat_b4_四副露': features['b4'], 
                        'feat_c_花色集中度': round(feat_c_concentration, 4),
                        'feat_d_摸切比例': round(feat_d_moqie_rate, 4), 
                        'feat_e_連續摸切強度': round(feat_e_moqie_strength, 4), 
                        'feat_f_摸切轉手切': round(feat_f_mo_to_shou, 4),
                        'feat_g_中張第一張被打出': features['g'], 
                        'feat_h_中張第二張被打出': features['h'], 
                        'feat_i_中張第三張被打出': features['i'], 
                        'feat_j_中張第四張被打出': features['j'],
                        'feat_k_字牌第一張被打出': features['k'],
                        'feat_l_字牌第二張被打出': features['l'],
                        'feat_m_字牌第三張被打出': features['m'],
                        'feat_n_字牌第四張被打出': features['n'],
                        'feat_o_邊張第一張被打出': features['o'],
                        'feat_p_邊張第二張被打出': features['p'],
                        'feat_q_邊張第三張被打出': features['q'],
                        'feat_r_邊張第四張被打出': features['r']
                    }
                })
    return tracker_log

if __name__ == "__main__":
    folder_path = Path("D:\\Project\\Mahjong\\Board\\Test")

    files = [f for f in folder_path.iterdir() if f.is_file()]
    
    master_log = []
    success_count = 0
    fail_count = 0
    failed_files = [] 

    print(f"📂 開始批次處理資料夾 (線性迴歸模式): {folder_path}")
    print(f"📝 總共找到 {len(files)} 個檔案，開始分析...\n")

    for file in files:
        try:
            game_log = track_game_state_linear(str(file))
            master_log.extend(game_log)
            success_count += 1
            print(f"✅ 已完成處理: {file.name}") # 只印出完成的檔案名稱
        except Exception as e:
            print(f"❌ 讀取 {file.name} 時發生錯誤: {e}")
            fail_count += 1
            failed_files.append(file.name) 

    if failed_files:
        error_file_name = "Linear_Failed_Files_Log.txt"
        with open(error_file_name, 'w', encoding='utf-8') as f:
            f.write("以下牌譜檔案讀取或分析失敗：\n")
            for fname in failed_files:
                f.write(f"{fname}\n")
        
        print("\n" + "!"*40)
        print("⚠️ 錯誤檔案清單 ⚠️")
        print("!"*40)
        for fname in failed_files:
            print(f" - {fname}")
        print(f"\n已將上述 {len(failed_files)} 個錯誤檔名存至: {error_file_name}")

    if master_log:
        correct_predictions = 0
        total_predictions = len(master_log)
        error_log = [] 

        for log in master_log:
            is_actually_tenpai = 1 if log['實際向聽數'] <= 0 else 0
            is_predicted_tenpai = 1 if log['預測聽牌分數'] >= 0.5 else 0
            
            if is_actually_tenpai == is_predicted_tenpai:
                correct_predictions += 1
            else:
                error_item = log.copy()
                if is_predicted_tenpai == 1 and is_actually_tenpai == 0:
                    error_item['錯誤類型'] = "誤報 (實際未聽，猜測已聽)"
                else:
                    error_item['錯誤類型'] = "漏報 (實際已聽，猜測未聽)"
                error_log.append(error_item)
                
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        print("\n" + "="*40)
        print("Linear 預測準確度報告")
        print("="*40)
        print(f"成功分析盤面數: {success_count}")
        if fail_count > 0:
            print(f"失敗盤面數: {fail_count} (請查看 {error_file_name})")
        print(f"總分析步數 (樣本數): {total_predictions}")
        print(f"預測正確次數: {correct_predictions}")
        print(f"判斷錯誤次數: {len(error_log)}")
        print(f"整體準確率 (Accuracy): {accuracy:.2%}")
        print("="*40)

        excel_name = "Batch_Linear_Tracking_Result.xlsx"
        flat_log = []
        for log in master_log:
            flat_item = log.copy()
            features = flat_item.pop('特徵值')
            flat_item.update(features)
            flat_log.append(flat_item)
            
        df = pd.DataFrame(flat_log)
        df.to_excel(excel_name, index=False)
        
        json_name = "Batch_Linear_Tracking_Result.json"
        with open(json_name, 'w', encoding='utf-8') as f:
            json.dump(master_log, f, ensure_ascii=False, indent=4)
            
        print(f"\n📁 線性迴歸完整資料已儲存！")
        print(f" 📊 Excel 報表: {excel_name}")
        print(f" 📄 JSON 檔案: {json_name}")

        if error_log:
            error_excel_name = "Batch_Linear_Error_Tracking.xlsx"
            flat_error_log = []
            for log in error_log:
                flat_item = log.copy()
                features = flat_item.pop('特徵值')
                flat_item.update(features)
                flat_error_log.append(flat_item)
            
            df_error = pd.DataFrame(flat_error_log)
            cols = list(df_error.columns)
            cols.insert(4, cols.pop(cols.index('錯誤類型')))
            df_error = df_error[cols]
            df_error.to_excel(error_excel_name, index=False)
            print(f" ❌ 錯誤判斷報表 (僅列出猜錯的步驟): {error_excel_name}")

    else:
        print("\n沒有成功產出任何追蹤紀錄。")