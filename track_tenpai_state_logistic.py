import math
import json
import pandas as pd
from pathlib import Path
from analyze import getRound
from fileProcess import RoundState, States, playerState

# ==========================================
# AUTO-UPDATE-START
WEIGHTS_BY_MELD = {
    0: {
        'a': 0.9117, 'c': 0.2616, 'd': 0.8004, 'e': 0.0040, 'f': -0.0058,
        'g': 0.4674, 'h': 0.1241, 'i': 0.0165, 'j': -0.0063,
        'k': -0.6873, 'l': -0.2569, 'm': -0.0553, 'n': 0.0139,
        'o': 0.3212, 'p': 0.1780, 'q': 0.0892, 'r': 0.0195
    },
    1: {
        'a': 1.1172, 'c': 0.2970, 'd': 0.3431, 'e': -0.0763, 'f': -0.0725,
        'g': 0.2991, 'h': 0.1010, 'i': 0.0211, 'j': -0.0175,
        'k': -0.2724, 'l': -0.1560, 'm': -0.0754, 'n': -0.0629,
        'o': 0.0727, 'p': -0.0208, 'q': -0.0448, 'r': -0.0168
    },
    2: {
        'a': 0.8475, 'c': 0.2471, 'd': 0.2097, 'e': -0.0407, 'f': -0.0966,
        'g': 0.2379, 'h': 0.0392, 'i': 0.0213, 'j': -0.0135,
        'k': -0.1122, 'l': -0.0721, 'm': -0.0699, 'n': -0.0784,
        'o': -0.0057, 'p': -0.0699, 'q': -0.0797, 'r': -0.0630
    },
    3: {
        'a': 0.7558, 'c': 0.2262, 'd': 0.1740, 'e': -0.1029, 'f': -0.0480,
        'g': 0.1858, 'h': 0.0528, 'i': -0.0338, 'j': -0.0365,
        'k': -0.0295, 'l': -0.0686, 'm': -0.0653, 'n': -0.0513,
        'o': -0.0416, 'p': -0.0642, 'q': -0.0466, 'r': -0.0308
    },
    4: {
        'a': 0.4204, 'c': 0.4111, 'd': -0.0228, 'e': 0.2688, 'f': 0.0087,
        'g': 0.1829, 'h': -0.0602, 'i': 0.0248, 'j': -0.0789,
        'k': 0.0390, 'l': 0.0178, 'm': -0.0503, 'n': -0.0691,
        'o': 0.0346, 'p': -0.0927, 'q': -0.1965, 'r': -0.0368
    },
}
MEANS_BY_MELD = {
    0: {
        'a': 4.6088, 'c': 0.5429, 'd': 0.1051, 'e': 0.0037, 'f': 0.0064,
        'g': 0.1646, 'h': 0.0483, 'i': 0.0101, 'j': 0.0013,
        'k': 0.2318, 'l': 0.1501, 'm': 0.0656, 'n': 0.0156,
        'o': 0.2029, 'p': 0.0812, 'q': 0.0240, 'r': 0.0041
    },
    1: {
        'a': 6.2978, 'c': 0.4429, 'd': 0.1890, 'e': 0.0140, 'f': 0.0185,
        'g': 0.2061, 'h': 0.0607, 'i': 0.0125, 'j': 0.0011,
        'k': 0.1170, 'l': 0.0953, 'm': 0.0575, 'n': 0.0207,
        'o': 0.2539, 'p': 0.1266, 'q': 0.0413, 'r': 0.0072
    },
    2: {
        'a': 8.2161, 'c': 0.4315, 'd': 0.2358, 'e': 0.0232, 'f': 0.0257,
        'g': 0.2683, 'h': 0.0990, 'i': 0.0214, 'j': 0.0024,
        'k': 0.0422, 'l': 0.0575, 'm': 0.0523, 'n': 0.0256,
        'o': 0.2203, 'p': 0.1429, 'q': 0.0570, 'r': 0.0110
    },
    3: {
        'a': 9.5483, 'c': 0.4590, 'd': 0.2683, 'e': 0.0330, 'f': 0.0304,
        'g': 0.2912, 'h': 0.1267, 'i': 0.0330, 'j': 0.0037,
        'k': 0.0224, 'l': 0.0478, 'm': 0.0512, 'n': 0.0282,
        'o': 0.1724, 'p': 0.1432, 'q': 0.0659, 'r': 0.0142
    },
    4: {
        'a': 10.6837, 'c': 0.4796, 'd': 0.2870, 'e': 0.0387, 'f': 0.0363,
        'g': 0.3084, 'h': 0.1394, 'i': 0.0494, 'j': 0.0048,
        'k': 0.0135, 'l': 0.0502, 'm': 0.0590, 'n': 0.0231,
        'o': 0.1394, 'p': 0.1355, 'q': 0.0574, 'r': 0.0183
    },
}
SCALES_BY_MELD = {
    0: {
        'a': 3.2902, 'c': 0.3767, 'd': 0.1741, 'e': 0.0270, 'f': 0.0305,
        'g': 0.3708, 'h': 0.2143, 'i': 0.1001, 'j': 0.0355,
        'k': 0.4220, 'l': 0.3572, 'm': 0.2477, 'n': 0.1240,
        'o': 0.4022, 'p': 0.2732, 'q': 0.1530, 'r': 0.0639
    },
    1: {
        'a': 3.0118, 'c': 0.3105, 'd': 0.1735, 'e': 0.0508, 'f': 0.0461,
        'g': 0.4045, 'h': 0.2388, 'i': 0.1110, 'j': 0.0338,
        'k': 0.3214, 'l': 0.2936, 'm': 0.2328, 'n': 0.1424,
        'o': 0.4352, 'p': 0.3326, 'q': 0.1990, 'r': 0.0847
    },
    2: {
        'a': 2.8406, 'c': 0.2059, 'd': 0.1614, 'e': 0.0625, 'f': 0.0489,
        'g': 0.4431, 'h': 0.2986, 'i': 0.1446, 'j': 0.0492,
        'k': 0.2010, 'l': 0.2327, 'm': 0.2226, 'n': 0.1578,
        'o': 0.4145, 'p': 0.3500, 'q': 0.2319, 'r': 0.1044
    },
    3: {
        'a': 2.6955, 'c': 0.1530, 'd': 0.1558, 'e': 0.0732, 'f': 0.0493,
        'g': 0.4543, 'h': 0.3326, 'i': 0.1787, 'j': 0.0610,
        'k': 0.1479, 'l': 0.2133, 'm': 0.2205, 'n': 0.1655,
        'o': 0.3777, 'p': 0.3503, 'q': 0.2481, 'r': 0.1184
    },
    4: {
        'a': 2.5589, 'c': 0.1286, 'd': 0.1451, 'e': 0.0734, 'f': 0.0505,
        'g': 0.4618, 'h': 0.3464, 'i': 0.2167, 'j': 0.0690,
        'k': 0.1156, 'l': 0.2184, 'm': 0.2356, 'n': 0.1502,
        'o': 0.3464, 'p': 0.3422, 'q': 0.2325, 'r': 0.1341
    },
}

INTERCEPTS_BY_MELD = {
    0: -5.0935,
    1: -2.3211,
    2: -0.6806,
    3: 0.5255,
    4: 1.8711,
}

# AUTO-UPDATE-END (請勿刪除此行)
# ==========================================

def calculate_probability(features, current_meld_count):
    meld = min(current_meld_count, 4)
    
    W = WEIGHTS_BY_MELD[meld]
    M = MEANS_BY_MELD[meld]
    S = SCALES_BY_MELD[meld]
    intercept = INTERCEPTS_BY_MELD[meld]
    
    keys = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']
    
    z = intercept
    for k in keys:
        scaled_val = (features[k] - M[k]) / S[k] if S[k] != 0 else 0
        z += W[k] * scaled_val

    z = max(min(z, 500), -500) 
    return 1 / (1 + math.exp(-z))

def track_game_state(file_path):
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

                discard_nth = 0
                tile_name = "" 
                
                is_zhong = False
                is_zi = False
                is_bian = False
                
                if tile_str.isdigit():
                    card_num = int(tile_str)
                    suit = card_num // 100
                    face_value = (card_num // 10) % 10

                    if suit in (1, 2, 3) and 3 <= face_value <= 7:
                        is_zhong = True
                    elif suit == 4:
                        is_zi = True
                    elif suit in (1, 2, 3) and face_value in (1, 2, 8, 9):
                        is_bian = True

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

                    if suit == 1:
                        stats['discard_wan'] += 1
                    elif suit == 2:
                        stats['discard_tong'] += 1
                    elif suit == 3:
                        stats['discard_tiao'] += 1
                    elif suit == 4:
                        stats['discard_zi'] += 1

                td = stats['total_discard']
                turn = stats['turn_count']
                def safe_div(a, b): return a / b if b > 0 else 0.0

                number_tiles = stats['discard_wan'] + stats['discard_tong'] + stats['discard_tiao']
                max_suit = max(stats['discard_wan'], stats['discard_tong'], stats['discard_tiao'])
                
                feat_c_concentration = 1.0 - safe_div(max_suit, number_tiles)
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

                pred_prob = calculate_probability(features, stats['meld_count'])

                player_state_obj = states.get_player(states.state[j], actor_loc)
                actual_shanten = player_state_obj.shantenCount
                action_name = "手切" if action == "HD" else "摸切"

                # 這裡已經移除了終端機的洗版列印

                tracker_log.append({
                    '檔案名稱': file_name,
                    'Step_ID': j,
                    '玩家': actor_loc,
                    '動作': action_name,
                    '丟棄牌': tile_str,
                    '實際向聽數': actual_shanten,
                    '預測聽牌機率': round(pred_prob, 4),
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

    print(f"📂 開始批次處理資料夾: {folder_path}")
    print(f"📝 總共找到 {len(files)} 個檔案，開始分析...\n")

    for file in files:
        try:
            game_log = track_game_state(str(file))
            master_log.extend(game_log)
            success_count += 1
            print(f"✅ 已完成處理: {file.name}") # 只印出完成的檔案名稱
        except Exception as e:
            print(f"❌ 讀取 {file.name} 時發生錯誤: {e}")
            fail_count += 1
            failed_files.append(file.name)

    if failed_files:
        error_file_name = "Logistic_Failed_Files_Log.txt"
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
            is_predicted_tenpai = 1 if log['預測聽牌機率'] >= 0.5 else 0
            
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
        print("Logistic 預測準確度報告")
        print("="*40)
        print(f"成功分析盤面數: {success_count}")
        if fail_count > 0:
            print(f"失敗盤面數: {fail_count} (請查看 Logistic_Failed_Files_Log.txt)")
        print(f"總分析步數 (樣本數): {total_predictions}")
        print(f"預測正確次數: {correct_predictions}")
        print(f"判斷錯誤次數: {len(error_log)}")
        print(f"整體準確率 (Accuracy): {accuracy:.2%}")
        print("="*40)

        excel_name = "Batch_Logistic_Tracking_Result.xlsx"
        flat_log = []
        for log in master_log:
            flat_item = log.copy()
            features = flat_item.pop('特徵值')
            flat_item.update(features)
            flat_log.append(flat_item)
            
        df = pd.DataFrame(flat_log)
        df.to_excel(excel_name, index=False)
        
        json_name = "Batch_Logistic_Tracking_Result.json"
        with open(json_name, 'w', encoding='utf-8') as f:
            json.dump(master_log, f, ensure_ascii=False, indent=4)
            
        print(f"\n📁 完整資料已儲存！")
        print(f" 📊 Excel 報表: {excel_name}")
        print(f" 📄 JSON 檔案: {json_name}")

        if error_log:
            error_excel_name = "Batch_Logistic_Error_Tracking.xlsx"
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