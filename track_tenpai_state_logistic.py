import math
import json
import pandas as pd
from pathlib import Path
from analyze import getRound
from fileProcess import RoundState, States, playerState

# --- 模型參數 (請依訓練結果填入最新的數值) ---
INTERCEPT = -2.4607  
WEIGHTS = {
    'a': 1.3989, 'b': 0.5679, 'c': 0.4445, 'd': 0.1506,
    'e': -0.1088, 'f': -0.0370, 'g': 0.3046, 'h': 0.1465,
    'i': 0.0047, 'j': 0.0027, 'k': -0.4535, 'l': -0.1225,
    'm': -0.0442, 'n': -0.0505, 'o': 0.1197, 'p': 0.0888,
    'q': -0.0016, 'r': -0.0105
}

MEANS = {
    'a': 5.5966, 'b': 0.8766, 'c': 0.5281, 'd': 0.1803,
    'e': 0.0127, 'f': 0.0165, 'g': 0.1947, 'h': 0.0552,
    'i': 0.0086, 'j': 0.0008, 'k': 0.1716, 'l': 0.1207,
    'm': 0.0670, 'n': 0.0231, 'o': 0.2185, 'p': 0.1003,
    'q': 0.0333, 'r': 0.0063
}

SCALES = {
    'a': 3.3296, 'b': 0.9683, 'c': 0.3470, 'd': 0.1870,
    'e': 0.0484, 'f': 0.0436, 'g': 0.3959, 'h': 0.2283,
    'i': 0.0924, 'j': 0.0278, 'k': 0.3771, 'l': 0.3258,
    'm': 0.2500, 'n': 0.1502, 'o': 0.4133, 'p': 0.3003,
    'q': 0.1794, 'r': 0.0791
}

def calculate_probability(features):
    scaled = {k: (features[k] - MEANS[k]) / SCALES[k] if SCALES[k] != 0 else 0 for k in features}

    z = INTERCEPT \
        + WEIGHTS['a'] * scaled['a'] \
        + WEIGHTS['b'] * scaled['b'] \
        + WEIGHTS['c'] * scaled['c'] \
        + WEIGHTS['d'] * scaled['d'] \
        + WEIGHTS['e'] * scaled['e'] \
        + WEIGHTS['f'] * scaled['f'] \
        + WEIGHTS['g'] * scaled['g'] \
        + WEIGHTS['h'] * scaled['h'] \
        + WEIGHTS['i'] * scaled['i'] \
        + WEIGHTS['j'] * scaled['j'] \
        + WEIGHTS['k'] * scaled['k'] \
        + WEIGHTS['l'] * scaled['l'] \
        + WEIGHTS['m'] * scaled['m'] \
        + WEIGHTS['n'] * scaled['n'] \
        + WEIGHTS['o'] * scaled['o'] \
        + WEIGHTS['p'] * scaled['p'] \
        + WEIGHTS['q'] * scaled['q'] \
        + WEIGHTS['r'] * scaled['r']

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
                    'b': stats['meld_count'], 
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

                pred_prob = calculate_probability(features)

                player_state_obj = states.get_player(states.state[j], actor_loc)
                actual_shanten = player_state_obj.shantenCount
                action_name = "手切" if action == "HD" else "摸切"

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
                        'feat_b_吃碰數': stats['meld_count'],
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
    folder_path = Path("D:\\Project\\Mahjong\\Board\\TCGA MJ 2025\\3")

    files = [f for f in folder_path.iterdir() if f.is_file()]
    
    master_log = []
    success_count = 0
    fail_count = 0
    failed_files = []

    print(f"📂 開始批次處理資料夾: {folder_path}")
    print(f"📝 總共找到 {len(files)} 個檔案，開始分析...\n")

    for file in files:
        try:
            print(f"處理中: {file.name}")
            game_log = track_game_state(str(file))
            master_log.extend(game_log)
            success_count += 1
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