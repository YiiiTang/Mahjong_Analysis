import pandas as pd
import json
from pathlib import Path
from analyze import getRound
from fileProcess import RoundState, States, playerState

# --- 模型參數 (請依訓練結果填入最新的數值) ---
INTERCEPT = 0.1865  
WEIGHTS = {
    'a': 0.1681, 'b': 0.0745, 'c': 0.0601, 'd': 0.0019,
    'e': -0.0023, 'f': 0.0046, 'g': 0.0257, 'h': 0.0173,
    'i': 0.0020, 'j': 0.0029, 'k': -0.0120, 'l': -0.0142,
    'm': -0.0105, 'n': -0.0087, 'o': 0.0022, 'p': -0.0033,
    'q': -0.0062, 'r': -0.0020
}

MEANS = {
    'a': 5.6080, 'b': 0.8770, 'c': 0.5261, 'd': 0.1816,
    'e': 0.0128, 'f': 0.0165, 'g': 0.1937, 'h': 0.0550,
    'i': 0.0089, 'j': 0.0010, 'k': 0.1681, 'l': 0.1219,
    'm': 0.0651, 'n': 0.0236, 'o': 0.2216, 'p': 0.1014,
    'q': 0.0334, 'r': 0.0062
}

SCALES = {
    'a': 3.3261, 'b': 0.9675, 'c': 0.3467, 'd': 0.1883,
    'e': 0.0480, 'f': 0.0433, 'g': 0.3952, 'h': 0.2280,
    'i': 0.0939, 'j': 0.0321, 'k': 0.3739, 'l': 0.3272,
    'm': 0.2467, 'n': 0.1518, 'o': 0.4154, 'p': 0.3018,
    'q': 0.1798, 'r': 0.0785
}

def calculate_linear_score(features):
    scaled = {k: (features[k] - MEANS[k]) / SCALES[k] if SCALES[k] != 0 else 0 for k in features}
    
    score = INTERCEPT \
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
                pred_score = calculate_linear_score(features)

                actual_shanten = states.get_player(states.state[j], actor_loc).shantenCount
                tracker_log.append({
                    '檔案名稱': file_name, 'Step_ID': j, '玩家': actor_loc, '動作': "手切" if action == "HD" else "摸切",
                    '實際向聽數': actual_shanten, '預測聽牌分數': round(pred_score, 4),
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

    print(f"📂 開始批次處理資料夾 (線性迴歸模式): {folder_path}")
    print(f"📝 總共找到 {len(files)} 個檔案，開始分析...\n")

    for file in files:
        try:
            print(f"處理中: {file.name}")
            game_log = track_game_state_linear(str(file))
            master_log.extend(game_log)
            success_count += 1
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