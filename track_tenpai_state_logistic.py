import math
import json
import pandas as pd
from pathlib import Path
from analyze import getRound
from fileProcess import RoundState, States, playerState

# --- 模型參數 (請依訓練結果填入) ---
INTERCEPT = -2.7291  
WEIGHTS = {
    'a': 1.2849, 'b': 0.5394, 'c': 0.4233, 'd': 0.3736,
    'e': -0.8156, 'f': -0.0841, 'g': -0.0894, 'h': 0.0351    
}

MEANS = {
    'a': 5.5966, 'b': 0.8766, 'c': 0.1406, 'd': 0.5281,
    'e': 0.5972, 'f': 0.1803, 'g': 0.0127, 'h': 0.0165
}
SCALES = {
    'a': 3.3296, 'b': 0.9683, 'c': 0.1708, 'd': 0.3470,
    'e': 0.3043, 'f': 0.1870, 'g': 0.0484, 'h': 0.0436
}

def calculate_probability(features):
    # 執行標準化轉換
    scaled = {k: (features[k] - MEANS[k]) / SCALES[k] if SCALES[k] != 0 else 0 for k in features}

    z = INTERCEPT \
        + WEIGHTS['a'] * scaled['a'] \
        + WEIGHTS['b'] * scaled['b'] \
        + WEIGHTS['c'] * scaled['c'] \
        + WEIGHTS['d'] * scaled['d'] \
        + WEIGHTS['e'] * scaled['e'] \
        + WEIGHTS['f'] * scaled['f'] \
        + WEIGHTS['g'] * scaled['g'] \
        + WEIGHTS['h'] * scaled['h']

    z = max(min(z, 500), -500) 
    return 1 / (1 + math.exp(-z))

def track_game_state(file_path):
    """分析單一盤面，回傳該盤面的所有追蹤紀錄"""
    states = getRound(file_path)
    players = ['E', 'S', 'W', 'N']
    file_name = Path(file_path).name
    
    player_stats = {
        loc: {
            'turn_count': 0, 'meld_count': 0, 'total_discard': 0,
            'discard_3_to_7': 0, 'discard_wan': 0, 'discard_tong': 0,
            'discard_tiao': 0, 'discard_zi': 0, 'moqie_count': 0,
            'current_continuous_moqie': 0, 'max_continuous_moqie': 0,
            'moqie_to_shouqie_count': 0, 'last_discard_type': None
        } for loc in players
    }

    tracker_log = []

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

                if tile_str.isdigit():
                    card_num = int(tile_str)
                    suit = card_num // 100
                    face_value = (card_num // 10) % 10

                    if suit in (1, 2, 3) and 3 <= face_value <= 7:
                        stats['discard_3_to_7'] += 1

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

                feat_c = safe_div(stats['discard_3_to_7'], td)
                number_tiles = stats['discard_wan'] + stats['discard_tong'] + stats['discard_tiao']
                max_suit = max(stats['discard_wan'], stats['discard_tong'], stats['discard_tiao'])
                feat_d = 1.0 - safe_div(max_suit, number_tiles)
                feat_e = safe_div(stats['discard_zi'], td)
                feat_f = safe_div(stats['moqie_count'], td)
                feat_g = safe_div(max(stats['max_continuous_moqie'] - 2, 0), turn)
                feat_h = safe_div(stats['moqie_to_shouqie_count'], turn)

                features = {
                    'a': turn, 'b': stats['meld_count'], 'c': feat_c,
                    'd': feat_d, 'e': feat_e, 'f': feat_f,
                    'g': feat_g, 'h': feat_h
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
                        '當下巡數': turn,
                        '當下副露': stats['meld_count'],
                        '當下連續摸切': stats['current_continuous_moqie'],
                        '中張比例': round(feat_c, 4),
                        '花色集中度': round(feat_d, 4),
                        '字牌比例': round(feat_e, 4),
                        '摸切比例': round(feat_f, 4),
                        '連續摸切強度': round(feat_g, 4),
                        '摸切轉手切': round(feat_h, 4)
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

        for log in master_log:
            is_actually_tenpai = 1 if log['實際向聽數'] <= 0 else 0
            is_predicted_tenpai = 1 if log['預測聽牌機率'] >= 0.5 else 0
            
            if is_actually_tenpai == is_predicted_tenpai:
                correct_predictions += 1
                
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        print("\n" + "="*40)
        print("Logistic 預測準確度報告")
        print("="*40)
        print(f"成功分析盤面數: {success_count}")
        if fail_count > 0:
            print(f"失敗盤面數: {fail_count} (請查看 Logistic_Failed_Files_Log.txt)")
        print(f"總分析步數 (樣本數): {total_predictions}")
        print(f"預測正確次數: {correct_predictions}")
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
    else:
        print("\n沒有成功產出任何追蹤紀錄。")