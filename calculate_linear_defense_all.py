def calculate_ting_probability(intercept, a, b, c, d, e, 
                               turn_count, meld_count, 
                               middle_cards_discarded, 
                               total_discarded, 
                               max_suit_discarded, 
                               total_suit_discarded, 
                               honor_cards_discarded, 
                               player_total_discarded):
    """
    計算聽牌機率 Pi
    """
    
    part_c = c * (middle_cards_discarded / total_discarded) if total_discarded != 0 else 0
    
    part_d_ratio = (max_suit_discarded / total_suit_discarded) if total_suit_discarded != 0 else 0
    part_d = d * (1 - part_d_ratio)
    
    part_e = e * (honor_cards_discarded / player_total_discarded) if player_total_discarded != 0 else 0
    
    pi_score = intercept + (a * turn_count) + (b * meld_count) + part_c + part_d + part_e
    
    # 套用 min(1, score) 確保機率不超過 1
    # 如果機率不能為負數，建議改用 max(0, min(1, pi_score))
    return min(1.0, pi_score)

# --- 使用範例 ---
params = {
    "intercept": -0.1562, # 截距 (範例值)
    "a": 0.0415,         # 巡數權重
    "b": 0.1052,          # 副露數權重
    "c": 0.2408,          # 中張數比例權重
    "d": 0.2290,         # 花色分布權重
    "e": -0.1635,          # 字牌比例權重
    
    "turn_count": 6,              # 巡數
    "meld_count": 1,               # 副露數 (碰、吃、明槓)
    "middle_cards_discarded": 1,   # 中張數 (3~7)
    "total_discarded": 6,         # 總丟牌數
    "max_suit_discarded": 3,       # 最多被丟的花色數量
    "total_suit_discarded": 3,     # 總丟花色牌數 (該家)
    "honor_cards_discarded": 3,    # 丟字牌數
    "player_total_discarded": 6   # 總丟牌數 (該家)
}

result = calculate_ting_probability(**params)
print(f"聽牌機率 Pi: {result:.2%}")