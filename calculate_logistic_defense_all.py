import math

# sigmoid function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def tenpai_probability(
    jun = 6,                 # 巡數
    furo = 1,              # 副露數
    mid_tiles = 1,         # 中張數(3~7)
    total_discards = 6,    # 總丟牌數
    max_suit_discards = 3, # 最多被丟的花色數量
    total_suit_discards = 3, # 總丟花色牌數
    honor_discards = 3,      # 丟字牌數

    # 權重
    w0 = -2.8961,  
    a = 0.3440,
    b = 0.7942,
    c = 2.0349,
    d = -2.7171,
    e = -1.5216
):
    
    # 防止除以0
    if total_discards == 0:
        total_discards = 1
    if total_suit_discards == 0:
        total_suit_discards = 1

    # feature 計算
    feature_mid_ratio = mid_tiles / total_discards
    feature_suit_bias = 1 - (max_suit_discards / total_suit_discards)
    feature_honor_ratio = honor_discards / total_discards

    # linear combination (z)
    z = (
        w0
        + a * jun
        + b * furo
        + c * feature_mid_ratio
        + d * feature_suit_bias
        + e * feature_honor_ratio
    )

    # sigmoid → 機率
    prob = sigmoid(z)
    print(f"Sigmoid 聽牌機率: {prob:.2%}")

    return prob

# --- 測試數值 ---
tenpai_probability()