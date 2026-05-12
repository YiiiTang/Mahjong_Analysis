import pandas as pd
import json
from pathlib import Path
from analyze import getRound
from fileProcess import RoundState, States, playerState

# ==========================================
# AUTO-UPDATE-START
WEIGHTS_BY_MELD = {
    0: {
        'early': {
            'a': 0.0161, 'c': 0.0080, 'd': 5.0287, 'e': 5.3558, 'f': 7.3575,
            'g': 0.0054, 'h': 0.0020, 'i': 0.0004, 'j': 0.0187, 'k': 0.0053,
            'l': 0.0021, 'm': 0.0017, 'n': 0.0094, 'o': 0.0065, 'p': 0.0034,
            'q': 0.0015, 'r': 0.0080, 's': 0.0040, 't': 0.0033, 'u': 0.0011,
            'v': 0.0089, 'w': 0.0040, 'x': 0.0024, 'y': -0.0002
        },
        'late': {
            'a': 0.0095, 'c': 0.0166, 'd': -42.1652, 'e': -34.2739, 'f': -38.5357,
            'g': 0.1272, 'h': -0.0050, 'i': -0.0127, 'j': 0.0033, 'k': -0.0053,
            'l': -0.0071, 'm': -0.0033, 'n': -0.0070, 'o': -0.0114, 'p': -0.0118,
            'q': -0.0091, 'r': -0.0057, 's': -0.0077, 't': -0.0054, 'u': -0.0016,
            'v': 0.0025, 'w': -0.0022, 'x': -0.0073, 'y': -0.0070
        },
    },
    1: {
        'early': {
            'a': 0.0470, 'c': 0.0240, 'd': 1.2009, 'e': 1.6166, 'f': 1.9099,
            'g': 0.0006, 'h': 0.0045, 'i': 0.0023, 'j': -0.0021, 'k': -0.0026,
            'l': -0.0028, 'm': -0.0004, 'n': 0.0084, 'o': 0.0006, 'p': 0.0001,
            'q': -0.0013, 'r': 0.0004, 's': -0.0058, 't': -0.0022, 'u': -0.0031,
            'v': 0.0018, 'w': 0.0000, 'x': -0.0030, 'y': 0.0003
        },
        'late': {
            'a': 0.0712, 'c': 0.0333, 'd': -5.1235, 'e': -5.5015, 'f': -5.8387,
            'g': 0.0778, 'h': -0.0384, 'i': -0.0131, 'j': 0.3130, 'k': 0.2289,
            'l': 0.1216, 'm': 0.0355, 'n': 0.0861, 'o': 0.1460, 'p': 0.1724,
            'q': 0.1335, 'r': 0.1574, 's': 0.1837, 't': 0.1554, 'u': 0.0822,
            'v': 0.2053, 'w': 0.1830, 'x': 0.1143, 'y': 0.0600
        },
    },
    2: {
        'early': {
            'a': 0.0698, 'c': 0.0215, 'd': 4.3592, 'e': 4.8865, 'f': 5.6846,
            'g': -0.0057, 'h': -0.0041, 'i': 0.0022, 'j': 0.0059, 'k': -0.0066,
            'l': -0.0038, 'm': -0.0062, 'n': 0.0101, 'o': 0.0059, 'p': 0.0005,
            'q': -0.0065, 'r': -0.0059, 's': -0.0091, 't': -0.0111, 'u': -0.0057,
            'v': 0.0096, 'w': -0.0042, 'x': 0.0010, 'y': 0.0031
        },
        'late': {
            'a': 0.0810, 'c': 0.0379, 'd': -11.4594, 'e': -11.8301, 'f': -12.2131,
            'g': 0.0378, 'h': -0.0158, 'i': -0.0245, 'j': 0.1631, 'k': 0.0964,
            'l': 0.0540, 'm': 0.0149, 'n': 0.0362, 'o': 0.0665, 'p': 0.0738,
            'q': 0.0587, 'r': 0.0665, 's': 0.0801, 't': 0.0613, 'u': 0.0291,
            'v': 0.0991, 'w': 0.0832, 'x': 0.0445, 'y': 0.0194
        },
    },
    3: {
        'early': {
            'a': 0.0781, 'c': 0.0196, 'd': 4.5541, 'e': 4.9394, 'f': 5.3757,
            'g': -0.0071, 'h': -0.0007, 'i': 0.0158, 'j': 0.0207, 'k': -0.0055,
            'l': -0.0001, 'm': -0.0119, 'n': 0.0067, 'o': -0.0067, 'p': -0.0058,
            'q': 0.0014, 'r': -0.0148, 's': -0.0182, 't': -0.0067, 'u': -0.0047,
            'v': 0.0076, 'w': 0.0057, 'x': -0.0017, 'y': 0.0031
        },
        'late': {
            'a': 0.0757, 'c': 0.0351, 'd': -42.8572, 'e': -45.0283, 'f': -45.1805,
            'g': 0.0137, 'h': -0.0207, 'i': -0.0068, 'j': -0.0961, 'k': -0.0864,
            'l': -0.0653, 'm': -0.0220, 'n': -0.0322, 'o': -0.0497, 'p': -0.0605,
            'q': -0.0489, 'r': -0.0603, 's': -0.0661, 't': -0.0629, 'u': -0.0383,
            'v': -0.0601, 'w': -0.0674, 'x': -0.0462, 'y': -0.0208
        },
    },
    4: {
        'early': {
            'a': 0.0701, 'c': 0.0286, 'd': -112.5403, 'e': -113.7580, 'f': -133.1500,
            'g': -0.0272, 'h': -0.0076, 'i': -0.0377, 'j': 0.0267, 'k': -0.0175,
            'l': -0.0029, 'm': -0.0000, 'n': -0.0156, 'o': 0.0212, 'p': -0.0161,
            'q': 0.0213, 'r': -0.0159, 's': -0.0456, 't': -0.0228, 'u': 0.0000,
            'v': 0.0377, 'w': -0.0075, 'x': -0.0234, 'y': 0.0000
        },
        'late': {
            'a': 0.0009, 'c': 0.0402, 'd': -21.0823, 'e': -21.9371, 'f': -22.6662,
            'g': 0.0168, 'h': 0.0166, 'i': -0.0127, 'j': -0.0819, 'k': -0.0838,
            'l': -0.0507, 'm': -0.0229, 'n': -0.0118, 'o': -0.0327, 'p': -0.0363,
            'q': -0.0323, 'r': -0.0433, 's': -0.0473, 't': -0.0507, 'u': -0.0328,
            'v': -0.0380, 'w': -0.0574, 'x': -0.0551, 'y': -0.0078
        },
    },
}
MEANS_BY_MELD = {
    0: {
        'early': {
            'a': 3.5866, 'c': 0.5536, 'd': 0.1564, 'e': 0.2279, 'f': 0.6157,
            'g': 0.0987, 'h': 0.0020, 'i': 0.0044, 'j': 0.1594, 'k': 0.0318,
            'l': 0.0040, 'm': 0.0004, 'n': 0.2642, 'o': 0.1635, 'p': 0.0639,
            'q': 0.0120, 'r': 0.1279, 's': 0.0462, 't': 0.0105, 'u': 0.0013,
            'v': 0.0879, 'w': 0.0231, 'x': 0.0037, 'y': 0.0003
        },
        'late': {
            'a': 10.8702, 'c': 0.4761, 'd': 0.3413, 'e': 0.3664, 'f': 0.2923,
            'g': 0.1437, 'h': 0.0142, 'i': 0.0180, 'j': 0.1950, 'k': 0.1467,
            'l': 0.0471, 'm': 0.0065, 'n': 0.0290, 'o': 0.0689, 'p': 0.0811,
            'q': 0.0396, 'r': 0.0536, 's': 0.0799, 't': 0.0514, 'u': 0.0137,
            'v': 0.0745, 'w': 0.0724, 'x': 0.0341, 'y': 0.0053
        },
    },
    1: {
        'early': {
            'a': 5.0224, 'c': 0.4373, 'd': 0.1209, 'e': 0.2891, 'f': 0.5900,
            'g': 0.1518, 'h': 0.0052, 'i': 0.0106, 'j': 0.2023, 'k': 0.0404,
            'l': 0.0047, 'm': 0.0003, 'n': 0.1457, 'o': 0.1084, 'p': 0.0548,
            'q': 0.0135, 'r': 0.1606, 's': 0.0771, 't': 0.0192, 'u': 0.0017,
            'v': 0.1259, 'w': 0.0382, 'x': 0.0068, 'y': 0.0005
        },
        'late': {
            'a': 10.6396, 'c': 0.4617, 'd': 0.2291, 'e': 0.3786, 'f': 0.3923,
            'g': 0.3141, 'h': 0.0431, 'i': 0.0457, 'j': 0.2199, 'k': 0.1307,
            'l': 0.0393, 'm': 0.0042, 'n': 0.0166, 'o': 0.0486, 'p': 0.0715,
            'q': 0.0452, 'r': 0.0556, 's': 0.0845, 't': 0.0601, 'u': 0.0167,
            'v': 0.0845, 'w': 0.0801, 'x': 0.0351, 'y': 0.0074
        },
    },
    2: {
        'early': {
            'a': 6.1722, 'c': 0.3991, 'd': 0.1804, 'e': 0.3396, 'f': 0.4800,
            'g': 0.1726, 'h': 0.0054, 'i': 0.0130, 'j': 0.2851, 'k': 0.0673,
            'l': 0.0081, 'm': 0.0004, 'n': 0.0607, 'o': 0.0661, 'p': 0.0417,
            'q': 0.0116, 'r': 0.1417, 's': 0.0791, 't': 0.0250, 'u': 0.0025,
            'v': 0.1511, 'w': 0.0521, 'x': 0.0070, 'y': 0.0005
        },
        'late': {
            'a': 10.8573, 'c': 0.4727, 'd': 0.2470, 'e': 0.3699, 'f': 0.3830,
            'g': 0.3176, 'h': 0.0463, 'i': 0.0420, 'j': 0.2477, 'k': 0.1430,
            'l': 0.0372, 'm': 0.0046, 'n': 0.0171, 'o': 0.0469, 'p': 0.0639,
            'q': 0.0426, 'r': 0.0474, 's': 0.0796, 't': 0.0533, 'u': 0.0161,
            'v': 0.0824, 'w': 0.0793, 'x': 0.0330, 'y': 0.0055
        },
    },
    3: {
        'early': {
            'a': 6.7931, 'c': 0.4194, 'd': 0.2233, 'e': 0.3561, 'f': 0.4206,
            'g': 0.1742, 'h': 0.0043, 'i': 0.0106, 'j': 0.3526, 'k': 0.0845,
            'l': 0.0106, 'm': 0.0006, 'n': 0.0302, 'o': 0.0514, 'p': 0.0322,
            'q': 0.0103, 'r': 0.1162, 's': 0.0724, 't': 0.0236, 'u': 0.0033,
            'v': 0.1407, 'w': 0.0585, 'x': 0.0115, 'y': 0.0012
        },
        'late': {
            'a': 11.1844, 'c': 0.4807, 'd': 0.2659, 'e': 0.3714, 'f': 0.3626,
            'g': 0.3250, 'h': 0.0496, 'i': 0.0419, 'j': 0.2576, 'k': 0.1498,
            'l': 0.0434, 'm': 0.0064, 'n': 0.0176, 'o': 0.0480, 'p': 0.0633,
            'q': 0.0374, 'r': 0.0471, 's': 0.0723, 't': 0.0491, 'u': 0.0152,
            'v': 0.0753, 'w': 0.0766, 'x': 0.0341, 'y': 0.0065
        },
    },
    4: {
        'early': {
            'a': 7.2185, 'c': 0.4239, 'd': 0.2560, 'e': 0.3420, 'f': 0.4020,
            'g': 0.1653, 'h': 0.0024, 'i': 0.0106, 'j': 0.4523, 'k': 0.0800,
            'l': 0.0092, 'm': 0.0000, 'n': 0.0154, 'o': 0.0492, 'p': 0.0615,
            'q': 0.0062, 'r': 0.0708, 's': 0.0769, 't': 0.0092, 'u': 0.0000,
            'v': 0.1323, 'w': 0.0308, 'x': 0.0062, 'y': 0.0000
        },
        'late': {
            'a': 11.5961, 'c': 0.4936, 'd': 0.2966, 'e': 0.3543, 'f': 0.3490,
            'g': 0.3191, 'h': 0.0476, 'i': 0.0421, 'j': 0.2743, 'k': 0.1543,
            'l': 0.0520, 'm': 0.0071, 'n': 0.0123, 'o': 0.0520, 'p': 0.0547,
            'q': 0.0326, 'r': 0.0450, 's': 0.0635, 't': 0.0344, 'u': 0.0159,
            'v': 0.0794, 'w': 0.0802, 'x': 0.0317, 'y': 0.0088
        },
    },
}
SCALES_BY_MELD = {
    0: {
        'early': {
            'a': 2.1540, 'c': 0.4021, 'd': 0.2525, 'e': 0.2685, 'f': 0.3691,
            'g': 0.1758, 'h': 0.0206, 'i': 0.0276, 'j': 0.3660, 'k': 0.1755,
            'l': 0.0632, 'm': 0.0199, 'n': 0.4409, 'o': 0.3698, 'p': 0.2446,
            'q': 0.1088, 'r': 0.3339, 's': 0.2099, 't': 0.1020, 'u': 0.0354,
            'v': 0.2831, 'w': 0.1501, 'x': 0.0605, 'y': 0.0177
        },
        'late': {
            'a': 1.7614, 'c': 0.1216, 'd': 0.1776, 'e': 0.1444, 'f': 0.1623,
            'g': 0.1578, 'h': 0.0498, 'i': 0.0416, 'j': 0.3962, 'k': 0.3539,
            'l': 0.2118, 'm': 0.0803, 'n': 0.1678, 'o': 0.2533, 'p': 0.2729,
            'q': 0.1950, 'r': 0.2252, 's': 0.2711, 't': 0.2209, 'u': 0.1161,
            'v': 0.2626, 'w': 0.2592, 'x': 0.1815, 'y': 0.0729
        },
    },
    1: {
        'early': {
            'a': 1.9408, 'c': 0.3455, 'd': 0.1624, 'e': 0.2275, 'f': 0.2736,
            'g': 0.1638, 'h': 0.0322, 'i': 0.0398, 'j': 0.4017, 'k': 0.1969,
            'l': 0.0683, 'm': 0.0172, 'n': 0.3528, 'o': 0.3109, 'p': 0.2276,
            'q': 0.1152, 'r': 0.3671, 's': 0.2668, 't': 0.1374, 'u': 0.0418,
            'v': 0.3317, 'w': 0.1918, 'x': 0.0820, 'y': 0.0222
        },
        'late': {
            'a': 1.6655, 'c': 0.1319, 'd': 0.1352, 'e': 0.1427, 'f': 0.1501,
            'g': 0.1443, 'h': 0.0811, 'i': 0.0549, 'j': 0.4142, 'k': 0.3371,
            'l': 0.1943, 'm': 0.0649, 'n': 0.1279, 'o': 0.2150, 'p': 0.2577,
            'q': 0.2077, 'r': 0.2291, 's': 0.2781, 't': 0.2376, 'u': 0.1282,
            'v': 0.2781, 'w': 0.2714, 'x': 0.1839, 'y': 0.0859
        },
    },
    2: {
        'early': {
            'a': 1.5288, 'c': 0.2459, 'd': 0.1694, 'e': 0.1936, 'f': 0.2277,
            'g': 0.1492, 'h': 0.0306, 'i': 0.0414, 'j': 0.4515, 'k': 0.2506,
            'l': 0.0898, 'm': 0.0197, 'n': 0.2388, 'o': 0.2484, 'p': 0.2000,
            'q': 0.1072, 'r': 0.3488, 's': 0.2699, 't': 0.1560, 'u': 0.0496,
            'v': 0.3581, 'w': 0.2221, 'x': 0.0835, 'y': 0.0213
        },
        'late': {
            'a': 1.7554, 'c': 0.1237, 'd': 0.1352, 'e': 0.1386, 'f': 0.1425,
            'g': 0.1389, 'h': 0.0830, 'i': 0.0527, 'j': 0.4317, 'k': 0.3501,
            'l': 0.1893, 'm': 0.0677, 'n': 0.1295, 'o': 0.2113, 'p': 0.2446,
            'q': 0.2019, 'r': 0.2126, 's': 0.2706, 't': 0.2246, 'u': 0.1260,
            'v': 0.2750, 'w': 0.2702, 'x': 0.1788, 'y': 0.0741
        },
    },
    3: {
        'early': {
            'a': 1.2105, 'c': 0.1911, 'd': 0.1615, 'e': 0.1789, 'f': 0.1968,
            'g': 0.1367, 'h': 0.0266, 'i': 0.0362, 'j': 0.4778, 'k': 0.2782,
            'l': 0.1026, 'm': 0.0243, 'n': 0.1710, 'o': 0.2209, 'p': 0.1766,
            'q': 0.1012, 'r': 0.3204, 's': 0.2592, 't': 0.1519, 'u': 0.0569,
            'v': 0.3477, 'w': 0.2347, 'x': 0.1067, 'y': 0.0344
        },
        'late': {
            'a': 1.8877, 'c': 0.1195, 'd': 0.1332, 'e': 0.1398, 'f': 0.1401,
            'g': 0.1375, 'h': 0.0855, 'i': 0.0520, 'j': 0.4373, 'k': 0.3569,
            'l': 0.2038, 'm': 0.0794, 'n': 0.1317, 'o': 0.2138, 'p': 0.2436,
            'q': 0.1898, 'r': 0.2119, 's': 0.2591, 't': 0.2160, 'u': 0.1223,
            'v': 0.2640, 'w': 0.2659, 'x': 0.1814, 'y': 0.0805
        },
    },
    4: {
        'early': {
            'a': 0.9535, 'c': 0.1630, 'd': 0.1539, 'e': 0.1555, 'f': 0.1819,
            'g': 0.1242, 'h': 0.0177, 'i': 0.0361, 'j': 0.4977, 'k': 0.2713,
            'l': 0.0956, 'm': 1.0000, 'n': 0.1231, 'o': 0.2163, 'p': 0.2403,
            'q': 0.0782, 'r': 0.2564, 's': 0.2665, 't': 0.0956, 'u': 1.0000,
            'v': 0.3388, 'w': 0.1727, 'x': 0.0782, 'y': 1.0000
        },
        'late': {
            'a': 2.0256, 'c': 0.1109, 'd': 0.1277, 'e': 0.1326, 'f': 0.1368,
            'g': 0.1297, 'h': 0.0787, 'i': 0.0508, 'j': 0.4461, 'k': 0.3613,
            'l': 0.2221, 'm': 0.0837, 'n': 0.1104, 'o': 0.2221, 'p': 0.2273,
            'q': 0.1777, 'r': 0.2072, 's': 0.2438, 't': 0.1822, 'u': 0.1250,
            'v': 0.2703, 'w': 0.2717, 'x': 0.1753, 'y': 0.0935
        },
    },
}

INTERCEPTS_BY_MELD = {
    0: {
        'early': 0.0137,
        'late': 0.0863
    },
    1: {
        'early': 0.0805,
        'late': 0.4033
    },
    2: {
        'early': 0.2251,
        'late': 0.5488
    },
    3: {
        'early': 0.4221,
        'late': 0.7159
    },
    4: {
        'early': 0.7200,
        'late': 0.8827
    },
}

# AUTO-UPDATE-END (請勿刪除此行)
# ==========================================

def calculate_linear_score(features, current_meld_count):
    meld = min(current_meld_count, 4)
    phase = 'early' if features['a'] <= 8 else 'late'
    
    W = WEIGHTS_BY_MELD[meld][phase]
    M = MEANS_BY_MELD[meld][phase]
    S = SCALES_BY_MELD[meld][phase]
    intercept = INTERCEPTS_BY_MELD[meld][phase]
    
    keys = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
    
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
            'discard_3_to_7': 0, 'discard_1289': 0,
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
                is_bian_19 = False
                is_bian_28 = False
                
                if tile_str and tile_str.isdigit():
                    card_num = int(tile_str)
                    suit, val = card_num // 100, (card_num // 10) % 10
                    
                    if suit in (1, 2, 3):
                        if 3 <= val <= 7:
                            stats['discard_3_to_7'] += 1
                            is_zhong = True
                        elif val in (1, 9):
                            stats['discard_1289'] += 1
                            is_bian_19 = True
                        elif val in (2, 8):
                            stats['discard_1289'] += 1
                            is_bian_28 = True
                    elif suit == 4:
                        stats['discard_zi'] += 1
                        is_zi = True
                    
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

                td, turn = stats['total_discard'], stats['turn_count']
                def safe_div(a, b): return a / b if b > 0 else 0.0

                num_tiles = stats['discard_wan'] + stats['discard_tong'] + stats['discard_tiao']
                max_s = max(stats['discard_wan'], stats['discard_tong'], stats['discard_tiao'])
                
                feat_c_concentration = round(1.0 - safe_div(max_s, num_tiles), 4)
                feat_d_prop_zhong = round(safe_div(stats['discard_3_to_7'], td), 4)
                feat_e_prop_bian = round(safe_div(stats['discard_1289'], td), 4)
                feat_f_prop_zi = round(safe_div(stats['discard_zi'], td), 4)
                feat_g_moqie_rate = round(safe_div(stats['moqie_count'], td), 4)
                feat_h_moqie_strength = round(safe_div(max(stats['max_continuous_moqie'] - 2, 0), turn), 4)
                feat_i_mo_to_shou = round(safe_div(stats['moqie_to_shouqie_count'], turn), 4)

                features = {
                    'a': turn, 
                    'c': feat_c_concentration, 
                    'd': feat_d_prop_zhong,
                    'e': feat_e_prop_bian,
                    'f': feat_f_prop_zi,
                    'g': feat_g_moqie_rate, 
                    'h': feat_h_moqie_strength, 
                    'i': feat_i_mo_to_shou,
                    'j': 1 if (is_zhong and discard_nth == 1) else 0,
                    'k': 1 if (is_zhong and discard_nth == 2) else 0,
                    'l': 1 if (is_zhong and discard_nth == 3) else 0,
                    'm': 1 if (is_zhong and discard_nth == 4) else 0,
                    'n': 1 if (is_zi and discard_nth == 1) else 0,
                    'o': 1 if (is_zi and discard_nth == 2) else 0,
                    'p': 1 if (is_zi and discard_nth == 3) else 0,
                    'q': 1 if (is_zi and discard_nth == 4) else 0,
                    'r': 1 if (is_bian_19 and discard_nth == 1) else 0,
                    's': 1 if (is_bian_19 and discard_nth == 2) else 0,
                    't': 1 if (is_bian_19 and discard_nth == 3) else 0,
                    'u': 1 if (is_bian_19 and discard_nth == 4) else 0,
                    'v': 1 if (is_bian_28 and discard_nth == 1) else 0,
                    'w': 1 if (is_bian_28 and discard_nth == 2) else 0,
                    'x': 1 if (is_bian_28 and discard_nth == 3) else 0,
                    'y': 1 if (is_bian_28 and discard_nth == 4) else 0
                }
                
                pred_score = calculate_linear_score(features, stats['meld_count'])

                actual_shanten = states.get_player(states.state[j], actor_loc).shantenCount
                action_name = "手切" if action == "HD" else "摸切"

                tracker_log.append({
                    '檔案名稱': file_name, 'Step_ID': j, '玩家': actor_loc, '動作': action_name, '丟棄牌': tile_str,
                    '實際向聽數': actual_shanten, '預測聽牌分數': round(pred_score, 4),
                    '特徵值': {
                        'feat_a_巡數': turn, 
                        'feat_b_吃碰數': stats['meld_count'],
                        'feat_c_花色集中度': feat_c_concentration,
                        'feat_d_中張比例(3 ~ 7)': feat_d_prop_zhong,
                        'feat_e_邊張比例(1、2、8、9)': feat_e_prop_bian,
                        'feat_f_字牌比例': feat_f_prop_zi,
                        'feat_g_摸切比例': feat_g_moqie_rate, 
                        'feat_h_連續摸切強度': feat_h_moqie_strength, 
                        'feat_i_摸切轉手切': feat_i_mo_to_shou,
                        'feat_j_中張第一張被打出': features['j'], 
                        'feat_k_中張第二張被打出': features['k'], 
                        'feat_l_中張第三張被打出': features['l'], 
                        'feat_m_中張第四張被打出': features['m'],
                        'feat_n_字牌第一張被打出': features['n'],
                        'feat_o_字牌第二張被打出': features['o'],
                        'feat_p_字牌第三張被打出': features['p'],
                        'feat_q_字牌第四張被打出': features['q'],
                        'feat_r_邊張(1、9)第一張被打出': features['r'],
                        'feat_s_邊張(1、9)第二張被打出': features['s'],
                        'feat_t_邊張(1、9)第三張被打出': features['t'],
                        'feat_u_邊張(1、9)第四張被打出': features['u'],
                        'feat_v_邊張(2、8)第一張被打出': features['v'],
                        'feat_w_邊張(2、8)第二張被打出': features['w'],
                        'feat_x_邊張(2、8)第三張被打出': features['x'],
                        'feat_y_邊張(2、8)第四張被打出': features['y']
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
            print(f"✅ 已完成處理: {file.name}") 
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