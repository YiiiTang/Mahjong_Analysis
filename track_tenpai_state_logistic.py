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
        'early': {
            'a': 1.2342, 'c': 0.2738, 'd': -0.0808, 'e': 0.2366, 'f': -0.1167,
            'g': 0.6591, 'h': -0.0378, 'i': -0.1144, 'j': 0.4803, 'k': 0.0201,
            'l': 0.0296, 'm': 0.0210, 'n': -0.4436, 'o': -0.1748, 'p': -0.0716,
            'q': -0.0439, 'r': 0.0800, 's': 0.0202, 't': 0.0306, 'u': 0.0057,
            'v': 0.2130, 'w': 0.0482, 'x': 0.0263, 'y': -0.0915
        },
        'late': {
            'a': 0.1119, 'c': 0.2411, 'd': 0.0292, 'e': 0.0223, 'f': -0.0522,
            'g': 1.5731, 'h': -0.2439, 'i': -0.1971, 'j': 0.2024, 'k': 0.0828,
            'l': -0.0509, 'm': -0.0877, 'n': -0.1201, 'o': -0.2063, 'p': -0.1284,
            'q': -0.0918, 'r': 0.0083, 's': -0.0230, 't': 0.0117, 'u': -0.0163,
            'v': 0.1357, 'w': 0.0597, 'x': -0.0148, 'y': -0.1500
        },
    },
    1: {
        'early': {
            'a': 0.9310, 'c': 0.1946, 'd': 0.5636, 'e': 0.1597, 'f': -0.4686,
            'g': 0.0610, 'h': 0.0066, 'i': -0.0137, 'j': 0.0804, 'k': -0.0221,
            'l': -0.0384, 'm': -0.0021, 'n': -0.0378, 'o': -0.0516, 'p': 0.0361,
            'q': -0.0104, 'r': -0.0110, 's': -0.0781, 't': -0.0102, 'u': -0.0412,
            'v': 0.0765, 'w': 0.0097, 'x': -0.0281, 'y': -0.0001
        },
        'late': {
            'a': 0.3352, 'c': 0.1563, 'd': 0.3680, 'e': -0.0410, 'f': -0.2932,
            'g': 0.3849, 'h': -0.1977, 'i': -0.0664, 'j': 0.1523, 'k': -0.0093,
            'l': -0.0395, 'm': -0.0495, 'n': -0.0044, 'o': -0.0027, 'p': -0.0255,
            'q': -0.0569, 'r': -0.0010, 's': -0.0311, 't': -0.0604, 'u': -0.0365,
            'v': 0.0736, 'w': -0.0259, 'x': -0.0659, 'y': 0.0087
        },
    },
    2: {
        'early': {
            'a': 0.5922, 'c': 0.1504, 'd': 0.5107, 'e': 0.0814, 'f': -0.4489,
            'g': -0.0274, 'h': -0.0212, 'i': 0.0003, 'j': 0.0808, 'k': -0.0125,
            'l': -0.0197, 'm': -0.1156, 'n': -0.0025, 'o': 0.0210, 'p': 0.0098,
            'q': -0.0176, 'r': -0.0619, 's': -0.0904, 't': -0.0724, 'u': -0.0330,
            'v': 0.0813, 'w': -0.0177, 'x': -0.0031, 'y': 0.0277
        },
        'late': {
            'a': 0.3827, 'c': 0.1601, 'd': 0.3283, 'e': -0.0122, 'f': -0.3002,
            'g': 0.1732, 'h': -0.0869, 'i': -0.1260, 'j': 0.1298, 'k': -0.0671,
            'l': -0.0180, 'm': -0.0209, 'n': -0.0323, 'o': 0.0041, 'p': -0.0046,
            'q': -0.0245, 'r': 0.0008, 's': -0.0254, 't': -0.0452, 'u': -0.0392,
            'v': 0.0674, 'w': -0.0004, 'x': -0.0508, 'y': -0.0162
        },
    },
    3: {
        'early': {
            'a': 0.4313, 'c': 0.0998, 'd': 0.4500, 'e': -0.0500, 'f': -0.3203,
            'g': -0.0420, 'h': -0.0020, 'i': 0.0903, 'j': 0.0914, 'k': -0.0099,
            'l': -0.0200, 'm': -0.1597, 'n': 0.0162, 'o': -0.0432, 'p': -0.0430,
            'q': 0.0059, 'r': -0.0440, 's': -0.1069, 't': -0.0320, 'u': -0.0180,
            'v': 0.0421, 'w': 0.0492, 'x': -0.0030, 'y': 0.0142
        },
        'late': {
            'a': 0.4134, 'c': 0.1546, 'd': 0.3086, 'e': 0.0113, 'f': -0.3052,
            'g': 0.0531, 'h': -0.0808, 'i': -0.0464, 'j': 0.0608, 'k': 0.0064,
            'l': -0.0732, 'm': 0.0002, 'n': 0.0037, 'o': 0.0108, 'p': -0.0053,
            'q': -0.0019, 'r': -0.0086, 's': 0.0073, 't': -0.0591, 'u': -0.0534,
            'v': 0.0484, 'w': -0.0352, 'x': -0.0325, 'y': 0.0351
        },
    },
    4: {
        'early': {
            'a': 0.4274, 'c': 0.1550, 'd': 0.4208, 'e': 0.1175, 'f': -0.4540,
            'g': -0.1021, 'h': -0.1043, 'i': -0.2231, 'j': 0.1893, 'k': -0.1244,
            'l': -0.0334, 'm': 0.0000, 'n': -0.0331, 'o': 0.1117, 'p': -0.0193,
            'q': 0.2052, 'r': -0.1047, 's': -0.1962, 't': -0.1042, 'u': 0.0000,
            'v': 0.1137, 'w': -0.0912, 'x': -0.1446, 'y': 0.0000
        },
        'late': {
            'a': -0.0118, 'c': 0.3799, 'd': 0.4010, 'e': -0.0627, 'f': -0.3191,
            'g': -0.0073, 'h': 0.4594, 'i': -0.0072, 'j': -0.0778, 'k': -0.2652,
            'l': -0.1864, 'm': -0.1044, 'n': 0.3326, 'o': 0.0292, 'p': -0.0168,
            'q': 0.0539, 'r': -0.0525, 's': -0.0792, 't': -0.1591, 'u': -0.1400,
            'v': 0.0683, 'w': -0.1818, 'x': -0.2398, 'y': 0.2679
        },
    },
}
MEANS_BY_MELD = {
    0: {
        'early': {
            'a': 3.5842, 'c': 0.5535, 'd': 0.1571, 'e': 0.2290, 'f': 0.6139,
            'g': 0.0994, 'h': 0.0020, 'i': 0.0044, 'j': 0.1600, 'k': 0.0317,
            'l': 0.0041, 'm': 0.0003, 'n': 0.2628, 'o': 0.1640, 'p': 0.0637,
            'q': 0.0122, 'r': 0.1274, 's': 0.0464, 't': 0.0108, 'u': 0.0011,
            'v': 0.0884, 'w': 0.0231, 'x': 0.0036, 'y': 0.0003
        },
        'late': {
            'a': 10.8828, 'c': 0.4768, 'd': 0.3420, 'e': 0.3661, 'f': 0.2919,
            'g': 0.1426, 'h': 0.0141, 'i': 0.0177, 'j': 0.1930, 'k': 0.1467,
            'l': 0.0472, 'm': 0.0059, 'n': 0.0286, 'o': 0.0704, 'p': 0.0828,
            'q': 0.0398, 'r': 0.0546, 's': 0.0792, 't': 0.0502, 'u': 0.0142,
            'v': 0.0734, 'w': 0.0724, 'x': 0.0350, 'y': 0.0053
        },
    },
    1: {
        'early': {
            'a': 5.0301, 'c': 0.4365, 'd': 0.1212, 'e': 0.2894, 'f': 0.5893,
            'g': 0.1525, 'h': 0.0051, 'i': 0.0106, 'j': 0.2037, 'k': 0.0406,
            'l': 0.0047, 'm': 0.0002, 'n': 0.1438, 'o': 0.1083, 'p': 0.0547,
            'q': 0.0136, 'r': 0.1611, 's': 0.0777, 't': 0.0192, 'u': 0.0016,
            'v': 0.1260, 'w': 0.0377, 'x': 0.0067, 'y': 0.0005
        },
        'late': {
            'a': 10.6380, 'c': 0.4616, 'd': 0.2278, 'e': 0.3782, 'f': 0.3940,
            'g': 0.3142, 'h': 0.0427, 'i': 0.0459, 'j': 0.2193, 'k': 0.1297,
            'l': 0.0412, 'm': 0.0042, 'n': 0.0174, 'o': 0.0475, 'p': 0.0718,
            'q': 0.0450, 'r': 0.0549, 's': 0.0856, 't': 0.0600, 'u': 0.0169,
            'v': 0.0844, 'w': 0.0804, 'x': 0.0347, 'y': 0.0069
        },
    },
    2: {
        'early': {
            'a': 6.1768, 'c': 0.4000, 'd': 0.1806, 'e': 0.3388, 'f': 0.4806,
            'g': 0.1732, 'h': 0.0056, 'i': 0.0129, 'j': 0.2804, 'k': 0.0686,
            'l': 0.0081, 'm': 0.0004, 'n': 0.0615, 'o': 0.0664, 'p': 0.0416,
            'q': 0.0116, 'r': 0.1447, 's': 0.0809, 't': 0.0240, 'u': 0.0024,
            'v': 0.1502, 'w': 0.0518, 'x': 0.0071, 'y': 0.0004
        },
        'late': {
            'a': 10.8681, 'c': 0.4733, 'd': 0.2468, 'e': 0.3711, 'f': 0.3820,
            'g': 0.3173, 'h': 0.0461, 'i': 0.0418, 'j': 0.2489, 'k': 0.1419,
            'l': 0.0383, 'm': 0.0045, 'n': 0.0165, 'o': 0.0475, 'p': 0.0623,
            'q': 0.0424, 'r': 0.0477, 's': 0.0799, 't': 0.0552, 'u': 0.0158,
            'v': 0.0828, 'w': 0.0795, 'x': 0.0310, 'y': 0.0054
        },
    },
    3: {
        'early': {
            'a': 6.8016, 'c': 0.4194, 'd': 0.2230, 'e': 0.3560, 'f': 0.4209,
            'g': 0.1746, 'h': 0.0041, 'i': 0.0102, 'j': 0.3500, 'k': 0.0828,
            'l': 0.0103, 'm': 0.0007, 'n': 0.0314, 'o': 0.0517, 'p': 0.0329,
            'q': 0.0122, 'r': 0.1127, 's': 0.0732, 't': 0.0222, 'u': 0.0030,
            'v': 0.1434, 'w': 0.0606, 'x': 0.0115, 'y': 0.0015
        },
        'late': {
            'a': 11.1805, 'c': 0.4809, 'd': 0.2672, 'e': 0.3699, 'f': 0.3628,
            'g': 0.3249, 'h': 0.0484, 'i': 0.0425, 'j': 0.2563, 'k': 0.1520,
            'l': 0.0419, 'm': 0.0068, 'n': 0.0181, 'o': 0.0483, 'p': 0.0651,
            'q': 0.0342, 'r': 0.0499, 's': 0.0728, 't': 0.0470, 'u': 0.0152,
            'v': 0.0763, 'w': 0.0743, 'x': 0.0342, 'y': 0.0073
        },
    },
    4: {
        'early': {
            'a': 7.2423, 'c': 0.4327, 'd': 0.2620, 'e': 0.3416, 'f': 0.3964,
            'g': 0.1662, 'h': 0.0030, 'i': 0.0117, 'j': 0.4538, 'k': 0.0885,
            'l': 0.0077, 'm': 0.0000, 'n': 0.0192, 'o': 0.0577, 'p': 0.0577,
            'q': 0.0038, 'r': 0.0654, 's': 0.0846, 't': 0.0115, 'u': 0.0000,
            'v': 0.1077, 'w': 0.0346, 'x': 0.0077, 'y': 0.0000
        },
        'late': {
            'a': 11.6483, 'c': 0.4911, 'd': 0.2944, 'e': 0.3552, 'f': 0.3504,
            'g': 0.3239, 'h': 0.0502, 'i': 0.0417, 'j': 0.2679, 'k': 0.1477,
            'l': 0.0540, 'm': 0.0088, 'n': 0.0121, 'o': 0.0529, 'p': 0.0573,
            'q': 0.0309, 'r': 0.0463, 's': 0.0617, 't': 0.0364, 'u': 0.0143,
            'v': 0.0794, 'w': 0.0860, 'x': 0.0342, 'y': 0.0077
        },
    },
}
SCALES_BY_MELD = {
    0: {
        'early': {
            'a': 2.1529, 'c': 0.4018, 'd': 0.2531, 'e': 0.2690, 'f': 0.3701,
            'g': 0.1764, 'h': 0.0206, 'i': 0.0275, 'j': 0.3666, 'k': 0.1752,
            'l': 0.0637, 'm': 0.0183, 'n': 0.4401, 'o': 0.3703, 'p': 0.2442,
            'q': 0.1097, 'r': 0.3334, 's': 0.2103, 't': 0.1034, 'u': 0.0339,
            'v': 0.2839, 'w': 0.1502, 'x': 0.0596, 'y': 0.0183
        },
        'late': {
            'a': 1.7763, 'c': 0.1216, 'd': 0.1780, 'e': 0.1443, 'f': 0.1627,
            'g': 0.1570, 'h': 0.0487, 'i': 0.0413, 'j': 0.3946, 'k': 0.3538,
            'l': 0.2121, 'm': 0.0764, 'n': 0.1666, 'o': 0.2559, 'p': 0.2756,
            'q': 0.1955, 'r': 0.2272, 's': 0.2700, 't': 0.2184, 'u': 0.1185,
            'v': 0.2608, 'w': 0.2591, 'x': 0.1838, 'y': 0.0729
        },
    },
    1: {
        'early': {
            'a': 1.9382, 'c': 0.3447, 'd': 0.1627, 'e': 0.2270, 'f': 0.2730,
            'g': 0.1638, 'h': 0.0319, 'i': 0.0397, 'j': 0.4027, 'k': 0.1973,
            'l': 0.0685, 'm': 0.0147, 'n': 0.3509, 'o': 0.3108, 'p': 0.2274,
            'q': 0.1159, 'r': 0.3676, 's': 0.2677, 't': 0.1371, 'u': 0.0400,
            'v': 0.3318, 'w': 0.1905, 'x': 0.0815, 'y': 0.0222
        },
        'late': {
            'a': 1.6654, 'c': 0.1318, 'd': 0.1353, 'e': 0.1423, 'f': 0.1500,
            'g': 0.1442, 'h': 0.0805, 'i': 0.0548, 'j': 0.4138, 'k': 0.3359,
            'l': 0.1987, 'm': 0.0649, 'n': 0.1308, 'o': 0.2127, 'p': 0.2582,
            'q': 0.2073, 'r': 0.2278, 's': 0.2798, 't': 0.2374, 'u': 0.1289,
            'v': 0.2779, 'w': 0.2718, 'x': 0.1831, 'y': 0.0826
        },
    },
    2: {
        'early': {
            'a': 1.5326, 'c': 0.2465, 'd': 0.1696, 'e': 0.1930, 'f': 0.2279,
            'g': 0.1488, 'h': 0.0308, 'i': 0.0412, 'j': 0.4492, 'k': 0.2527,
            'l': 0.0898, 'm': 0.0202, 'n': 0.2402, 'o': 0.2489, 'p': 0.1997,
            'q': 0.1072, 'r': 0.3518, 's': 0.2727, 't': 0.1532, 'u': 0.0485,
            'v': 0.3573, 'w': 0.2215, 'x': 0.0838, 'y': 0.0202
        },
        'late': {
            'a': 1.7529, 'c': 0.1238, 'd': 0.1351, 'e': 0.1380, 'f': 0.1421,
            'g': 0.1391, 'h': 0.0826, 'i': 0.0526, 'j': 0.4324, 'k': 0.3490,
            'l': 0.1919, 'm': 0.0669, 'n': 0.1275, 'o': 0.2127, 'p': 0.2418,
            'q': 0.2014, 'r': 0.2131, 's': 0.2712, 't': 0.2284, 'u': 0.1247,
            'v': 0.2756, 'w': 0.2705, 'x': 0.1732, 'y': 0.0735
        },
    },
    3: {
        'early': {
            'a': 1.2141, 'c': 0.1924, 'd': 0.1601, 'e': 0.1782, 'f': 0.1971,
            'g': 0.1367, 'h': 0.0253, 'i': 0.0356, 'j': 0.4770, 'k': 0.2755,
            'l': 0.1012, 'm': 0.0272, 'n': 0.1744, 'o': 0.2215, 'p': 0.1783,
            'q': 0.1098, 'r': 0.3162, 's': 0.2604, 't': 0.1472, 'u': 0.0543,
            'v': 0.3505, 'w': 0.2386, 'x': 0.1064, 'y': 0.0384
        },
        'late': {
            'a': 1.8969, 'c': 0.1190, 'd': 0.1330, 'e': 0.1398, 'f': 0.1398,
            'g': 0.1367, 'h': 0.0839, 'i': 0.0523, 'j': 0.4366, 'k': 0.3590,
            'l': 0.2004, 'm': 0.0824, 'n': 0.1333, 'o': 0.2144, 'p': 0.2467,
            'q': 0.1817, 'r': 0.2176, 's': 0.2598, 't': 0.2116, 'u': 0.1224,
            'v': 0.2655, 'w': 0.2623, 'x': 0.1817, 'y': 0.0850
        },
    },
    4: {
        'early': {
            'a': 0.9481, 'c': 0.1571, 'd': 0.1567, 'e': 0.1583, 'f': 0.1864,
            'g': 0.1254, 'h': 0.0197, 'i': 0.0378, 'j': 0.4979, 'k': 0.2840,
            'l': 0.0874, 'm': 1.0000, 'n': 0.1373, 'o': 0.2332, 'p': 0.2332,
            'q': 0.0619, 'r': 0.2472, 's': 0.2783, 't': 0.1068, 'u': 1.0000,
            'v': 0.3100, 'w': 0.1828, 'x': 0.0874, 'y': 1.0000
        },
        'late': {
            'a': 2.0357, 'c': 0.1126, 'd': 0.1277, 'e': 0.1323, 'f': 0.1345,
            'g': 0.1284, 'h': 0.0817, 'i': 0.0507, 'j': 0.4429, 'k': 0.3548,
            'l': 0.2261, 'm': 0.0935, 'n': 0.1095, 'o': 0.2239, 'p': 0.2325,
            'q': 0.1730, 'r': 0.2101, 's': 0.2407, 't': 0.1872, 'u': 0.1189,
            'v': 0.2703, 'w': 0.2804, 'x': 0.1817, 'y': 0.0875
        },
    },
}

INTERCEPTS_BY_MELD = {
    0: {
        'early': -5.8499,
        'late': -3.0905
    },
    1: {
        'early': -3.3621,
        'late': -0.4767
    },
    2: {
        'early': -1.5980,
        'late': 0.2320
    },
    3: {
        'early': -0.3839,
        'late': 1.0659
    },
    4: {
        'early': 1.1765,
        'late': 2.3559
    },
}

# AUTO-UPDATE-END (請勿刪除此行)
# ==========================================

def calculate_probability(features, current_meld_count):
    meld = min(current_meld_count, 4)
    phase = 'early' if features['a'] <= 8 else 'late'
    
    W = WEIGHTS_BY_MELD[meld][phase]
    M = MEANS_BY_MELD[meld][phase]
    S = SCALES_BY_MELD[meld][phase]
    intercept = INTERCEPTS_BY_MELD[meld][phase]
    
    keys = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
    
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

                stats['last_discard_type'] = action

                discard_nth = 0
                tile_name = "" 
                
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

                    if suit == 1: stats['discard_wan'] += 1
                    elif suit == 2: stats['discard_tong'] += 1
                    elif suit == 3: stats['discard_tiao'] += 1

                td = stats['total_discard']
                turn = stats['turn_count']
                def safe_div(a, b): return a / b if b > 0 else 0.0

                number_tiles = stats['discard_wan'] + stats['discard_tong'] + stats['discard_tiao']
                max_suit = max(stats['discard_wan'], stats['discard_tong'], stats['discard_tiao'])
                
                feat_c_concentration = round(1.0 - safe_div(max_suit, number_tiles), 4)
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

                pred_prob = calculate_probability(features, stats['meld_count'])

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

    print(f"📂 開始批次處理資料夾: {folder_path}")
    print(f"📝 總共找到 {len(files)} 個檔案，開始分析...\n")

    for file in files:
        try:
            game_log = track_game_state(str(file))
            master_log.extend(game_log)
            success_count += 1
            print(f"✅ 已完成處理: {file.name}") 
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