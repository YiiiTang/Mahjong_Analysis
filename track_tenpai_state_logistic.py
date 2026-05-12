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
            'a': 1.2023, 'c': 0.2977, 'd': -0.0907, 'e': 0.2336, 'f': -0.1077,
            'g': 0.5136, 'h': 0.0837, 'i': -0.0280, 'j': -0.0480, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.4974, 'l': 0.0263, 'm': 0.0327, 'n': 0.0213,
            'o': -0.4535, 'p': -0.1838, 'q': -0.0877, 'r': -0.0500, 's': 0.0835,
            't': 0.0212, 'u': 0.0298, 'v': 0.0049, 'w': 0.2248, 'x': 0.0500,
            'y': 0.0240, 'z': -0.0944
        },
        'late': {
            'a': 0.0794, 'c': 0.2618, 'd': 0.0028, 'e': 0.0222, 'f': -0.0235,
            'g': 1.0644, 'h': 0.1863, 'i': 0.2552, 'j': -0.0355, 'z1': 0.1863,
            'z2': -0.2873, 'k': 0.2205, 'l': 0.0916, 'm': -0.0474, 'n': -0.0778,
            'o': -0.1227, 'p': -0.2197, 'q': -0.1517, 'r': -0.1061, 's': -0.0035,
            't': -0.0324, 'u': 0.0037, 'v': -0.0183, 'w': 0.1488, 'x': 0.0740,
            'y': -0.0175, 'z': -0.1462
        },
    },
    1: {
        'early': {
            'a': 0.9282, 'c': 0.1993, 'd': 0.5615, 'e': 0.1589, 'f': -0.4666,
            'g': 0.0421, 'h': 0.0273, 'i': -0.0304, 'j': 0.0133, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.0850, 'l': -0.0208, 'm': -0.0384, 'n': -0.0023,
            'o': -0.0417, 'p': -0.0570, 'q': 0.0307, 'r': -0.0136, 's': -0.0083,
            't': -0.0772, 'u': -0.0100, 'v': -0.0411, 'w': 0.0796, 'x': 0.0111,
            'y': -0.0283, 'z': 0.0002
        },
        'late': {
            'a': 0.3174, 'c': 0.1614, 'd': 0.3651, 'e': -0.0458, 'f': -0.2862,
            'g': 0.2015, 'h': 0.0581, 'i': 0.0866, 'j': 0.0109, 'z1': 0.0581,
            'z2': -0.1102, 'k': 0.1547, 'l': -0.0071, 'm': -0.0378, 'n': -0.0491,
            'o': -0.0058, 'p': -0.0003, 'q': -0.0234, 'r': -0.0561, 's': 0.0007,
            't': -0.0343, 'u': -0.0573, 'v': -0.0362, 'w': 0.0746, 'x': -0.0210,
            'y': -0.0643, 'z': 0.0052
        },
    },
    2: {
        'early': {
            'a': 0.5882, 'c': 0.1530, 'd': 0.5086, 'e': 0.0810, 'f': -0.4471,
            'g': -0.0625, 'h': 0.0322, 'i': -0.0400, 'j': 0.0378, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.0850, 'l': -0.0115, 'm': -0.0200, 'n': -0.1179,
            'o': -0.0052, 'p': 0.0161, 'q': 0.0051, 'r': -0.0209, 's': -0.0600,
            't': -0.0904, 'u': -0.0740, 'v': -0.0335, 'w': 0.0829, 'x': -0.0174,
            'y': -0.0028, 'z': 0.0272
        },
        'late': {
            'a': 0.3785, 'c': 0.1649, 'd': 0.3231, 'e': -0.0115, 'f': -0.2961,
            'g': 0.0545, 'h': 0.0859, 'i': 0.0959, 'j': -0.0699, 'z1': 0.0859,
            'z2': -0.1242, 'k': 0.1383, 'l': -0.0624, 'm': -0.0163, 'n': -0.0182,
            'o': -0.0347, 'p': 0.0017, 'q': -0.0098, 'r': -0.0312, 's': 0.0047,
            't': -0.0247, 'u': -0.0434, 'v': -0.0417, 'w': 0.0716, 'x': 0.0022,
            'y': -0.0495, 'z': -0.0166
        },
    },
    3: {
        'early': {
            'a': 0.4308, 'c': 0.0981, 'd': 0.4505, 'e': -0.0495, 'f': -0.3212,
            'g': -0.0407, 'h': -0.0041, 'i': -0.0407, 'j': 0.1146, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.0914, 'l': -0.0105, 'm': -0.0200, 'n': -0.1605,
            'o': 0.0166, 'p': -0.0433, 'q': -0.0428, 'r': 0.0056, 's': -0.0440,
            't': -0.1076, 'u': -0.0330, 'v': -0.0170, 'w': 0.0431, 'x': 0.0492,
            'y': -0.0025, 'z': 0.0141
        },
        'late': {
            'a': 0.3783, 'c': 0.1634, 'd': 0.3002, 'e': 0.0114, 'f': -0.2978,
            'g': -0.1083, 'h': 0.1334, 'i': 0.0673, 'j': 0.0656, 'z1': 0.1334,
            'z2': -0.1272, 'k': 0.0762, 'l': 0.0071, 'm': -0.0704, 'n': -0.0030,
            'o': 0.0002, 'p': -0.0009, 'q': -0.0216, 'r': -0.0133, 's': -0.0016,
            't': 0.0031, 'u': -0.0656, 'v': -0.0517, 'w': 0.0527, 'x': -0.0363,
            'y': -0.0390, 'z': 0.0329
        },
    },
    4: {
        'early': {
            'a': 0.4274, 'c': 0.1524, 'd': 0.4269, 'e': 0.1131, 'f': -0.4555,
            'g': -0.1063, 'h': -0.0572, 'i': -0.0110, 'j': -0.2081, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.1738, 'l': -0.1240, 'm': -0.0324, 'n': 0.0000,
            'o': -0.0374, 'p': 0.1090, 'q': -0.0008, 'r': 0.2083, 's': -0.1008,
            't': -0.1901, 'u': -0.1039, 'v': 0.0000, 'w': 0.1159, 'x': -0.0876,
            'y': -0.1423, 'z': 0.0000
        },
        'late': {
            'a': -0.0622, 'c': 0.3977, 'd': 0.4008, 'e': -0.0591, 'f': -0.3230,
            'g': -0.2220, 'h': 0.4541, 'i': 0.2764, 'j': 0.2110, 'z1': 0.4541,
            'z2': -0.1182, 'k': -0.0238, 'l': -0.2498, 'm': -0.2030, 'n': -0.0897,
            'o': 0.3157, 'p': -0.0221, 'q': -0.0720, 'r': 0.0156, 's': -0.0751,
            't': -0.1201, 'u': -0.1681, 'v': -0.1485, 'w': 0.0838, 'x': -0.1945,
            'y': -0.2409, 'z': 0.2714
        },
    },
}
MEANS_BY_MELD = {
    0: {
        'early': {
            'a': 3.5842, 'c': 0.5535, 'd': 0.1571, 'e': 0.2290, 'f': 0.6139,
            'g': 0.0994, 'h': 0.1938, 'i': 0.0123, 'j': 0.0271, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.1600, 'l': 0.0317, 'm': 0.0041, 'n': 0.0003,
            'o': 0.2628, 'p': 0.1640, 'q': 0.0637, 'r': 0.0122, 's': 0.1274,
            't': 0.0464, 'u': 0.0108, 'v': 0.0011, 'w': 0.0884, 'x': 0.0231,
            'y': 0.0036, 'z': 0.0003
        },
        'late': {
            'a': 10.8828, 'c': 0.4768, 'd': 0.3420, 'e': 0.3661, 'f': 0.2919,
            'g': 0.1426, 'h': 0.3658, 'i': 0.0387, 'j': 0.1915, 'z1': 0.3658,
            'z2': 0.0802, 'k': 0.1930, 'l': 0.1467, 'm': 0.0472, 'n': 0.0059,
            'o': 0.0286, 'p': 0.0704, 'q': 0.0828, 'r': 0.0398, 's': 0.0546,
            't': 0.0792, 'u': 0.0502, 'v': 0.0142, 'w': 0.0734, 'x': 0.0724,
            'y': 0.0350, 'z': 0.0053
        },
    },
    1: {
        'early': {
            'a': 5.0301, 'c': 0.4365, 'd': 0.1212, 'e': 0.2894, 'f': 0.5893,
            'g': 0.1525, 'h': 0.3572, 'i': 0.0303, 'j': 0.0700, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.2037, 'l': 0.0406, 'm': 0.0047, 'n': 0.0002,
            'o': 0.1438, 'p': 0.1083, 'q': 0.0547, 'r': 0.0136, 's': 0.1611,
            't': 0.0777, 'u': 0.0192, 'v': 0.0016, 'w': 0.1260, 'x': 0.0377,
            'y': 0.0067, 'z': 0.0005
        },
        'late': {
            'a': 10.6380, 'c': 0.4616, 'd': 0.2278, 'e': 0.3782, 'f': 0.3940,
            'g': 0.3142, 'h': 1.0459, 'i': 0.1041, 'j': 0.5041, 'z1': 1.0459,
            'z2': 0.2416, 'k': 0.2193, 'l': 0.1297, 'm': 0.0412, 'n': 0.0042,
            'o': 0.0174, 'p': 0.0475, 'q': 0.0718, 'r': 0.0450, 's': 0.0549,
            't': 0.0856, 'u': 0.0600, 'v': 0.0169, 'w': 0.0844, 'x': 0.0804,
            'y': 0.0347, 'z': 0.0069
        },
    },
    2: {
        'early': {
            'a': 6.1768, 'c': 0.4000, 'd': 0.1806, 'e': 0.3388, 'f': 0.4806,
            'g': 0.1732, 'h': 0.4353, 'i': 0.0379, 'j': 0.0924, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.2804, 'l': 0.0686, 'm': 0.0081, 'n': 0.0004,
            'o': 0.0615, 'p': 0.0664, 'q': 0.0416, 'r': 0.0116, 's': 0.1447,
            't': 0.0809, 'u': 0.0240, 'v': 0.0024, 'w': 0.1502, 'x': 0.0518,
            'y': 0.0071, 'z': 0.0004
        },
        'late': {
            'a': 10.8681, 'c': 0.4733, 'd': 0.2468, 'e': 0.3711, 'f': 0.3820,
            'g': 0.3173, 'h': 1.2038, 'i': 0.1003, 'j': 0.4724, 'z1': 1.2038,
            'z2': 0.2705, 'k': 0.2489, 'l': 0.1419, 'm': 0.0383, 'n': 0.0045,
            'o': 0.0165, 'p': 0.0475, 'q': 0.0623, 'r': 0.0424, 's': 0.0477,
            't': 0.0799, 'u': 0.0552, 'v': 0.0158, 'w': 0.0828, 'x': 0.0795,
            'y': 0.0310, 'z': 0.0054
        },
    },
    3: {
        'early': {
            'a': 6.8016, 'c': 0.4194, 'd': 0.2230, 'e': 0.3560, 'f': 0.4209,
            'g': 0.1746, 'h': 0.4826, 'i': 0.0314, 'j': 0.0765, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.3500, 'l': 0.0828, 'm': 0.0103, 'n': 0.0007,
            'o': 0.0314, 'p': 0.0517, 'q': 0.0329, 'r': 0.0122, 's': 0.1127,
            't': 0.0732, 'u': 0.0222, 'v': 0.0030, 'w': 0.1434, 'x': 0.0606,
            'y': 0.0115, 'z': 0.0015
        },
        'late': {
            'a': 11.1805, 'c': 0.4809, 'd': 0.2672, 'e': 0.3699, 'f': 0.3628,
            'g': 0.3249, 'h': 1.3053, 'i': 0.1061, 'j': 0.4939, 'z1': 1.3053,
            'z2': 0.2932, 'k': 0.2563, 'l': 0.1520, 'm': 0.0419, 'n': 0.0068,
            'o': 0.0181, 'p': 0.0483, 'q': 0.0651, 'r': 0.0342, 's': 0.0499,
            't': 0.0728, 'u': 0.0470, 'v': 0.0152, 'w': 0.0763, 'x': 0.0743,
            'y': 0.0342, 'z': 0.0073
        },
    },
    4: {
        'early': {
            'a': 7.2423, 'c': 0.4327, 'd': 0.2620, 'e': 0.3416, 'f': 0.3964,
            'g': 0.1662, 'h': 0.4000, 'i': 0.0385, 'j': 0.0885, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.4538, 'l': 0.0885, 'm': 0.0077, 'n': 0.0000,
            'o': 0.0192, 'p': 0.0577, 'q': 0.0577, 'r': 0.0038, 's': 0.0654,
            't': 0.0846, 'u': 0.0115, 'v': 0.0000, 'w': 0.1077, 'x': 0.0346,
            'y': 0.0077, 'z': 0.0000
        },
        'late': {
            'a': 11.6483, 'c': 0.4911, 'd': 0.2944, 'e': 0.3552, 'f': 0.3504,
            'g': 0.3239, 'h': 1.4190, 'i': 0.1058, 'j': 0.5061, 'z1': 1.4190,
            'z2': 0.3208, 'k': 0.2679, 'l': 0.1477, 'm': 0.0540, 'n': 0.0088,
            'o': 0.0121, 'p': 0.0529, 'q': 0.0573, 'r': 0.0309, 's': 0.0463,
            't': 0.0617, 'u': 0.0364, 'v': 0.0143, 'w': 0.0794, 'x': 0.0860,
            'y': 0.0342, 'z': 0.0077
        },
    },
}
SCALES_BY_MELD = {
    0: {
        'early': {
            'a': 2.1529, 'c': 0.4018, 'd': 0.2531, 'e': 0.2690, 'f': 0.3701,
            'g': 0.1764, 'h': 0.5082, 'i': 0.1103, 'j': 0.1637, 'z1': 1.0000,
            'z2': 1.0000, 'k': 0.3666, 'l': 0.1752, 'm': 0.0637, 'n': 0.0183,
            'o': 0.4401, 'p': 0.3703, 'q': 0.2442, 'r': 0.1097, 's': 0.3334,
            't': 0.2103, 'u': 0.1034, 'v': 0.0339, 'w': 0.2839, 'x': 0.1502,
            'y': 0.0596, 'z': 0.0183
        },
        'late': {
            'a': 1.7763, 'c': 0.1216, 'd': 0.1780, 'e': 0.1443, 'f': 0.1627,
            'g': 0.1570, 'h': 0.8803, 'i': 0.1930, 'j': 0.4471, 'z1': 0.8803,
            'z2': 0.2717, 'k': 0.3946, 'l': 0.3538, 'm': 0.2121, 'n': 0.0764,
            'o': 0.1666, 'p': 0.2559, 'q': 0.2756, 'r': 0.1955, 's': 0.2272,
            't': 0.2700, 'u': 0.2184, 'v': 0.1185, 'w': 0.2608, 'x': 0.2591,
            'y': 0.1838, 'z': 0.0729
        },
    },
    1: {
        'early': {
            'a': 1.9382, 'c': 0.3447, 'd': 0.1627, 'e': 0.2270, 'f': 0.2730,
            'g': 0.1638, 'h': 0.6896, 'i': 0.1713, 'j': 0.2578, 'z1': 1.0000,
            'z2': 1.0000, 'k': 0.4027, 'l': 0.1973, 'm': 0.0685, 'n': 0.0147,
            'o': 0.3509, 'p': 0.3108, 'q': 0.2274, 'r': 0.1159, 's': 0.3676,
            't': 0.2677, 'u': 0.1371, 'v': 0.0400, 'w': 0.3318, 'x': 0.1905,
            'y': 0.0815, 'z': 0.0222
        },
        'late': {
            'a': 1.6654, 'c': 0.1318, 'd': 0.1353, 'e': 0.1423, 'f': 0.1500,
            'g': 0.1442, 'h': 1.3638, 'i': 0.3054, 'j': 0.6087, 'z1': 1.3638,
            'z2': 0.4281, 'k': 0.4138, 'l': 0.3359, 'm': 0.1987, 'n': 0.0649,
            'o': 0.1308, 'p': 0.2127, 'q': 0.2582, 'r': 0.2073, 's': 0.2278,
            't': 0.2798, 'u': 0.2374, 'v': 0.1289, 'w': 0.2779, 'x': 0.2718,
            'y': 0.1831, 'z': 0.0826
        },
    },
    2: {
        'early': {
            'a': 1.5326, 'c': 0.2465, 'd': 0.1696, 'e': 0.1930, 'f': 0.2279,
            'g': 0.1488, 'h': 0.7380, 'i': 0.1911, 'j': 0.2912, 'z1': 1.0000,
            'z2': 1.0000, 'k': 0.4492, 'l': 0.2527, 'm': 0.0898, 'n': 0.0202,
            'o': 0.2402, 'p': 0.2489, 'q': 0.1997, 'r': 0.1072, 's': 0.3518,
            't': 0.2727, 'u': 0.1532, 'v': 0.0485, 'w': 0.3573, 'x': 0.2215,
            'y': 0.0838, 'z': 0.0202
        },
        'late': {
            'a': 1.7529, 'c': 0.1238, 'd': 0.1351, 'e': 0.1380, 'f': 0.1421,
            'g': 0.1391, 'h': 1.4803, 'i': 0.3004, 'j': 0.5987, 'z1': 1.4803,
            'z2': 0.4442, 'k': 0.4324, 'l': 0.3490, 'm': 0.1919, 'n': 0.0669,
            'o': 0.1275, 'p': 0.2127, 'q': 0.2418, 'r': 0.2014, 's': 0.2131,
            't': 0.2712, 'u': 0.2284, 'v': 0.1247, 'w': 0.2756, 'x': 0.2705,
            'y': 0.1732, 'z': 0.0735
        },
    },
    3: {
        'early': {
            'a': 1.2141, 'c': 0.1924, 'd': 0.1601, 'e': 0.1782, 'f': 0.1971,
            'g': 0.1367, 'h': 0.7528, 'i': 0.1744, 'j': 0.2672, 'z1': 1.0000,
            'z2': 1.0000, 'k': 0.4770, 'l': 0.2755, 'm': 0.1012, 'n': 0.0272,
            'o': 0.1744, 'p': 0.2215, 'q': 0.1783, 'r': 0.1098, 's': 0.3162,
            't': 0.2604, 'u': 0.1472, 'v': 0.0543, 'w': 0.3505, 'x': 0.2386,
            'y': 0.1064, 'z': 0.0384
        },
        'late': {
            'a': 1.8969, 'c': 0.1190, 'd': 0.1330, 'e': 0.1398, 'f': 0.1398,
            'g': 0.1367, 'h': 1.5522, 'i': 0.3080, 'j': 0.6140, 'z1': 1.5522,
            'z2': 0.4552, 'k': 0.4366, 'l': 0.3590, 'm': 0.2004, 'n': 0.0824,
            'o': 0.1333, 'p': 0.2144, 'q': 0.2467, 'r': 0.1817, 's': 0.2176,
            't': 0.2598, 'u': 0.2116, 'v': 0.1224, 'w': 0.2655, 'x': 0.2623,
            'y': 0.1817, 'z': 0.0850
        },
    },
    4: {
        'early': {
            'a': 0.9481, 'c': 0.1571, 'd': 0.1567, 'e': 0.1583, 'f': 0.1864,
            'g': 0.1254, 'h': 0.7027, 'i': 0.1923, 'j': 0.2840, 'z1': 1.0000,
            'z2': 1.0000, 'k': 0.4979, 'l': 0.2840, 'm': 0.0874, 'n': 1.0000,
            'o': 0.1373, 'p': 0.2332, 'q': 0.2332, 'r': 0.0619, 's': 0.2472,
            't': 0.2783, 'u': 0.1068, 'v': 1.0000, 'w': 0.3100, 'x': 0.1828,
            'y': 0.0874, 'z': 1.0000
        },
        'late': {
            'a': 2.0357, 'c': 0.1126, 'd': 0.1277, 'e': 0.1323, 'f': 0.1345,
            'g': 0.1284, 'h': 1.6202, 'i': 0.3076, 'j': 0.6183, 'z1': 1.6202,
            'z2': 0.4668, 'k': 0.4429, 'l': 0.3548, 'm': 0.2261, 'n': 0.0935,
            'o': 0.1095, 'p': 0.2239, 'q': 0.2325, 'r': 0.1730, 's': 0.2101,
            't': 0.2407, 'u': 0.1872, 'v': 0.1189, 'w': 0.2703, 'x': 0.2804,
            'y': 0.1817, 'z': 0.0875
        },
    },
}

INTERCEPTS_BY_MELD = {
    0: {
        'early': -5.8070,
        'late': -3.0245
    },
    1: {
        'early': -3.3626,
        'late': -0.4741
    },
    2: {
        'early': -1.5980,
        'late': 0.2331
    },
    3: {
        'early': -0.3839,
        'late': 1.0705
    },
    4: {
        'early': 1.1763,
        'late': 2.4104
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
    
    keys = ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'z1', 'z2', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    z_val = intercept
    for k in keys:
        scaled_val = (features[k] - M[k]) / S[k] if S[k] != 0 else 0
        z_val += W.get(k, 0) * scaled_val

    z_val = max(min(z_val, 500), -500) 
    return 1 / (1 + math.exp(-z_val))

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
            'moqie_to_shouqie_count': 0, 'last_discard_type': None,
            'discard_history': []
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
                turn = stats['turn_count']
                tile_str = step_data[3] if len(step_data) > 3 else ""

                is_prev_2_moqie = 0
                if turn >= 9:
                    if len(stats['discard_history']) >= 2 and stats['discard_history'][-1] == 'MD' and stats['discard_history'][-2] == 'MD':
                        is_prev_2_moqie = 1

                is_mo_to_shou_now = 0 

                if action == 'MD':
                    stats['moqie_count'] += 1
                    stats['current_continuous_moqie'] += 1
                    if stats['current_continuous_moqie'] > stats['max_continuous_moqie']:
                        stats['max_continuous_moqie'] = stats['current_continuous_moqie']
                elif action == 'HD':
                    if stats['current_continuous_moqie'] >= 2:
                        stats['moqie_to_shouqie_count'] += 1
                        is_mo_to_shou_now = 1
                    stats['current_continuous_moqie'] = 0

                stats['last_discard_type'] = action
                stats['discard_history'].append(action)

                recent_moqie_count_from_turn_9 = stats['current_continuous_moqie'] if turn >= 9 else 0

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
                def safe_div(a, b): return a / b if b > 0 else 0.0

                number_tiles = stats['discard_wan'] + stats['discard_tong'] + stats['discard_tiao']
                max_suit = max(stats['discard_wan'], stats['discard_tong'], stats['discard_tiao'])
                
                feat_val_concentration = round(1.0 - safe_div(max_suit, number_tiles), 4)
                feat_val_prop_zhong = round(safe_div(stats['discard_3_to_7'], td), 4)
                feat_val_prop_bian_1289 = round(safe_div(stats['discard_1289'], td), 4)
                feat_val_prop_zi = round(safe_div(stats['discard_zi'], td), 4)
                feat_val_moqie_rate = round(safe_div(stats['moqie_count'], td), 4)

                features = {
                    'a': turn, 
                    'c': feat_val_concentration,
                    'd': feat_val_prop_zhong,
                    'e': feat_val_prop_bian_1289,
                    'f': feat_val_prop_zi,
                    'g': feat_val_moqie_rate, 
                    'h': stats['current_continuous_moqie'], 
                    'i': is_mo_to_shou_now,
                    'j': stats['moqie_to_shouqie_count'],
                    'z1': recent_moqie_count_from_turn_9,
                    'z2': is_prev_2_moqie,
                    'k': 1 if (is_zhong and discard_nth == 1) else 0,
                    'l': 1 if (is_zhong and discard_nth == 2) else 0,
                    'm': 1 if (is_zhong and discard_nth == 3) else 0,
                    'n': 1 if (is_zhong and discard_nth == 4) else 0,
                    'o': 1 if (is_zi and discard_nth == 1) else 0,
                    'p': 1 if (is_zi and discard_nth == 2) else 0,
                    'q': 1 if (is_zi and discard_nth == 3) else 0,
                    'r': 1 if (is_zi and discard_nth == 4) else 0,
                    's': 1 if (is_bian_19 and discard_nth == 1) else 0,
                    't': 1 if (is_bian_19 and discard_nth == 2) else 0,
                    'u': 1 if (is_bian_19 and discard_nth == 3) else 0,
                    'v': 1 if (is_bian_19 and discard_nth == 4) else 0,
                    'w': 1 if (is_bian_28 and discard_nth == 1) else 0,
                    'x': 1 if (is_bian_28 and discard_nth == 2) else 0,
                    'y': 1 if (is_bian_28 and discard_nth == 3) else 0,
                    'z': 1 if (is_bian_28 and discard_nth == 4) else 0
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
                        'feat_c_花色集中度': features['c'],
                        'feat_d_中張比例(3 ~ 7)': features['d'],
                        'feat_e_邊張比例(1、2、8、9)': features['e'],
                        'feat_f_字牌比例': features['f'],
                        'feat_g_摸切比例': features['g'],
                        'feat_h_目前連續摸切': features['h'],
                        'feat_i_摸切轉手切': features['i'],
                        'feat_j_摸切轉手切次數': features['j'],
                        'feat_z1_第9巡起最近連續摸切次數': features['z1'],
                        'feat_z2_第9巡起前兩巡連續摸切': features['z2'],
                        'feat_k_中張第一張被打出': features['k'],
                        'feat_l_中張第二張被打出': features['l'],
                        'feat_m_中張第三張被打出': features['m'],
                        'feat_n_中張第四張被打出': features['n'],
                        'feat_o_字牌第一張被打出': features['o'],
                        'feat_p_字牌第二張被打出': features['p'],
                        'feat_q_字牌第三張被打出': features['q'],
                        'feat_r_字牌第四張被打出': features['r'],
                        'feat_s_邊張(1、9)第一張被打出': features['s'],
                        'feat_t_邊張(1、9)第二張被打出': features['t'],
                        'feat_u_邊張(1、9)第三張被打出': features['u'],
                        'feat_v_邊張(1、9)第四張被打出': features['v'],
                        'feat_w_邊張(2、8)第一張被打出': features['w'],
                        'feat_x_邊張(2、8)第二張被打出': features['x'],
                        'feat_y_邊張(2、8)第三張被打出': features['y'],
                        'feat_z_邊張(2、8)第四張被打出': features['z']
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