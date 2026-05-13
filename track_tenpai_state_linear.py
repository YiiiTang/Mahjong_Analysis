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
            'a': 0.0149, 'c': 0.0087, 'd': 5.1899, 'e': 5.5268, 'f': 7.5934,
            'h': 0.0081, 'i': -0.0016, 'j': 0.0044, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.0182, 'l': 0.0050, 'm': 0.0020, 'n': 0.0016,
            'o': 0.0075, 'p': 0.0048, 'q': 0.0019, 'r': 0.0004, 's': 0.0077,
            't': 0.0037, 'u': 0.0029, 'v': 0.0010, 'w': 0.0087, 'x': 0.0038,
            'y': 0.0022, 'z': -0.0002
        },
        'late': {
            'a': 0.0085, 'c': 0.0175, 'd': -42.2095, 'e': -34.3093, 'f': -38.5740,
            'h': 0.0243, 'i': 0.0205, 'j': 0.0079, 'z1': 0.0243,
            'z2': -0.0185, 'k': 0.0054, 'l': -0.0036, 'm': -0.0065, 'n': -0.0029,
            'o': -0.0070, 'p': -0.0119, 'q': -0.0133, 'r': -0.0106, 's': -0.0049,
            't': -0.0070, 'u': -0.0049, 'v': -0.0013, 'w': 0.0042, 'x': -0.0010,
            'y': -0.0068, 'z': -0.0070
        },
    },
    1: {
        'early': {
            'a': 0.0461, 'c': 0.0245, 'd': 1.0339, 'e': 1.3828, 'f': 1.6294,
            'h': 0.0055, 'i': -0.0038, 'j': 0.0077, 'z1': -0.0000,
            'z2': -0.0000, 'k': -0.0014, 'l': -0.0024, 'm': -0.0027, 'n': -0.0004,
            'o': 0.0077, 'p': -0.0001, 'q': -0.0008, 'r': -0.0019, 's': 0.0009,
            't': -0.0056, 'u': -0.0022, 'v': -0.0031, 'w': 0.0023, 'x': 0.0002,
            'y': -0.0031, 'z': 0.0004
        },
        'late': {
            'a': 0.0681, 'c': 0.0346, 'd': -3.8956, 'e': -4.2065, 'f': -4.4739,
            'h': 0.0139, 'i': 0.0230, 'j': 0.0014, 'z1': 0.0139,
            'z2': -0.0308, 'k': 0.3095, 'l': 0.2258, 'm': 0.1198, 'n': 0.0350,
            'o': 0.0843, 'p': 0.1440, 'q': 0.1694, 'r': 0.1308, 's': 0.1553,
            't': 0.1803, 'u': 0.1531, 'v': 0.0810, 'w': 0.2029, 'x': 0.1807,
            'y': 0.1124, 'z': 0.0586
        },
    },
    2: {
        'early': {
            'a': 0.0693, 'c': 0.0216, 'd': 5.6130, 'e': 6.3196, 'f': 7.3703,
            'h': 0.0025, 'i': -0.0084, 'j': 0.0089, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.0063, 'l': -0.0065, 'm': -0.0038, 'n': -0.0061,
            'o': 0.0098, 'p': 0.0056, 'q': 0.0000, 'r': -0.0068, 's': -0.0059,
            't': -0.0090, 'u': -0.0111, 'v': -0.0058, 'w': 0.0098, 'x': -0.0042,
            'y': 0.0010, 'z': 0.0030
        },
        'late': {
            'a': 0.0797, 'c': 0.0387, 'd': -10.9983, 'e': -11.3560, 'f': -11.7252,
            'h': 0.0157, 'i': 0.0130, 'j': -0.0122, 'z1': 0.0157,
            'z2': -0.0202, 'k': 0.1623, 'l': 0.0952, 'm': 0.0530, 'n': 0.0149,
            'o': 0.0349, 'p': 0.0645, 'q': 0.0712, 'r': 0.0562, 's': 0.0659,
            't': 0.0784, 'u': 0.0602, 'v': 0.0278, 'w': 0.0981, 'x': 0.0821,
            'y': 0.0436, 'z': 0.0190
        },
    },
    3: {
        'early': {
            'a': 0.0781, 'c': 0.0195, 'd': 4.5798, 'e': 4.9676, 'f': 5.4065,
            'h': -0.0039, 'i': -0.0035, 'j': 0.0165, 'z1': -0.0000,
            'z2': 0.0000, 'k': 0.0204, 'l': -0.0056, 'm': 0.0000, 'n': -0.0119,
            'o': 0.0069, 'p': -0.0062, 'q': -0.0053, 'r': 0.0017, 's': -0.0149,
            't': -0.0183, 'u': -0.0068, 'v': -0.0047, 'w': 0.0075, 'x': 0.0057,
            'y': -0.0017, 'z': 0.0030
        },
        'late': {
            'a': 0.0704, 'c': 0.0363, 'd': -41.9087, 'e': -44.0323, 'f': -44.1816,
            'h': 0.0146, 'i': 0.0108, 'j': 0.0070, 'z1': 0.0146,
            'z2': -0.0175, 'k': -0.0873, 'l': -0.0802, 'm': -0.0614, 'n': -0.0207,
            'o': -0.0302, 'p': -0.0468, 'q': -0.0574, 'r': -0.0462, 's': -0.0561,
            't': -0.0620, 'u': -0.0601, 'v': -0.0359, 'w': -0.0553, 'x': -0.0631,
            'y': -0.0439, 'z': -0.0197
        },
    },
    4: {
        'early': {
            'a': 0.0734, 'c': 0.0276, 'd': -100.5175, 'e': -101.6110, 'f': -118.9443,
            'h': -0.0492, 'i': -0.0206, 'j': -0.0368, 'z1': 0.0000,
            'z2': -0.0000, 'k': 0.0187, 'l': -0.0179, 'm': -0.0011, 'n': -0.0000,
            'o': -0.0110, 'p': 0.0281, 'q': -0.0055, 'r': 0.0239, 's': -0.0169,
            't': -0.0446, 'u': -0.0207, 'v': 0.0000, 'w': 0.0341, 'x': -0.0061,
            'y': -0.0246, 'z': 0.0000
        },
        'late': {
            'a': -0.0032, 'c': 0.0402, 'd': -22.4040, 'e': -23.3072, 'f': -24.0815,
            'h': 0.0239, 'i': 0.0103, 'j': 0.0065, 'z1': 0.0239,
            'z2': -0.0049, 'k': -0.0790, 'l': -0.0823, 'm': -0.0517, 'n': -0.0217,
            'o': -0.0135, 'p': -0.0347, 'q': -0.0380, 'r': -0.0340, 's': -0.0431,
            't': -0.0479, 'u': -0.0502, 'v': -0.0329, 'w': -0.0377, 'x': -0.0573,
            'y': -0.0554, 'z': -0.0082
        },
    },
}
MEANS_BY_MELD = {
    0: {
        'early': {
            'a': 3.5866, 'c': 0.5536, 'd': 0.1564, 'e': 0.2279, 'f': 0.6157,
            'h': 0.1932, 'i': 0.0124, 'j': 0.0271, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.1594, 'l': 0.0318, 'm': 0.0040, 'n': 0.0004,
            'o': 0.2642, 'p': 0.1635, 'q': 0.0639, 'r': 0.0120, 's': 0.1279,
            't': 0.0462, 'u': 0.0105, 'v': 0.0013, 'w': 0.0879, 'x': 0.0231,
            'y': 0.0037, 'z': 0.0003
        },
        'late': {
            'a': 10.8702, 'c': 0.4761, 'd': 0.3413, 'e': 0.3664, 'f': 0.2923,
            'h': 0.3660, 'i': 0.0388, 'j': 0.1948, 'z1': 0.3660,
            'z2': 0.0796, 'k': 0.1950, 'l': 0.1467, 'm': 0.0471, 'n': 0.0065,
            'o': 0.0290, 'p': 0.0689, 'q': 0.0811, 'r': 0.0396, 's': 0.0536,
            't': 0.0799, 'u': 0.0514, 'v': 0.0137, 'w': 0.0745, 'x': 0.0724,
            'y': 0.0341, 'z': 0.0053
        },
    },
    1: {
        'early': {
            'a': 5.0224, 'c': 0.4373, 'd': 0.1209, 'e': 0.2891, 'f': 0.5900,
            'h': 0.3576, 'i': 0.0303, 'j': 0.0701, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.2023, 'l': 0.0404, 'm': 0.0047, 'n': 0.0003,
            'o': 0.1457, 'p': 0.1084, 'q': 0.0548, 'r': 0.0135, 's': 0.1606,
            't': 0.0771, 'u': 0.0192, 'v': 0.0017, 'w': 0.1259, 'x': 0.0382,
            'y': 0.0068, 'z': 0.0005
        },
        'late': {
            'a': 10.6396, 'c': 0.4617, 'd': 0.2291, 'e': 0.3786, 'f': 0.3923,
            'h': 1.0472, 'i': 0.1044, 'j': 0.5019, 'z1': 1.0472,
            'z2': 0.2432, 'k': 0.2199, 'l': 0.1307, 'm': 0.0393, 'n': 0.0042,
            'o': 0.0166, 'p': 0.0486, 'q': 0.0715, 'r': 0.0452, 's': 0.0556,
            't': 0.0845, 'u': 0.0601, 'v': 0.0167, 'w': 0.0845, 'x': 0.0801,
            'y': 0.0351, 'z': 0.0074
        },
    },
    2: {
        'early': {
            'a': 6.1722, 'c': 0.3991, 'd': 0.1804, 'e': 0.3396, 'f': 0.4800,
            'h': 0.4333, 'i': 0.0374, 'j': 0.0927, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.2851, 'l': 0.0673, 'm': 0.0081, 'n': 0.0004,
            'o': 0.0607, 'p': 0.0661, 'q': 0.0417, 'r': 0.0116, 's': 0.1417,
            't': 0.0791, 'u': 0.0250, 'v': 0.0025, 'w': 0.1511, 'x': 0.0521,
            'y': 0.0070, 'z': 0.0005
        },
        'late': {
            'a': 10.8573, 'c': 0.4727, 'd': 0.2470, 'e': 0.3699, 'f': 0.3830,
            'h': 1.2080, 'i': 0.1009, 'j': 0.4740, 'z1': 1.2080,
            'z2': 0.2714, 'k': 0.2477, 'l': 0.1430, 'm': 0.0372, 'n': 0.0046,
            'o': 0.0171, 'p': 0.0469, 'q': 0.0639, 'r': 0.0426, 's': 0.0474,
            't': 0.0796, 'u': 0.0533, 'v': 0.0161, 'w': 0.0824, 'x': 0.0793,
            'y': 0.0330, 'z': 0.0055
        },
    },
    3: {
        'early': {
            'a': 6.7931, 'c': 0.4194, 'd': 0.2233, 'e': 0.3561, 'f': 0.4206,
            'h': 0.4735, 'i': 0.0304, 'j': 0.0801, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.3526, 'l': 0.0845, 'm': 0.0106, 'n': 0.0006,
            'o': 0.0302, 'p': 0.0514, 'q': 0.0322, 'r': 0.0103, 's': 0.1162,
            't': 0.0724, 'u': 0.0236, 'v': 0.0033, 'w': 0.1407, 'x': 0.0585,
            'y': 0.0115, 'z': 0.0012
        },
        'late': {
            'a': 11.1844, 'c': 0.4807, 'd': 0.2659, 'e': 0.3714, 'f': 0.3626,
            'h': 1.3153, 'i': 0.1069, 'j': 0.4876, 'z1': 1.3153,
            'z2': 0.2961, 'k': 0.2576, 'l': 0.1498, 'm': 0.0434, 'n': 0.0064,
            'o': 0.0176, 'p': 0.0480, 'q': 0.0633, 'r': 0.0374, 's': 0.0471,
            't': 0.0723, 'u': 0.0491, 'v': 0.0152, 'w': 0.0753, 'x': 0.0766,
            'y': 0.0341, 'z': 0.0065
        },
    },
    4: {
        'early': {
            'a': 7.2185, 'c': 0.4239, 'd': 0.2560, 'e': 0.3420, 'f': 0.4020,
            'h': 0.4031, 'i': 0.0369, 'j': 0.0800, 'z1': 0.0000,
            'z2': 0.0000, 'k': 0.4523, 'l': 0.0800, 'm': 0.0092, 'n': 0.0000,
            'o': 0.0154, 'p': 0.0492, 'q': 0.0615, 'r': 0.0062, 's': 0.0708,
            't': 0.0769, 'u': 0.0092, 'v': 0.0000, 'w': 0.1323, 'x': 0.0308,
            'y': 0.0062, 'z': 0.0000
        },
        'late': {
            'a': 11.5961, 'c': 0.4936, 'd': 0.2966, 'e': 0.3543, 'f': 0.3490,
            'h': 1.3827, 'i': 0.1129, 'j': 0.5097, 'z1': 1.3827,
            'z2': 0.3210, 'k': 0.2743, 'l': 0.1543, 'm': 0.0520, 'n': 0.0071,
            'o': 0.0123, 'p': 0.0520, 'q': 0.0547, 'r': 0.0326, 's': 0.0450,
            't': 0.0635, 'u': 0.0344, 'v': 0.0159, 'w': 0.0794, 'x': 0.0802,
            'y': 0.0317, 'z': 0.0088
        },
    },
}
SCALES_BY_MELD = {
    0: {
        'early': {
            'a': 2.1540, 'c': 0.4021, 'd': 0.2525, 'e': 0.2685, 'f': 0.3691,
            'h': 0.5081, 'i': 0.1105, 'j': 0.1637, 'z1': 1.0000,
            'z2': 1.0000, 'k': 0.3660, 'l': 0.1755, 'm': 0.0632, 'n': 0.0199,
            'o': 0.4409, 'p': 0.3698, 'q': 0.2446, 'r': 0.1088, 's': 0.3339,
            't': 0.2099, 'u': 0.1020, 'v': 0.0354, 'w': 0.2831, 'x': 0.1501,
            'y': 0.0605, 'z': 0.0177
        },
        'late': {
            'a': 1.7614, 'c': 0.1216, 'd': 0.1776, 'e': 0.1444, 'f': 0.1623,
            'h': 0.8844, 'i': 0.1930, 'j': 0.4513, 'z1': 0.8844,
            'z2': 0.2706, 'k': 0.3962, 'l': 0.3539, 'm': 0.2118, 'n': 0.0803,
            'o': 0.1678, 'p': 0.2533, 'q': 0.2729, 'r': 0.1950, 's': 0.2252,
            't': 0.2711, 'u': 0.2209, 'v': 0.1161, 'w': 0.2626, 'x': 0.2592,
            'y': 0.1815, 'z': 0.0729
        },
    },
    1: {
        'early': {
            'a': 1.9408, 'c': 0.3455, 'd': 0.1624, 'e': 0.2275, 'f': 0.2736,
            'h': 0.6923, 'i': 0.1714, 'j': 0.2581, 'z1': 1.0000,
            'z2': 1.0000, 'k': 0.4017, 'l': 0.1969, 'm': 0.0683, 'n': 0.0172,
            'o': 0.3528, 'p': 0.3109, 'q': 0.2276, 'r': 0.1152, 's': 0.3671,
            't': 0.2668, 'u': 0.1374, 'v': 0.0418, 'w': 0.3317, 'x': 0.1918,
            'y': 0.0820, 'z': 0.0222
        },
        'late': {
            'a': 1.6655, 'c': 0.1319, 'd': 0.1352, 'e': 0.1427, 'f': 0.1501,
            'h': 1.3757, 'i': 0.3058, 'j': 0.6081, 'z1': 1.3757,
            'z2': 0.4290, 'k': 0.4142, 'l': 0.3371, 'm': 0.1943, 'n': 0.0649,
            'o': 0.1279, 'p': 0.2150, 'q': 0.2577, 'r': 0.2077, 's': 0.2291,
            't': 0.2781, 'u': 0.2376, 'v': 0.1282, 'w': 0.2781, 'x': 0.2714,
            'y': 0.1839, 'z': 0.0859
        },
    },
    2: {
        'early': {
            'a': 1.5288, 'c': 0.2459, 'd': 0.1694, 'e': 0.1936, 'f': 0.2277,
            'h': 0.7322, 'i': 0.1898, 'j': 0.2920, 'z1': 1.0000,
            'z2': 1.0000, 'k': 0.4515, 'l': 0.2506, 'm': 0.0898, 'n': 0.0197,
            'o': 0.2388, 'p': 0.2484, 'q': 0.2000, 'r': 0.1072, 's': 0.3488,
            't': 0.2699, 'u': 0.1560, 'v': 0.0496, 'w': 0.3581, 'x': 0.2221,
            'y': 0.0835, 'z': 0.0213
        },
        'late': {
            'a': 1.7554, 'c': 0.1237, 'd': 0.1352, 'e': 0.1386, 'f': 0.1425,
            'h': 1.4862, 'i': 0.3012, 'j': 0.6001, 'z1': 1.4862,
            'z2': 0.4447, 'k': 0.4317, 'l': 0.3501, 'm': 0.1893, 'n': 0.0677,
            'o': 0.1295, 'p': 0.2113, 'q': 0.2446, 'r': 0.2019, 's': 0.2126,
            't': 0.2706, 'u': 0.2246, 'v': 0.1260, 'w': 0.2750, 'x': 0.2702,
            'y': 0.1788, 'z': 0.0741
        },
    },
    3: {
        'early': {
            'a': 1.2105, 'c': 0.1911, 'd': 0.1615, 'e': 0.1789, 'f': 0.1968,
            'h': 0.7536, 'i': 0.1718, 'j': 0.2725, 'z1': 1.0000,
            'z2': 1.0000, 'k': 0.4778, 'l': 0.2782, 'm': 0.1026, 'n': 0.0243,
            'o': 0.1710, 'p': 0.2209, 'q': 0.1766, 'r': 0.1012, 's': 0.3204,
            't': 0.2592, 'u': 0.1519, 'v': 0.0569, 'w': 0.3477, 'x': 0.2347,
            'y': 0.1067, 'z': 0.0344
        },
        'late': {
            'a': 1.8877, 'c': 0.1195, 'd': 0.1332, 'e': 0.1398, 'f': 0.1401,
            'h': 1.5671, 'i': 0.3090, 'j': 0.6102, 'z1': 1.5671,
            'z2': 0.4565, 'k': 0.4373, 'l': 0.3569, 'm': 0.2038, 'n': 0.0794,
            'o': 0.1317, 'p': 0.2138, 'q': 0.2436, 'r': 0.1898, 's': 0.2119,
            't': 0.2591, 'u': 0.2160, 'v': 0.1223, 'w': 0.2640, 'x': 0.2659,
            'y': 0.1814, 'z': 0.0805
        },
    },
    4: {
        'early': {
            'a': 0.9535, 'c': 0.1630, 'd': 0.1539, 'e': 0.1555, 'f': 0.1819,
            'h': 0.6843, 'i': 0.1886, 'j': 0.2713, 'z1': 1.0000,
            'z2': 1.0000, 'k': 0.4977, 'l': 0.2713, 'm': 0.0956, 'n': 1.0000,
            'o': 0.1231, 'p': 0.2163, 'q': 0.2403, 'r': 0.0782, 's': 0.2564,
            't': 0.2665, 'u': 0.0956, 'v': 1.0000, 'w': 0.3388, 'x': 0.1727,
            'y': 0.0782, 'z': 1.0000
        },
        'late': {
            'a': 2.0256, 'c': 0.1109, 'd': 0.1277, 'e': 0.1326, 'f': 0.1368,
            'h': 1.5780, 'i': 0.3164, 'j': 0.6211, 'z1': 1.5780,
            'z2': 0.4669, 'k': 0.4461, 'l': 0.3613, 'm': 0.2221, 'n': 0.0837,
            'o': 0.1104, 'p': 0.2221, 'q': 0.2273, 'r': 0.1777, 's': 0.2072,
            't': 0.2438, 'u': 0.1822, 'v': 0.1250, 'w': 0.2703, 'x': 0.2717,
            'y': 0.1753, 'z': 0.0935
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
    
    keys = ['a', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'z1', 'z2', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    score = intercept
    for k in keys:
        scaled_val = (features[k] - M[k]) / S[k] if S[k] != 0 else 0
        score += W.get(k, 0) * scaled_val
        
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

                td = stats['total_discard']
                def safe_div(a, b): return a / b if b > 0 else 0.0

                num_tiles = stats['discard_wan'] + stats['discard_tong'] + stats['discard_tiao']
                max_s = max(stats['discard_wan'], stats['discard_tong'], stats['discard_tiao'])
                
                feat_val_concentration = round(1.0 - safe_div(max_s, num_tiles), 4)
                feat_val_prop_zhong = round(safe_div(stats['discard_3_to_7'], td), 4)
                feat_val_prop_bian_1289 = round(safe_div(stats['discard_1289'], td), 4)
                feat_val_prop_zi = round(safe_div(stats['discard_zi'], td), 4)

                features = {
                    'a': turn, 
                    'c': feat_val_concentration, 
                    'd': feat_val_prop_zhong,
                    'e': feat_val_prop_bian_1289,
                    'f': feat_val_prop_zi,
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
                
                pred_score = calculate_linear_score(features, stats['meld_count'])

                actual_shanten = states.get_player(states.state[j], actor_loc).shantenCount
                action_name = "手切" if action == "HD" else "摸切"

                tracker_log.append({
                    '檔案名稱': file_name, 'Step_ID': j, '玩家': actor_loc, '動作': action_name, '丟棄牌': tile_str,
                    '實際向聽數': actual_shanten, '預測聽牌分數': round(pred_score, 4),
                    '特徵值': {
                        'feat_a_巡數': turn,
                        'feat_c_花色集中度': features['c'],
                        'feat_d_中張比例(3 ~ 7)': features['d'],
                        'feat_e_邊張比例(1、2、8、9)': features['e'],
                        'feat_f_字牌比例': features['f'],
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