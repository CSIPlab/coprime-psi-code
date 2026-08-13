import numpy as np 
import math 
from itertools import combinations

def get_coprime_paris(min_s, max_s, len_s_list,limit = 50):
    """
    Generates a list of coprime sets.
    
    Args:
        min_s: Minimum integer value in a set.
        max_s: Maximum integer value in a set.
        len_s_list: A list of desired lengths, e.g., [3, 4, 5].
    """
    def is_coprime_set(arr):
        # The set is coprime if the GCD of all elements is 1
        res = arr[0]
        for x in arr[1:]:
            res = math.gcd(res, x)
        return res == 1

    all_results = []
    
    numbers = list(range(min_s, max_s + 1))
    
    for length in len_s_list:
        count = 0
        
        for combo in combinations(numbers, length):
            if is_coprime_set(combo):
                all_results.append(list(combo))
                count += 1
            
            if limit > 0 and count >= limit:
                break
                
    return all_results   
 
SHIFT_LISTS = [


    # optimal shifts and 1 
    [1], 
    [24,25], # sqrt N/2 assuming N is 1200
    [22,23], # sqrt N/2 assuming N is 1024
    [1,63], # 1 and large 
    [62,63], # large only
    
    # =====================================================
    # size 2 — small / small (coprime, local shifts)
    # =====================================================
    [2, 3],
    [3, 5],
    [4, 7],
    [5, 9],
    [6, 11],
    [7, 10],
    [8, 15],
    [9, 14],

    # =====================================================
    # size 2 — small / large (coprime, asymmetric baselines)
    # =====================================================
    [2, 21],
    [3, 20],
    [4, 25],
    [5, 24],
    [6, 23],
    [7, 26],
    [8, 27],
    [9, 28],
    [11, 22],
    [13, 24],

    # =====================================================
    # size 2 — large / large (coprime, global shifts)
    # =====================================================
    [21, 22],
    [22, 27],
    [23, 24],
    [25, 26],
    [27, 28],

    # =====================================================
    # size 3 — mixed (coprime, good spread)
    # =====================================================
    [5, 12, 19],
    [7, 16, 23],
    [9, 20, 23],
    [11, 18, 25],
    [13, 17, 28],
    [8, 15, 22],

    # =====================================================
    # size 3 — small / small / small (coprime)
    # =====================================================
    [2, 3, 5],
    [3, 4, 7],
    [5, 7, 11],

    # =====================================================
    # size 3 — small / large / large (coprime)
    # =====================================================
    [3, 22, 25],
    [5, 24, 29],  # 28
    [7, 23, 26],
    [9, 25, 28],

    # =====================================================
    # size 3 — large / large / large (<30, coprime)
    # =====================================================
    [21, 23, 25],
    [22, 25, 27],
    [23, 26, 27],
    [24, 25, 28],

    # =====================================================
    # size 4 — mixed, well-conditioned
    # =====================================================
    [5, 9, 16, 23],
    [7, 10, 19, 26],
    [11, 14, 17, 25],
    [6, 13, 22, 27],

    # =====================================================
    # size 4 — small-heavy (coprime)
    # =====================================================
    [2, 3, 5, 7],
    [3, 4, 7, 11],

    # =====================================================
    # size 4 — small / large balanced
    # =====================================================
    [3, 8, 21, 25],
    [5, 12, 23, 28],
    [7, 14, 19, 26],

    # =====================================================
    # size 4 — large-heavy (<30, coprime)
    # =====================================================
    [18, 23, 25, 28],
    [19, 21, 24, 27],
    [20, 23, 26, 27],
] 



SHIFT_GROUPS = [
    [1],     # 8 measurements
    [16, 17], # 16 measurements
    # [16, 17, 22],   # 24 measurements
    [23, 31, 16, 17], # 32 measurements
    # [61, 33, 23, 18, 16], # 40 measurements
    # [22, 15, 61, 12, 33, 17], # 48 measurements
    # [11, 17, 12, 18, 2, 22, 32], # 56 measurements
    [21, 33, 11, 22, 16, 13, 17, 63], # 64 measurements
    # [15, 62, 33, 32, 31, 17, 12, 1, 61], # 72 measurements
    # [17, 3, 23, 63, 18, 16, 13, 11, 61, 2], # 80 measurements
    # [15, 16, 62, 12, 23, 17, 61, 2, 22, 13, 31], # 88 measurements
    # [33, 11, 18, 61, 22, 2, 17, 13, 23, 62, 21, 12], # 96 measurements
    # [32, 3, 23, 21, 18, 11, 12, 62, 31, 22, 1, 61, 33], # 104 measurements
    # [3, 63, 12, 61, 32, 1, 2, 22, 17, 15, 21, 33, 16, 23], # 112 measurements
    # [15, 31, 17, 32, 16, 21, 62, 22, 13, 2, 23, 18, 11, 12, 3], # 120 measurements
    # [15, 16, 1, 17, 2, 22, 11, 13, 3, 31, 63, 61, 62, 32, 18, 33], # 128 measurements
]