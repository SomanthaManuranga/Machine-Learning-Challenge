import argparse
import numpy as np
import pandas as pd
import operator 

def framework(pairs, arr):
    """
    Args:
       - pairs:  a list of (cond, calc) tuples. calc() must be an executable
       - arr: a numpy array with the features in order feat_1, feat_2, ...
    
    Executes the first calc() whose cond returns True.
    Returns None if no condition matches.
    """
    targets = []

    for i in range(arr.shape[0]):
        row = arr[i]
        for cond, calc in pairs:
            if cond_eval(cond, row):
                targets.append(calc(row))
                break
        
    return targets


def cond_eval(condition, arr):
    """evaluate a condition"""
    ops = {
         ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    if condition is None:
        return True
    
    op = ops[condition[1]]
    return op(arr[condition[0]], condition[2])


def main(args):
    """
    Implementation for predicting target02.
    
    ============================================================
    DISCOVERED RULES (Grid Search Analysis - R² = 0.915):
    ============================================================
    
    Key Features Identified:
    - feat_4 (index 4): Controller variable (65.65% importance)
    - feat_185 (index 185): Calculation variable (18.25% importance)
    - feat_13 (index 13): Calculation variable (13.04% importance)
    
    3-Region Rule Structure:
    
    Region 1: feat_4 <= 0.2    →  -2*feat_185 - feat_13
    Region 2: 0.2 < feat_4 <= 0.7  →  2*feat_185 - feat_13
    Region 3: feat_4 > 0.7    →  -feat_185 + feat_13
    ============================================================
    """
    
    # Condition 1: feat_4 <= 0.2
    condition1 = (4, "<=", 0.2)
    def calc1(arr):
        return -2 * arr[185] - arr[13]
    
    # Condition 2: feat_4 <= 0.7 (implicitly > 0.2)
    condition2 = (4, "<=", 0.7)
    def calc2(arr):
        return 2 * arr[185] - arr[13]
    
    # Condition 3: Default (feat_4 > 0.7)
    condition3 = None
    def calc3(arr):
        return -arr[185] + arr[13]
    
    pair_list = [
        (condition1, calc1),
        (condition2, calc2),
        (condition3, calc3)
    ]
    
    data_array = pd.read_csv(args.eval_file_path).values
    
    return framework(pair_list, data_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework Task 2")
    parser.add_argument("--eval_file_path", required=True, help="Path to EVAL_<ID>.csv")
    args = parser.parse_args()

    target02 = main(args)