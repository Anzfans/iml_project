"""
Paired T-Test for Model Comparison
Usage:
    python src/paired_ttest.py --models xgboost svm --balance          # run from scratch
    python src/paired_ttest.py --models xgboost svm --balance --load   # load saved results
"""

import numpy as np
from scipy import stats
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.KfoldTraininer import run_stratified_kfold_cv, get_result_filename, load_results


def paired_ttest(scores1, scores2, name1, name2, metric, alpha=0.05):
    """Perform paired t-test and print results."""
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    diff = np.mean(scores1) - np.mean(scores2)
    
    print(f"\n{metric.upper()}: {name1} vs {name2}")
    print(f"  {name1}: {np.mean(scores1):.4f} ± {np.std(scores1):.4f}")
    print(f"  {name2}: {np.mean(scores2):.4f} ± {np.std(scores2):.4f}")
    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.6f}")
    print(f"  {'*** Significant' if p_value < alpha else 'Not significant'} (α={alpha})")
    if p_value < alpha:
        print(f"  → {name1 if diff > 0 else name2} is better")


def get_results(model, balance, load_saved, folds, seed):
    """Get results either by loading from file or running CV."""
    if load_saved:
        filepath = get_result_filename(model, balance)
        if os.path.exists(filepath):
            print(f"Loading saved results from: {filepath}")
            return load_results(filepath)
        else:
            print(f"No saved results found at {filepath}, running CV...")
    return run_stratified_kfold_cv(model, balance, folds, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paired T-Test for Model Comparison')
    parser.add_argument('--models', '-m', nargs=2, required=True, choices=['xgboost', 'svm', 'logistic'])
    parser.add_argument('--balance1', '-b1', action='store_true', help='Balance for model 1')
    parser.add_argument('--balance2', '-b2', action='store_true', help='Balance for model 2')
    parser.add_argument('--balance', '-b', action='store_true', help='Balance for both models')
    parser.add_argument('--load', '-l', action='store_true', help='Load saved results instead of retraining')
    parser.add_argument('--folds', '-k', type=int, default=5)
    parser.add_argument('--seed', '-s', type=int, default=42)
    args = parser.parse_args()
    
    bal1 = args.balance or args.balance1
    bal2 = args.balance or args.balance2
    
    name1 = f"{args.models[0].upper()}{'_bal' if bal1 else ''}"
    name2 = f"{args.models[1].upper()}{'_bal' if bal2 else ''}"
    
    print(f"\n{'='*60}\nGetting results for {name1}...\n{'='*60}")
    res1 = get_results(args.models[0], bal1, args.load, args.folds, args.seed)
    
    print(f"\n{'='*60}\nGetting results for {name2}...\n{'='*60}")
    res2 = get_results(args.models[1], bal2, args.load, args.folds, args.seed)
    
    print(f"\n{'='*60}\nPaired T-Test Results\n{'='*60}")
    for metric in ['accuracy', 'recall']:
        paired_ttest(res1['fold_metrics'][metric], res2['fold_metrics'][metric], name1, name2, metric)
