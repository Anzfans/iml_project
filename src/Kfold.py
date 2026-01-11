"""
Stratified K-Fold Cross-Validation for Multiple Models (XGBoost, SVM, Logistic Regression)
Outputs: Accuracy, Recall

Usage:
    python src/Kfold.py --model xgboost --balance
    python src/Kfold.py --model svm --no-balance
    python src/Kfold.py --model logistic --balance
    python src/Kfold.py --model all --balance --save
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import argparse
import sys
import os
import json
import time

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import basic_preprocess, get_target_mapping, preprocess_for_model


def get_result_filename(model_name, class_balance):
    """Generate result filename based on configuration."""
    balance_suffix = '_balanced' if class_balance else ''
    return f'results/{model_name}{balance_suffix}_kfold_results.json'


def save_results(results, filepath):
    """Save results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    save_data = {
        'model_name': results['model_name'],
        'class_balance': results['class_balance'],
        'fold_metrics': results['fold_metrics'],
        'mean_metrics': results['mean_metrics'],
        'std_metrics': results['std_metrics'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
    }
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to: {filepath}")


def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    data['confusion_matrix'] = np.array(data['confusion_matrix'])
    return data


def get_model(model_name, class_balance=False, random_state=42, scale_pos_weight=None):
    """
    Get model instance based on model name and class balance setting.
    
    Parameters:
    -----------
    model_name : str
        Name of the model ('xgboost', 'svm', 'logistic')
    class_balance : bool
        Whether to apply class balancing
    random_state : int
        Random seed for reproducibility
    scale_pos_weight : float, optional
        For XGBoost: count(negative) / count(positive). Only used when class_balance=True.
    
    Returns:
    --------
    model : sklearn estimator
        Model instance
    """
    
    if model_name == 'xgboost':
        params = {
            'n_estimators': 2000,
            'learning_rate': 0.005,
            'max_depth': 7,
            'eval_metric': 'logloss',
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'random_state': random_state,
            'use_label_encoder': False
        }
        if class_balance and scale_pos_weight is not None:
            params['scale_pos_weight'] = scale_pos_weight
        return XGBClassifier(**params)
    
    elif model_name == 'svm':
        params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'probability': True,  # Required for predict_proba
            'random_state': random_state
        }
        if class_balance:
            params['class_weight'] = 'balanced'
        return SVC(**params)
    
    elif model_name == 'logistic':
        params = {
            'max_iter': 1000,
            'solver': 'lbfgs',
            'random_state': random_state
        }
        if class_balance:
            params['class_weight'] = 'balanced'
        return LogisticRegression(**params)
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from 'xgboost', 'svm', 'logistic'")


def run_stratified_kfold_cv(model_name='xgboost', class_balance=False, n_splits=5, random_state=42):
    """
    Run Stratified K-Fold Cross-Validation with specified model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model ('xgboost', 'svm', 'logistic')
    class_balance : bool
        Whether to apply class balancing
    n_splits : int
        Number of folds for cross-validation
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing all metrics and predictions
    """
    
    # ==================== 1. Load and Preprocess Data ====================
    print("=" * 60)
    print(f"Model: {model_name.upper()}")
    print(f"Class Balance: {'Enabled' if class_balance else 'Disabled'}")
    print("Loading and preprocessing data...")
    print("=" * 60)
    
    df_train_raw = pd.read_csv('data/raw/train.csv')
    df_train_basic = basic_preprocess(df_train_raw)
    
    # Get target encoding mapping
    target_enc_cols = ['job', 'marital', 'education', 'default', 'housing', 
                       'loan', 'contact', 'month', 'poutcome', 'day_of_week']
    mapping = get_target_mapping(df_train_basic, target_enc_cols)
    
    # Preprocess for XGBoost
    df_processed = preprocess_for_model(df_train_basic, 'XG', mapping)
    
    # Split features and target
    X = df_processed.drop('target', axis=1)
    y = df_processed['target']
    
    # Fixed scale_pos_weight for XGBoost
    scale_pos_weight = 10
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    print(f"Positive class ratio: {y.mean():.4f}")
    if class_balance and model_name == 'xgboost':
        print(f"XGBoost scale_pos_weight: {scale_pos_weight}")
    
    # ==================== 2. Initialize Stratified K-Fold ====================
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Storage for metrics
    fold_metrics = {
        'accuracy': [],
        'recall': [],
        'train_time': []
    }
    
    # Storage for predictions (for overall confusion matrix)
    all_y_true = []
    all_y_pred = []
    
    # ==================== 3. Cross-Validation Loop ====================
    print("\n" + "=" * 60)
    print(f"Starting {n_splits}-Fold Stratified Cross-Validation")
    print("=" * 60)
    
    # Initialize scaler for SVM and Logistic Regression
    scaler = StandardScaler() if model_name in ['svm', 'logistic'] else None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        
        # Split data
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
        print(f"Train positive ratio: {y_train.mean():.4f}, Val positive ratio: {y_val.mean():.4f}")
        
        # Scale features for SVM and Logistic Regression
        if scaler is not None:
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
        
        # Initialize model
        model = get_model(model_name, class_balance, random_state, scale_pos_weight)
        
        # Train model with timing
        start_time = time.time()
        if model_name == 'xgboost':
            model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        acc = accuracy_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        
        # Store metrics
        fold_metrics['accuracy'].append(acc)
        fold_metrics['recall'].append(rec)
        fold_metrics['train_time'].append(train_time)
        
        # Store predictions
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())
        
        # Print fold results
        print(f"Accuracy:  {acc:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"Train Time: {train_time:.2f}s")
    
    # ==================== 4. Summary Statistics ====================
    print("\n" + "=" * 60)
    print("Cross-Validation Summary")
    print("=" * 60)
    
    print(f"\n{'Metric':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 52)
    for metric, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        if metric == 'train_time':
            print(f"{metric.upper():<12} {mean_val:>9.2f}s {std_val:>9.2f}s {min_val:>9.2f}s {max_val:>9.2f}s")
        else:
            print(f"{metric.upper():<12} {mean_val:>10.4f} {std_val:>10.4f} {min_val:>10.4f} {max_val:>10.4f}")
    
    # Print total training time
    total_time = sum(fold_metrics['train_time'])
    print(f"\nTotal Training Time: {total_time:.2f}s")
    
    # ==================== 5. Overall Confusion Matrix ====================
    print("\n" + "=" * 60)
    print("Overall Confusion Matrix (aggregated across all folds)")
    print("=" * 60)
    
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"\n{'':>15} Predicted No  Predicted Yes")
    print(f"{'Actual No':<15} {cm[0][0]:>10} {cm[0][1]:>13}")
    print(f"{'Actual Yes':<15} {cm[1][0]:>10} {cm[1][1]:>13}")
    
    print("\n" + "=" * 60)
    print("Classification Report (aggregated)")
    print("=" * 60)
    print(classification_report(all_y_true, all_y_pred, target_names=['No (0)', 'Yes (1)']))
    
    # ==================== 6. Plot Metrics Bar Chart ====================
    # Create model display name and file suffix
    model_display = model_name.upper()
    balance_suffix = '_balanced' if class_balance else ''
    file_prefix = f'{model_name}{balance_suffix}'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Only show accuracy and recall
    selected_metrics = ['accuracy', 'recall']
    means = [np.mean(fold_metrics[m]) for m in selected_metrics]
    stds = [np.std(fold_metrics[m]) for m in selected_metrics]
    
    x = np.arange(len(selected_metrics))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', 
                  edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in selected_metrics], fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{model_display} {n_splits}-Fold CV Metrics (Balance={class_balance})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    metrics_path = f'results/{file_prefix}_kfold_metrics.png'
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Metrics bar chart saved to: {metrics_path}")
    
    # ==================== 7. Return Results ====================
    results = {
        'model_name': model_name,
        'class_balance': class_balance,
        'fold_metrics': fold_metrics,
        'mean_metrics': {k: np.mean(v) for k, v in fold_metrics.items()},
        'std_metrics': {k: np.std(v) for k, v in fold_metrics.items()},
        'confusion_matrix': cm,
        'all_y_true': all_y_true,
        'all_y_pred': all_y_pred
    }
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Stratified K-Fold Cross-Validation for Multiple Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/Kfold.py --model xgboost --balance
    python src/Kfold.py --model svm --no-balance
    python src/Kfold.py --model logistic --balance
    python src/Kfold.py --model all --balance
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['xgboost', 'svm', 'logistic', 'all'],
        default='xgboost',
        help='Model to use for cross-validation (default: xgboost)'
    )
    
    parser.add_argument(
        '--balance', '-b',
        action='store_true',
        dest='balance',
        help='Enable class balancing (scale_pos_weight for XGBoost, class_weight for SVM/Logistic)'
    )
    
    parser.add_argument(
        '--no-balance',
        action='store_false',
        dest='balance',
        help='Disable class balancing (default)'
    )
    
    parser.add_argument(
        '--folds', '-k',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to JSON file for later use'
    )
    
    parser.set_defaults(balance=False)
    
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("Stratified K-Fold Cross-Validation")
    print("=" * 60)
    print(f"Model(s): {args.model}")
    print(f"Class Balance: {args.balance}")
    print(f"Folds: {args.folds}")
    print(f"Random Seed: {args.seed}")
    print("=" * 60)
    
    # Determine which models to run
    if args.model == 'all':
        models = ['xgboost', 'svm', 'logistic']
    else:
        models = [args.model]
    
    # Store all results
    all_results = {}
    
    # Run cross-validation for each model
    for model_name in models:
        print(f"\n{'#' * 60}")
        print(f"# Running {model_name.upper()}")
        print(f"{'#' * 60}")
        
        results = run_stratified_kfold_cv(
            model_name=model_name,
            class_balance=args.balance,
            n_splits=args.folds,
            random_state=args.seed
        )
        all_results[model_name] = results
        
        # Save results if requested
        if args.save:
            result_path = get_result_filename(model_name, args.balance)
            save_results(results, result_path)
    
    # Print summary comparison if multiple models
    if len(models) > 1:
        print("\n" + "=" * 80)
        print("Model Comparison Summary")
        print("=" * 80)
        print(f"\n{'Model':<12} {'Accuracy':>10} {'Recall':>10}")
        print("-" * 34)
        for model_name, results in all_results.items():
            m = results['mean_metrics']
            print(f"{model_name.upper():<12} {m['accuracy']:>10.4f} {m['recall']:>10.4f}")
    
    print("\n" + "=" * 60)
    print("Cross-Validation Complete!")
    print("=" * 60)
