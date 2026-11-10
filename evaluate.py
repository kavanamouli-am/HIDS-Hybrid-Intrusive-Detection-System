import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Import custom functions
from data_preprocessing import clean_columns, transform_data
from signature_engine import evaluate_signature_batch
from ml_engine import predict_ml_prob
from fusion_layer import compute_fusion_score, predict_fusion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc

def run_evaluation(test_csv_path="UNSW_NB15_testing-set.csv", models_dir="models"):
    """
    Evaluates the trained HIDS models on a given test dataset.
    """
    print("="*60)
    # Check if files exist
    if not os.path.exists(test_csv_path):
        print(f"Error: Test dataset file '{test_csv_path}' not found!")
        return
        
    preprocessors_path = os.path.join(models_dir, "preprocessors.pkl")
    features_path = os.path.join(models_dir, "selected_features.json")
    signature_tree_path = os.path.join(models_dir, "signature_tree_model.pkl")
    xgboost_path = os.path.join(models_dir, "xgboost_model.pkl")
    fusion_config_path = os.path.join(models_dir, "fusion_config.json")
    
    paths = [preprocessors_path, features_path, signature_tree_path, xgboost_path, fusion_config_path]
    if not all(os.path.exists(p) for p in paths):
        print("Error: Some model files are missing! Run the training pipeline (pipeline.py) first.")
        for p in paths:
            print(f"  - {p}: {'Found' if os.path.exists(p) else 'MISSING'}")
        return
        
    print("Loading preprocessor and models from disk...")
    # Load Preprocessors
    with open(preprocessors_path, 'rb') as f:
        preprocessors = pickle.load(f)
        
    # Load Selected Features list
    with open(features_path, 'r') as f:
        selected_features = json.load(f)
        
    # Load Decision Tree Signature Model
    with open(signature_tree_path, 'rb') as f:
        dt_model = pickle.load(f)
        
    # Load XGBoost Model
    with open(xgboost_path, 'rb') as f:
        xgb_model = pickle.load(f)
        
    # Load Fusion Layer Configuration
    with open(fusion_config_path, 'r') as f:
        fusion_config = json.load(f)
        
    print("All models loaded successfully.")
    print("="*60)
    
    # Load Test Data
    print(f"Loading test set: {test_csv_path}...")
    df_test = pd.read_csv(test_csv_path)
    print(f"Loaded {df_test.shape[0]} test connections with {df_test.shape[1]} columns.")
    
    # Preprocess test data
    print("Preprocessing dataset and engineering features...")
    X_test_proc, y_test = transform_data(df_test, preprocessors)
    
    # Slice to top 20 selected features
    X_test_sel = X_test_proc[selected_features]
    
    # 1. Run Signature Engine
    print("Evaluating Signature Detection Engine (Decision Tree)...")
    sig_scores = evaluate_signature_batch(X_test_sel, model=dt_model)
    
    # 2. Run Machine Learning Engine
    print("Evaluating Machine Learning Detection Engine (XGBoost)...")
    xgb_probs = predict_ml_prob(xgb_model, X_test_sel)
    
    # 3. Weighted Fusion
    print("Combining models via Weighted Fusion Layer...")
    fusion_scores = compute_fusion_score(sig_scores, xgb_probs)
    
    # Get optimal decision threshold
    optimal_threshold = fusion_config.get("optimal_threshold", 0.75)
    print(f"Using tuned decision threshold: {optimal_threshold:.4f}")
    
    # Final Predictions
    y_pred = predict_fusion(sig_scores, xgb_probs, threshold=optimal_threshold)
    
    # 4. Generate Classification Performance Metrics
    print("\n" + "="*50)
    print("HIDS EVALUATION REPORT")
    print("="*50)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, fusion_scores)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Total Test Instances: {len(y_test)}")
    print(f"Accuracy:            {acc:.5%}")
    print(f"Precision:           {prec:.5%}")
    print(f"Recall:              {rec:.5%}")
    print(f"F1 Score:            {f1:.5%}")
    print(f"ROC-AUC Score:       {auc:.5%}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"], digits=5))
    
    # Individual engines performance for comparison
    dt_pred = (sig_scores >= 0.5).astype(int)
    xgb_pred = (xgb_probs >= 0.5).astype(int)
    
    print("="*50)
    print("INDIVIDUAL ENGINES COMPARISON (Default Threshold = 0.5)")
    print("="*50)
    print(f"Decision Tree (Signature) - Acc: {accuracy_score(y_test, dt_pred):.4%}, Rec: {recall_score(y_test, dt_pred):.4%}, F1: {f1_score(y_test, dt_pred):.4%}")
    print(f"XGBoost (ML Engine)       - Acc: {accuracy_score(y_test, xgb_pred):.4%}, Rec: {recall_score(y_test, xgb_pred):.4%}, F1: {f1_score(y_test, xgb_pred):.4%}")
    print(f"Weighted Fusion (HIDS)    - Acc: {acc:.4%}, Rec: {rec:.4%}, F1: {f1:.4%}")
    print("="*50)
    
    # Save Report
    report = {
        "test_dataset": test_csv_path,
        "total_instances": int(len(y_test)),
        "decision_threshold": float(optimal_threshold),
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc)
        },
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1])
        },
        "individual_engines": {
            "decision_tree": {
                "accuracy": float(accuracy_score(y_test, dt_pred)),
                "recall": float(recall_score(y_test, dt_pred)),
                "f1_score": float(f1_score(y_test, dt_pred))
            },
            "xgboost": {
                "accuracy": float(accuracy_score(y_test, xgb_pred)),
                "recall": float(recall_score(y_test, xgb_pred)),
                "f1_score": float(f1_score(y_test, xgb_pred))
            }
        }
    }
    
    # Generate Advanced Visualizations
    print("Generating advanced evaluation visualizations...")
    os.makedirs("static", exist_ok=True)
    
    # 1. Comparative ROC Curve
    fpr_dt, tpr_dt, _ = roc_curve(y_test, sig_scores)
    roc_auc_dt = sk_auc(fpr_dt, tpr_dt)
    
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
    roc_auc_xgb = sk_auc(fpr_xgb, tpr_xgb)
    
    fpr_fusion, tpr_fusion, _ = roc_curve(y_test, fusion_scores)
    roc_auc_fusion = sk_auc(fpr_fusion, tpr_fusion)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_dt, tpr_dt, label=f'Signature (Decision Tree) (AUC = {roc_auc_dt:.4f})', color='#f59e0b', lw=2)
    plt.plot(fpr_xgb, tpr_xgb, label=f'ML Anomaly (XGBoost) (AUC = {roc_auc_xgb:.4f})', color='#3b82f6', lw=2)
    plt.plot(fpr_fusion, tpr_fusion, label=f'Weighted Fusion (HIDS) (AUC = {roc_auc_fusion:.4f})', color='#4f46e5', lw=2.5)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=11)
    plt.ylabel('True Positive Rate (TPR)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=13, pad=15)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig("static/roc_curve.png", dpi=300)
    plt.close()
    print("Comparative ROC Curve plot saved to static/roc_curve.png")
    
    # 2. Performance Metrics vs Threshold Tuning Curve
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = []
    accuracies = []
    recalls = []
    
    for t in thresholds:
        preds = (fusion_scores >= t).astype(int)
        f1_scores.append(f1_score(y_test, preds))
        accuracies.append(accuracy_score(y_test, preds))
        recalls.append(recall_score(y_test, preds))
        
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, accuracies, label='Accuracy', color='#10b981', lw=2)
    plt.plot(thresholds, recalls, label='Recall (Detection Rate)', color='#a855f7', lw=2)
    plt.plot(thresholds, f1_scores, label='F1-Score', color='#4f46e5', lw=2.5)
    plt.axvline(x=optimal_threshold, color='#e11d48', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.2f})', lw=1.5)
    plt.xlabel('Fusion Decision Threshold', fontsize=11)
    plt.ylabel('Performance Metric Score', fontsize=11)
    plt.title('HIDS Performance Metrics vs. Decision Threshold', fontsize=13, pad=15)
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig("static/threshold_tuning.png", dpi=300)
    plt.close()
    print("Threshold tuning analysis plot saved to static/threshold_tuning.png")
    
    report_path = os.path.join(models_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\nEvaluation metrics successfully logged to: {report_path}")
    print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate HIDS models on a dataset.")
    parser.add_argument("--test_csv", type=str, default="UNSW_NB15_testing-set.csv", help="Path to testing CSV")
    args = parser.parse_args()
    
    run_evaluation(test_csv_path=args.test_csv)
