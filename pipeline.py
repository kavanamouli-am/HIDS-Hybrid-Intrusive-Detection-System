import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil

# Import modules
from data_preprocessing import fit_preprocessors, transform_data, save_preprocessors
from feature_selection import select_features, save_selected_features
from signature_engine import (
    train_signature_tree,
    extract_rules_from_tree,
    save_signature_rules,
    evaluate_signature_batch,
    verify_rules_correctness
)
from ml_engine import train_ml_engine, predict_ml_prob, save_ml_model
from fusion_layer import (
    compute_fusion_score,
    predict_fusion,
    tune_threshold,
    evaluate_predictions
)
from explainability import fit_shap_explainer, generate_global_shap_plot

# Fast ML models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Train multiple fast models (Logistic Regression, Naive Bayes, ExtraTrees, XGBoost)
    and evaluate them on test data.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
        "NaiveBayes": GaussianNB(),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        results[name] = metrics
        print(f"{name} Results: {metrics}")

    return results

def display_results(results):
    """
    Print comparison table of model metrics.
    """
    print("\n=== Model Comparison ===")
    print("{:<20} {:<10} {:<10} {:<10} {:<10}".format("Model", "Accuracy", "Precision", "Recall", "F1"))
    for model, metrics in results.items():
        print("{:<20} {:.4f}   {:.4f}   {:.4f}   {:.4f}".format(
            model,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"]
        ))

def run_pipeline(train_path="UNSW_NB15_training-set.csv", test_path="UNSW_NB15_testing-set.csv", models_dir="models"):
    print("="*60)
    print("STARTING HYBRID INTRUSION DETECTION SYSTEM TRAINING PIPELINE")
    print("="*60)

    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading datasets...\n  Train: {train_path}\n  Test: {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 2. Preprocessing
    print("\n[Step 1] Preprocessing and Feature Engineering...")
    preprocessors = fit_preprocessors(train_df)
    save_preprocessors(preprocessors, os.path.join(models_dir, "preprocessors.pkl"))

    X_train, y_train = transform_data(train_df, preprocessors)
    X_test, y_test = transform_data(test_df, preprocessors)

    # 3. Feature Selection
    print("\n[Step 2] Feature Selection using Random Forest...")
    selected_features, full_ranking = select_features(X_train, y_train, n_features=20)
    save_selected_features(selected_features, os.path.join(models_dir, "selected_features.json"))

    # Plot feature importance
    top_ranking = full_ranking[:20]
    names = [x[0] for x in top_ranking][::-1]
    importances = [x[1] for x in top_ranking][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(names, importances, color='#3b82f6')
    plt.title("Random Forest Feature Importance (Top 20)", fontsize=14, pad=15)
    plt.xlabel("Gini Importance", fontsize=12)
    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/rf_importance.png", dpi=300)
    plt.close()

    # Slice datasets
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # 4. Signature Engine
    print("\n[Step 3] Signature Engine...")
    dt_model = train_signature_tree(X_train_sel, y_train, max_depth=8)
    rules = extract_rules_from_tree(dt_model, selected_features)
    save_signature_rules(rules, os.path.join(models_dir, "signature_rules.json"))
    save_ml_model(dt_model, os.path.join(models_dir, "signature_tree_model.pkl"))
    verify_rules_correctness(X_test_sel, dt_model, rules)

    # 5. Machine Learning Engine (XGBoost)
    print("\n[Step 4] Machine Learning Engine (XGBoost)...")
    xgb_model = train_ml_engine(X_train_sel, y_train, tune_hyperparameters=True)
    save_ml_model(xgb_model, os.path.join(models_dir, "xgboost_model.pkl"))

    # 6. Fusion Threshold Tuning
    print("\n[Step 5] Tuning Fusion Threshold...")
    sig_prob_test = evaluate_signature_batch(X_test_sel, model=dt_model)
    ml_prob_test = predict_ml_prob(xgb_model, X_test_sel)
    optimal_threshold, tuning_metrics = tune_threshold(y_test, sig_prob_test, ml_prob_test)

    fusion_config = {
        "formula": "0.4 * signature_score + 0.6 * xgb_probability",
        "weights": {"signature": 0.4, "xgb": 0.6},
        "optimal_threshold": optimal_threshold
    }
    with open(os.path.join(models_dir, "fusion_config.json"), 'w') as f:
        json.dump(fusion_config, f, indent=4)

    # 7. Final Evaluation
    print("\n[Step 6] Running Final Evaluation...")
    y_pred_test = predict_fusion(sig_prob_test, ml_prob_test, threshold=optimal_threshold)
    fusion_score_test = compute_fusion_score(sig_prob_test, ml_prob_test)
    eval_metrics = evaluate_predictions(y_test, y_pred_test, fusion_score_test, output_dir=models_dir)

    if os.path.exists(os.path.join(models_dir, "confusion_matrix.png")):
        shutil.copy(os.path.join(models_dir, "confusion_matrix.png"), "static/confusion_matrix.png")

    # 8. SHAP Explainability
    print("\n[Step 7] Generating SHAP Explanations...")
    shap_explainer = fit_shap_explainer(xgb_model)
    save_ml_model(shap_explainer, os.path.join(models_dir, "shap_explainer.pkl"))
    generate_global_shap_plot(shap_explainer, X_train_sel, "static/shap_summary.png")

    # 9. Train & Compare Additional Models
    print("\n[Step 8] Training Multiple Models for Comparison...")
    results = train_and_evaluate_models(X_train_sel, y_train, X_test_sel, y_test)
    display_results(results)
    
        # Save comparison results to JSON for dashboard
    comparison_path = os.path.join(models_dir, "model_comparison.json")
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Model comparison results saved to {comparison_path}")

    # Copy to static folder for Flask dashboard access
    shutil.copy(comparison_path, "static/model_comparison.json")
    print("Model comparison results copied to static/model_comparison.json")

    return eval_metrics

if __name__ == "__main__":
    run_pipeline()
