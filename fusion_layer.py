import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def compute_fusion_score(signature_score, ml_probability):
    """
    Computes the weighted fusion score:
      final_score = 0.4 * signature_score + 0.6 * ml_probability
    """
    return 0.4 * signature_score + 0.6 * ml_probability

def predict_fusion(signature_score, ml_probability, threshold=0.5):
    """
    Generates binary prediction based on weighted fusion and a decision threshold.
    """
    score = compute_fusion_score(signature_score, ml_probability)
    return (score >= threshold).astype(int)

def tune_threshold(y_true, signature_score, ml_probability):
    """
    Grid searches the decision threshold to find the optimal threshold that:
    1. Satisfies target requirements (Accuracy > 93%, Recall > 95%, F1 Score > 93%)
    2. Otherwise, maximizes F1 score.
    Returns the best threshold and a dict of its metrics.
    """
    print("Tuning decision threshold...")
    fusion_score = compute_fusion_score(signature_score, ml_probability)
    
    best_thresh = 0.5
    best_f1 = -1.0
    best_metrics = {}
    
    # Grid of candidate thresholds
    thresholds = np.linspace(0.3, 0.9, 61)
    
    satisfying_candidates = []
    
    for thresh in thresholds:
        y_pred = (fusion_score >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        
        metrics = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1)
        }
        
        # Check if it satisfies requirements: Accuracy > 0.93, Recall > 0.95, F1 > 0.93
        if acc >= 0.93 and rec >= 0.95 and f1 >= 0.93:
            satisfying_candidates.append((thresh, metrics))
            
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = metrics
            
    # If we have thresholds that satisfy the strict targets, pick the one with highest F1/Accuracy
    if satisfying_candidates:
        # Sort by F1 score descending
        satisfying_candidates.sort(key=lambda x: (x[1]['f1_score'], x[1]['accuracy']), reverse=True)
        chosen_thresh, chosen_metrics = satisfying_candidates[0]
        print(f"Found threshold satisfying targets: Thresh={chosen_thresh:.3f}")
        return float(chosen_thresh), chosen_metrics
    else:
        print(f"Could not find threshold satisfying all strict targets. Selecting F1-maximizing threshold: Thresh={best_thresh:.3f}")
        return float(best_thresh), best_metrics

def evaluate_predictions(y_true, y_pred, y_prob, output_dir=None):
    """
    Computes Accuracy, Precision, Recall, F1 Score, ROC-AUC.
    Plots and saves confusion matrix to output_dir if specified.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
        
    metrics = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(auc)
    }
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy:  {acc:.4%}")
    print(f"Precision: {prec:.4%}")
    print(f"Recall:    {rec:.4%}")
    print(f"F1 Score:  {f1:.4%}")
    print(f"ROC-AUC:   {auc:.4%}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics as JSON
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # Plot Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=14, pad=15)
        plt.colorbar()
        
        classes = ['Normal', 'Attack']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0, fontsize=11)
        plt.yticks(tick_marks, classes, fontsize=11)
        
        # Annotate numbers
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontsize=12, weight='bold')
                
        plt.ylabel('True Label', fontsize=12, labelpad=10)
        plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Confusion Matrix plot saved to {cm_path}")
        
    return metrics

if __name__ == "__main__":
    print("Testing Fusion Layer Module...")
    # Dummy data
    y_true = np.array([0]*50 + [1]*50)
    sig_prob = np.random.uniform(0.1, 0.4, size=100)
    sig_prob[50:] = np.random.uniform(0.6, 0.9, size=50)
    ml_prob = np.random.uniform(0.1, 0.4, size=100)
    ml_prob[50:] = np.random.uniform(0.6, 0.9, size=50)
    
    thresh, metrics = tune_threshold(y_true, sig_prob, ml_prob)
    print(f"Optimal Threshold: {thresh:.3f}, Metrics: {metrics}")
    
    y_pred = predict_fusion(sig_prob, ml_prob, threshold=thresh)
    fusion_score = compute_fusion_score(sig_prob, ml_prob)
    
    eval_metrics = evaluate_predictions(y_true, y_pred, fusion_score, output_dir="models")
    
    # Cleanup
    if os.path.exists("models/metrics.json"):
        os.remove("models/metrics.json")
    if os.path.exists("models/confusion_matrix.png"):
        os.remove("models/confusion_matrix.png")
    print("Success!")
