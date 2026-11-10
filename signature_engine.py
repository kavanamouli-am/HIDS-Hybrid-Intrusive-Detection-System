import os
import json
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def train_signature_tree(X_train, y_train, max_depth=8, random_state=42):
    """
    Trains a DecisionTreeClassifier on the training set to serve as the basis
    for rule-based signature extraction.
    """
    print(f"Training DecisionTreeClassifier Signature Engine (max_depth={max_depth})...")
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=20,
        random_state=random_state
    )
    dt.fit(X_train, y_train)
    print("Decision Tree training complete.")
    return dt

def extract_rules_from_tree(tree, feature_names):
    """
    Traverses the DecisionTreeClassifier structure and extracts rules
    corresponding to each path from root to leaf.
    Returns a list of dictionaries containing condition paths, predictions, and confidence scores.
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]
    
    rules = []
    
    def recurse(node, conditions):
        # Check if internal node or leaf node
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = float(tree_.threshold[node])
            
            # Left child: feature <= threshold
            left_cond = conditions + [{"feature": name, "operator": "<=", "threshold": threshold}]
            recurse(tree_.children_left[node], left_cond)
            
            # Right child: feature > threshold
            right_cond = conditions + [{"feature": name, "operator": ">", "threshold": threshold}]
            recurse(tree_.children_right[node], right_cond)
        else:
            # Leaf node
            value = tree_.value[node][0]
            total = float(sum(value))
            # Probability/confidence of attack (class 1)
            prob_1 = float(value[1] / total) if total > 0.0 else 0.0
            prediction = 1 if prob_1 >= 0.5 else 0
            
            rules.append({
                "leaf_id": int(node),
                "conditions": conditions,
                "prediction": prediction,
                "confidence": prob_1
            })
            
    recurse(0, [])
    print(f"Extracted {len(rules)} rules from the Decision Tree.")
    return rules

def save_signature_rules(rules, filepath):
    """
    Saves the list of rules as a JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(rules, f, indent=4)
    print(f"Signature rules saved to {filepath}")

def load_signature_rules(filepath):
    """
    Loads signature rules from a JSON file.
    """
    with open(filepath, 'r') as f:
        rules = json.load(f)
    return rules

def evaluate_signature_row(row, rules):
    """
    Evaluates a single sample row against the signature rules.
    Returns the confidence score (class 1 probability) of the matching rule.
    """
    for rule in rules:
        match = True
        for cond in rule["conditions"]:
            feat = cond["feature"]
            op = cond["operator"]
            thresh = cond["threshold"]
            
            val = row[feat]
            if op == "<=":
                if not (val <= thresh):
                    match = False
                    break
            elif op == ">":
                if not (val > thresh):
                    match = False
                    break
        if match:
            return rule["confidence"]
            
    return 0.5  # Fallback (should not be reached if rules are exhaustive)

def evaluate_signature_batch(X, model=None, rules=None):
    """
    Evaluates a batch of samples.
    If `model` (DecisionTreeClassifier) is provided, uses vectorized prediction (highly efficient).
    If `rules` (JSON structure) is provided, evaluates row-by-row via logical rules.
    """
    if model is not None:
        # Vectorized path using sklearn
        return model.predict_proba(X)[:, 1]
    elif rules is not None:
        # Rule-by-rule path
        scores = []
        for _, row in X.iterrows():
            scores.append(evaluate_signature_row(row, rules))
        return np.array(scores)
    else:
        raise ValueError("Either model or rules must be provided to evaluate signature.")

def verify_rules_correctness(X, model, rules):
    """
    Verifies that the extracted rules match the model outputs exactly.
    Prints check statistics and returns True if correct, False otherwise.
    """
    print("Verifying rule extraction consistency...")
    scores_model = evaluate_signature_batch(X, model=model)
    scores_rules = evaluate_signature_batch(X, rules=rules)
    
    max_diff = np.max(np.abs(scores_model - scores_rules))
    is_consistent = max_diff < 1e-6
    print(f"Rule verification: Max difference = {max_diff:.8f}. Consistent = {is_consistent}")
    return is_consistent

if __name__ == "__main__":
    print("Testing Signature Engine...")
    # Create sample dummy data
    np.random.seed(42)
    features = ["feat_a", "feat_b"]
    X = pd.DataFrame(np.random.randn(200, 2), columns=features)
    y = ((X["feat_a"] > 0) & (X["feat_b"] < 0.5)).astype(int)
    
    # Train tree
    dt = train_signature_tree(X, y, max_depth=3)
    
    # Extract rules
    rules = extract_rules_from_tree(dt, features)
    print("Example rule:\n", json.dumps(rules[0], indent=2))
    
    # Save/load
    test_filepath = "models/test_rules.json"
    save_signature_rules(rules, test_filepath)
    loaded_rules = load_signature_rules(test_filepath)
    
    # Verify correctness
    verify_rules_correctness(X, dt, loaded_rules)
    
    # Cleanup
    if os.path.exists(test_filepath):
        os.remove(test_filepath)
    print("Success!")
