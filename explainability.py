import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import pandas as pd
import numpy as np

def fit_shap_explainer(model):
    """
    Fits a SHAP TreeExplainer on the trained XGBoost model.
    """
    print("Fitting SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    return explainer

def generate_global_shap_plot(explainer, X, output_path):
    """
    Generates a global SHAP summary plot and saves it as a PNG image.
    Uses a subset of samples if X is large to speed up calculation.
    """
    print(f"Generating global SHAP summary plot for {min(len(X), 500)} samples...")
    # Sample if dataset is very large
    X_sample = X if len(X) <= 500 else X.sample(500, random_state=42)
    
    # Compute SHAP values
    shap_values = explainer(X_sample)
    
    # Plot summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Feature Importance Summary", fontsize=14, pad=15)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Global SHAP summary plot saved to {output_path}")

def generate_local_shap_plot(explainer, X_sample, output_html_path):
    """
    Generates an interactive HTML force plot for a single sample prediction.
    """
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    
    # Compute SHAP values for this single sample
    shap_values = explainer(X_sample)
    
    # Create the force plot
    # Handles multi-class output if necessary, but here we assume binary (1D output)
    expected_value = explainer.expected_value
    # In some SHAP versions, expected_value is a list or array for binary classifier.
    # We take the element corresponding to the prediction, or if it is scalar, use it directly.
    if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
        # For binary classifier, sometimes tree explainer expected_value is a 2-element array (class 0, class 1)
        expected_value_scalar = expected_value[1]
        shap_vals_row = shap_values.values[0][:, 1] if len(shap_values.values.shape) > 2 else shap_values.values[0]
    else:
        expected_value_scalar = expected_value
        shap_vals_row = shap_values.values[0]
        
    # Generate HTML force plot
    plot = shap.force_plot(
        expected_value_scalar,
        shap_vals_row,
        X_sample.iloc[0],
        matplotlib=False,
        link='logit' # converts log-odds output to probability
    )
    
    # Save to file
    shap.save_html(output_html_path, plot)
    print(f"Local SHAP force plot HTML saved to {output_html_path}")

if __name__ == "__main__":
    print("Testing Explainability Module...")
    from xgboost import XGBClassifier
    # Dummy data
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y = np.random.randint(0, 2, size=100)
    
    model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    
    explainer = fit_shap_explainer(model)
    
    # Global plot test
    test_global_path = "models/test_shap_summary.png"
    generate_global_shap_plot(explainer, X, test_global_path)
    
    # Local plot test
    test_local_path = "models/test_local_shap.html"
    generate_local_shap_plot(explainer, X.iloc[[0]], test_local_path)
    
    # Cleanup
    if os.path.exists(test_global_path):
        os.remove(test_global_path)
    if os.path.exists(test_local_path):
        os.remove(test_local_path)
        
    print("Success!")
