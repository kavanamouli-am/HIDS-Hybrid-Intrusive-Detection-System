import os
import json
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

# Import custom modules
from data_preprocessing import clean_columns, engineer_features, transform_data
from signature_engine import evaluate_signature_batch, load_signature_rules
from ml_engine import predict_ml_prob
from fusion_layer import compute_fusion_score, predict_fusion, evaluate_predictions
from explainability import generate_local_shap_plot

app = Flask(__name__)
app.secret_key = "hids_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for caching loaded models and processed data
MODELS_DIR = "models"
PREPROCESSORS = None
SELECTED_FEATURES = None
SIGNATURE_RULES = None
SIGNATURE_TREE_MODEL = None
XGBOOST_MODEL = None
FUSION_CONFIG = None
SHAP_EXPLAINER = None

# Global store for uploaded batch data to allow SHAP explanation lookup
# (In a multi-user system we would use a session/database, but for local run a global dictionary is perfect)
UPLOADED_BATCH_X = None
UPLOADED_BATCH_RAW = None

def load_all_models():
    """
    Loads all models, preprocessors, config files and explainers.
    """
    global PREPROCESSORS, SELECTED_FEATURES, SIGNATURE_RULES, SIGNATURE_TREE_MODEL, XGBOOST_MODEL, FUSION_CONFIG, SHAP_EXPLAINER
    
    print("Loading models and preprocessors from disk...")
    try:
        # Load Preprocessor
        with open(os.path.join(MODELS_DIR, "preprocessors.pkl"), 'rb') as f:
            PREPROCESSORS = pickle.load(f)
            
        # Load Selected Features
        with open(os.path.join(MODELS_DIR, "selected_features.json"), 'r') as f:
            SELECTED_FEATURES = json.load(f)
            
        # Load Signature Rules
        SIGNATURE_RULES = load_signature_rules(os.path.join(MODELS_DIR, "signature_rules.json"))
        
        # Load Decision Tree Signature Model
        with open(os.path.join(MODELS_DIR, "signature_tree_model.pkl"), 'rb') as f:
            SIGNATURE_TREE_MODEL = pickle.load(f)
            
        # Load XGBoost Model
        with open(os.path.join(MODELS_DIR, "xgboost_model.pkl"), 'rb') as f:
            XGBOOST_MODEL = pickle.load(f)
            
        # Load Fusion Configuration
        with open(os.path.join(MODELS_DIR, "fusion_config.json"), 'r') as f:
            FUSION_CONFIG = json.load(f)
            
        # Load SHAP Explainer
        with open(os.path.join(MODELS_DIR, "shap_explainer.pkl"), 'rb') as f:
            SHAP_EXPLAINER = pickle.load(f)
            
        print("All models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Attempt to load models on startup
load_all_models()

@app.route('/')
def index():
    """
    Overview Home route showing global metrics from training run.
    """
    # Load training metrics
    metrics = {}
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
    # Load fusion config
    threshold = 0.75
    if FUSION_CONFIG:
        threshold = FUSION_CONFIG.get("optimal_threshold", 0.75)
        
    # Check if models are loaded
    models_ready = (PREPROCESSORS is not None and XGBOOST_MODEL is not None)
    
    return render_template(
        'dashboard.html',
        metrics=metrics,
        threshold=threshold,
        models_ready=models_ready
    )

@app.route('/detection', methods=['GET', 'POST'])
def detection():
    """
    Handles CSV upload and preloaded sample files, running the detection pipeline,
    and rendering prediction results.
    """
    global UPLOADED_BATCH_X, UPLOADED_BATCH_RAW
    
    if PREPROCESSORS is None or XGBOOST_MODEL is None:
        flash("Models are not loaded. Please train the pipeline first.", "danger")
        return redirect(url_for('index'))
        
    predictions_data = []
    batch_metrics = None
    file_processed = False
    has_labels = False
    
    # Check if a preloaded sample was requested
    sample_file = request.args.get('sample')
    
    # List of available preloaded samples to show in the view
    preloaded_samples = [
        {"filename": "normal_traffic.csv", "name": "Normal Traffic", "type": "Safe"},
        {"filename": "exploits_attacks.csv", "name": "Exploits Attack", "type": "Threat"},
        {"filename": "dos_attacks.csv", "name": "DoS Attack", "type": "Threat"},
        {"filename": "reconnaissance_attacks.csv", "name": "Reconnaissance", "type": "Threat"},
        {"filename": "generic_attacks.csv", "name": "Generic Attack", "type": "Threat"},
        {"filename": "fuzzers_attacks.csv", "name": "Fuzzers Attack", "type": "Threat"},
        {"filename": "analysis_attacks.csv", "name": "Analysis Attack", "type": "Threat"},
        {"filename": "backdoor_attacks.csv", "name": "Backdoor Attack", "type": "Threat"},
        {"filename": "mixed_traffic.csv", "name": "Mixed Traffic", "type": "Mixed"}
    ]
    
    df_raw = None
    source_name = None
    
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            flash("No file part in request.", "warning")
            return redirect(request.url)
            
        file = request.files['csv_file']
        if file.filename == '':
            flash("No file selected.", "warning")
            return redirect(request.url)
            
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            try:
                df_raw = pd.read_csv(filepath)
                source_name = file.filename
                file_processed = True
            except Exception as e:
                flash(f"Error reading uploaded file: {e}", "danger")
                
    elif request.method == 'GET' and sample_file:
        # Verify it's a valid sample
        valid_filenames = [s["filename"] for s in preloaded_samples]
        if sample_file in valid_filenames:
            filepath = os.path.join("samples", sample_file)
            if os.path.exists(filepath):
                try:
                    df_raw = pd.read_csv(filepath)
                    source_name = sample_file
                    file_processed = True
                except Exception as e:
                    flash(f"Error reading sample file: {e}", "danger")
            else:
                flash("Requested sample file not found on disk.", "danger")
        else:
            flash("Invalid sample file requested.", "warning")
            
    if file_processed and df_raw is not None:
        try:
            if df_raw.empty:
                flash("The selected dataset is empty.", "warning")
                return redirect(url_for('detection'))
                
            # Strip columns
            df_raw = clean_columns(df_raw)
            
            # Run Preprocessing
            X_proc, y_true = transform_data(df_raw, PREPROCESSORS)
            
            # Slice to selected features
            X_sel = X_proc[SELECTED_FEATURES]
            
            # Cache in memory for SHAP lookup (limit display to 2000 rows for speed and layout stability)
            max_display = 2000
            UPLOADED_BATCH_X = X_sel.head(max_display).copy()
            UPLOADED_BATCH_RAW = df_raw.head(max_display).copy()
            
            # Predict Signature Scores
            sig_scores = evaluate_signature_batch(X_sel, model=SIGNATURE_TREE_MODEL)
            
            # Predict ML Probabilities
            ml_probs = predict_ml_prob(XGBOOST_MODEL, X_sel)
            
            # Fusion prediction
            threshold = FUSION_CONFIG.get("optimal_threshold", 0.75)
            fusion_scores = compute_fusion_score(sig_scores, ml_probs)
            final_preds = (fusion_scores >= threshold).astype(int)
            
            # Create result structure for display
            for i in range(min(len(df_raw), max_display)):
                row_dict = df_raw.iloc[i].to_dict()
                row_dict['idx'] = i
                
                # Add all 20 selected feature values (including engineered ones)
                for col in SELECTED_FEATURES:
                    val = X_sel.iloc[i][col]
                    if isinstance(val, (float, np.floating)):
                        row_dict[col] = round(float(val), 4)
                    else:
                        row_dict[col] = val
                        
                row_dict['signature_score'] = round(float(sig_scores[i]), 4)
                row_dict['ml_probability'] = round(float(ml_probs[i]), 4)
                row_dict['fusion_score'] = round(float(fusion_scores[i]), 4)
                row_dict['final_prediction'] = int(final_preds[i])
                
                if y_true is not None:
                    row_dict['true_label'] = int(y_true[i])
                    
                predictions_data.append(row_dict)
                
            # Compute batch evaluation metrics if ground truth label is present
            if y_true is not None and not np.isnan(y_true).all():
                has_labels = True
                batch_metrics = evaluate_predictions(y_true, final_preds, fusion_scores)
            
        except Exception as e:
            flash(f"Error processing data: {e}", "danger")
            import traceback
            traceback.print_exc()
            
    return render_template(
        'detection.html',
        file_uploaded=file_processed,
        predictions=predictions_data,
        batch_metrics=batch_metrics,
        has_labels=has_labels,
        preloaded_samples=preloaded_samples,
        active_sample=sample_file,
        selected_features=SELECTED_FEATURES
    )

@app.route('/explain/<int:row_idx>')
def explain(row_idx):
    """
    Renders an interactive SHAP force plot for a specific row in the uploaded batch.
    """
    global UPLOADED_BATCH_X
    
    if UPLOADED_BATCH_X is None or SHAP_EXPLAINER is None:
        return "No uploaded batch data found or SHAP explainer not loaded. Run detection first."
        
    if row_idx < 0 or row_idx >= len(UPLOADED_BATCH_X):
        return f"Invalid row index: {row_idx}"
        
    try:
        # Get the specific sample row
        sample_row = UPLOADED_BATCH_X.iloc[[row_idx]]
        
        # Output file path for this force plot
        plot_html_name = f"temp_force_plot_{row_idx}.html"
        output_html_path = os.path.join(app.static_folder, plot_html_name)
        
        # Generate and save HTML force plot
        generate_local_shap_plot(SHAP_EXPLAINER, sample_row, output_html_path)
        
        # Render a simple template containing an iframe to load this file
        return render_template('explain.html', plot_url=url_for('static', filename=plot_html_name), row_idx=row_idx)
    except Exception as e:
        import traceback
        return f"Error generating SHAP explanation: {e}<br><pre>{traceback.format_exc()}</pre>"

@app.route('/rules')
def rules():
    """
    View route showing signature rules extracted from Decision Tree.
    """
    return render_template('rules.html', rules=SIGNATURE_RULES)

@app.route('/reload_models')
def reload_models():
    """
    API endpoint to reload models.
    """
    success = load_all_models()
    if success:
        flash("Models reloaded successfully!", "success")
    else:
        flash("Failed to reload models.", "danger")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
