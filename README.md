# EcoNetToolkit — simple models for ecological data

EcoNetToolkit lets you train a shallow neural network or classical models on your tabular ecological data using a simple YAML file.

- CSV input with automatic preprocessing (impute, scale, encode)
- Model zoo: MLP (shallow), Random Forest, SVM, XGBoost, Logistic Regression
- Repeated training with different seeds for stable estimates
- Metrics, including for unbalanced datasets (balanced accuracy, PR AUC)
- Configure the project from a single config file

## Setup

### macOS and Linux: Create and activate a virtual environment

In your terminal, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To leave the venv later, run `deactivate`.

### Windows: Install Anaconda and create an environment

[Install Anaconda (Windows 64‑bit) from the official website](https://www.anaconda.com/download), using the default settings. After installation, open “Anaconda Prompt” from the Start Menu. In the prompt, run:

```bash
conda env create -f environment.yml
conda activate econet
```

If the `conda` command isn’t recognised, make sure you’re in the Anaconda Prompt.

### Configure and Run

1. Inspect and edit the example config.

    `configs/example_config.yaml` shows all options with a toy dataset in `data/sample.csv`.

2. Run:

    ```bash
    python run.py --config configs/example_config.yaml
    ```

    Outputs are written to `outputs/` by default:
    
    **Single model outputs:**
    - `report_<model>.json`: per-seed metrics for each model
    - `confusion_matrix_<model>.png`: confusion matrix for each model
    - `pr_curve_<model>.png`: precision-recall curve for each model
    - `model_<name>_seed<N>.joblib`: trained models saved per seed
    
    **Multi-model comparison outputs:**
    - `report_all_models.json`: combined metrics across all models
    - `comparison_*.png`: side-by-side boxplots comparing models (accuracy, f1, etc.)
    - `pr_curve_comparison.png`: overlaid precision-recall curves for all models

3. Inspect saved models (optional):

    ```bash
    python inspect_models.py outputs/model_*.joblib
    ```

    This shows model type, parameters, training iterations, and other metadata. The `.joblib` files contain serialised scikit-learn models that you can load and use for predictions:

    ```python
    import joblib
    model = joblib.load('outputs/model_mlp_seed0.joblib')
    predictions = model.predict(X_new)  # X_new must be preprocessed the same way
    ```

## Config reference (YAML)

You can train **single or multiple models** for comparison. See `configs/example_config.yaml` for comprehensive examples of all model types and their parameters.

### Simple example (single model)

```yaml
problem_type: classification

data:
    path: data/sample.csv
    features: [f1, f2, habitat]
    label: label
    test_size: 0.2
    val_size: 0.2
    random_state: 0
    scaling: standard
    impute_strategy: mean

models:
  - name: mlp
    params:
      hidden_layer_sizes: [32, 16]
      max_iter: 300
      early_stopping: true

training:
    repetitions: 5
    random_seed: 0

output_dir: outputs
```

### Multi-model comparison example

Train and compare multiple models at once (see `configs/multi_model_config.yaml`):

```yaml
models:
  - name: logistic
    params:
      C: 1.0
      max_iter: 1000
  
  - name: random_forest
    params:
      n_estimators: 100
      max_depth: null
  
  - name: mlp
    params:
      hidden_layer_sizes: [32, 16]
      max_iter: 300
      early_stopping: true
```

### Available models and key parameters

**MLP (Multi-Layer Perceptron)**
- `hidden_layer_sizes`: List of layer sizes, e.g., `[32, 16]`
- `max_iter`: Maximum iterations
- `early_stopping`: Stop when validation plateaus
- `alpha`: L2 regularization
- `learning_rate_init`: Initial learning rate

**Random Forest**
- `n_estimators`: Number of trees
- `max_depth`: Max tree depth (`null` = unlimited)
- `min_samples_split`: Min samples to split
- `max_features`: Features per split (`sqrt`, `log2`, or `null`)

**SVM (Support Vector Machine)**
- `C`: Regularization parameter
- `kernel`: `rbf`, `linear`, `poly`, or `sigmoid`
- `gamma`: Kernel coefficient (`scale` or `auto`)

**XGBoost**
- `n_estimators`: Boosting rounds
- `max_depth`: Max tree depth
- `learning_rate`: Step size (eta)
- `subsample`: Training instance ratio
- `colsample_bytree`: Feature ratio

**Logistic Regression**
- `C`: Inverse regularization strength
- `max_iter`: Max solver iterations
- `solver`: `lbfgs`, `liblinear`, `newton-cg`, etc.
- `penalty`: `l1`, `l2`, `elasticnet`, or `null`

Notes
- For classification with two classes, we compute ROC-AUC and PR AUC if the
	model can produce probabilities (e.g., MLP, RandomForest, SVM with probability=True).
- Balanced accuracy is helpful when one class is much rarer than the other.
- If you have many classes, macro-averaged Precision/Recall/F1 summarise across them.

## Tips for your own data

- Ensure your `features:` list includes only columns available in your CSV.
- Text categories are automatically one-hot encoded.
- If your dataset is very imbalanced, consider `class_weight: balanced` in `model.params`
	for `logistic` or `svm`, or tune `scale_pos_weight` for `xgboost`.

## Testing

Run the test suite to ensure everything works correctly:

```bash
pytest tests/test_ecosci.py -v
```

Or run with coverage:

```bash
pytest tests/test_ecosci.py -v --cov=ecosci --cov-report=html
```

The tests verify:
- Data loading and preprocessing (scaling, encoding, splits)
- Model instantiation and training for all model types
- Metric computation produces sane values (0-1 ranges, no NaN/inf)
- Full end-to-end pipeline runs without errors
- Models produce reasonable accuracy (better than random)

## Troubleshooting

- ImportError for xgboost: install it with `pip install xgboost` or switch to another model.
- Shapes or column errors: double-check your `features:` and `label:` names.
- No probabilities for some models: not all models support `predict_proba`; plots that need probabilities are skipped automatically.

## Development layout

- `run.py` — simple entrypoint script
- `ecosci/` — package with modules:
	- `config.py` (YAML reader)
	- `data.py` (CSV loader + preprocessing)
	- `models.py` (ModelZoo)
	- `trainer.py` (seeded training loop, saving models)
	- `eval.py` (metrics and plots)
- `configs/` — example configuration
- `data/` — sample CSV for quick testing
