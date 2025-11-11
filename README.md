# EcoNetToolkit — simple models for ecological data

[![CI](https://github.com/ysims/EcoNetToolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/ysims/EcoNetToolkit/actions/workflows/ci.yml)
[![Lint](https://github.com/ysims/EcoNetToolkit/actions/workflows/lint.yml/badge.svg)](https://github.com/ysims/EcoNetToolkit/actions/workflows/lint.yml)

EcoNetToolkit lets you train a shallow neural network or classical models on your tabular ecological data using a simple YAML file.

- CSV input with automatic preprocessing (impute, scale, encode)
- Model zoo: MLP (shallow), Random Forest, SVM, XGBoost, Logistic Regression, Linear Regression
- Repeated training with different seeds for stable estimates
- Metrics, including for unbalanced datasets (balanced accuracy, PR AUC)
- Configure the project from a single config file

## Getting Started

### macOS and Linux (Terminal)

For these steps, open a new terminal and enter the commands in the command line. 

1. Clone the repository and move into the directory:

    ```bash
    git clone https://github.com/ysims/EcoNetToolkit.git
    cd EcoNetToolkit
    ```

2. In your terminal, run:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    To leave the venv later, run `deactivate`.

If you have already followed these steps during a previous session, reactivate the virtual environment by opening a terminal in the `EcoNetToolkit` directory and run:

```bash
source .venv/bin/activate
```

### Windows (Anaconda)

[Install Anaconda (Windows 64‑bit) from the official website](https://www.anaconda.com/download), using the default settings. After installation, open 'Anaconda Prompt' from the Start Menu. In the prompt, run the following steps.

1. Get Git (this may already be installed):

    ```bash
    conda install git
    ```

2. Clone the repository and move into the directory:

    ```bash
    git clone https://github.com/ysims/EcoNetToolkit.git
    cd EcoNetToolkit
    ```

3. Create the conda environment and activate it:

    ```bash
    conda env create -f environment.yml
    conda activate econet
    ```

If the `conda` command isn’t recognised, make sure you’re in the Anaconda Prompt.

If you have already followed these steps during a previous session, reactivate the conda environment by opening an Anaconda Prompt in the `EcoNetToolkit` directory and run:

```bash
conda activate econet
```

### Configure and Run

All commands should be run in the terminal (macOS and Linux) or the Anaconda prompt (Windows).

EcoNetToolkit includes two example datasets to help you get started.

**Classification Example: Palmer Penguins**

Predict penguin species from morphological measurements:

```bash
python run.py --config configs/penguins_config.yaml
```

This demonstrates multi-class classification (3 species: Adelie, Chinstrap, Gentoo) using features like bill length, flipper length, and body mass.

**Regression Example: Possum Morphology**

Predict possum age from morphological measurements:

```bash
python run.py --config configs/possum_config.yaml
```

This demonstrates continuous variable prediction using head length, skull width, and other physical measurements.

#### Outputs

Outputs are organised into folders based on your config file name. For example, running `configs/possum_config.yaml` creates:

```
outputs/
└── possum_config/                    # Named after your config file
    ├── random_forest/                # Model-specific subfolder
    │   ├── model_random_forest_seed42.joblib
    │   ├── model_random_forest_seed43.joblib
    │   ├── ...
    │   ├── report_random_forest.json
    │   ├── confusion_matrix_random_forest.png   (classification only)
    │   ├── pr_curve_random_forest.png           (classification only)
    │   └── residual_plot_random_forest.png      (regression only)
    ├── xgboost/
    ├── mlp/
    ├── svm/
    ├── linear/
    ├── report_all_models.json        # Combined results across all models
    ├── comparison_mse.png             # Comparison plots (regression)
    ├── comparison_r2.png
    ├── comparison_accuracy.png        # Comparison plots (classification)
    ├── comparison_f1.png
    └── pr_curve_comparison.png        # Combined PR curves (classification)
```

**Model-specific outputs** (in each model subfolder):
- `model_<name>_seed<N>.joblib`: trained models for each random seed
- `report_<model>.json`: per-seed metrics (MSE, R², accuracy, F1, etc.)
- `confusion_matrix_<model>.png`: confusion matrix heatmap (classification)
- `pr_curve_<model>.png`: precision-recall curve (classification)
- `residual_plot_<model>.png`: predicted vs actual and residuals (regression)

**Multi-model comparison outputs** (in the config folder root):
- `report_all_models.json`: combined metrics across all models and seeds
- `comparison_*.png`: side-by-side boxplots comparing model performance
- `pr_curve_comparison.png`: overlaid precision-recall curves (classification)

### Inspecting Saved Models

```bash
python inspect_models.py outputs/possum_config/random_forest/model_*.joblib
```

This shows model type, parameters, training iterations, and other metadata. The `.joblib` files contain serialised scikit-learn models that you can load and use for predictions:

```python
import joblib
model = joblib.load('outputs/possum_config/random_forest/model_random_forest_seed42.joblib')
predictions = model.predict(X_new)  # X_new must be preprocessed the same way
```

## Config reference (YAML)

You can train **single or multiple models** for comparison. See `configs/penguins_config.yaml` for comprehensive examples of all model types and their parameters.

### Simple example (single model, classification)

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
    repetitions: 5      # or num_seeds: 5
    random_seed: 0

# Optional: specify output directory (defaults to outputs/<config_name>/)
output:
    dir: outputs/my_experiment
```

**Note:** If `output.dir` is not specified, outputs are automatically saved to `outputs/<config_name>/` where `<config_name>` is derived from your config file name.

### Available models and key parameters

**MLP (Multi-Layer Perceptron)**
- `hidden_layer_sizes`: List of layer sizes, e.g., `[32, 16]`
- `max_iter`: Maximum iterations
- `early_stopping`: Stop when validation plateaus
- `n_iter_no_change`: Patience - epochs to wait without improvement (default: 10)
- `validation_fraction`: Fraction of training data for validation
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

**Logistic Regression** (classification only)
- `C`: Inverse regularization strength
- `max_iter`: Max solver iterations
- `solver`: `lbfgs`, `liblinear`, `newton-cg`, etc.
- `penalty`: `l1`, `l2`, `elasticnet`, or `null`

**Linear Regression** (regression only)
- `fit_intercept`: Whether to calculate the intercept (default: `true`)
- `normalize`: Whether to normalize features (deprecated, use `scaling` in data config)

### Notes on metrics

**Classification:**
- Primary ranking metric: **Cohen's kappa** (accounts for chance agreement, robust for imbalanced data)
- Also reported: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC

**Regression:**
- Primary ranking metric: **MSE** (Mean Squared Error, lower is better)
- Also reported: RMSE, MAE, R², MAPE

### Additional notes

- For classification with two classes, ROC-AUC and PR-AUC are computed if the model can produce probabilities (e.g., MLP, RandomForest, SVM with `probability=True`).
- For multi-class problems, macro-averaged Precision/Recall/F1 summarize performance across all classes.
- Models are ranked by Cohen's kappa (classification) or MSE (regression) to identify the best performer.

## Using your own data

1. Place your CSV file in the `data` folder.
2. Make a `yaml` config file in the `configs` folder for your data. 

    Use one of the existing config files (penguin for classification; possum for regression) as a basis for your data. Change the CSV path to point to your CSV file and change the features and label parameters to match the columns in your CSV file. The parameters for the different models should be tuned for your problem.

    If you are unsure how to make the `yaml` file, try providing ChatGPT (or your favourite LLM) with your CSV file (or the first few rows) and link to this repository and ask it to make a config file for your data. Consider data privacy before doing this. 


Some tips: 

- Ensure your `features:` list includes only columns available in your CSV.
- Text categories are automatically one-hot encoded.
- If your dataset is very imbalanced, consider `class_weight: balanced` in `model.params`
	for `logistic` or `svm`, or tune `scale_pos_weight` for `xgboost`.

## Testing

Testing is provided for development purposes and is used by the CI system when pull requests are created.

### Unit Tests

Run the test suite to ensure everything works correctly:

```bash
python run_tests.py all -v
```

Or run with coverage:

```bash
python run_tests.py all -v --cov=ecosci --cov-report=html
```

The tests verify:
- Data loading and preprocessing (scaling, encoding, splits)
- Model instantiation and training for all model types
- Metric computation produces sane values (0-1 ranges, no NaN/inf)
- Full end-to-end pipeline runs without errors
- Models produce reasonable accuracy (better than random)

### End-to-End Testing

Test the full pipeline with the included example datasets:

**Classification (Penguins):**
```bash
python run.py --config configs/penguins_config.yaml
```

**Regression (Possum):**
```bash
python run.py --config configs/possum_config.yaml
```

These demonstrate that the toolkit works correctly for both problem types and generates appropriate metrics and visualizations.

## Troubleshooting

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
