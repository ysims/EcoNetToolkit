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
    - `report.json`: per-seed metrics
    - `metric_*.png`: boxplots across seeds
    - `confusion_matrix.png`: confusion heatmap (last run)
    - `pr_curve.png`: precision-recall curve if binary and probabilities available
    - `model_*.joblib`: saved models per seed

## Config reference (YAML)

```yaml
problem_type: classification  # or regression

data:
    path: data/sample.csv       # location of your CSV file
    features: [f1, f2, habitat] # list of input columns in your CSV
    label: label                # the column to predict
    test_size: 0.2              # fraction held out for final testing
    val_size: 0.2               # fraction of remaining data used as validation
    random_state: 0             # controls the train/val/test split
    scaling: standard           # standard|minmax
    impute_strategy: mean       # mean|median|most_frequent

model:
    name: mlp                   # mlp|random_forest|svm|xgboost|logistic
    params:
        hidden_layer_sizes: [32]  # only for mlp; a shallow single hidden layer
        max_iter: 300             # training iterations for mlp
        early_stopping: true      # let sklearn stop when val score stops improving
        validation_fraction: 0.1  # fraction inside the mlp used for early stopping

training:
    repetitions: 3              # if seeds not provided, run base_seed..+n-1
    random_seed: 0              # base seed (for splits and models)
    # seeds: [1, 2, 3]         # optional explicit seeds list overrides repetitions

output_dir: outputs
```

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
