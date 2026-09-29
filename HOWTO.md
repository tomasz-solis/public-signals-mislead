# How to run the analysis

Takes 5 to 10 minutes. Run everything from the project root.

## Setup

1. Clone the repo:

   ```bash
   git clone https://github.com/tomasz-solis/public-signals-mislead.git
   cd public-signals-mislead
   ```

2. Create a local virtualenv and install the project:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   python -m pip install -e .
   python -m pip install -e '.[dev]'   # optional, for tests
   ```

3. Optional, only to re-collect Reddit data: copy `.env.example` to `.env` and fill in the Reddit API credentials. The analysis runs without this because the collected data is already in the repo.

4. Check the data is there:

   ```bash
   ls data/validation/labeled_features.csv
   ls data/trends/
   ```

## Run the analysis

### 1. Apply decision context

```bash
python scripts/apply_outcomes.py
```

Expected summary:

```text
Supported: 16
Pulled back: 3
Unknown action: 17

Known business outcome coverage:
  Positive: 7
  Negative: 2
  Unknown: 27
```

This adds the two fields the rest of the repo depends on: `company_action` (what the public record suggests the company did) and `business_outcome` (what the public record can prove about value). They are kept apart on purpose.

### 2. Run the tests

```bash
python -m pytest tests -v
```

All tests should pass.

### 3. Run the statistics

```bash
python src/analysis/statistical_analysis.py
```

What to look for:

- `Search Decay - Supported vs Pulled Back` uses Mann-Whitney U as the main test.
- The Bonferroni threshold is shown.
- The power analysis explains that only very large effects are detectable.
- The sensitivity section shows how the high-decay share changes from 50% to 95%.
- The framework validation compares the decision rules with a majority-class baseline.
- The signal ablation table shows whether combined signals beat single-signal rules.
- The observability note explains that `company_action` is easier to see than true value.

Current headline output:

```text
Sample sizes: 16 supported, 3 pulled back, 36 total
Bonferroni threshold for 3 tests: p < 0.017
...
KEY FINDING: 11 supported features above 80% decay (69%)
```

The script also updates `data/validation/statistical_results.csv`.

### 4. Generate the charts

```bash
python scripts/generate_visualizations.py
```

It prints `OK VISUALIZATIONS COMPLETE` and writes HTML files to `results/figures/`. Open one with `open results/figures/decay_vs_action.html` (macOS), `start` (Windows) or `xdg-open` (Linux).

For a walkthrough instead of commands, read `README.md`, `documentation/HOW_PRODUCT_TEAMS_SHOULD_USE_THIS.md` and `documentation/ARCHITECTURE.md`.

## One command

```bash
./run_analysis.sh
```

It picks `python` or `python3` and installs the package in editable mode if needed.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python: command not found` | Use `python3`. |
| `(venv)` shows but `python` isn't found | The virtualenv was created before the repo was moved or renamed. Delete `venv` and repeat setup step 2. |
| `ModuleNotFoundError` | Reinstall: `python -m pip install -e .` |
| `pytest` is missing | Install the dev extra: `python -m pip install -e '.[dev]'` |
| Charts are missing after cloning | They are generated, not tracked. Run `python scripts/generate_visualizations.py`. |

Why is `business_outcome` so often `UNKNOWN`? By design. The repo is about decision risk, not predicting business outcomes. Public sources often show whether a feature was kept or pulled back, but rarely its real value.

## Re-collecting data (optional)

Not needed for normal use; work from the provided data. The pipeline is still there.

Google Trends:

```bash
python src/data_collection/create_batches.py
python src/data_collection/collect_trends_data.py --full --input data/raw/batches/batch_1_of_5.csv
python src/data_collection/merge_batches.py
python src/data_collection/recalculate_with_peaks.py --input data/trends/MERGED_trends_data.csv
```

Collection runs write logs to `data/collection.log`.

Reddit validation:

```bash
python src/data_collection/reddit/validate_features.py --companies "Netflix"
```
