# How to Run This Analysis

**Time:** 5-10 minutes  
**Difficulty:** Easy

If your machine exposes `python3` instead of `python`, substitute that in the commands below.

## Setup

### 1. Clone or Download

```bash
git clone https://github.com/yourusername/public-signals-mislead.git
cd public-signals-mislead
```

### 2. Install the Project

```bash
python -m pip install -e .
```

Optional, if you want to run tests too:

```bash
python -m pip install -e '.[dev]'
```

### 2.5 Environment Setup (Optional)

This is only needed if you want to re-collect Reddit data.

```bash
cp .env.example .env
```

Fill in the Reddit API credentials in `.env`. The analysis itself runs fine without this because the collected data is already in the repo.

### 3. Verify the Data Files

```bash
ls data/validation/labeled_features.csv
ls data/trends/
```

## Run the Analysis

### Step 1: Apply Decision Context

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

This step adds two fields that the rest of the repo depends on:

- `company_action`: what the public record suggests the company did
- `business_outcome`: what the public record can actually prove about value

Those fields are intentionally separate. Removal is observable. True value often is not.

### Step 2: Run the Test Suite

```bash
python -m pytest tests -v
```

Expected result:

```text
============================== 7 passed in ...
```

### Step 3: Run the Statistical Analysis

```bash
python src/analysis/statistical_analysis.py
```

What to look for:

- `Search Decay - Supported vs Pulled Back` uses Mann-Whitney U as the primary test
- the Bonferroni threshold is shown explicitly
- the power analysis explains that the study can only detect very large effects
- the sensitivity section shows how the high-decay share changes from 50% to 95%
- the framework validation compares the decision rules against a majority-class baseline
- the signal ablation table shows whether combined signals outperform single-signal rules
- the observability note explains that `company_action` is more visible than true value

Current headline output:

```text
Sample sizes: 16 supported, 3 pulled back, 36 total
Bonferroni threshold for 3 tests: p < 0.017
...
KEY FINDING: 11 supported features above 80% decay (69%)
```

The script also updates `data/validation/statistical_results.csv`.

If you want the repo walkthrough instead of just the commands, start with:

- `README.md`
- `documentation/HOW_PRODUCT_TEAMS_SHOULD_USE_THIS.md`
- `documentation/ARCHITECTURE.md`

### Step 4: Generate the Charts

```bash
python scripts/generate_visualizations.py
```

Expected result:

```text
✓ VISUALIZATIONS COMPLETE
```

The HTML files are written to `results/figures/`.

### Step 5: Open a Chart

```bash
open results/figures/decay_vs_action.html
```

Use `start` on Windows or `xdg-open` on Linux.

## One-Command Option

```bash
./run_analysis.sh
```

That script now detects `python` vs `python3` automatically and installs the local package in editable mode when needed.

## Troubleshooting

### `python: command not found`

Use `python3` instead.

### `ModuleNotFoundError`

Reinstall the local package:

```bash
python -m pip install -e .
```

### `pytest` is missing

Install the dev extra:

```bash
python -m pip install -e '.[dev]'
```

### Charts are missing after clone

They are generated artifacts now, not tracked files. Rebuild them with:

```bash
python scripts/generate_visualizations.py
```

### Why does the repo keep `business_outcome = UNKNOWN` so often?

That is by design. The repo is framed around product-decision risk, not full business-outcome prediction. Public sources often reveal whether a feature was kept or pulled back, but they usually do not reveal the real value of the feature.

## Optional Recollection

You do not need this for the portfolio version, but the pipeline is still there.

### Google Trends

```bash
python src/data_collection/create_batches.py
python src/data_collection/collect_trends_data.py --full --input data/raw/batches/batch_1_of_5.csv
python src/data_collection/merge_batches.py
python src/data_collection/recalculate_with_peaks.py --input data/trends/MERGED_trends_data.csv
```

Collection runs now write structured logs to `data/collection.log`.

### Reddit Validation

```bash
python src/data_collection/reddit/validate_features.py --companies "Netflix"
```

Bottom line: for normal use, skip recollection and work from the provided data.
