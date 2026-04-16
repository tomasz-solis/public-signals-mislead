# Architecture

## Pipeline Overview

```text
data/raw/feature_inventory.csv
        |
        v
src/data_collection/create_batches.py
        |
        v
src/data_collection/collect_trends_data.py
        |
        v
src/data_collection/merge_batches.py
        |
        v
src/data_collection/recalculate_with_peaks.py
        |
        +--> data/trends/MERGED_trends_data_PEAK_metrics.csv
        |
        v
src/data_collection/reddit/validate_features.py
        |
        v
scripts/create_labeled_dataset.py
        |
        v
scripts/apply_outcomes.py
        |
        v
data/validation/labeled_features.csv
        |
        +--> src/analysis/statistical_analysis.py
        |
        +--> scripts/generate_visualizations.py
```

## What Each Stage Does

`create_batches.py`
- Splits the inventory into collection-friendly batches because Google Trends rate-limits aggressively.

`collect_trends_data.py`
- Collects raw Google Trends interest-over-time data.
- Computes launch-based decay metrics used earlier in the pipeline.

`merge_batches.py`
- Merges batch outputs into one trends dataset.
- Applies an override rule when extended-window files exist for the same feature.

`recalculate_with_peaks.py`
- Recomputes decay from the actual peak date rather than the launch date.
- This is the more defensible version of decay because many features peak after launch.

`reddit/validate_features.py`
- Collects Reddit mentions and simple sentiment signals.
- These are public reaction signals, not proof of product value.

`create_labeled_dataset.py`
- Joins trends metrics and Reddit signals into one analysis table.
- Adds public-signal labels only.

`apply_outcomes.py`
- Adds public decision context from `config/outcomes.py`.
- Separates `company_action` from `business_outcome`.

`statistical_analysis.py`
- Compares supported vs pulled-back features.
- Runs group tests, effect sizes, bootstrap CI, power analysis, threshold sensitivity, framework validation, and signal ablation.

`generate_visualizations.py`
- Produces local interactive HTML charts for exploration.
- Also writes static SVG previews for GitHub-rendered docs.

## Key Design Choices

### 1. Company action and business outcome are separate

This repo does not force every feature into a fake success/failure label.

- `company_action` is what the public record suggests the company did
- `business_outcome` is what the public record can actually prove about value

That separation is the backbone of the analysis.

### 2. Peak-based decay matters

Launch-date decay is easy to calculate and often wrong.

If a feature peaks later because of word of mouth, marketing, or a tentpole event, launch-date decay understates persistence. The peak-based recalculation step exists to avoid that distortion.

### 3. Public signals are treated as noisy inputs

Search interest, mention volume, and sentiment can reveal:

- attention
- confusion
- backlash
- cultural salience

They cannot directly reveal:

- retention lift
- monetization impact
- segment-level strategic value

That is why the repo frames the output as decision support, not product truth.

### 4. Static and interactive outputs are kept separate

- Interactive Plotly HTML stays in `results/figures/` and is regenerated locally
- Static SVG previews live in `documentation/assets/` for GitHub readability

This keeps the repo clean without hiding the visuals from reviewers.
