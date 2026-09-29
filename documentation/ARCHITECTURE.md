# Architecture

## Pipeline

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

## What each stage does

| Stage | What it does |
|---|---|
| `create_batches.py` | Splits the inventory into batches, because Google Trends rate-limits hard. |
| `collect_trends_data.py` | Collects raw Google Trends interest over time and computes launch-based decay (used earlier in the pipeline). |
| `merge_batches.py` | Merges batch outputs into one dataset. Extended-window files override the default for the same feature. |
| `recalculate_with_peaks.py` | Recomputes decay from the actual peak date instead of the launch date. This is the more defensible measure, because many features peak after launch. |
| `reddit/validate_features.py` | Collects Reddit mentions and simple sentiment. These are public reaction signals, not proof of value. |
| `create_labeled_dataset.py` | Joins trends metrics and Reddit signals into one table. Adds public-signal labels only. |
| `apply_outcomes.py` | Adds public decision context from `config/outcomes.py`, keeping `company_action` apart from `business_outcome`. |
| `statistical_analysis.py` | Compares supported and pulled-back features: group tests, effect sizes, bootstrap CIs, power analysis, threshold sensitivity, framework validation and signal ablation. |
| `generate_visualizations.py` | Builds local interactive HTML charts and static SVG previews for GitHub. |

## Design choices

1. Company action and business outcome are separate. The repo doesn't force every feature into a fake success or failure label. `company_action` is what the public record suggests the company did; `business_outcome` is what it can prove about value.
2. Decay is measured from the peak. Launch-date decay is easy and often wrong. If a feature peaks later (word of mouth, marketing, a big event), launch-date decay understates persistence.
3. Public signals are noisy inputs. Search interest, mention volume and sentiment can show attention, confusion, backlash and cultural visibility. They can't show retention lift, monetisation or strategic value for a segment. So the output is decision support, not product truth.
4. Static and interactive charts are kept apart. Interactive Plotly HTML is regenerated locally in `results/figures/`. Static SVG previews live in `documentation/assets/` so reviewers see them on GitHub.
