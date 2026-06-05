# Utility Scripts

## Usage Order

### One-Time Setup (Already Done)
1. `create_labeled_dataset.py` - Merge inventory + Reddit results
   - Only run if you collect new Reddit data
   - Creates `data/validation/labeled_features.csv`

### Every Analysis Run
2. `apply_outcomes.py` - Apply public decision context
```bash
   python scripts/apply_outcomes.py
```
   - Adds `company_action` and `business_outcome`
   - Keeps those two labels separate on purpose
   - Treats ambiguous cases as `UNKNOWN` instead of forcing false certainty

3. Run analysis
```bash
   python src/analysis/statistical_analysis.py
```
   - Compares supported vs pulled-back features
   - Treats business outcome as partial coverage, not as the main label

4. Generate charts
```bash
   python scripts/generate_visualizations.py
```

## Quick Run (All at Once)
```bash
./run_analysis.sh
```
