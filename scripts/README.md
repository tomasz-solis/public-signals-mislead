# Scripts

Run in this order.

1. `create_labeled_dataset.py`: merges the feature inventory with Reddit results into `data/validation/labeled_features.csv`. Already done; only rerun after collecting new Reddit data.
2. `apply_outcomes.py`: adds `company_action` and `business_outcome`, kept separate on purpose. Ambiguous cases stay `UNKNOWN` instead of forcing a label.

   ```bash
   python scripts/apply_outcomes.py
   ```

3. Run the analysis. It compares supported and pulled-back features and treats business outcome as partial coverage, not the main label.

   ```bash
   python src/analysis/statistical_analysis.py
   ```

4. Generate the charts:

   ```bash
   python scripts/generate_visualizations.py
   ```

Or run all of it with `./run_analysis.sh`.
