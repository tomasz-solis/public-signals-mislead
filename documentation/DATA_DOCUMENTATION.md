# Data documentation

Sources, collection method and how to reproduce the data.

## Sources

### Feature inventory (`data/raw/feature_inventory.csv`)

Compiled by hand from company press releases (Spotify Newsroom, Netflix About, YouTube Blog, Disney Newsroom) and public earnings call transcripts. 50 features, launched November 2021 to December 2024.

| Column | Meaning |
|---|---|
| `feature_id` | Unique id (1 to 50) |
| `feature_name` | Readable name |
| `company` | Platform (Spotify, Netflix, YouTube and others) |
| `launch_date` | Official launch date (YYYY-MM-DD) |
| `feature_type` | Category (AI, Monetization, Content and others) |
| `google_trends_keyword` | Search term used for collection |
| `announcement_source` | URL of the official announcement |
| `expected_stickiness` | Initial hypothesis |
| `notes` | Extra context |

### Google Trends (`data/trends/`)

Collected with `pytrends`, an unofficial Google Trends API.

- Window: 14 days before launch to 98 days (14 weeks) after, 112 days per feature. The window is long because some features peaked 55 days after launch.
- Rate limits: batches of 10 features, 10 seconds between requests, 24 hours between batches. About 5 days in total.
- Geography: global. Netflix, Spotify and Disney+ are international, password sharing was worldwide news, and the analysis compares relative patterns, not absolute volumes.

### Reddit (`data/validation/`)

Collected with PRAW (Python Reddit API Wrapper), with a public JSON fallback, from company subreddits (r/netflix, r/spotify and others) over 30 to 90 days after launch. PRAW allows about 60 requests a minute and public JSON about 30, so all companies take 1 to 2 hours.

Sentiment is lexicon matching against 30 hand-picked positive and negative keywords. That is a crude baseline on purpose: if a noisy measure of an already noisy signal can't separate supported from pulled-back features, a better classifier is unlikely to rescue it. The keyword list is in `src/data_collection/reddit/reddit_validator.py`.

## Processing

1. `merge_batches.py` combines the 5 batches and removes duplicates into `MERGED_trends_data.csv`.
2. `recalculate_with_peaks.py` finds each feature's actual peak date and computes decay from it into `MERGED_trends_data_PEAK_metrics.csv`.

Why peak-based: features don't peak on launch day. Paramount+ Live Sports launched on 1 March and peaked on 25 March (24 days later). Apple Music Family launched on 15 October and peaked on 29 November (45 days later). People search when they hit a problem (a big game, a pricing question), not when a feature launches. Launch-based decay came out at 0.7% (falsely "sticky"); peak-based decay is 73% (correctly "novelty").

## Final metrics

| Metric | Definition |
|---|---|
| `days_to_peak` | Days from launch to peak |
| `peak_interest` | Google Trends score at peak (0 to 100, normalised) |
| `week_4_interest` | Average interest 21 to 28 days after peak |
| `week_8_interest` | Average interest 56 to 63 days after peak |
| `decay_rate_w4` | (peak - week_4) / peak |
| `decay_rate_w8` | (peak - week_8) / peak |

Classification: `sticky` below 30% decay, `mixed` 30% to 70%, `novelty` above 70%, `unknown` with too little data.

## Limitations

1. Search volume isn't usage. Falling search can mean adoption (users learned the feature and use it by habit) or abandonment (curiosity wore off). Cross-check with company metrics and Reddit sentiment.
2. Google Trends normalises each feature to peak = 100 within its timeframe. Decay patterns compare across features; absolute volumes don't. "Netflix password sharing" and "Disney+ Parental Controls" both peak at 100, but Netflix likely has about 1000x the volume.
3. Sampling bias, mostly fixed. Rate limits first blocked high-volume features while letting low-volume ones through. Batch collection with 24-hour gaps captured the full set. Features with very low volume still show no data, and 5 to 8 were excluded.
4. Patterns may differ by market. This is global behaviour.
5. Features launched November 2021 to December 2024. The newest may still be in the awareness phase.

## Reproducing

```bash
# 1. Clone
git clone https://github.com/tomasz-solis/public-signals-mislead
cd public-signals-mislead

# 2. Install
pip install -r requirements.txt

# 3. Create batches
python src/data_collection/create_batches.py

# 4. Collect (5 days, one batch per day)
python src/data_collection/collect_trends_data.py --full --input data/raw/batches/batch_1_of_5.csv
# Wait 24 hours
python src/data_collection/collect_trends_data.py --full --input data/raw/batches/batch_2_of_5.csv
# Repeat for batches 3-5

# 5. Merge and analyse
python src/data_collection/merge_batches.py
python src/data_collection/recalculate_with_peaks.py --input data/trends/MERGED_trends_data.csv
```

Trends data can differ slightly by collection date because Google updates volumes and sampling. The core sticky vs novelty patterns should hold.

## What's in the repo

Included: the feature inventory, final metrics, sample trends data (5 features) and all code. Not included: full raw trends data (about 2 to 5 MB) and intermediate batch files, because the scripts can rebuild them and the repo keeps data under 1 MB. For the full dataset, contact me or run the collection (5 days because of rate limits).

## Quality checks

- Launch dates checked against official sources.
- Keywords tested by hand on trends.google.com.
- Duplicates removed after merging batches.
- Peak dates inspected visually.
- Decay calculations spot-checked.

Known issue: Netflix Extra Member has no search volume (the keyword is too specific). Other low-volume features are excluded.

## References

- Google Trends: [trends.google.com](https://trends.google.com)
- pytrends: [github.com/GeneralMills/pytrends](https://github.com/GeneralMills/pytrends)
- Google Trends method: [support.google.com/trends/answer/4365533](https://support.google.com/trends/answer/4365533)
- Company sources: see `feature_inventory.csv`

Last updated December 2024. Global coverage. Features from November 2021 to December 2024.

## Contact

Tomasz Solis · tomasz.solis@gmail.com · [LinkedIn](https://www.linkedin.com/in/tomaszsolis) · [GitHub](https://github.com/tomasz-solis)
