# Case study: Netflix password sharing

Search interest fell 93.3% within four weeks of peak, and Netflix then reported 9.3M paid net additions. The public signal and the business outcome pointed in opposite directions.

## The public signal

In May 2023 Netflix started enforcing its password-sharing crackdown across markets. Within four weeks of peak, Google Trends shows searches for "Netflix password sharing" down 93.3%.

Reddit was noisy: 37 mentions in the tracked window, 29.7% negative by our keyword lexicon and 10.8% positive. The other 59.5% were neutral (questions, workarounds, plain descriptions). The overall classification was UNCERTAIN: the signals were too mixed to call.

An analyst reading only public signals would see collapsed search interest, negative-leaning Reddit and no clear verdict, and might conclude the feature was struggling and consider a rollback. That would have been wrong.

## What actually happened

Netflix reported 9.3 million paid net additions in Q1 2024 and named the crackdown and the Extra Member add-on as main drivers. It kept the policy, expanded it globally and called it a growth success in several shareholder letters.

Source: [Netflix Q1 2024 shareholder letter](https://ir.netflix.net/financials/quarterly-earnings/default.aspx) (Tier 1 evidence).

## Why the public signal misled

1. Search decay measured curiosity, not usage. People searched to understand the new rules. Once they understood (and complied, bought Extra Member or cancelled), there was no reason to search again. The decay reflects answered questions, not product failure.
2. Reddit negativity was selection bias. Angry people posted. People who paid the extra $8 and moved on didn't. Complaint volume doesn't scale with business harm.
3. The metric that mattered was invisible. Netflix cared about how many sharing households converted to paid accounts. That number, inside the 9.3M net additions, was never going to show up in Google Trends or Reddit.

## The contrast: Disney+ GroupWatch

Similar public signal, different product path:

| Signal | Netflix Password Sharing | Disney+ GroupWatch |
|--------|--------------------------|-------------------|
| Search decay | 93.3% | 100.0% |
| Reddit mentions | 37 | 12 |
| Reddit negative ratio | 29.7% | 41.7% |
| Reddit positive ratio | 10.8% | 0.0% |
| Company action | Supported | Pulled back |
| Business outcome | Positive (9.3M subs) | Unknown |

GroupWatch was quietly removed in September 2023, according to a help-center notice. No earnings mention, no stated audience impact, no revenue attribution. Whether it had value for a niche audience is unknown.

Source: [Disney+ help-center notice via ComicBook](https://comicbook.com/irl/news/disney-plus-groupwatch-feature-no-longer-available/) (Tier 2 evidence).

Both show steep decay and negative attention, and their product paths split completely. That is what this repo studies.

## Where it fits in the wider analysis

Across 36 subscription features, 69% of those companies kept supporting still show more than 80% search decay (95% CI 44% to 86%, n=16). Netflix Password Sharing is one of them, and the clearest case of heavy decay alongside strong business results.

The decision framework classifies Netflix Password Sharing correctly as supported (true positive) and GroupWatch correctly as pulled back (true negative). Its two misses are Games and App-Only Membership, both false positives (predicted supported, actually pulled back).

## Internal data that would have changed the analysis

As the analyst on this decision, I'd have asked for:

- Conversion: the share of sharing households that moved to paid accounts or Extra Member.
- Churn by segment: did cancellations spike among sharers, and did they come back within 90 days?
- Revenue per user: net ARPU after lost sharers and new subscribers.
- Retention cohorts: 30/60/90-day retention of converted accounts vs organically acquired ones.
- Cost of enforcement: engineering, support tickets and brand cost.

None of it is visible from outside, and all of it is needed before recommending a rollback.

## The lesson

This isn't about Netflix being right or Disney being wrong. It's about what public data can and can't tell you.

Public signals settle faster than product value. Search interest fades in weeks; retention takes quarters to measure. A feature can look dead in Google Trends while driving the best subscriber quarter in years.

Rule of thumb: outside concern without internal evidence should trigger an investigation, not a rollback.

## Data sources

- Search decay: Google Trends via `pytrends`, peak-based method (`src/data_collection/recalculate_with_peaks.py`).
- Reddit sentiment: keyword lexicon on company subreddit mentions (`src/data_collection/reddit/reddit_validator.py`). The [sentiment method note](README.md#sentiment-method) explains why the crude method is deliberate.
- Business outcome: Netflix Q1 2024 shareholder letter ([source](https://ir.netflix.net/financials/quarterly-earnings/default.aspx)).
- GroupWatch removal: Disney+ help-center notice ([source](https://comicbook.com/irl/news/disney-plus-groupwatch-feature-no-longer-available/)).
- Framework validation: `framework_error_analysis()` in `src/analysis/statistical_analysis.py`.
