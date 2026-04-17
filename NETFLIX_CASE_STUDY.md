# Case Study: Netflix Password Sharing

## The Public Signal

In May 2023, Netflix began enforcing its password sharing crackdown across markets. Within four weeks of peak search interest, Google Trends data shows a 93.3% decay in searches for "Netflix password sharing."

Reddit discussion was noisy: 37 mentions in the tracked window, with 29.7% classified as negative by our keyword lexicon. Only 10.8% read as positive. The remaining 59.5% were neutral — people asking questions, sharing workarounds, or describing the change without strong sentiment. The overall classification for this feature landed as UNCERTAIN, meaning the public signals were too mixed to call.

If you were a product analyst reading only the public signals, the story writes itself:

- search interest collapsed (93.3% decay)
- Reddit skewed negative (29.7% negative vs 10.8% positive)
- classification: UNCERTAIN — the signals don't resolve into a clear verdict

The reasonable conclusion: this feature is struggling. Consider rollback.

That conclusion would have been wrong.

## What Actually Happened

Netflix reported 9.3 million paid net additions in Q1 2024, directly citing the password sharing crackdown and the Extra Member add-on as primary drivers. The company kept the policy, expanded it globally, and described it as a growth success in multiple shareholder letters.

Source: [Netflix Q1 2024 shareholder letter](https://ir.netflix.net/financials/quarterly-earnings/default.aspx) (Tier 1 evidence).

The public signal and the business outcome pointed in opposite directions.

## Why The Public Signal Misled

Three things happened that external data could not capture:

**1. Search decay measured curiosity, not usage.** People searched "Netflix password sharing" to understand the new rules. Once they understood — whether they complied, bought Extra Member, or cancelled — there was no reason to search again. The 93.3% decay reflects resolved information needs, not product failure.

**2. Reddit negativity was selection bias.** People who were angry posted. People who paid the extra $8 and moved on did not write about it. The 29.7% negative ratio overstates the actual churn impact because complaint volume does not scale linearly with business harm.

**3. The metric that mattered was invisible.** What Netflix cared about was conversion: how many previously-sharing households converted to paid accounts. That number — embedded in the 9.3M net additions — was never going to appear in Google Trends or Reddit.

## The Contrast: Disney+ GroupWatch

Disney+ GroupWatch shows a similar public-signal pattern but a different product path.

| Signal | Netflix Password Sharing | Disney+ GroupWatch |
|--------|--------------------------|-------------------|
| Search decay | 93.3% | 100.0% |
| Reddit mentions | 37 | 12 |
| Reddit negative ratio | 29.7% | 41.7% |
| Reddit positive ratio | 10.8% | 0.0% |
| Company action | Supported | Pulled back |
| Business outcome | Positive (9.3M subs) | Unknown |

GroupWatch was quietly removed in September 2023 based on a help-center notice. No public earnings mention, no stated audience impact, no revenue attribution. Whether the feature had value for a niche audience remains unknown.

Source: [Disney+ help-center notice via ComicBook](https://comicbook.com/irl/news/disney-plus-groupwatch-feature-no-longer-available/) (Tier 2 evidence).

Both features show steep decay. Both drew negative attention. The product paths diverged completely. That is the problem this repo studies.

## Where This Fits In The Broader Analysis

Across the full dataset of 36 subscription features, 69% of the features companies continued to support still show more than 80% search decay (95% CI: 44%–86%, n=16). Netflix Password Sharing is one of them — and it's the single clearest case where heavy decay coincided with strong business results.

The decision framework correctly classifies Netflix Password Sharing as supported (true positive). It also correctly classifies GroupWatch as pulled back (true negative). The framework's two misses are Games and App-Only Membership — both false positives (predicted supported, actually pulled back).

## What Internal Data Would Have Changed The Analysis

If I had been the analyst on the Netflix decision, I would have asked for:

- **Conversion rate**: what percentage of previously-sharing households converted to paid accounts or Extra Member
- **Churn by segment**: did cancellations spike among sharers, and did they return within 90 days
- **Revenue per user change**: net ARPU impact after accounting for lost sharers and gained subscribers
- **Retention cohort**: 30/60/90-day retention of newly-converted accounts vs organically acquired ones
- **Cost of enforcement**: engineering, support tickets, and brand-perception cost of the crackdown

None of that is observable from outside. All of it is necessary before recommending rollback.

## The Transferable Lesson

This is not a story about Netflix being right or Disney being wrong. It is a story about what public data can and cannot tell you.

Public signals resolve faster than product value does. Search interest decays in weeks. Retention impact takes quarters to measure. A feature can look dead in Google Trends while silently driving the best subscriber quarter in a company's recent history.

The practical rule: external concern without internal evidence should trigger investigation, not rollback.

## Data Sources

- Search decay: Google Trends data collected via `pytrends`, peak-based methodology (`src/data_collection/recalculate_with_peaks.py`)
- Reddit sentiment: keyword-based lexicon applied to company subreddit mentions (`src/data_collection/reddit/reddit_validator.py`). See the [sentiment methodology note](../README.md#sentiment-methodology) for why the crude method is a deliberate choice.
- Business outcome: Netflix Q1 2024 shareholder letter ([source](https://ir.netflix.net/financials/quarterly-earnings/default.aspx))
- GroupWatch removal: Disney+ help-center notice ([source](https://comicbook.com/irl/news/disney-plus-groupwatch-feature-no-longer-available/))
- Framework validation: `src/analysis/statistical_analysis.py` → `framework_error_analysis()`
