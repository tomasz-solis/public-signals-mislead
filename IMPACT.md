# Project impact

When search interest or public commentary drops, how much should a product team read into it? Not much on its own. Public signals are useful early warnings but weak evidence of product value. In this sample, features that companies kept supporting often looked poor on public signals. That doesn't prove they succeeded; it means the outside record is too thin to treat search decay or online reaction as a verdict.

## Claim

The repo supports this:

- Public signals are noisy inputs for product decisions.
- Company action is often visible from outside.
- True business outcome often isn't.
- A rollback is visible, but it doesn't prove the feature never had value.
- A supported feature can still show steep decay in public attention.

It doesn't prove any team made the wrong call. That would need internal usage, retention, margin, strategy and operational data.

## Evidence

Dataset: 36 subscription features across major streaming and subscription platforms. 20 have public decision context, 19 have an action label for the main comparison, and 9 have a known business outcome.

| Result | Value |
|---|---|
| Supported features, average search decay | `83.7%` |
| Pulled-back features, average search decay | `92.1%` |
| Mann-Whitney U p-value | `0.284` |
| Supported features with more than 80% decay | `69%` |

The pulled-back group is small (`n=3`), so the study is underpowered for modest effects. Read it with caution.

## Why it matters

Product teams see some version of this: search interest collapsed, Reddit or press turned negative, the feature stopped being discussed, and leadership wants to know whether to keep investing. Those signals should start an investigation, not end it. A good decision brings in internal usage, cohort retention, revenue effect, support load, strategic fit and the cost of undoing the work.

## Analytical choices

The strongest part is the labelling discipline:

- `company_action` records what the company appears to do.
- `business_outcome` records what the public record can prove.
- `UNKNOWN` is a real state when evidence is thin.

That avoids forcing every feature into success or failure and keeps the statistical claim small and defensible.

## What it demonstrates

| Area | What the repo does |
|---|---|
| Decision analysis | Frames the question around decision quality, challenges a tempting causal story, separates observable action from hidden value, reports power limits openly |
| Statistics | Mann-Whitney U for the small-sample primary comparison, Welch's t-test and effect sizes as context, bootstrap confidence intervals, power analysis, Spearman correlation for bounded non-normal features |
| Product judgment | Treats public commentary as a weak signal, keeps ambiguous cases instead of over-labelling, turns the result into a rule: investigate before rolling back |

## How to describe it

Safest summary:

> A decision-support analysis of 36 subscription features showing why public signals such as search decay and online commentary should not be used as standalone evidence for product value.

The operating lesson:

> External signals are useful for triage. They are not enough for a product verdict without internal context.

Don't claim it predicts product success or proves teams decided badly. The evidence doesn't support that. The value is disciplined reasoning with incomplete information.
