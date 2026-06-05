# Project Impact

This project is a decision-quality analysis of public product signals. It asks a
simple product question: when search interest or public commentary drops, how
much should a team infer from that?

The answer is deliberately narrow. Public signals can be useful early-warning
inputs, but they are weak standalone evidence for product value. In this sample,
features that companies continued to support often looked poor in public-signal
terms. That does not prove the features succeeded. It means the outside record is
too thin to treat search decay or online reaction as a verdict.

## Claim

The repo supports this claim:

- public signals are noisy inputs for product decisions
- company action is often observable from outside the company
- true business outcome is often not observable
- a rollback is visible, but it does not prove the feature never had value
- a supported feature can still have steep public attention decay

The repo does not prove that any product team made the wrong call. That would
require internal usage, retention, margin, strategic, and operational data. The
point is to separate what the public record can show from what it cannot.

## Evidence

Dataset:

- 36 subscription features across major streaming and subscription platforms
- 20 features with public decision context
- 19 features with an action label usable in the main comparison
- 9 features with a known business outcome

Main result:

- Supported features: `83.7%` average search decay
- Pulled-back features: `92.1%` average search decay
- Mann-Whitney U p-value: `0.284`
- `69%` of supported features still show more than `80%` decay

The pulled-back group is small (`n=3`), so the study is underpowered for modest
effects. The right reading is caution, not certainty.

## Why It Matters

Product teams often see a version of this problem:

- search interest collapsed
- Reddit or press reaction turned negative
- the feature stopped being discussed publicly
- leadership wants to know whether to keep investing

Those signals should start an investigation. They should not end it. A mature
decision would bring in internal usage, cohort retention, revenue effect,
support load, strategic fit, and the cost of undoing the work.

## Analytical Choices

The strongest part of the repo is the labeling discipline:

- `company_action` captures what the company appears to do
- `business_outcome` captures what the public record can actually prove
- `UNKNOWN` is kept as a real state when the evidence is thin

That avoids the common mistake of forcing every feature into a clean
success/failure label. It also keeps the statistical claim smaller and more
defensible.

## What This Demonstrates

Decision analysis:

- frames the question around decision quality, not model novelty
- challenges a tempting causal story
- separates observable action from hidden value
- reports the power limits instead of hiding them

Statistical work:

- Mann-Whitney U for the small-sample primary comparison
- Welch's t-test and effect sizes as secondary context
- bootstrap confidence intervals
- power analysis
- Spearman correlation for bounded, non-normal features

Product judgment:

- treats public commentary as a weak signal rather than product truth
- keeps ambiguous cases instead of over-labeling them
- converts the result into a decision rule: investigate before rollback

## How To Position The Repo

The safest summary is:

> A decision-support analysis of 36 subscription features showing why public
> signals such as search decay and online commentary should not be used as
> standalone evidence for product value.

The sharper operating lesson is:

> External signals are useful for triage. They are not enough for a product
> verdict without internal context.

Avoid claiming that the project predicts product success or proves teams made bad
decisions. The evidence does not support that. The value is the discipline of
reasoning under incomplete information.
