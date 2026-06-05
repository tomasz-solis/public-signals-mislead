# Decision Framework: When Public Signals Turn Negative

Use this when a feature looks weak in search, social, or public discussion and the team is tempted to call for rollback.

![Static preview of the decision framework](assets/decision_matrix_preview.svg)

## Classify The External Signal

| Signal pattern | What it may mean | What it does **not** prove |
|----------------|------------------|-----------------------------|
| Search spike, then steep decay | Launch attention normalized | The feature failed |
| High complaint volume | Confusion, friction, or reputational risk | Low value across all users |
| Low public discussion | Quiet adoption or low cultural salience | Low product value |
| Public removal discourse | Perceived dissatisfaction | Correctness of rollback |

## Ask The Three Questions In Order

1. What is observable externally?
2. What did the company visibly do next?
3. What internal evidence do we have about value?

Do not answer question three with question one.

## Match The Decision To The Evidence

| Evidence state | Recommended posture |
|----------------|---------------------|
| External concern, no internal data yet | Investigate. Do not recommend rollback yet. |
| External concern, strong internal adoption or retention | Fix UX, messaging, or targeting before considering rollback. |
| External concern, weak internal usage, weak retention, high cost | Rollback becomes plausible. |
| External calm, strong internal value | Keep supporting. Public quiet is not a failure signal. |
| Mixed external signals, mixed internal value | Narrow the feature, retarget it, or reduce investment instead of all-or-nothing thinking. |

## Require These Internal Inputs Before Rollback

- adoption by eligible users
- repeat usage after first exposure
- retention or churn effect
- monetization impact where relevant
- cost and maintenance burden
- segment-level value, especially for strategic cohorts

## Questions A PM Or Director Should Ask

- Are we reacting to public perception or to measured product value?
- Which user segments would lose something meaningful if we remove this?
- Are we seeing a feature problem, a messaging problem, or a discoverability problem?
- What internal metric would have to be true for rollback to be justified?
- If we remove this, what belief are we acting on, and what evidence supports it?

## Bottom Line

Public signals are useful as alerts.

They are weak as verdicts.

Use them to decide what to investigate next, not what to kill.
