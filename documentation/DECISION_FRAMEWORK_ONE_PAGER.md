# Decision framework: when public signals turn negative

Use this when a feature looks weak in search, social or public discussion and the team is tempted to roll it back. Public signals are good alerts and weak verdicts: use them to decide what to investigate, not what to kill.

![Static preview of the decision framework](assets/decision_matrix_preview.svg)

## 1. Classify the outside signal

| Signal pattern | What it may mean | What it doesn't prove |
|----------------|------------------|-----------------------|
| Search spike, then steep decay | Launch attention settled | That the feature failed |
| High complaint volume | Confusion, friction or reputational risk | Low value for all users |
| Little public discussion | Quiet adoption or low cultural visibility | Low product value |
| Public talk about removal | Perceived dissatisfaction | That a rollback is right |

## 2. Ask three questions, in order

1. What is observable from outside?
2. What did the company visibly do next?
3. What internal evidence do we have about value?

Don't answer question 3 with question 1.

## 3. Match the decision to the evidence

| Evidence | Posture |
|----------------|---------------------|
| Outside concern, no internal data yet | Investigate. Don't recommend a rollback yet. |
| Outside concern, strong internal adoption or retention | Fix UX, messaging or targeting before considering a rollback. |
| Outside concern, weak usage, weak retention, high cost | A rollback becomes plausible. |
| Outside calm, strong internal value | Keep supporting. Public quiet isn't failure. |
| Mixed outside signals, mixed internal value | Narrow the feature, retarget it or cut investment, rather than all or nothing. |

## 4. Internal inputs required before a rollback

- Adoption by eligible users.
- Repeat usage after first exposure.
- Retention or churn effect.
- Monetisation impact, where relevant.
- Cost and maintenance burden.
- Value by segment, especially strategic cohorts.

## Questions a PM or director should ask

- Are we reacting to public perception or to measured product value?
- Which user segments would lose something meaningful if we remove this?
- Is this a feature problem, a messaging problem or a discoverability problem?
- What internal metric would have to be true to justify a rollback?
- If we remove it, what belief are we acting on, and what supports it?
