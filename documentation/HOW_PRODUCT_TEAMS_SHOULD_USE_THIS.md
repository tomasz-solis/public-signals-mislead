# How A Product Team Should Use This Repo

This repo is useful when a team is feeling pressure to interpret public reaction quickly.

Typical trigger moments:

- search interest fell sharply after launch
- Reddit, X, or YouTube comments turned loud and negative
- a feature looks like it has "lost momentum"
- leaders want to know whether the team should pull back

The repo is not built to answer "should we roll this back?" on its own.

It is built to answer a narrower and more useful question:

Are we overreading external signals that do not actually tell us whether the feature is worth keeping?

## How To Use It Well

### 1. Use it as a challenge to the obvious story

If the room is converging too quickly on:

- "interest collapsed, so the feature is dead"
- "people complained, so it was a mistake"

this repo is the reason to slow down.

The analysis shows that steep search decay is common even for features companies keep supporting. That alone should break the shortcut from public reaction to product verdict.

### 2. Separate three questions

Do not let these collapse into one:

- What are people saying in public?
- What did the company visibly do next?
- What value did the feature actually create?

Those are different questions. This repo only observes the first two reliably.

### 3. Use public signals to target investigation

Public signals can still be useful. They help identify:

- confusion
- unmet expectations
- positioning issues
- onboarding friction
- reputation risk

That means they are good for asking better follow-up questions. They are weak as standalone evidence for a rollback recommendation.

### 4. Ask for internal evidence before recommending action

If the decision on the table is "remove, scale back, or stop investing," external evidence is not enough. Pull internal adoption, retention, monetization, and cost signals first.

See: [What internal data I would require before recommending rollback](INTERNAL_DATA_FOR_ROLLBACK.md)

## What A Strong Product Discussion Sounds Like

Weak version:

- "Search interest dropped 90%, so users have moved on."

Stronger version:

- "Search interest dropped 90%, but this repo shows that pattern is common even among supported features. Before we call for rollback, I want adoption, repeat usage, retention, and cost data."

Weak version:

- "Reddit hates it, so we should undo the launch."

Stronger version:

- "Reddit reaction tells us something about perception, but not enough about value. Let's separate PR risk, onboarding friction, and actual product performance."

## What This Repo Helps A Team Avoid

- rolling back a feature because public attention normalized after launch
- treating loud but narrow backlash as representative of total user value
- confusing company action with proof of business value
- filling missing internal evidence with assumptions

## Recommended Companion Artifact

For an operating version that can be shared with PMs, directors, or leadership, use:

- [One-page decision framework](DECISION_FRAMEWORK_ONE_PAGER.md)
