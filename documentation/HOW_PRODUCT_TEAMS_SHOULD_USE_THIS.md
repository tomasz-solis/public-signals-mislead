# How a product team should use this repo

Use it when the team feels pressure to read public reaction quickly: search interest fell sharply after launch, Reddit, X or YouTube comments turned loud and negative, a feature seems to have "lost momentum", or leaders want to know whether to pull back.

It doesn't answer "should we roll this back?" on its own. It answers a narrower question: are we overreading outside signals that don't tell us whether the feature is worth keeping?

## How to use it

1. Challenge the obvious story. If the room is settling on "interest collapsed, so it's dead" or "people complained, so it was a mistake", this is the reason to slow down. Steep search decay is common even for features companies keep supporting.
2. Keep three questions apart: what people say in public, what the company visibly did next, and what value the feature created. This repo only observes the first two reliably.
3. Use public signals to aim the investigation. They are good at pointing to confusion, unmet expectations, positioning problems, onboarding friction and reputation risk. They are weak evidence for a rollback on their own.
4. Get internal evidence before recommending action. If the decision is to remove, scale back or stop investing, pull internal adoption, retention, monetisation and cost first. See [What internal data I'd need before recommending a rollback](INTERNAL_DATA_FOR_ROLLBACK.md).

## Weak vs strong arguments

| Weak | Stronger |
|---|---|
| "Search interest dropped 90%, so users have moved on." | "Search interest dropped 90%, but that pattern is common even for supported features. Before calling for a rollback, I want adoption, repeat usage, retention and cost data." |
| "Reddit hates it, so we should undo the launch." | "Reddit tells us about perception, not value. Let's separate PR risk, onboarding friction and actual product performance." |

## What it helps a team avoid

- Rolling back a feature because public attention settled after launch.
- Treating loud but narrow backlash as representative of all users.
- Mistaking company action for proof of business value.
- Filling missing internal evidence with assumptions.

To share an operating version with PMs, directors or leadership, use the [one-page decision framework](DECISION_FRAMEWORK_ONE_PAGER.md).
