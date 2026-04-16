# Project Impact 

## Summary

This repo is strongest when it is framed as a product-decision project, not a prediction project.

The counter-intuitive finding is simple:

- public buzz fades fast
- public commentary is noisy
- teams can still be looking at a feature with real user or business value

That makes the repo memorable. It is not another "I trained a model" portfolio piece. It is an argument about judgment: external signals look decisive, but without internal context they are poor grounds for product decisions.

## What The Repo Actually Shows

The careful claim is:

- public signals are weak standalone inputs for product decisions
- company action is often observable from the outside
- business outcome is often not
- a rollback is visible, but that does not prove the feature never had value

The repo does **not** prove that product teams definitely made bad decisions. It shows why they can be misled if they lean too hard on outside signals without adoption, retention, revenue, or strategic context.

That distinction matters. It keeps the thesis sharp instead of overstated.

## Why This Is Portfolio-Worthy

### 1. It has a real point of view

Most projects show method. This one shows judgment.

The thesis is counter-intuitive but concrete: heavy search decay is common even among features a company continues to support.

### 2. It solves a real product problem

Product teams regularly face some version of:

- buzz collapsed, should we panic?
- users are loud, should we roll it back?
- the trend line looks dead, does that mean the feature is dead?

This repo gives a better response: public signals can start an investigation, but they should not end it.

### 3. It separates observable facts from inferred truth

That is the most mature part of the project.

- `company_action` captures what the company appears to do
- `business_outcome` captures what the public record can actually prove
- `UNKNOWN` is treated as a real state, not something to hand-wave away

That is much stronger than forcing every feature into a fake success/failure label.

## Quantified Results

### Dataset Scale

- 36 subscription features analyzed across major streaming and subscription platforms
- 20 features with public decision context
- 19 features with an action label usable in the main comparison
- 9 features with a known business outcome

### Main Statistical Result

<<<<<<< HEAD
Multi-signal validation achieves reasonable accuracy:
- High decay + positive sentiment + high mentions = ADOPTION
- High decay + negative sentiment = ABANDONMENT
=======
- Supported features: `83.7%` average search decay
- Pulled-back features: `92.1%` average search decay
- Mann-Whitney U p-value: `0.284`
- `69%` of supported features still show more than `80%` decay

### Honest Limitation

- the pulled-back group is tiny (`n=3`)
- power analysis shows the study can only detect very large effects
- the correct conclusion is caution, not certainty

That limitation does not weaken the repo. It strengthens the credibility of the framing.

## Business Value

### For Product Teams

Before:

- "Search interest dropped 90%, so the feature must be dying."
- "Reddit is angry, so rollback is probably the right call."

After:

- "Search decay is common even for supported features."
- "Public reaction is one input, not the decision itself."
- "We need internal usage, retention, and strategic context before acting."

### For Data Teams

The reusable idea is not the exact dataset. It is the framing discipline:

- separate signal from outcome
- separate observable action from hidden value
- treat missing truth as unknown, not as implied failure
- report what the sample can actually support

## What This Demonstrates

### Decision Science

- framed the problem around decision quality, not just metric movement
- challenged a tempting but weak causal story
- built a decision matrix for handling ambiguous external signals
- kept the conclusion narrow when the data could not support more

### Statistical Analysis

- used Mann-Whitney U for the small-sample primary test
- kept Welch's t-test and effect sizes for context
- added bootstrap confidence intervals and power analysis
- used Spearman correlation for bounded, non-normal features
- surfaced multiple-testing limits instead of hiding them

### Data Engineering

- packaged the project with `pyproject.toml`
- added focused tests around the analysis layer
- validated schemas between pipeline steps
- improved operational hygiene with `.gitignore`, `.env.example`, and structured logging

### Product Thinking

- recognized that "removed" and "failed" are not the same thing
- preserved ambiguous cases like `Watch Party` instead of forcing false certainty
- treated public commentary as noisy evidence, not as product truth

## What Makes The Repo Stand Out

### The finding is surprising

People expect a clean answer like "high decay means failure." The repo shows why that story collapses under scrutiny.

### The methodology is honest

It does not pretend the public record contains the whole truth. That restraint is part of the value.

### The output is usable

This is not just analysis for analysis's sake. The repo turns the finding into guidance a product team could actually use.

## Interview Talking Points

### "Walk me through this project"

Use this version:

"I analyzed 36 subscription features to see whether public signals like search decay and Reddit reaction were useful as standalone product inputs. They weren't. Supported features often looked just as 'dead' in public as pulled-back ones. So I reframed the project around a stricter distinction: company action is often observable, true business value usually is not. The outcome is a cautionary decision-support analysis, not a prediction model."

### "What assumption did you challenge?"

"The easy assumption was that high search decay means failure. The data did not support that. Netflix Password Sharing and Disney+ GroupWatch can both look bad in public-signal terms while representing very different product realities."

### "What did you learn?"

"The hardest part was not the code. It was refusing to turn weak public evidence into fake certainty. That is why the repo now separates `company_action` from `business_outcome` and keeps `UNKNOWN` when the public record is too thin."

## Staff-Level Positioning

The strongest way to position this repo is not:

- "I predicted success and failure from public data"
- "I proved teams made bad decisions"

It is:

- "I used ambiguous external data to build a decision-quality framework."
- "I focused on what is observable, what is unknown, and where teams overread noisy signals."
- "I treated the contribution as decision framing under incomplete information, not as a prediction exercise."

That positioning is more senior because it shows restraint, operating judgment, and awareness of evidence limits.

## Portfolio Blurb

Use this when you need a short version:

> Built a decision-science case study showing that public signals like search decay and online chatter are weak standalone inputs for product decisions. The project analyzes 36 subscription features, separates observable `company_action` from mostly hidden `business_outcome`, and turns noisy external data into a practical framework for when teams should investigate further instead of rushing to rollback.

Shorter version:

> Decision-support analysis of 36 subscription features showing why public signals can mislead product teams when used without internal context.

## Interview Narrative

Use this structure in interviews.

### 1. The setup

"I wanted to study a mistake product teams can make from the outside: treating public signals like Google Trends or Reddit reaction as if they were direct evidence of product value."

### 2. The tension

"The problem is that public reaction is observable, but real business value usually is not. So I restructured the repo around that distinction instead of forcing everything into a fake success/failure label."

### 3. The analytical move

"I separated `company_action` from `business_outcome`, kept ambiguous cases as `UNKNOWN`, and tested whether public signals actually distinguish supported from pulled-back features. They did not do that cleanly."

### 4. The product takeaway

"The takeaway is not 'ignore external data.' It is 'use external data as a prompt for investigation, not as a product verdict.'"

### 5. Why it matters

"That is the kind of decision-science work I want to do more of: helping teams reason better under incomplete information instead of pretending noisy signals are stronger than they are."

## LinkedIn Post Hooks

Use stronger language in the hook, then keep the body precise.

**Option 1**
"Public signals mislead product teams when used without internal context. I analyzed 36 subscription features and found that 69% of supported features still showed more than 80% search decay."

**Option 2**
"A feature can look dead in Google Trends and still be worth keeping. I built a repo around one product question: what can public signals actually tell us, and where do they start to mislead?"

**Option 3**
"Disney+ GroupWatch and Netflix Password Sharing can both look bad from the outside. That is the problem. Public signals often move faster than product value gets resolved."

## Resume Bullets

- Built a decision-support analysis showing public signals are weak standalone inputs for product decisions, using Google Trends and Reddit data across 36 subscription features
- Designed a labeling framework that separates observable `company_action` from mostly hidden `business_outcome`, preventing false certainty in ambiguous feature cases
- Implemented a statistical workflow with Mann-Whitney U, bootstrap confidence intervals, power analysis, and sensitivity checks to test whether public signals distinguish supported from pulled-back features
- Created a reusable product-decision matrix that treats external buzz as a prompt for investigation rather than a verdict

## The Real Value

This repo shows five things that matter in senior product analytics work:

1. challenge the obvious story
2. test it with the right level of rigor
3. keep unknowns visible instead of hiding them
4. separate what is observed from what is inferred
5. turn analysis into a practical decision frame

That is what makes the project stronger than a generic model demo. It has a clear claim, a real limitation, and a point of view worth discussing.
>>>>>>> d291a23 (docs(portfolio): add architecture, walkthrough, and chart previews)
