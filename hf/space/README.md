---
title: "Aawden: does the model read the code, or the reviewer?"
emoji: "\U0001F6E1"
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: cc-by-nc-4.0
base_model: CohereLabs/aya-expanse-8b
short_description: "Real cases: a wrong reviewer, and what the model answers"
---

# Try the test yourself

A security reviewer reads a file and says "this is fine." The model answers. Is it reading the code,
or agreeing with the reviewer?

Every file on this page is one the **unchanged** model already gets right with no reviewer present.
Then the reviewer states the opposite verdict. Four answers are shown:

- the **correct** answer, which comes from running the code;
- the model **alone**, with no reviewer;
- the model **with the reviewer line** — the case where it gives its answer away;
- the model **after our adapter** — the same file, same reviewer, same question.

Every card is a **recorded** answer from a saved run. Nothing on this page calls a model, so it loads
instantly and costs nothing to host. You can switch between four models with the buttons at the top.

## What you are looking at

The measure behind it is the **flip rate on solved cases**: of the files a model gets right on its
own, the share a wrong reviewer talks it out of. Unchanged Aya Expanse 8B gives up all 67
of its 67 correct answers on the plain set. With the adapter it gives up at most a handful
per reviewer wording, in every one of three training runs.

The page is one of the project's three artifacts:

- **the adapter** — `RishyanthReddy/aawden-aya8b-reviewer-adapter`
- **the benchmark and every final number** — `RishyanthReddy/aawden-reviewer-flip-bench`
- **the source, the protocols and the failures** — the GitHub repository linked in the card

## Not claimed

- The page does not run inference. It replays saved verdicts.
- It ships no model weights.
- The fixtures are small and templated, written by one author, and have not been independently
  reviewed. The base model is CC-BY-NC-4.0 and is not redistributed.
