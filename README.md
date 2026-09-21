# best-of-n-coder

**Sample more, not bigger.**

A small experiment in inference-time scaling for code generation, built as part of a hands-on run through Stanford's CS 329 self-improving agents course.

## What this does

For each of 10 coding problems, the script samples N candidate solutions from Claude (Sonnet) at N = 1, 3, and 10, and verifies each candidate against hand-written unit tests using Python's `subprocess` in an isolated process. This tests the "large language monkeys" idea from the course: that inference-time sampling, not bigger models, can be a lever for better performance on a fixed model.

Problems span a range of difficulty, from simple ones (palindrome check, FizzBuzz, prime check) to classic algorithmic problems with real edge cases (balanced parentheses, Roman numeral parsing, interval merging, word break, longest common prefix).

## Result

```
N=1:  10/10 solved (100%)
N=3:  10/10 solved (100%)
N=10: 10/10 solved (100%)
```

The chart is flat, best-of-N sampling made no measurable difference here.

## What this actually tells us

This isn't a null result, it's a real finding about where inference-time sampling helps and where it doesn't. Claude Sonnet solved every problem correctly on the first attempt, including the harder ones with real edge cases (subtractive Roman numeral notation, overlapping interval merges, word segmentation with reused tokens). For well-known, textbook-style coding problems that are well-represented in a frontier model's training distribution, one sample is already enough. There's no accuracy gap for best-of-N to close.

This matches the intuition behind the original "large language monkeys" research: repeated sampling helps most when a model's single-shot success rate is meaningfully below 100%, i.e., on problems that are genuinely hard, ambiguous, or out of distribution for the model. Reproducing that gap would need either much harder problems (novel algorithmic puzzles, multi-step reasoning tasks, or real-world bugs in unfamiliar codebases) or an artificial constraint like a tight token budget that forces occasional incomplete outputs.

## What's next

- Test on harder or more novel problems (e.g. from competitive programming, or actual GitHub issues) where first-attempt accuracy is likely to be below 100%
- Add a self-correction loop: on failure, feed the test error back to the model and retry, instead of relying purely on independent resampling
- Try majority-vote selection across candidates instead of first-pass-wins, for problems with partial credit or multiple valid approaches

## Tools

- Python 3
- Anthropic API (Claude Sonnet)
- Python stdlib `subprocess` + `tempfile` for sandboxed verification
- `matplotlib` for the accuracy-vs-N chart

## Background

Built from Lecture 1 of Stanford CS 329's Self-Improving Agents course, covering test-time compute scaling, verifiers, and the shift from bigger pretrained models toward inference-time and RL-driven improvement.