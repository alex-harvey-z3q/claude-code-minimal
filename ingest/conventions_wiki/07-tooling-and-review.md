# 07. Tooling and Review Expectations

## Standard tooling

Repositories should adopt a consistent baseline, typically:

- Black
- Ruff
- mypy or pyright
- pytest
- pre-commit

## Pre-commit hooks

Common hooks should include:

- formatting
- linting
- import cleanup or sorting
- trailing whitespace cleanup
- end-of-file fixes

## CI expectations

At minimum, pull requests should pass:

- formatting checks
- lint checks
- type checks where enabled
- relevant tests

## Code review standards

### Review for maintainability

A review should consider:

- clarity of names and structure
- correctness and edge cases
- test coverage at the right level
- operational concerns such as logs and metrics
- backwards compatibility where relevant

### Prefer actionable feedback

Good review comments are specific and solution-oriented.

Examples:

- “Can we split validation from persistence here to make each path easier to test?”
- “This exception loses the original cause. Please chain the underlying exception.”

### Separate preference from policy

State whether feedback is:

- required by the style guide or repository conventions
- a strong recommendation
- a minor preference

## Pull request guidance

Pull requests should be:

- focused in scope
- clearly described
- small enough to review effectively

Large refactors should be broken into incremental steps where practical.
