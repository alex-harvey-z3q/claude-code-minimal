# 01. Core Principles

## Purpose

Our Python code should be easy to read, safe to change, and straightforward to operate. The best code is usually not the most clever code. It is the code that another engineer can understand and modify quickly with confidence.

## Principles

### Readability over cleverness

Prefer explicit, boring code over compressed or highly abstract code.

```python
# Preferred
if user is None:
    return None

return user.email
```

```python
# Avoid
return None if user is None else user.email
```

### Consistency over personal preference

A consistent codebase is easier to maintain than a collection of individually “clean” files with different styles.

### Simplicity over premature abstraction

Do not create layers, helper classes, or frameworks until there is a demonstrated need.

### Correctness before optimization

Get the behavior right first. Optimize only when there is measured evidence that performance matters.

### Small, focused units

Functions, classes, and modules should each have a narrow responsibility.

### Explicit boundaries

Be clear about:

- inputs and outputs
- side effects
- failure modes
- ownership of resources

## What good code looks like

Good Python code in our organization is:

- easy to scan
- well named
- typed where useful
- tested at the right level
- observable in production
- unsurprising in behavior

## What to avoid

Avoid code that is:

- too clever to maintain
- tightly coupled across modules
- dependent on hidden global state
- missing error context
- difficult to test in isolation
