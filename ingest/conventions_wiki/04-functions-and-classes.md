# 04. Functions and Classes

## Functions

### Keep functions focused

A function should do one thing at one level of abstraction.

Signs a function should be split:

- it mixes validation, business logic, persistence, and formatting
- it has multiple unrelated branches
- it requires extensive comments to explain the sequence

### Prefer explicit return types

```python
def find_user(user_id: str) -> User | None:
    ...
```

### Limit parameter count

Prefer a small number of parameters. When many values naturally travel together, introduce a typed object.

```python
@dataclass(frozen=True)
class EmailRequest:
    to: str
    subject: str
    body: str
```

### Avoid boolean flag parameters when possible

Boolean flags often indicate a function is doing more than one job.

```python
# Avoid
render_report(data, include_metadata=True)

# Better
render_report(data)
render_report_with_metadata(data)
```

### Use guard clauses

Prefer early returns over deeply nested conditionals.

```python
def get_primary_email(user: User | None) -> str | None:
    if user is None:
        return None

    if not user.emails:
        return None

    return user.emails[0]
```

## Classes

### Use classes when state and behavior belong together

Do not create classes just to group functions.

### Prefer composition over inheritance

Inheritance should model a real subtype relationship. Most code is clearer with composition.

### Keep constructors lightweight

`__init__` should assign fields, validate basic invariants, and avoid heavy I/O.

### Class size guidance

A class with too many public methods often has too many responsibilities.

## Dataclasses

Use `@dataclass` for simple data containers.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Money:
    currency: str
    amount_cents: int
```

Use `frozen=True` when immutability is desirable.

## Properties

Use `@property` sparingly. Prefer methods when computation is non-trivial, slow, or has side effects.

## Module-level design

Prefer plain functions for stateless transformations and lightweight orchestration. Introduce a class only when it improves clarity or encapsulates meaningful state.
