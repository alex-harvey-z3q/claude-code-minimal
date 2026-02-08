# 02. Formatting and Structure

## Formatting baseline

We follow these defaults unless a repository specifies otherwise:

- **Black** for formatting
- **Ruff** for linting and import organization
- 88 character line length
- 4 spaces per indentation level
- UTF-8 encoding

## General formatting rules

### Use trailing commas in multiline literals

This improves diffs and keeps formatter behavior stable.

```python
items = [
    "alpha",
    "beta",
    "gamma",
]
```

### Prefer one statement per line

Do not compress multiple operations onto one line.

```python
# Preferred
count += 1
log_count(count)
```

```python
# Avoid
count += 1; log_count(count)
```

### Use blank lines intentionally

- 2 blank lines between top-level declarations
- 1 blank line between methods when needed for readability
- Separate logical sections within long functions

## Imports

### Import order

Order imports in this sequence:

1. Standard library
2. Third-party packages
3. First-party/internal packages

```python
import json
from pathlib import Path

import requests

from northstar.auth.tokens import TokenService
```

### Import rules

- Prefer absolute imports
- Avoid wildcard imports
- Import modules or named symbols consistently within a file
- Remove unused imports

```python
# Preferred
from northstar.payments.client import PaymentsClient
```

```python
# Avoid
from northstar.payments.client import *
```

## File organization

### Keep modules cohesive

Each module should have a clear responsibility. If a file contains unrelated concerns, split it.

### Suggested top-level order in a module

1. module docstring if useful
2. imports
3. constants
4. type aliases / protocol definitions
5. exceptions
6. classes
7. public functions
8. private helpers

### Avoid oversized modules

As a rule of thumb, when a file becomes hard to navigate or understand in one screenful-at-a-time reading session, consider splitting it.

## Script entry points

CLI or executable modules should use a clear entry point.

```python
def main() -> int:
    print("Hello")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```
