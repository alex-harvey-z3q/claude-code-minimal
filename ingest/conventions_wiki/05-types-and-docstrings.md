# 05. Type Hints and Docstrings

## Type hints

### Expectation

New and actively maintained Python code should include type hints for:

- public functions
- method signatures
- complex module-level constants where helpful
- dataclass and model fields

### Why we type code

Type hints improve:

- editor support
- refactoring safety
- readability of interfaces
- static analysis in CI

### Guidelines

- Use modern built-in generic syntax where supported: `list[str]`, `dict[str, int]`
- Prefer precise types over `Any`
- Use `Protocol`, `TypedDict`, or dataclasses for structured contracts when appropriate
- Use `X | None` instead of `Optional[X]` in modern Python code unless repository constraints say otherwise

```python
def parse_tags(raw_tags: list[str]) -> set[str]:
    return {tag.strip().lower() for tag in raw_tags}
```

### `Any` usage

Use `Any` only at boundaries where the type is truly unknown or dynamic, and contain it quickly.

## Docstrings

### When docstrings are required

Docstrings are expected for:

- public modules when context is useful
- public classes
- public functions whose purpose, arguments, or behavior is not obvious

Docstrings are usually not needed for short private helpers with self-explanatory names.

### Style

Use triple double quotes.

```python
def generate_invoice(user_id: str) -> Invoice:
    """Generate an invoice for the given user."""
```

### Content guidelines

Docstrings should explain:

- what the function or class does
- important constraints
- side effects
- non-obvious exceptions

Do not repeat information already obvious from the signature.

```python
def save_report(report: Report, destination: Path) -> None:
    """Persist a report to disk.

    Raises:
        OSError: If the report cannot be written.
    """
```

## Comments

### Use comments to explain why, not what

```python
# Preferred: explains intent
# We cap retries here to avoid duplicate partner charges.
```

```python
# Avoid: restates code
# Increment retry count.
retry_count += 1
```
