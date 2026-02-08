# 06. Errors, Logging, and Testing

## Error handling

### Raise specific exceptions

Use precise exception types whenever possible.

```python
if timeout_seconds <= 0:
    raise ValueError("timeout_seconds must be positive")
```

### Preserve context

When re-raising, chain exceptions so the original cause is visible.

```python
try:
    payload = json.loads(raw_payload)
except json.JSONDecodeError as exc:
    raise InvalidPayloadError("Could not parse partner payload") from exc
```

### Do not swallow exceptions silently

At minimum, add context through logging or explicit handling.

```python
# Avoid
try:
    sync_data()
except Exception:
    pass
```

### Catch broadly only at boundaries

Broad exception handling may be acceptable at:

- process boundaries
- worker loop boundaries
- API framework boundaries
- CLI entry points

Even there, log useful context and fail predictably.

## Logging

### Use structured, actionable logs

Logs should help answer:

- what happened
- where it happened
- what identifiers are relevant
- whether the system recovered

```python
logger.info(
    "Invoice sent",
    extra={"invoice_id": invoice.id, "user_id": invoice.user_id},
)
```

### Never log secrets

Do not log:

- access tokens
- passwords
- API keys
- full payment details
- sensitive personal data unless explicitly approved

### Log level guidance

- `debug`: diagnostic details for development
- `info`: important expected system events
- `warning`: unusual but recoverable situations
- `error`: failures affecting a specific operation
- `critical`: severe failures threatening service health

## Testing

### Test behavior, not implementation details

Prefer tests that describe expected outcomes over tests tightly coupled to internal structure.

### Keep tests readable

Use descriptive test names.

```python
def test_generate_invoice_returns_unpaid_invoice_for_new_account() -> None:
    ...
```

### Arrange, Act, Assert

Structure tests clearly.

### Avoid over-mocking

Mock boundaries to external systems, not every internal collaborator.

### Test pyramid guidance

Aim for a healthy mix of:

- unit tests for business logic
- integration tests for component interaction
- end-to-end tests for critical user flows only
