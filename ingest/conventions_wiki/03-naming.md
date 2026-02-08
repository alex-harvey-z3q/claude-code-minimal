# 03. Naming Conventions

## General rule

Names should reveal intent. A reader should be able to guess what a symbol represents without tracing its whole implementation.

## Modules and packages

- Use short, lowercase names
- Separate words with underscores when needed
- Avoid vague names like `utils`, `helpers`, or `common` unless the scope is truly narrow and well understood

Examples:

- `invoice_service.py`
- `retry_policy.py`
- `feature_flags/`

## Variables

- Use descriptive nouns
- Prefer domain language over technical shorthand
- Avoid single-letter names except in tight, obvious scopes

```python
invoice_total = calculate_total(invoice)
for index, item in enumerate(items):
    ...
```

## Functions and methods

Use verb phrases for actions.

Examples:

- `load_config`
- `send_invoice`
- `is_expired`
- `build_response`

Boolean-returning functions should read naturally as predicates.

Examples:

- `is_active`
- `has_access`
- `should_retry`

## Classes

Use PascalCase nouns.

Examples:

- `InvoiceService`
- `RetryPolicy`
- `UserRepository`

## Constants

Use uppercase with underscores for true constants.

```python
DEFAULT_TIMEOUT_SECONDS = 30
MAX_RETRY_ATTEMPTS = 5
```

Do not use all-caps for values that are configuration-like but frequently overridden or environment-specific.

## Private names

Use a single leading underscore for internal helpers.

```python
def _parse_timestamp(value: str) -> datetime:
    ...
```

Double underscore name mangling is rarely necessary and should generally be avoided.

## Abbreviations

Avoid abbreviations unless they are widely understood in the team or domain.

Prefer:

- `configuration` only when the longer name improves clarity, otherwise `config`
- `identifier` or `id` depending on local convention
- `request_count` instead of `req_cnt`
