# 08. Examples

## Example: unclear vs clear naming

### Avoid

```python
def proc(d, f):
    if f:
        return [x for x in d if x.ok]
    return d
```

### Preferred

```python
def filter_active_records(records: list[Record], only_active: bool) -> list[Record]:
    if not only_active:
        return records

    return [record for record in records if record.is_active]
```

## Example: nested logic vs guard clauses

### Avoid

```python
def get_shipping_country(order: Order) -> str | None:
    if order.customer is not None:
        if order.customer.address is not None:
            return order.customer.address.country
    return None
```

### Preferred

```python
def get_shipping_country(order: Order) -> str | None:
    if order.customer is None:
        return None

    if order.customer.address is None:
        return None

    return order.customer.address.country
```

## Example: broad exception swallowing

### Avoid

```python
def refresh_cache() -> None:
    try:
        cache.populate()
    except Exception:
        logger.warning("Cache refresh failed")
```

### Preferred

```python
def refresh_cache() -> None:
    try:
        cache.populate()
    except CacheConnectionError as exc:
        logger.warning("Cache refresh failed", exc_info=exc)
        raise
```

## Example: docstring that adds value

```python
def reserve_inventory(order_id: str) -> None:
    """Reserve inventory for an order.

    This operation is not idempotent across warehouse providers.
    Callers must ensure duplicate reservations cannot occur.
    """
```

## Example: typed boundary object

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CreateUserInput:
    email: str
    display_name: str
    send_welcome_email: bool
```
