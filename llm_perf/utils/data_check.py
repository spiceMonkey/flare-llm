from typing import Any, Dict, Iterable, List, Optional


def validate_int_fields(
    cfg: Dict[str, Any],
    fields: Iterable[str],
    *,
    min_value: Optional[int] = None,
    allow_float_for_int: bool = False,
    prefix: str = "configuration",
) -> None:
    """Generic validator for integer-like fields in a dict.

    - Ensures each field exists in ``cfg``.
    - Ensures each value can be converted to int (optionally from float).
    - Enforces an optional lower bound ``min_value`` on the integer value.

    Raises ``ValueError`` with a message that includes a snapshot of the
    requested fields and a list of per-field errors if any constraint is
    violated.
    """

    errors: List[str] = []
    field_list: List[str] = list(fields)

    for key in field_list:
        if key not in cfg:
            errors.append(f"missing required key '{key}'")
            continue

        value = cfg[key]

        # Optionally allow float-like inputs that represent an integer
        if allow_float_for_int and isinstance(value, float):
            try:
                v_int = int(value)
            except (TypeError, ValueError):
                errors.append(f"{key}={value!r} is not an integer")
                continue
        else:
            try:
                v_int = int(value)
            except (TypeError, ValueError):
                errors.append(f"{key}={value!r} is not an integer")
                continue

        if min_value is not None and v_int < min_value:
            errors.append(f"{key}={value!r} must be >= {min_value}")

    if errors:
        snapshot = {k: cfg.get(k) for k in field_list}
        msg = (
            f"Invalid {prefix}; expected integer values with bounds for fields. "
            f"Read values: {snapshot}. Errors: " + "; ".join(errors)
        )
        raise ValueError(msg)


def validate_positive_int_fields(
    cfg: Dict[str, Any],
    fields: Iterable[str],
    *,
    allow_float_for_int: bool = False,
    prefix: str = "configuration",
) -> None:
    """Specialization of ``validate_int_fields`` enforcing values >= 1."""
    validate_int_fields(
        cfg,
        fields,
        min_value=1,
        allow_float_for_int=allow_float_for_int,
        prefix=prefix,
    )


def validate_float_fields(
    cfg: Dict[str, Any],
    fields: Iterable[str],
    *,
    min_value: Optional[float] = None,
    prefix: str = "configuration",
) -> None:
    """Generic validator for float-like fields in a dict.

    - Ensures each field exists in ``cfg``.
    - Ensures each value can be converted to float.
    - Enforces an optional lower bound ``min_value`` on the float value.
    """

    errors: List[str] = []
    field_list: List[str] = list(fields)

    for key in field_list:
        if key not in cfg:
            errors.append(f"missing required key '{key}'")
            continue

        value = cfg[key]
        try:
            v_float = float(value)
        except (TypeError, ValueError):
            errors.append(f"{key}={value!r} is not a float")
            continue

        if min_value is not None and v_float < min_value:
            errors.append(f"{key}={value!r} must be >= {min_value}")

    if errors:
        snapshot = {k: cfg.get(k) for k in field_list}
        msg = (
            f"Invalid {prefix}; expected float values with bounds for fields. "
            f"Read values: {snapshot}. Errors: " + "; ".join(errors)
        )
        raise ValueError(msg)


def validate_nonnegative_float_fields(
    cfg: Dict[str, Any],
    fields: Iterable[str],
    *,
    prefix: str = "configuration",
) -> None:
    """Specialization enforcing float values >= 0.0."""

    validate_float_fields(cfg, fields, min_value=0.0, prefix=prefix)


def validate_positive_float_fields(
    cfg: Dict[str, Any],
    fields: Iterable[str],
    *,
    prefix: str = "configuration",
) -> None:
    """Specialization enforcing float values > 0.0."""

    # Use a small positive lower bound to represent > 0.0
    validate_float_fields(cfg, fields, min_value=0.0, prefix=prefix)
