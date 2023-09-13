def int_div(a: int, *b: int) -> int:
    """Divide a by b. Return an integer."""
    try:
        y = a
        for x in b:
            y = y / x
    except Exception as exc:
        raise ValueError(f"Inputs: a={a}, b={b}") from exc
    return int(y)


def int_mul(a: int, *b: int) -> int:
    """Multiply a by b. Return an integer."""
    y = a
    for x in b:
        y *= x
    return int(y)


def add_int(a: int, *b: int) -> int:
    """Add a and b. Return an integer."""
    y = a
    for x in b:
        y += x
    return int(y)


def int_max(a: int, *b: int) -> int:
    """Return the maximum of a and b. Return an integer."""
    y = a
    for x in b:
        y = max(x, y)
    return int(y)
