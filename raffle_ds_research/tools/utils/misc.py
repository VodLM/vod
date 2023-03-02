from __future__ import annotations


def int_div(a, *b):
    try:
        y = a
        for x in b:
            y = y / x
    except Exception as exc:
        raise ValueError(f"Inputs: a={a}, b={b}") from exc
    return int(y)


def int_mul(a, *b):
    y = a
    for x in b:
        y *= x
    return int(y)


def int_max(a, *b):
    y = a
    for x in b:
        y = max(x, y)
    return int(y)
