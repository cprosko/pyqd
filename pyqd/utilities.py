"""Helper functions for the pyqd package."""


def is_iterable(obj):
    """Returns whether obj is iterable or not."""
    try:
        _ = iter(obj)
        return True
    except TypeError:
        return False


def binom(n, k):
    if not 0 <= k <= n:
        return 0
    b = 1
    for t in range(min(k, n - k)):
        b *= n
        b /= t + 1
        n -= 1
    return int(b)