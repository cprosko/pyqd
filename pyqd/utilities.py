"""Helper functions for the pyqd package."""


import numpy as np


def is_iterable(obj):
    """Returns whether obj is iterable or not."""
    try:
        _ = iter(obj)
        return True
    except TypeError:
        return False


def ensure_iterable(*args):
    """Get iterable version of arguments as numpy.ndarrays"""
    return tuple(np.array([arg]).flatten() for arg in args)


def binom(n, k):
    if not 0 <= k <= n:
        return 0
    b = 1
    for t in range(min(k, n - k)):
        b *= n
        b /= t + 1
        n -= 1
    return int(b)
