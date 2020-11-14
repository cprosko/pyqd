"""Helper functions for the pyqd package."""

def is_iterable(obj):
    try:
        iterator = iter(obj)
        return True
    except TypeError:
        return False
