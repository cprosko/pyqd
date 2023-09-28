def partitions(n: int, min_summand: int = 1):
    """Generator for tuples of summands totaling integer n.

    Arguments:
    ----------
    n (int): Integer for which tuples of summands summing to n will be yielded.

    Keyword Arguments:
    ------------------
    min_summand (int): Minimum summand to be considered. Must be larger than 0.
    """
    if min_summand <= 0:
        raise Exception(f"min_summand ({min_summand}) must be greater than zero!")
    if min_summand > n:
        raise Exception(f"min_summand ({min_summand}) must be smaller than n ({n})!")
    if n <= 0:
        raise Exception(f"n ({n}) must be larger than 0!")
    yield (n,)
    for i in range(min_summand, n // 2 + 1):
        for p in partitions(n - i, i):
            yield (i,) + p
