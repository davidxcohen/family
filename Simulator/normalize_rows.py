def normalize_rows(x: numpy.ndarray):
    """
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return x/numpy.linalg.norm(x, ord=2, axis=1, keepdims=True)