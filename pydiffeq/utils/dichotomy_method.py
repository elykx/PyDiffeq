import numpy as np


def dichotomy_method(lower_bound, upper_bound, k, tol=1e-6):
    while np.any(upper_bound - lower_bound > tol):
        mid = (lower_bound + upper_bound) / 2
        if np.any((mid - k) * (lower_bound - k)) < 0:
            upper_bound = mid
        else:
            lower_bound = mid
    return (lower_bound + upper_bound) / 2
