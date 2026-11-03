"""New reference implementation of paper Eq. 6-8, not original experiment code.
Requires independent train/test units and exchangeability within exact test strata.
"""
import numpy as np
from scipy.stats import rankdata

def overlap_weights(y, strata):
    y = np.asarray(y)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Binary labels required")
    _, groups = np.unique(np.asarray(strata), axis=0, return_inverse=True)
    counts = np.bincount(groups)
    propensity = np.bincount(groups, weights=y) / counts
    weights = np.where(y == 1, 1 - propensity[groups], propensity[groups])
    if weights.sum() == 0:
        raise ValueError("No training overlap")
    return weights / weights.mean()

def permutation_test(frozen_scores, y, strata, permutations=999, seed=7):
    if permutations < 1:
        raise ValueError("At least one permutation required")
    y = np.asarray(y)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Binary labels required")
    ranks = rankdata(frozen_scores, method="average")
    _, groups = np.unique(np.asarray(strata), axis=0, return_inverse=True)
    indices = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    observed = float(y @ ranks)
    exceed = 0
    for _ in range(permutations):
        permuted = y.copy()
        for ix in indices:
            permuted[ix] = rng.permutation(y[ix])
        exceed += int(float(permuted @ ranks) >= observed)
    information = sum(float(np.sum((y[ix] - y[ix].mean()) ** 2)) for ix in indices)
    return {"p_value": (1 + exceed) / (permutations + 1),
            "permutations": permutations, "test_information": information,
            "mixed_test_strata": int(sum(0 < y[ix].sum() < len(ix) for ix in indices))}
