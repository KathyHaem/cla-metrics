import numpy as np
from scipy.stats import kendalltau


def _row_distance(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    """Row-wise distance between corresponding rows of a and b, shape (n, dim) -> (n,).

    Only relative ordering of distances matters for TSI (Kendall's tau is invariant to
    monotonic transforms), so "cosine distance" here is just 1 - cosine similarity.
    """
    if metric == "euclidean":
        return np.linalg.norm(a - b, axis=1)
    elif metric == "cosine":
        sims = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        return 1 - sims
    else:
        raise ValueError(f"Unknown metric: {metric!r}, expected 'euclidean' or 'cosine'")


def _pairwise_dist_from_anchor(reps: np.ndarray, anchor: int, metric: str = "euclidean") -> np.ndarray:
    """Distance from reps[anchor] to every other row, anchor excluded."""
    d = _row_distance(reps, np.broadcast_to(reps[anchor], reps.shape), metric)
    return np.delete(d, anchor)


def tsi_score(x: np.ndarray, y: np.ndarray, metric: str = "euclidean") -> float:
    """ Triplet Similarity Index (TSI), from Soares et al. (2026),
    "Scalable and Interpretable Representation Alignment with Ordinal Similarity"
    https://arxiv.org/html/2606.16379

    x, y: paired representations, shape (num_examples, hidden_size) each, where
    x[i] and y[i] are representations of the same underlying sentence in the two
    languages/layers being compared.
    metric: "euclidean" (default) or "cosine" distance for the ordinal comparisons.

    For an anchor point i and any other two points j, k, define the ordinal
    comparison O(i, j, k) = sign(d(i, j) - d(i, k)), i.e. whether j or k is closer
    to i. TSI is the fraction of all (anchor, j, k) triplets for which this
    ordering agrees between the X and Y representation spaces -- the probability
    that a random triplet's relative ordering is preserved under the X -> Y
    mapping.

    Exact computation is naively O(N^3) (all triplets), but per anchor, agreement
    over all (j, k) pairs is exactly a Kendall rank correlation between the
    anchor's distances to every other point in X vs. in Y: with continuous-valued
    distances (ties ~never occur), P(agree) for anchor i is (tau_i + 1) / 2. This
    gets us the paper's claimed O(N^2 log N) exact algorithm by relying on
    scipy's O(N log N) Kendall's tau implementation per anchor.
    """
    n = x.shape[0]
    assert y.shape[0] == n, f"x and y must have the same number of examples, got {n} and {y.shape[0]}"
    assert n >= 3, "TSI needs at least 3 examples to form a triplet"

    per_anchor_agreement = np.empty(n)
    for i in range(n):
        dx_i = _pairwise_dist_from_anchor(x, i, metric)
        dy_i = _pairwise_dist_from_anchor(y, i, metric)
        tau, _ = kendalltau(dx_i, dy_i)
        per_anchor_agreement[i] = (tau + 1) / 2

    return per_anchor_agreement.mean().item()


def tsi_score_approx(x: np.ndarray, y: np.ndarray, epsilon: float = 0.01, delta: float = 0.05,
                      n_samples: int = None, seed: int = 42, metric: str = "euclidean") -> float:
    """ Approximate TSI via uniform triplet sampling (Corollary 4 of the TSI paper, see tsi_score).

    Sampling ceil(1 / (2 * epsilon^2) * log(2 / delta)) triplets gives an estimate with additive
    error at most epsilon with probability at least 1 - delta, independent of N. Useful when N is
    large enough that the exact O(N^2 log N) computation in tsi_score is too slow.
    metric: "euclidean" (default) or "cosine" distance for the ordinal comparisons.
    """
    n = x.shape[0]
    assert y.shape[0] == n, f"x and y must have the same number of examples, got {n} and {y.shape[0]}"
    assert n >= 3, "TSI needs at least 3 examples to form a triplet"

    if n_samples is None:
        n_samples = int(np.ceil(1 / (2 * epsilon ** 2) * np.log(2 / delta)))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_samples, 3))
    dup = (idx[:, 0] == idx[:, 1]) | (idx[:, 0] == idx[:, 2]) | (idx[:, 1] == idx[:, 2])
    while dup.any():
        idx[dup] = rng.integers(0, n, size=(dup.sum(), 3))
        dup = (idx[:, 0] == idx[:, 1]) | (idx[:, 0] == idx[:, 2]) | (idx[:, 1] == idx[:, 2])
    anchor, j, k = idx[:, 0], idx[:, 1], idx[:, 2]

    dx_ij = _row_distance(x[anchor], x[j], metric)
    dx_ik = _row_distance(x[anchor], x[k], metric)
    dy_ij = _row_distance(y[anchor], y[j], metric)
    dy_ik = _row_distance(y[anchor], y[k], metric)

    agree = np.sign(dx_ij - dx_ik) == np.sign(dy_ij - dy_ik)
    return agree.mean().item()
