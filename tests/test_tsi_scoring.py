import numpy as np
import pytest

from tsi.tsi_scoring import tsi_score, tsi_score_approx


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_identical_spaces_score_one(rng):
    x = rng.normal(size=(60, 16))
    assert tsi_score(x, x) == pytest.approx(1.0)
    assert tsi_score(x, x, metric="cosine") == pytest.approx(1.0)


def test_isometry_preserves_score(rng):
    # a rotation + uniform scale preserves all pairwise distance orderings exactly
    x = rng.normal(size=(60, 16))
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    y = (x @ q) * 3.0
    assert tsi_score(x, y) == pytest.approx(1.0)
    assert tsi_score(x, y, metric="cosine") == pytest.approx(1.0)


def test_independent_spaces_score_near_half(rng):
    x = rng.normal(size=(200, 16))
    y = rng.normal(size=(200, 16))
    assert tsi_score(x, y) == pytest.approx(0.5, abs=0.05)
    assert tsi_score(x, y, metric="cosine") == pytest.approx(0.5, abs=0.05)


def test_symmetric_in_its_arguments(rng):
    x = rng.normal(size=(40, 8))
    y = rng.normal(size=(40, 8))
    assert tsi_score(x, y) == pytest.approx(tsi_score(y, x))
    assert tsi_score(x, y, metric="cosine") == pytest.approx(tsi_score(y, x, metric="cosine"))


def test_approx_matches_exact(rng):
    x = rng.normal(size=(80, 16))
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    y = x @ q + rng.normal(scale=0.5, size=(80, 16))

    exact = tsi_score(x, y)
    approx = tsi_score_approx(x, y, n_samples=20000, seed=1)
    assert approx == pytest.approx(exact, abs=0.02)

    exact = tsi_score(x, y, metric="cosine")
    approx = tsi_score_approx(x, y, n_samples=20000, seed=1, metric="cosine")
    assert approx == pytest.approx(exact, abs=0.02)


def test_too_few_examples_raises():
    x = np.zeros((2, 4))
    with pytest.raises(AssertionError):
        tsi_score(x, x)


def test_mismatched_example_counts_raises(rng):
    x = rng.normal(size=(10, 4))
    y = rng.normal(size=(11, 4))
    with pytest.raises(AssertionError):
        tsi_score(x, y)
