import numpy as np

from src.analysis.optimizer import EntropyOptimizer


def test_entropy_optimizer_sum_to_one():
    """Ensures optimized weights always sum to 1.0."""
    matrix = np.random.rand(10, 4)
    constraints = [(0.1, 0.5), (0.1, 0.5), (0.1, 0.5), (0.1, 0.5)]

    weights = EntropyOptimizer.find_optimal_weights(matrix, constraints)

    assert np.isclose(weights.sum(), 1.0)
    assert all(0.1 <= w <= 0.5 for w in weights)

def test_entropy_optimizer_uniform_data():
    """Ensures uniform data results in uniform weights (if bounds allow)."""
    matrix = np.ones((5, 3))
    constraints = [(0.1, 0.9), (0.1, 0.9), (0.1, 0.9)]

    weights = EntropyOptimizer.find_optimal_weights(matrix, constraints)

    # Should be close to 0.33 each
    assert np.allclose(weights, [1/3, 1/3, 1/3], atol=1e-2)
