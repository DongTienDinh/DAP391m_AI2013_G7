import numpy as np
from scipy.optimize import minimize


class EntropyOptimizer:
    """Optimizes weights using Shannon Entropy Maximization (SLSQP)."""

    @staticmethod
    def find_optimal_weights(
        norm_matrix: np.ndarray, constraints: list[tuple[float, float]]
    ) -> np.ndarray:
        """
        Solves the entropy maximization problem subject to sum(w)=1 and bounds.
        """

        def neg_entropy(w: np.ndarray) -> float:
            scores = (norm_matrix * w).sum(axis=1)
            total = scores.sum()
            if total <= 0:
                return 0.0
            p = scores / total
            # Standardized entropy
            return (1.0 / np.log(len(p))) * np.sum(p * np.log(p + 1e-10))

        # Initial guess (center of bounds)
        w0 = np.array([(lo + hi) / 2.0 for lo, hi in constraints])
        w0 /= w0.sum()

        result = minimize(
            neg_entropy,
            w0,
            method="SLSQP",
            bounds=constraints,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        )

        return result.x
