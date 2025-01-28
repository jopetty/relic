from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
import numpy as np


def optimize_gp(X, y, bounds, n_restarts=5):
    """
    Fit a Gaussian Process with Matern kernel and find the minimum on a bounded hyperplane.

    Args:
        X: Training input points (n_samples, n_features)
        y: Training target values (n_samples,)
        bounds: List of (min, max) tuples for each dimension
        n_restarts: Number of random restarts for optimization

    Returns:
        tuple: (optimal_point, predicted_minimum_value)
    """
    # Initialize GP with Matern kernel
    kernel = Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,  # For kernel hyperparameter optimization
        random_state=42,
    )

    # Fit GP
    gp.fit(X, y)

    # Define objective function
    def objective(x):
        return gp.predict([x])[0]

    # Run optimization from multiple random starting points
    best_x = None
    best_val = np.inf

    for _ in range(n_restarts):
        # Random starting point
        x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])

        # Run optimization
        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x

    return best_x, best_val


# Example usage
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    X = np.random.uniform(-5, 5, (20, 2))
    y = -(X[:, 0] ** 2 + X[:, 1] ** 2) + np.random.normal(0, 0.1, 20)

    # Define bounds for optimization
    bounds = [(-5, 5), (-5, 5)]

    # Find minimum
    opt_point, min_val = optimize_gp(X, y, bounds)
    print(f"Optimal point: {opt_point}")
    print(f"Predicted minimum value: {min_val}")
