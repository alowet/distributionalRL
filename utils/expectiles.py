import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import root, minimize
import os
import sys


def expectile_loss_fn(expectiles, taus, samples):
    """Expectile loss function, corresponds to distributional TD model.
	 Returns a single scalar, the mean squared error"""
    # distributional TD model: delta_t = (r + \gamma V*) - V_i
    # expectile loss: delta = sample - expectile
    delta = (samples[None, :] - expectiles[:, None])

    # distributional TD model: alpha^+ delta if delta > 0, alpha^- delta otherwise
    # expectile loss: |taus - I_{delta <= 0}| * delta^2

    # Note: When used to decode we take the gradient of this loss,
    # and then evaluate the mean-squared gradient. That is because *samples* must
    # trade-off errors with all expectiles to zero out the gradient of the
    # expectile loss.
    indic = np.array(delta <= 0., dtype=np.float32)
    grad = -0.5 * np.abs(taus[:, None] - indic) * delta
    return np.mean(np.square(np.mean(grad, axis=-1)))


def expectile_grad_loss(expectiles, taus, dist):
    # returns a vector, one value for each expectile
    delta = dist[np.newaxis, :] - expectiles[:, np.newaxis]
    indic = np.array(delta <= 0., dtype=np.float32)
    grad = -2. * np.abs(taus[:, np.newaxis] - indic) * delta
    return np.mean(grad, axis=1)


def run_decoding(reversal_points, taus, minv=0., maxv=1., method=None,
                 max_samples=1000, max_epochs=3, N=100):
    """Run decoding given reversal points and asymmetries (taus)."""

    ind = list(np.argsort(reversal_points))
    points = reversal_points[ind]
    tau = taus[ind]

    # Robustified optimization to infer distribution
    # Generate max_epochs sets of samples,
    # each starting the optimization at the best of max_samples initial points.
    sampled_dist = []
    for _ in range(max_epochs):
        # Randomly search for good initial conditions
        # This significantly improves the minima found
        samples = np.random.uniform(minv, maxv, size=(max_samples, N))
        fvalues = np.array([expectile_loss_fn(points, tau, x0) for x0 in samples])

        # Perform loss minimizing on expectile loss (w.r.t samples)
        x0 = np.array(sorted(samples[fvalues.argmin()]))
        fn_to_minimize = lambda x: expectile_loss_fn(points, tau, x)
        result = minimize(fn_to_minimize, method=method, bounds=[(minv, maxv) for _ in x0], x0=x0)['x']

        sampled_dist.extend(result.tolist())

    return sampled_dist, expectile_loss_fn(points, tau, np.array(sampled_dist))


def infer_dist(expectiles=None, taus=None, dist=None):
    """
	Given kappa and two of the following three values (reversal_points, taus, and dist), we can always infer the third
	:param reversal_points: vector of expectiles or Huber quantiles
	:param taus: vector, with values between 0 and 1
	:param dist: the distribution for which we want to compute Huber quantiles
	:param kappa: parameter for Huber quantiles
	"""
    # infer expectiles
    if expectiles is None:
        fn_to_solve = lambda x: expectile_grad_loss(x, taus, dist)
        taus[taus < 0.] = 0.
        taus[taus > 1.] = 1.
        sol = root(fn_to_solve, x0=np.quantile(dist, taus), method='lm')

    # infer taus
    elif taus is None:
        fn_to_solve = lambda x: expectile_grad_loss(expectiles, x, dist)
        sol = root(fn_to_solve, x0=np.linspace(0.01, 0.99, len(expectiles)), method='lm')

    # impute distribution
    elif dist is None:
        fn_to_solve = lambda x: expectile_grad_loss(expectiles, taus, x)
        sol = root(fn_to_solve, x0=expectiles, method='lm', options={'maxiter': 100000})

    check_convergence(sol)

    # return the optimized value
    return sol['x']


def plot_imputation(empirical_dist, imputed_dist, ax, bw=.05):
    sns.kdeplot(empirical_dist, bw=bw, color='k', lw=3., shade=True, label="Empirical", ax=ax, legend=False)
    sns.rugplot(empirical_dist, color='k', ax=ax)
    sns.kdeplot(imputed_dist, bw=bw, color=plt.cm.plasma(0), lw=3., shade=True, label="Decoded", ax=ax, legend=False)
    sns.rugplot(imputed_dist, color=plt.cm.plasma(0), ax=ax)


def check_convergence(sol):
    # make sure optimization has converged
    if not sol['success']:
        raise_print(sol['message'])
