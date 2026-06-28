"""Negative Binomial distribution utilities for TARVI."""

import torch


def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Log-likelihood of the Negative Binomial distribution.

    Uses the parameterization where:
        NB(x | mu, theta) where mu is mean and theta is inverse dispersion.

    For continuous (normalized) data, uses a Gaussian approximation of NB
    that matches the NB mean and variance: N(mu, mu + mu^2/theta).

    Parameters
    ----------
    x
        Observed values (can be continuous/normalized).
    mu
        Mean of the distribution (positive).
    theta
        Inverse dispersion parameter (positive). Larger theta = less dispersion.
    eps
        Small value for numerical stability.

    Returns
    -------
    Log-probability of x under the distribution.
    """
    mu = torch.clamp(mu, min=eps)
    theta = torch.clamp(theta, min=eps)

    # NB variance = mu + mu^2/theta
    # Use Gaussian approximation which is valid for continuous data
    # and avoids lgamma issues with non-integer x
    variance = mu + mu.pow(2) / theta + eps
    log_prob = -0.5 * torch.log(2 * torch.pi * variance) - 0.5 * (x - mu).pow(2) / variance

    return log_prob
