"""Conditional squared-error risk of a fixed graph resolvent.

This is a bias--variance calculation for a noisy signal, not a theorem that the
fitted financial model improves QLIKE. The graph must be fixed in the stated
conditional experiment; graph estimation error is not silently ignored.
"""

import numpy as np
from fractional_diffusion import diffusion_operator


def diffusion_risk(signal, covariance, graph, strength):
    m, covariance = np.asarray(signal, float), np.asarray(covariance, float)
    operator = diffusion_operator(graph, strength)
    if m.shape != (len(operator),) or covariance.shape != operator.shape:
        raise ValueError('Signal, covariance and graph dimensions must agree.')
    if not np.isfinite(m).all() or not np.isfinite(covariance).all():
        raise ValueError('Inputs must be finite.')
    if not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance).min() < -1e-12:
        raise ValueError('Noise covariance must be symmetric positive semidefinite.')
    bias = operator @ m - m
    bias_squared = float(bias @ bias)
    variance = float(np.trace(operator @ covariance @ operator.T))
    baseline = float(np.trace(covariance))
    return dict(bias_squared=bias_squared, variance=variance,
                risk=bias_squared + variance, baseline_risk=baseline,
                risk_difference=bias_squared + variance - baseline)


def spectral_risk(signal, covariance, graph, strength):
    """Per-mode identity, including noncommuting graph/noise covariance."""
    diffusion_risk(signal, covariance, graph, strength)
    eigenvalues, vectors = np.linalg.eigh(graph)
    eigenvalues = np.maximum(eigenvalues, 0.)
    attenuation = 1 / (1 + strength * eigenvalues)
    theta = vectors.T @ signal
    noise = np.diag(vectors.T @ covariance @ vectors)
    return dict(eigenvalue=eigenvalues, attenuation=attenuation,
                signal_energy=theta**2, noise_variance=noise,
                risk_difference=(1 - attenuation)**2 * theta**2 - (1 - attenuation**2) * noise)
