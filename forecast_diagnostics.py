"""Post-hoc diagnostics of frozen forecasts; never refits a forecast model."""

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


def hac_meat(scores, lags):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores[:, None]
    n = len(scores)
    if not np.isfinite(scores).all() or not 0 <= lags < n:
        raise ValueError("Finite scores and 0 <= lags < observations required.")
    result = scores.T @ scores
    for lag in range(1, lags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        result += (1 - lag / (lags + 1)) * (cross + cross.T)
    return result


def dm_test(difference, lags):
    """Two-sided asymptotic DM test with Bartlett HAC on ordered origin losses."""
    d = np.asarray(difference, dtype=float)
    if d.ndim != 1 or len(d) < 20 or not np.isfinite(d).all():
        raise ValueError("At least 20 finite, ordered origin differences required.")
    n = len(d)
    se2 = float(hac_meat(d - d.mean(), lags)[0, 0]) / n**2
    if se2 <= np.finfo(float).eps * np.mean(d**2):
        return dict(mean_difference=float(d.mean()), dm=np.nan, p_value=np.nan,
                    standard_error=np.sqrt(max(se2, 0)), status="degenerate_loss_variance")
    statistic = float(d.mean() / np.sqrt(se2))
    return dict(mean_difference=float(d.mean()), dm=statistic,
                p_value=float(2 * norm.sf(abs(statistic))),
                standard_error=float(np.sqrt(se2)), status="ok")


def mz_test(predicted, observed, lags):
    """OLS calibration y = alpha + beta*x; HAC Wald test of (alpha,beta)=(0,1)."""
    x, y = np.asarray(predicted, float), np.asarray(observed, float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 20:
        raise ValueError("Aligned one-dimensional samples of size >= 20 required.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite calibration input.")
    design = np.column_stack([np.ones(len(x)), x])
    if np.linalg.matrix_rank(design) != 2:
        raise ValueError("Calibration design is rank deficient.")
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coef
    bread = np.linalg.inv(design.T @ design)
    covariance = bread @ hac_meat(design * residual[:, None], lags) @ bread
    covariance *= len(x) / (len(x) - 2)
    covariance = (covariance + covariance.T) / 2
    variance_y = np.sum((y - y.mean())**2)
    result = dict(alpha=float(coef[0]), beta=float(coef[1]),
                  se_alpha=float(np.sqrt(max(covariance[0, 0], 0))),
                  se_beta=float(np.sqrt(max(covariance[1, 1], 0))),
                  r_squared=float(1 - residual @ residual / variance_y) if variance_y else np.nan,
                  wald=np.nan, p_value=np.nan, status="singular_hac")
    if np.linalg.matrix_rank(covariance) == 2 and np.max(np.abs(residual)) > 1e-12:
        deviation = coef - [0, 1]
        wald = float(deviation @ np.linalg.solve(covariance, deviation))
        if wald >= 0:
            result.update(wald=wald, p_value=float(chi2.sf(wald, 2)), status="ok")
    return result


def holm(pvalues):
    """Adjusted p-values; undefined tests retain their place in family size."""
    p = np.asarray(pvalues, dtype=float)
    if np.any((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)):
        raise ValueError("p-values must be in [0,1].")
    work = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(work, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(work[order] * np.arange(len(p), 0, -1)))
    output = np.empty_like(p)
    output[order] = adjusted
    output[~np.isfinite(p)] = np.nan
    return output


def aligned_pair(forecasts, model, reference):
    keys = ["panel", "horizon", "origin", "target", "asset"]
    parts = []
    for name in (model, reference):
        part = forecasts[forecasts.method == name].set_index(keys).sort_index()
        if part.empty or not part.index.is_unique:
            raise ValueError("Missing or duplicated forecast keys.")
        parts.append(part)
    a, b = parts
    if not a.index.equals(b.index):
        raise ValueError("Forecast keys differ; refusing an inner join.")
    for col in ("realized_proxy", "reference_scale"):
        if not np.array_equal(a[col], b[col]):
            raise ValueError("Targets or scales differ.")
    if not np.isfinite(a.qlike).all() or not np.isfinite(b.qlike).all():
        raise ValueError("Nonfinite losses.")
    return (a.qlike - b.qlike).rename("difference").reset_index()


def win_counts(pair, tolerance=1e-12):
    """Each asset's number of origin wins/losses; aggregate sign is separate."""
    output = []
    for key, block in pair.groupby(["panel", "horizon", "asset"], sort=True):
        d = block.difference.to_numpy()
        output.append(dict(zip(["panel", "horizon", "asset"], key)) |
                      dict(origins=len(d), wins=int((d < -tolerance).sum()),
                           ties=int((abs(d) <= tolerance).sum()),
                           losses=int((d > tolerance).sum()), mean_difference=float(d.mean())))
    return pd.DataFrame(output)
