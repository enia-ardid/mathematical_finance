"""Analytic risk curves and a fixed-seed, positive-signal simulation; no market data."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diffusion_risk import diffusion_risk
from fractional_diffusion import diffusion_operator


def main(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    n, draws, degrees, seed = 8, 40000, 24, 20261026
    adjacency = np.zeros((n, n))
    for j in range(n):
        adjacency[j, (j + 1) % n] = adjacency[(j + 1) % n, j] = .5
    graph = np.diag(adjacency.sum(axis=1)) - adjacency
    rng = np.random.default_rng(seed)
    noise = rng.chisquare(degrees, size=(draws, n)) / degrees
    records = []
    for name, amplitude in [('smooth', .05), ('heterogeneous', .8)]:
        signal = 1 + amplitude * (-1.)**np.arange(n)
        observations = signal * noise
        covariance = np.diag(2 * signal**2 / degrees)
        raw_error = np.sum((observations - signal)**2, axis=1)
        for strength in np.linspace(0, 4, 41):
            analytic = diffusion_risk(signal, covariance, graph, strength)
            error = np.sum((observations @ diffusion_operator(graph, strength).T - signal)**2, axis=1)
            difference = error - raw_error
            se = float(difference.std(ddof=1) / np.sqrt(draws))
            assert abs(difference.mean() - analytic['risk_difference']) <= 6 * se + 1e-12
            records.append(dict(scenario=name, strength=strength, **analytic,
                                mc_difference=float(difference.mean()), mc_standard_error=se))
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'diffusion_risk.csv', index=False)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.2), layout='constrained', sharex=True)
    for ax, (name, block) in zip(axes, frame.groupby('scenario', sort=False)):
        ax.plot(block.strength, block.risk_difference, color='#0072B2', label='Exact risk difference')
        ax.plot(block.strength, block.bias_squared, color='#D55E00', linestyle='--', label='Squared bias')
        ax.plot(block.strength, block.variance - block.baseline_risk, color='#009E73', linestyle=':', label='Variance change')
        sparse = block.iloc[::5]
        ax.errorbar(sparse.strength, sparse.mc_difference, yerr=1.96*sparse.mc_standard_error,
                    fmt='o', ms=3, color='#000000', label='Simulation ± 1.96 SE')
        ax.axhline(0, color='#777777', linewidth=.7)
        ax.set(title=name.capitalize(), xlabel=r'Diffusion strength $\tau$', ylabel='Change in total squared-error risk')
    axes[0].legend(fontsize=7)
    fig.savefig(output / 'diffusion_risk.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(output / 'diffusion_risk.png', dpi=180)
    plt.close(fig)
    manifest = dict(kind='synthetic mechanism experiment, not empirical forecast evidence',
                    seed=seed, draws=draws, assets=n, chi_square_degrees=degrees,
                    amplitudes=[.05,.8], graph='cycle with each edge weight 0.5',
                    noise='independent positive chi-square measurements, variance 2*m_j^2/24',
                    code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in ['diffusion_risk.py','run_diffusion_risk.py','fractional_diffusion.py']},
                    outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(output.glob('diffusion_risk.*'))})
    (output / 'diffusion_risk_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(frame[frame.strength.isin([0,1,4])].to_string(index=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/diffusion-risk'))
    main(parser.parse_args().output)
