"""Check publication assets against saved records; never fit a forecasting model.

Run from the repository root after rendering and compiling. Pass --regenerated
with a separately rendered directory to check byte-for-byte reproduction.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_hashes(mapping, root=Path('.')):
    for name, expected in mapping.items():
        if sha(root / name) != expected:
            raise ValueError(f'Hash mismatch: {root / name}')


def pooled(name):
    frame = pd.read_csv(Path('paper') / name)
    return frame[frame.year.astype(str) == 'all'].pivot(
        index=['panel', 'horizon'], columns='method', values='qlike')


def table_rows(name):
    result = []
    for line in (Path('paper/tables') / name).read_text().splitlines():
        if ' & ' in line and not line.startswith(('Horizon', 'Panel', 'Model', 'Asset', 'Year', ' & ')):
            result.append([x.strip().removesuffix('\\\\').strip() for x in line.split(' & ')])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regenerated', type=Path)
    args = parser.parse_args()
    record = json.loads(Path('paper/submission_assets_manifest.json').read_text())
    for field in ['input_sha256', 'source_sha256']:
        check_hashes(record[field])
    check_hashes(record['output_sha256'], Path('paper'))
    if args.regenerated:
        check_hashes(record['output_sha256'], args.regenerated)
    subset = json.loads(Path('paper/prediction_figures_manifest.json').read_text())
    check_hashes(subset['outputs_sha256'], Path('paper'))
    diagnostic = json.loads(Path('paper/diagnostics_summary.json').read_text())
    for field in ['input_sha256', 'code_sha256']:
        check_hashes(diagnostic[field])
    assert sha('paper/diagnostics_protocol.json') == diagnostic['protocol_sha256']
    assert sha('paper/diagnostics_protocol_amendment.json') == diagnostic['amendment_sha256']
    assert json.loads(Path('paper/diagnostics_protocol_amendment.json').read_text())['prior_protocol_sha256'] == diagnostic['protocol_sha256']

    labels = {'Historical':'historical', 'EWMA':'ewma', 'GARCH':'garch',
              'Fractional only':'fractional_only', 'Exponential + graph':'exponential_diffusion',
              'Fractional + graph':'fractional_diffusion', 'Refined fractional':'persistent_asymmetric',
              'Persistent symmetric':'persistent_symmetric', 'Asymmetric, no graph':'asymmetric_no_graph',
              'Asymmetric exponential':'asymmetric_exponential', 'GJR-GARCH':'gjr_garch',
              'Fixed 50:50 ensemble':'equal_combination', 'FIGARCH':'figarch',
              'Network GARCH (weighted)':'network_garch', 'Network GARCH (binary)':'network_garch_binary',
              'Pooled GARCH':'pooled_garch'}
    score_cells = 0
    for table, source, panels in [
        ('main_scores.tex','metrics.csv',['index','sectors']),
        ('refinement_scores.tex','refinement_metrics.csv',['index','sectors']),
        ('external_scores.tex','external_metrics.csv',['international']),
        ('network_scores.tex','network_metrics.csv',['sectors','international'])]:
        data = pooled(source)
        rows = table_rows(table)
        methods = [labels[row[0]] for row in rows]
        for row in rows:
            for cell, key in zip(row[1:], [(p,h) for p in panels for h in [1,5,21]], strict=True):
                value = data.loc[key, labels[row[0]]]
                is_best = np.isclose(value, data.loc[key,methods].min(), rtol=0, atol=1e-12)
                assert ('textbf' in cell) == is_best, (table,row[0],key)
                assert re.search(r'\d+\.\d{4}', cell).group() == f'{value:.4f}'
                score_cells += 1
    annual = pd.read_csv('paper/metrics.csv')
    annual = annual[(annual.year != 'all') & (annual.method == 'fractional_diffusion')]
    for row in table_rows('annual_scores.tex'):
        for cell,(panel,horizon) in zip(row[1:],[(p,h) for p in ['index','sectors'] for h in [1,5,21]],strict=True):
            value = annual[(annual.year == row[0]) & (annual.panel == panel) & (annual.horizon == horizon)].qlike.item()
            assert cell == f'{value:.4f}'
    differences = pd.read_csv('paper/comparisons.csv')
    for row in table_rows('main_differences.tex'):
        item = differences[(differences.panel == row[0].lower()) & (differences.horizon == int(row[1])) &
                           (differences.reference == 'garch') & (differences.block_length == 10)].iloc[0]
        assert row[2] == f'{item.mean_difference:+.5f}'
        assert re.findall(r'[+-]\d+\.\d+',row[3]) == [f'{item.lower_95:+.5f}',f'{item.upper_95:+.5f}']

    initial, refined, external, network = [pooled(n) for n in ['metrics.csv','refinement_metrics.csv','external_metrics.csv','network_metrics.csv']]
    assert (initial.fractional_diffusion < initial.ewma).all()
    assert (initial.garch < initial.fractional_diffusion).all()
    assert (refined.persistent_asymmetric < refined.fractional_diffusion).sum() == 6
    assert (refined.persistent_asymmetric < refined.garch).sum() == 4
    assert (refined.gjr_garch < refined.persistent_asymmetric).sum() == 6
    assert (external.equal_combination < external.persistent_asymmetric).sum() == 9
    assert (external.equal_combination < external.figarch).sum() == 9
    assert (external.equal_combination < external.gjr_garch).sum() == 3
    assert (external.loc['international','asymmetric_no_graph'] < external.loc['international','persistent_asymmetric']).all()
    assert (network.loc['sectors','network_garch'] < network.loc['sectors','persistent_asymmetric']).all()
    assert (network.loc['international','network_garch'] > network.loc['international','persistent_asymmetric']).all()
    comparisons = pd.read_csv('paper/external_comparisons.csv')
    intervals = comparisons[(comparisons.model == 'equal_combination') & (comparisons.reference == 'gjr_garch')]
    assert len(intervals) == 18 and (intervals.lower_95 <= 0).all() and (intervals.upper_95 >= 0).all()

    dm = pd.read_csv('paper/diagnostics_dm.csv')
    mz = pd.read_csv('paper/diagnostics_mz.csv')
    wins = pd.read_csv('paper/diagnostics_asset_wins.csv')
    panel = pd.read_csv('paper/diagnostics_panel_wins.csv')
    assert len(dm) == 36 and len(mz) == 54 and len(wins) == 120 and len(panel) == 18
    assert (dm.status == 'ok').all() and (dm.p_value > .05).all()
    assert (mz.status == 'ok').all()
    reject = mz.holm_p_value < .05
    assert (reject == ((mz.panel == 'international') & (mz.horizon == 1))).all()
    short = {'Fractional':'persistent_asymmetric', 'Ensemble':'equal_combination', 'GJR':'gjr_garch'}
    def p_display(value):
        return '$<0.001$' if value < .001 else f'{value:.3f}'
    for row in table_rows('dm_tests.tex'):
        panel_name = {'Index':'index','Sectors':'sectors','Countries':'international'}[row[0]]
        values = dm[(dm.panel == panel_name) & (dm.horizon == int(row[1])) & (dm.model == short[row[2]])].set_index('hac_lags')
        primary = values.loc[4]
        assert row[3:] == [f'{primary.mean_difference:+.5f}', f'{primary.dm:.3f}',
                           p_display(primary.p_value), p_display(values.loc[9,'p_value']),
                           p_display(primary.holm_p_value)]
    for i,row in enumerate(table_rows('calibration_tests.tex')):
        panel_name = ['index','sectors','international'][i // 9]
        values = mz[(mz.panel == panel_name) & (mz.horizon == int(row[1])) & (mz.method == short[row[0]])].set_index('hac_lags')
        primary = values.loc[4]
        assert row[2:] == [f'{primary.alpha:.3f} ({primary.se_alpha:.3f})',
                           f'{primary.beta:.3f} ({primary.se_beta:.3f})', f'{primary.r_squared:.3f}',
                           p_display(primary.p_value), p_display(values.loc[9,'p_value']),
                           p_display(primary.holm_p_value)]
    assert (wins.wins + wins.ties + wins.losses == 186).all()
    for row in table_rows('asset_win_counts.tex'):
        values = wins[wins.asset == row[0]].set_index(['model','horizon'])
        expected = [str(int(values.loc[(m,h),'wins'])) for m in ['persistent_asymmetric','equal_combination'] for h in [1,5,21]]
        assert row[1:] == expected
    for _,r in panel.iterrows():
        asset = wins[(wins.panel == r.panel) & (wins.horizon == r.horizon) & (wins.model == r.model)]
        assert r.asset_wins == (asset.mean_difference < -1e-12).sum()
        assert r.assets == len(asset) and r.origin_wins == asset.wins.sum()

    tex = Path('paper/main.tex').read_text()
    bib = Path('paper/references.bib').read_text()
    cited = {key.strip() for group in re.findall(r'\\cite[pt]?\{([^}]+)\}',tex) for key in group.split(',')}
    keys = set(re.findall(r'@\w+\{([^,]+),',bib))
    bbl = Path('paper/main.bbl').read_text()
    ordered = re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}',bbl)
    assert cited == set(ordered) and cited <= keys
    # Compare first-author names, including given names when surnames coincide.
    first_authors = {}
    for entry in re.split(r'@\w+\{',bib)[1:]:
        key = entry.split(',',1)[0]
        author = re.search(r'author\s*=\s*\{([^}]+)',entry).group(1).split(' and ')[0]
        first_authors[key] = re.sub(r'[^a-z]','',author.lower())
    normalized = [first_authors[key] for key in ordered]
    assert normalized == sorted(normalized)
    aux = Path('paper/main.aux').read_text()
    labels_defined = set(re.findall(r'\\newlabel\{([^}]+)\}',aux))
    assert set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',tex)) <= labels_defined
    assert r'\newlabel{eq:vintage-return}{{26}' in aux
    assert r'T^{\prime}' in tex
    assert re.search(r'\\sum_\{k=0\}\^\{M-1\}\s*C_k',tex)
    log = Path('paper/main.log').read_text()
    assert not re.search(r'Overfull \\[hv]box|undefined references|Citation.*undefined|LaTeX Warning',log)
    for path in [Path('README.md'),Path('paper/README.md'),*Path('docs').glob('*.md')]:
        for link in re.findall(r'\]\(([^)\s]+)\)',path.read_text()):
            if '://' not in link and not link.startswith('#'):
                assert (path.parent / unquote(link.split('#')[0])).exists(), (path,link)
    mechanism = json.loads(Path('paper/diffusion_risk_manifest.json').read_text())
    check_hashes(mechanism['code_sha256'])
    for name, expected in mechanism['outputs_sha256'].items():
        path = Path('paper/figures' if name.endswith(('.pdf','.png')) else 'paper') / name
        assert sha(path) == expected, path
    risk = pd.read_csv('paper/diffusion_risk.csv')
    assert len(risk) == 82
    assert (abs(risk.mc_difference - risk.risk_difference) <= 6*risk.mc_standard_error + 1e-12).all()
    for scenario in ['smooth','heterogeneous']:
        row = risk[(risk.scenario == scenario) & (risk.strength == 1)].iloc[0]
        assert f'{row.baseline_risk:.6f}' in tex and f'{row.risk:.6f}' in tex
    prospective = json.loads(Path('paper/prospective_protocol.json').read_text())
    check_hashes(prospective['code_sha256'])
    assert len(prospective['origin_dates']) == 52 and prospective['lead_sessions'] == 1
    assert prospective['origin_dates'][0] == '2026-10-27'
    assert prospective['sessions'][-1]['date'] == '2027-12-03'
    report = dict(figure_files=24, table_fragments=10, verified_score_cells=score_cells,
                  mechanism_rows=len(risk), prospective_protocol_frozen=True,
                  references=len(cited), diagnostics=dict(dm_rows=len(dm),mz_rows=len(mz),asset_win_rows=len(wins)),
                  paired_GJR_intervals_containing_zero=len(intervals),
                  figures_and_tables_reproduced=bool(args.regenerated),
                  status='passed')
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/submission-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
