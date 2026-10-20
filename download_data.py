"""Download a fresh Yahoo Finance snapshot; existing files are never overwritten."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from temporal_memory import SECTORS, load_prices


def download(directory, start='2018-01-01', end='2025-01-01'):
    directory = Path(directory)
    raw = directory / 'raw'
    if raw.exists() and any(raw.iterdir()):
        raise FileExistsError('Choose an empty output directory to preserve existing snapshots.')
    symbols = ['^GSPC', *SECTORS]
    yf.set_tz_cache_location(str(directory / '.yfinance-cache'))
    panel = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    if panel.empty or not isinstance(panel.columns, pd.MultiIndex):
        raise ValueError('Download returned no usable price panel.')
    close = panel['Close']
    if set(symbols) - set(close.columns):
        raise ValueError('Download did not return all requested tickers.')
    raw.mkdir(parents=True, exist_ok=True)
    sp = close[['^GSPC']].rename(columns={'^GSPC': 'SP500'})
    sectors = close[list(SECTORS)]
    sp.rename_axis('Date').to_csv(raw / 'sp500_close.csv')
    sectors.rename_axis('Date').to_csv(raw / 'sector_etfs_close.csv')
    load_prices(directory)
    manifest = {'provider': 'Yahoo Finance via yfinance', 'yfinance': yf.__version__,
                'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
                'start_inclusive': start, 'end_exclusive': end, 'auto_adjust': True,
                'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.glob('*.csv')}}
    (directory / 'source.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved and validated {len(close)} price rows per asset before inception alignment.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data_finance'))
    parser.add_argument('--start', default='2018-01-01')
    parser.add_argument('--end', default='2025-01-01', help='Exclusive end date.')
    args = parser.parse_args()
    download(args.output, args.start, args.end)
