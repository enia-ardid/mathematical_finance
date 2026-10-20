"""Download the exact external ETF universe recorded before data inspection."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yfinance as yf

from temporal_memory import validate_prices


def download(directory):
    protocol_path = Path('paper/external_protocol.json')
    protocol = json.loads(protocol_path.read_text())
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Choose an empty directory; snapshots are never overwritten.')
    symbols = protocol['external_tickers']
    yf.set_tz_cache_location(str(directory / '.yfinance-cache'))
    downloaded_at = datetime.now(timezone.utc).isoformat()
    prices = yf.download(symbols, start=protocol['start_inclusive'],
                         end=protocol['end_exclusive'], auto_adjust=True,
                         progress=False)['Close'].loc[:, symbols]
    # All selected funds predate the requested period. No filling or ticker removal.
    validate_prices(prices)
    if any(prices[c].count() < 1500 for c in symbols):
        raise ValueError('Insufficient history; do not silently change the universe.')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'prices.csv'
    prices.rename_axis('Date').to_csv(path)
    manifest = dict(provider='Yahoo Finance via yfinance', yfinance=yf.__version__,
                    downloaded_at_utc=downloaded_at, auto_adjust=True, tickers=symbols,
                    start_inclusive=protocol['start_inclusive'], end_exclusive=protocol['end_exclusive'],
                    rows=len(prices), first_price=str(prices.index[0].date()),
                    last_price=str(prices.index[-1].date()),
                    sha256={'prices.csv': hashlib.sha256(path.read_bytes()).hexdigest()},
                    protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest())
    (directory / 'source.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {len(prices)} prices for the {len(symbols)} pre-specified ETFs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    download(parser.parse_args().output)
