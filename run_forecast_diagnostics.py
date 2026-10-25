"""Run recorded post-hoc diagnostics on saved forecasts, without model refits."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from forecast_diagnostics import aligned_pair, dm_test, holm, mz_test, win_counts


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(protocol_path, output, publish=False):
    protocol = json.loads(protocol_path.read_text())
    source_hashes = dict(protocol["code_sha256"])
    loss_tolerance = 1e-12
    amendment_path = protocol_path.with_name("diagnostics_protocol_amendment.json")
    amendment_hash = None
    if amendment_path.exists():
        amendment = json.loads(amendment_path.read_text())
        if amendment["prior_protocol_sha256"] != digest(protocol_path):
            raise ValueError("Amendment does not match the original protocol.")
        source_hashes.update(amendment["code_sha256"])
        loss_tolerance = amendment["loss_reconstruction_atol"]
        amendment_hash = digest(amendment_path)
    for section in ("input_sha256", "code_sha256"):
        for name, expected in (source_hashes if section == "code_sha256" else protocol[section]).items():
            if digest(name) != expected:
                raise ValueError(f"{section} mismatch: {name}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new diagnostics output directory.")
    f = pd.read_csv(protocol["forecasts"], parse_dates=["origin", "target", "last_fit"],
                    float_precision="round_trip")
    metrics = pd.read_csv("paper/external_metrics.csv")
    expected = metrics[metrics.year == "all"].set_index(["panel", "horizon", "method"]).qlike.sort_index()
    actual = f.groupby(["panel", "horizon", "method"]).qlike.mean().sort_index()
    pd.testing.assert_index_equal(expected.index, actual.index)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    for name in ("variance", "reference_scale", "realized_proxy"):
        if not np.isfinite(f[name]).all() or (f[name] < 0).any():
            raise ValueError(f"Invalid {name}.")
    if (f[["variance", "reference_scale"]] <= 0).any().any():
        raise ValueError("Positive forecasts and reference scales required.")
    np.testing.assert_allclose(f.qlike, np.log(f.variance/f.reference_scale)+f.realized_proxy/f.variance,
                               rtol=1e-12, atol=loss_tolerance)
    dm, wins = [], []
    for model in protocol["dm_models"]:
        pair = aligned_pair(f, model, protocol["dm_reference"])
        counts = win_counts(pair, protocol["tie_tolerance"])
        counts["model"], counts["reference"] = model, protocol["dm_reference"]
        wins.append(counts)
        for (panel, horizon), block in pair.groupby(["panel", "horizon"]):
            ordered = block.groupby("origin", sort=True).difference.mean()
            if len(ordered) != protocol["expected_origins"]:
                raise ValueError("Unexpected origin count.")
            for lag in protocol["hac_lags"]:
                dm.append(dict(panel=panel, horizon=horizon, model=model,
                               reference=protocol["dm_reference"], origins=len(ordered),
                               hac_lags=lag, **dm_test(ordered.to_numpy(), lag)))
    dm = pd.DataFrame(dm)
    dm["holm_p_value"] = dm.groupby("hac_lags").p_value.transform(lambda p: holm(p.to_numpy()))
    mz = []
    for method in protocol["calibration_models"]:
        block = f[f.method == method].copy()
        block["normalized_forecast"] = block.variance / block.reference_scale
        block["normalized_target"] = block.realized_proxy / block.reference_scale
        for (panel, horizon), group in block.groupby(["panel", "horizon"]):
            series = group.groupby("origin", sort=True)[["normalized_forecast", "normalized_target"]].mean()
            for lag in protocol["hac_lags"]:
                result = mz_test(series.normalized_forecast, series.normalized_target, lag)
                mz.append(dict(panel=panel, horizon=horizon, method=method,
                               origins=len(series), hac_lags=lag, **result))
    mz = pd.DataFrame(mz)
    mz["holm_p_value"] = mz.groupby("hac_lags").p_value.transform(lambda p: holm(p.to_numpy()))
    wins = pd.concat(wins, ignore_index=True)
    counts = wins.assign(asset_win=wins.mean_difference < -protocol["tie_tolerance"],
                         asset_tie=abs(wins.mean_difference) <= protocol["tie_tolerance"])
    panel_counts = counts.groupby(["panel", "horizon", "model", "reference"]).agg(
        asset_wins=("asset_win", "sum"), asset_ties=("asset_tie", "sum"),
        assets=("asset", "size"), origin_wins=("wins", "sum"),
        origin_ties=("ties", "sum"), asset_origins=("origins", "sum")).reset_index()
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [("dm", dm), ("mz", mz), ("asset_wins", wins), ("panel_wins", panel_counts)]:
        table.to_csv(output / f"diagnostics_{name}.csv", index=False)
    summary = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                   protocol_sha256=digest(protocol_path),
                   amendment_sha256=amendment_hash,
                   scope=protocol["status"],
                   rows=dict(dm=len(dm), mz=len(mz), asset_wins=len(wins), panel_wins=len(panel_counts)),
                   dm_status=dm.status.value_counts().to_dict(), mz_status=mz.status.value_counts().to_dict(),
                   input_sha256=protocol["input_sha256"], code_sha256=source_hashes)
    (output / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if publish:
        for path in sorted(output.glob("diagnostics_*")):
            shutil.copyfile(path, Path("paper") / path.name)
    print(json.dumps(summary["rows"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("paper/diagnostics_protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/submission-diagnostics"))
    parser.add_argument("--publish-assets", action="store_true")
    args = parser.parse_args()
    run(args.protocol, args.output, args.publish_assets)


if __name__ == "__main__":
    main()
