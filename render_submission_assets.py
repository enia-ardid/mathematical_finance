"""Regenerate publication tables and figures using saved records only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from fractional_diffusion import fractional_weights
import render_prediction_figures as paths

PAPER = Path("paper")
PANELS = {"index": "Index", "sectors": "Sectors", "international": "Countries"}
LABELS = {
    "historical": "Historical", "ewma": "EWMA", "garch": "GARCH",
    "fractional_only": "Fractional only",
    "exponential_diffusion": "Exponential + graph",
    "fractional_diffusion": "Fractional + graph",
    "persistent_asymmetric": "Refined fractional",
    "equal_combination": "Fixed 50:50 ensemble",
    "gjr_garch": "GJR-GARCH", "figarch": "FIGARCH",
    "network_garch": "Network GARCH (weighted)",
    "network_garch_binary": "Network GARCH (binary)",
    "pooled_garch": "Pooled GARCH",
    "persistent_symmetric": "Persistent symmetric",
    "asymmetric_no_graph": "Asymmetric, no graph",
    "asymmetric_exponential": "Asymmetric exponential",
}
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
SHORT = {"persistent_asymmetric": "Fractional", "equal_combination": "Ensemble", "gjr_garch": "GJR"}
METHOD_COLOR = {"fractional_diffusion": OI[5], "persistent_asymmetric": OI[0],
                "equal_combination": OI[1], "garch": OI[3], "gjr_garch": OI[2], "figarch": OI[4]}
TABLE_OUTPUTS = ["network_scores", "main_scores", "refinement_scores", "external_scores",
                 "main_differences", "annual_scores", "dm_tests", "calibration_tests",
                 "asset_win_counts", "findings"]
FIGURE_OUTPUTS = ["kernel_graph", "forecast_comparison", "variance_paths",
                 "refinement_comparison", "external_comparison", "prediction_index_horizons",
                 "prediction_panel_evolution", "prediction_sectors_a", "prediction_sectors_b",
                 "prediction_countries", "asset_win_counts"]


def score_table(metrics, panels, methods, path):
    """Right-aligned scores; bold the unrounded minimum among displayed models."""
    pooled = metrics[metrics.year.astype(str) == "all"]
    columns = [(p, h) for p in panels for h in (1, 5, 21)]
    table = pooled.pivot(index="method", columns=["panel", "horizon"], values="qlike")
    table = table.loc[methods, columns]
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("Nonfinite or missing scores.")
    lines = [r"\begin{tabular}{l" + "r" * len(columns) + "}", r"\toprule"]
    lines += [" & " + " & ".join(r"\multicolumn{3}{c}{" + PANELS[p] + "}" for p in panels) + r" \\",
              "Horizon & " + " & ".join(str(h) for _, h in columns) + r" \\", r"\midrule"]
    for method, row in table.iterrows():
        cells = []
        for col, value in row.items():
            s = f"{value:.4f}"
            if np.isclose(value, table[col].min(), rtol=0, atol=1e-12):
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(LABELS[method] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")
    return table


def network_table(destination=PAPER / "tables"):
    return score_table(pd.read_csv(PAPER / "network_metrics.csv"),
                       ["sectors", "international"],
                       ["persistent_asymmetric", "equal_combination", "gjr_garch",
                        "figarch", "network_garch", "network_garch_binary", "pooled_garch"],
                       destination / "network_scores.tex")


def write_table(path, alignment, header, rows):
    lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule",
             header + r" \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def pvalue(value):
    if not np.isfinite(value):
        return "--"
    return r"$<0.001$" if value < .001 else f"{value:.3f}"


def tables(destination):
    network_table(destination)
    initial = pd.read_csv(PAPER / "metrics.csv")
    score_table(initial, ["index", "sectors"],
                ["historical", "ewma", "garch", "fractional_only",
                 "exponential_diffusion", "fractional_diffusion"], destination / "main_scores.tex")
    refined = pd.read_csv(PAPER / "refinement_metrics.csv")
    methods = ["fractional_diffusion", "persistent_symmetric", "persistent_asymmetric",
               "asymmetric_no_graph", "asymmetric_exponential", "garch", "gjr_garch"]
    score_table(refined, ["index", "sectors"], methods, destination / "refinement_scores.tex")
    score_table(pd.read_csv(PAPER / "external_metrics.csv"), ["international"],
                ["ewma", "garch", "gjr_garch", "figarch", "asymmetric_no_graph",
                 "asymmetric_exponential", "persistent_asymmetric", "equal_combination"],
                destination / "external_scores.tex")
    comparisons = pd.read_csv(PAPER / "comparisons.csv")
    rows = []
    for panel in ["index", "sectors"]:
        for horizon in [1, 5, 21]:
            r = comparisons[(comparisons.panel==panel) & (comparisons.horizon==horizon)
                            & (comparisons.reference=="garch") & (comparisons.block_length==10)].iloc[0]
            rows.append(f"{PANELS[panel]} & {horizon} & {r.mean_difference:+.5f} & "
                        f"$[{r.lower_95:+.5f},\\ {r.upper_95:+.5f}]$" + r" \\")
    write_table(destination / "main_differences.tex", "lrrr",
                r"Panel & Horizon & Fractional $-$ GARCH & Nominal 95\% interval", rows)
    annual = initial[(initial.year!="all") & (initial.method=="fractional_diffusion")]
    annual = annual.pivot(index="year", columns=["panel", "horizon"], values="qlike")
    cols = [(p,h) for p in ["index","sectors"] for h in [1,5,21]]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{Index} & \multicolumn{3}{c}{Sectors} \\",
             r"Year & 1 & 5 & 21 & 1 & 5 & 21 \\", r"\midrule"]
    for year,row in annual.iterrows():
        lines.append(str(year)+" & "+" & ".join(f"{row[c]:.4f}" for c in cols)+r" \\")
    (destination/"annual_scores.tex").write_text("\n".join(lines+[r"\bottomrule",r"\end{tabular}"])+"\n")
    dm = pd.read_csv(PAPER / "diagnostics_dm.csv")
    mz = pd.read_csv(PAPER / "diagnostics_mz.csv")
    rows = []
    for panel in PANELS:
        for horizon in [1, 5, 21]:
            for model in ["persistent_asymmetric", "equal_combination"]:
                b = dm[(dm.panel==panel)&(dm.horizon==horizon)&(dm.model==model)].set_index("hac_lags")
                r = b.loc[4]
                rows.append(f"{PANELS[panel]} & {horizon} & {SHORT[model]} & {r.mean_difference:+.5f} & "
                            f"{r.dm:.3f} & {pvalue(r.p_value)} & {pvalue(b.loc[9,'p_value'])} & "
                            f"{pvalue(r.holm_p_value)}"+r" \\")
    write_table(destination/"dm_tests.tex", "lrlrrrrr",
                r"Panel & $h$ & Model & $\Delta$ & DM & $p_4$ & $p_9$ & Holm $p_4$", rows)
    rows = []
    for panel in PANELS:
        rows.append(r"\multicolumn{8}{l}{\textit{" + PANELS[panel] + r"}} \\")
        for horizon in [1, 5, 21]:
            for method in SHORT:
                b=mz[(mz.panel==panel)&(mz.horizon==horizon)&(mz.method==method)].set_index("hac_lags")
                r=b.loc[4]
                rows.append(f"{SHORT[method]} & {horizon} & {r.alpha:.3f} ({r.se_alpha:.3f}) & "
                            f"{r.beta:.3f} ({r.se_beta:.3f}) & {r.r_squared:.3f} & "
                            f"{pvalue(r.p_value)} & {pvalue(b.loc[9,'p_value'])} & {pvalue(r.holm_p_value)}"+r" \\")
    write_table(destination/"calibration_tests.tex", "lrrrrrrr",
                r"Model & $h$ & $\widehat\alpha$ (SE) & $\widehat\beta$ (SE) & $R^2$ & $p_4$ & $p_9$ & Holm $p_4$",rows)
    wins=pd.read_csv(PAPER/"diagnostics_asset_wins.csv")
    rows=[]
    for panel,assets in [("index",["SP500"]),("sectors",paths.SECTORS),("international",paths.COUNTRIES)]:
        rows.append(r"\multicolumn{7}{l}{\textit{"+PANELS[panel]+r"}} \\")
        for asset in assets:
            a=wins[(wins.panel==panel)&(wins.asset==asset)].set_index(["model","horizon"])
            values=[str(int(a.loc[(m,h),"wins"])) for m in ["persistent_asymmetric","equal_combination"] for h in [1,5,21]]
            rows.append(asset+" & "+" & ".join(values)+r" \\")
    head=r" & \multicolumn{3}{c}{Fractional wins} & \multicolumn{3}{c}{Ensemble wins} \\"+"\n"+r"Asset & 1 & 5 & 21 & 1 & 5 & 21"
    write_table(destination/"asset_win_counts.tex","lrrrrrr",head,rows)
    # This summary reports existing paired comparisons, not selected p-values.
    rows=[
        r"Persistence, asymmetry and horizon selection & Lower mean loss in 6/6 US cases & Joint refinement; development data \\",
        r"Fixed ensemble vs.\ fractional / FIGARCH & Lower mean loss in 9/9 cases for each & Improvement relative to each reference \\",
        r"Fixed ensemble vs.\ GJR & Lower mean loss in 3/9 cases & All paired intervals include zero \\",
        r"Diffusion on country ETFs & No-graph variant better at all 3 horizons & No general diffusion benefit \\",
        r"Fractional vs.\ weighted Network GARCH & Lower loss in countries; higher in sectors & All 6 paired intervals include zero \\",
        r"Country one-session calibration & All 3 models reject the joint calibration null & Post-hoc HAC/Wald diagnostic \\",
    ]
    write_table(destination/"findings.tex",r">{\raggedright\arraybackslash}p{.30\textwidth}"*3,
                "Question & Recorded finding & Interpretation",rows)


def kernel_graph(destination):
    fig,axes=plt.subplots(1,3,figsize=(10.8,3.2),layout="constrained")
    for i,d in enumerate((.2,.4,.6,.8)):
        axes[0].loglog(np.arange(1,253),fractional_weights(d),color=OI[i],label=fr"$d={d}$")
    axes[0].set(title="Finite fractional kernel",xlabel="Lag (sessions)",ylabel="Normalized weight")
    axes[0].legend(frameon=False,fontsize=8)
    p=fractional_weights(.4)
    tail = axes[1].inset_axes([.54, .62, .43, .29])
    for i,b in enumerate((0,.5,.9)):
        effective=np.zeros(504)
        for j in range(504):
            effective[j]=(1-b)*(p[j] if j<252 else 0)+(b*effective[j-1] if j else 0)
        positive=effective>0
        axes[1].loglog(np.arange(1,253),effective[:252],color=OI[i],label=fr"$b={b}$")
        tail.semilogy(np.arange(1,505)[positive],effective[positive],color=OI[i],lw=.9)
    tail.axvline(252,color="#000000",ls=":",lw=.7)
    tail.set(title="Tail through lag 504", xticks=[1,252,504], yticks=[1e-3,1e-40,1e-80])
    tail.set_title("Tail through lag 504",fontsize=6)
    tail.tick_params(labelsize=5,pad=1)
    axes[1].set(title=r"Effective memory ($d=0.4$)",xlabel="Lag (sessions)",ylabel=r"$\widetilde p_j$")
    axes[1].legend(frameon=False,fontsize=8,loc="lower left")
    audit=json.loads(Path("artifacts/paper-study/fit_audit.json").read_text())
    graph=-np.asarray(audit["sectors"][-1]["graph"])
    np.fill_diagonal(graph,0)
    cmap=LinearSegmentedColormap.from_list("okabe_blue",["#FFFFFF","#56B4E9","#0072B2"])
    im=axes[2].imshow(graph,cmap=cmap,vmin=0)
    axes[2].set(title="Sector graph at final refit",
                xticks=range(11),xticklabels=paths.SECTORS,yticks=range(11),yticklabels=paths.SECTORS)
    axes[2].tick_params(axis="x",rotation=90,labelsize=6)
    axes[2].tick_params(axis="y",labelsize=6)
    fig.colorbar(im,ax=axes[2],shrink=.8,label="Rescaled edge weight")
    paths.save(fig,destination,"kernel_graph")


def initial_paths(forecasts,destination):
    data=forecasts[(forecasts.panel=="index")&(forecasts.horizon==21)]
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    specs=[("fractional_diffusion","Fractional + graph",METHOD_COLOR["fractional_diffusion"],"-"),
           ("ewma","EWMA",OI[1],"--"),("garch","GARCH",METHOD_COLOR["garch"],"-.")]
    for col,ax in enumerate(axes):
        block=data if not col else data[data.origin<"2025-01-01"]
        observed=block[block.method=="fractional_diffusion"]
        ax.plot(observed.origin,100*np.sqrt(252*observed.realized_proxy/21),color="#000000",
                lw=1.2,alpha=.75,label="Observed proxy")
        for method,label,color,style in specs:
            b=block[block.method==method]
            ax.plot(b.origin,100*np.sqrt(252*b.variance/21),color=color,ls=style,lw=1.1,label=label)
        ax.set(title="2023–2024 detail" if col else "Full evaluation period",
               xlabel="Forecast origin",ylabel="Annualized volatility (%)")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if col: ax.set_xlim(pd.Timestamp("2023-01-01"),pd.Timestamp("2024-12-31"))
        ax.grid(axis="y",alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=4,frameon=False,fontsize=8)
    fig.subplots_adjust(top=.79,bottom=.17,left=.08,right=.98,wspace=.25)
    paths.save(fig,destination,"variance_paths")


def score_lines(metrics,panels,methods,name,destination):
    fig,axes=plt.subplots(1,len(panels),figsize=(9,3.4),squeeze=False)
    styles=["-","--","-.",":"]
    for ax,panel in zip(axes.flat,panels):
        a=metrics[(metrics.year=="all")&(metrics.panel==panel)].pivot(index="horizon",columns="method",values="qlike")
        for i,method in enumerate(methods):
            ax.plot(range(3),a.loc[[1,5,21],method],color=METHOD_COLOR[method],ls=styles[i%4],
                    marker=["o","s","^","D"][i%4],ms=4,label=LABELS[method])
        ax.set(xticks=range(3),xticklabels=[1,5,21],xlabel="Horizon (sessions)",title=PANELS[panel])
        ax.grid(axis="y",alpha=.2)
        ax.margins(y=.15)
    axes[0,0].set_ylabel("Mean normalized QLIKE")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc="upper center",frameon=False,fontsize=8)
    fig.subplots_adjust(top=.72,bottom=.16,left=.08,right=.99,wspace=.30)
    paths.save(fig,destination,name)


def win_figure(destination):
    w=pd.read_csv(PAPER/"diagnostics_panel_wins.csv")
    fig,axes=plt.subplots(1,3,figsize=(9,3.5))
    for ax,panel in zip(axes,PANELS):
        block=w[w.panel==panel].set_index(["model","horizon"])
        for i,m in enumerate(["persistent_asymmetric","equal_combination"]):
            values=[block.loc[(m,h),"asset_wins"] for h in [1,5,21]]
            bars=ax.bar(np.arange(3)+(i-.5)*.32,values,.32,color=OI[i],label=SHORT[m],hatch="" if i==0 else "//")
            ax.bar_label(bars,padding=3,fontsize=8)
        total=int(block.assets.iloc[0])
        ax.set(title=f"{PANELS[panel]} ({total} {'asset' if total == 1 else 'assets'})",xticks=range(3),xticklabels=[1,5,21],
               xlabel="Horizon (sessions)",ylim=(0,total+max(.5,total*.16)))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("Assets with lower mean QLIKE than GJR")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc="upper center",frameon=False)
    fig.subplots_adjust(top=.78,bottom=.16,left=.08,right=.99,wspace=.28)
    paths.save(fig,destination,"asset_win_counts")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=PAPER)
    args=parser.parse_args()
    destination=args.output/"figures"
    table_dir=args.output/"tables"
    destination.mkdir(parents=True,exist_ok=True)
    table_dir.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv("artifacts/external-study-complete/forecasts.csv",parse_dates=["origin","target","last_fit"],float_precision="round_trip")
    original=pd.read_csv("artifacts/paper-study/forecasts.csv",parse_dates=["origin","target","last_fit"],float_precision="round_trip")
    initial=pd.read_csv(PAPER/"metrics.csv")
    external=pd.read_csv(PAPER/"external_metrics.csv")
    actual=f.groupby(["panel","horizon","method"]).qlike.mean().sort_index()
    expected=external[external.year=="all"].set_index(["panel","horizon","method"]).qlike.sort_index()
    pd.testing.assert_index_equal(actual.index,expected.index)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    tables(table_dir)
    kernel_graph(destination)
    initial_paths(original,destination)
    paths.initial_comparison(initial,destination)
    paths.index_paths(f,destination)
    paths.panel_paths(f,destination)
    paths.asset_paths(f,"sectors",paths.SECTORS[:6],"prediction_sectors_a",destination)
    paths.asset_paths(f,"sectors",paths.SECTORS[6:],"prediction_sectors_b",destination)
    paths.asset_paths(f,"international",paths.COUNTRIES,"prediction_countries",destination)
    score_lines(pd.read_csv(PAPER/"refinement_metrics.csv"),["index","sectors"],
                ["fractional_diffusion","persistent_asymmetric","garch","gjr_garch"],"refinement_comparison",destination)
    score_lines(external,list(PANELS),["persistent_asymmetric","equal_combination","gjr_garch","figarch"],
                "external_comparison",destination)
    win_figure(destination)
    inputs=[Path("artifacts/external-study-complete/forecasts.csv"),Path("artifacts/paper-study/forecasts.csv"),
            Path("artifacts/paper-study/fit_audit.json"),*sorted(PAPER.glob("*.csv"))]
    outputs=[destination/f"{n}.{suffix}" for n in FIGURE_OUTPUTS for suffix in ("pdf","png")]
    outputs += [table_dir/f"{n}.tex" for n in TABLE_OUTPUTS]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    record=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                scope="Presentation and saved-diagnostic tables only; no model fit or forecast update.",
                input_sha256={str(p):sha(p) for p in inputs},
                source_sha256={p.name:sha(p) for p in [Path(__file__),Path(paths.__file__)]},
                output_sha256={str(p.relative_to(args.output)):sha(p) for p in outputs},
                palette="Okabe-Ito; white and light gray reserved for background/grid.",
                log_axes="Every asset path in Figures 8-10; strictly positive inputs checked, no floor or clipping.",
                iqr="Cross-asset 25th/75th percentiles of each displayed volatility series; not confidence intervals. Central curve is RMS marginal volatility.",
                zoom="Figures 3 and 6 retain full period and separately display 2023-2024.",
                effective_memory="Illustrative d=0.4, M=252, b=0/0.5/0.9; main panel through lag 252 and inset through lag 504, exact recurrence.",
                wins="Counts of assets with lower full-period mean QLIKE than GJR, not independent successes.")
    (args.output/"submission_assets_manifest.json").write_text(json.dumps(record,indent=2)+"\n")
    prediction_names = ["forecast_comparison", "prediction_index_horizons",
                        "prediction_panel_evolution", "prediction_sectors_a",
                        "prediction_sectors_b", "prediction_countries"]
    subset = dict(record)
    subset["scope"] = "Prediction-figure subset of submission_assets_manifest.json."
    subset["outputs_sha256"] = {name: value for name, value in record["output_sha256"].items()
                                if Path(name).stem in prediction_names}
    del subset["output_sha256"]
    subset["time_axis"] = "Target-window end; forecasts issued h observed sessions earlier."
    subset["transformation"] = "100*sqrt(252*variance/h); panel variances averaged before square root."
    subset["scope_of_paths"] = dict(index_horizons=[1, 5, 21], panel_horizons=[5, 21],
                                    asset_horizon=21, sectors=paths.SECTORS, countries=paths.COUNTRIES)
    (args.output/"prediction_figures_manifest.json").write_text(json.dumps(subset,indent=2)+"\n")
    print(f"Rendered {len(FIGURE_OUTPUTS)} figures and {len(TABLE_OUTPUTS)} tables from saved records.")


if __name__ == "__main__":
    main()
