"""
model_evaluation_and_viz.py
============================
Reads the fresh_inference_*.json output (plus per-symbol fresh_eval CSVs)
and produces a comprehensive, publication-quality PDF report:

  Section 0 – Cover / exec-summary page
  Section 1 – Training-loss summary (final / val / best) from metadata
  Section 2 – Per-metric comparison bar charts (all symbols)
  Section 3 – Win heatmap (LSTM vs Transformer per metric × symbol)
  Section 4 – Aggregate summary + win-share pie + MAPE violin
  Section 5 – Stocks vs Indices sub-group breakdown
  Section 6 – Absolute-error distribution histograms
  Section 7 – Full metrics table (all symbols, all metrics)
  Section 8 – Per-symbol actual vs predicted curves

Run from:
  d:\\major-project\\backend\\stockproject\\predictor\\
  python model_evaluation_and_viz.py              # auto-finds latest fresh_inference JSON
  python model_evaluation_and_viz.py  <json_path> # explicit path
"""

import os
import sys
import json
import glob
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
VIZ_DIR   = os.path.join(BASE_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

LSTM_COLOR        = "#4FC3F7"
TR_COLOR          = "#FF8A65"
NAIVE_COLOR       = "#A5D6A7"
ACTUAL_COLOR      = "#EEEEEE"
BG_COLOR          = "#0D1117"
GRID_COLOR        = "#21262D"
TEXT_COLOR        = "#E6EDF3"
ACCENT_COLOR      = "#58A6FF"
TIE_COLOR         = "#90A4AE"

METRIC_META = {
    "mae":         ("MAE (₹)",            False),   # (label, higher-is-better)
    "rmse":        ("RMSE (₹)",           False),
    "mape":        ("MAPE %",             False),
    "r2":          ("R²",                 True),
    "dir_acc":     ("Dir. Accuracy %",    True),
}

STOCK_SYMS = [
    "RELIANCE.NS","AXISBANK.NS","HDFCBANK.NS","ONGC.NS","SBIN.NS",
    "INFY.NS","TCS.NS","ICICIBANK.NS","KOTAKBANK.NS","ADANIPORTS.NS",
    "ADANIENT.NS","BAJFINANCE.NS","BHARTIARTL.NS",
]
INDEX_SYMS = ["^NSEI","^NSEBANK","^NSEMDCP50","^CNXAUTO"]

def sk(sym):  return sym.replace(".", "_").replace("^", "INDEX_")


# ──────────────────────────────────────────────────────────────
# Style helpers
# ──────────────────────────────────────────────────────────────
def dark_ax(ax, title="", xlabel="", ylabel="", grid=True):
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, which="both")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    if title:   ax.set_title(title,   color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=8)
    if xlabel:  ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=9)
    if ylabel:  ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=9)
    if grid:    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.8)


def new_fig(figsize=(14, 7), title=""):
    fig = plt.figure(figsize=figsize, facecolor=BG_COLOR)
    if title:
        fig.suptitle(title, color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.98)
    return fig


def legend(ax, **kw):
    leg = ax.legend(facecolor="#1C2128", edgecolor=GRID_COLOR,
                    labelcolor=TEXT_COLOR, fontsize=8, **kw)
    return leg


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────
def find_latest_inference_json():
    pattern = os.path.join(MODEL_DIR, "fresh_inference_*.json")
    files   = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_inference(path):
    with open(path) as f:
        return json.load(f)


def load_fresh_eval_csv(symbol, asset_type):
    path = os.path.join(MODEL_DIR, f"fresh_eval_{asset_type}_{sk(symbol)}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"])


def load_metadata_for(symbol, model_type, asset_type):
    path = os.path.join(MODEL_DIR, f"metadata_{model_type}_{asset_type}_{sk(symbol)}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def to_metrics_df(report):
    """Flatten fresh inference JSON into a long-form metrics DataFrame."""
    rows = []
    for r in report:
        if r.get("status") != "ok":
            continue
        sym  = r["symbol"]
        atyp = r.get("asset_type", "stocks")
        for model in ("lstm", "transformer"):
            row = {"symbol": sym, "asset_type": atyp, "model": model}
            for m in ("mae","rmse","mape","r2","dir_acc","dir_acc_nonflat"):
                row[m] = r.get(f"{model}_{m}", np.nan)
            for m in ("mae","rmse","mape","dir_acc"):
                row[f"naive_{m}"] = r.get(f"naive_{m}", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# COVER PAGE
# ──────────────────────────────────────────────────────────────
def cover_page(pdf, report, metrics_df):
    ok   = [r for r in report if r.get("status") == "ok"]
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lstm = metrics_df[metrics_df["model"] == "lstm"]
    tr   = metrics_df[metrics_df["model"] == "transformer"]

    fig = new_fig(figsize=(14, 10))
    ax  = fig.add_subplot(111); ax.axis("off"); ax.set_facecolor(BG_COLOR)

    def txt(text, y, color=TEXT_COLOR, size=12, weight="normal"):
        ax.text(0.5, y, text, transform=ax.transAxes,
                ha="center", va="center", color=color, fontsize=size, fontweight=weight)

    txt("LSTM  vs  Transformer", 0.90, ACCENT_COLOR, 30, "bold")
    txt("Stock Price Prediction — Fair Comparison Report", 0.81, TEXT_COLOR, 16)
    txt(f"Fresh inference run: {ts}", 0.73, "#8B949E", 11)
    txt(f"Symbols evaluated: {len(ok)} / {len(STOCK_SYMS+INDEX_SYMS)}   •   Test split: last 15% (chronological)", 0.65, TEXT_COLOR, 12)

    if not lstm.empty and not tr.empty:
        lm = lstm["mape"].mean();  tm = tr["mape"].mean()
        lr = lstm["r2"].mean();    tr_ = tr["r2"].mean()
        ld = lstm["dir_acc"].mean(); td = tr["dir_acc"].mean()
        txt(f"Avg MAPE   →   LSTM: {lm:.2f}%   |   Transformer: {tm:.2f}%", 0.56, TEXT_COLOR, 12)
        txt(f"Avg R²     →   LSTM: {lr:.3f}    |   Transformer: {tr_:.3f}",  0.49, TEXT_COLOR, 12)
        txt(f"Avg DirAcc →   LSTM: {ld:.1f}%   |   Transformer: {td:.1f}%",  0.42, TEXT_COLOR, 12)

    txt("", 0.36, TEXT_COLOR, 1)
    txt("Architecture — LSTM:", 0.30, LSTM_COLOR, 12, "bold")
    txt("3 × LSTM layers (128 → 64 → 32 units) + BatchNorm + Dropout + Dense head", 0.24, TEXT_COLOR, 10)
    txt("Architecture — Transformer:", 0.18, TR_COLOR, 12, "bold")
    txt("3 × Encoder blocks (Multi-Head Attention 8h×64 + GELU FFN 256) + Positional Embedding", 0.12, TEXT_COLOR, 10)
    txt("+ GlobalAveragePooling + MLP head (128 → 64 → 1)", 0.07, TEXT_COLOR, 10)
    txt("Train/Val/Test: 70%/15%/15% chronological  •  Seq len: 120  •  Seed: 42  •  Period: 5 years", 0.01, "#8B949E", 10)

    pdf.savefig(fig, facecolor=BG_COLOR)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# SECTION 1 – TRAINING LOSS SUMMARY (from saved metadata)
# ──────────────────────────────────────────────────────────────
def training_loss_section(pdf, all_syms):
    print("[VIZ] Training loss summary …")
    records = []
    for sym in all_syms:
        atyp = "indices" if sym.startswith("^") else "stocks"
        for mt in ("lstm", "transformer"):
            meta = load_metadata_for(sym, mt, atyp)
            if meta is None: continue
            th = meta.get("training_history", {})
            records.append({
                "symbol":           sym,
                "model":            mt,
                "final_train_loss": th.get("final_loss",     np.nan),
                "final_val_loss":   th.get("final_val_loss", np.nan),
                "best_val_loss":    th.get("best_val_loss",  np.nan),
            })
    df = pd.DataFrame(records)
    if df.empty: return

    syms  = list(dict.fromkeys([r["symbol"] for r in records]))   # preserve order
    short = [s.replace(".NS","").replace("^","") for s in syms]
    n     = len(syms)
    x     = np.arange(n)
    w     = 0.35

    def bar_vals(model_type, col):
        lk = df[df["model"] == model_type].set_index("symbol")
        return [float(lk.loc[s, col]) if s in lk.index else np.nan for s in syms]

    # ── 1A: best val loss ─────────────────────────────────────
    fig = new_fig(figsize=(18, 7), title="Training Summary — Best Validation Loss per Symbol")
    ax  = fig.add_subplot(111); ax.set_facecolor(BG_COLOR)
    ax.bar(x - w/2, bar_vals("lstm",        "best_val_loss"), w, label="LSTM",
           color=LSTM_COLOR, alpha=0.85, edgecolor=BG_COLOR)
    ax.bar(x + w/2, bar_vals("transformer", "best_val_loss"), w, label="Transformer",
           color=TR_COLOR,   alpha=0.85, edgecolor=BG_COLOR)
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8, color=TEXT_COLOR)
    dark_ax(ax, ylabel="Best Val Loss (MSE on scaled data)")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.4f"))
    legend(ax)
    fig.tight_layout(rect=[0,0,1,0.93]); pdf.savefig(fig, facecolor=BG_COLOR); plt.close(fig)

    # ── 1B: train vs val scatter (overfitting check) ──────────
    fig = new_fig(figsize=(13, 6), title="Overfitting Check — Final Train Loss vs Final Val Loss")
    ax  = fig.add_subplot(111); ax.set_facecolor(BG_COLOR)
    for mt, color, marker in [("lstm", LSTM_COLOR, "o"), ("transformer", TR_COLOR, "s")]:
        sub = df[df["model"] == mt].dropna(subset=["final_train_loss","final_val_loss"])
        ax.scatter(sub["final_train_loss"], sub["final_val_loss"],
                   c=color, marker=marker, s=90, alpha=0.85,
                   edgecolors="white", linewidths=0.4, label=mt.upper(), zorder=4)
        for _, row in sub.iterrows():
            lbl = row["symbol"].replace(".NS","").replace("^","")
            ax.annotate(lbl, (row["final_train_loss"], row["final_val_loss"]),
                        textcoords="offset points", xytext=(4,4),
                        fontsize=6, color=TEXT_COLOR, alpha=0.7)
    lims = [0, max(df["final_train_loss"].max(skipna=True),
                   df["final_val_loss"].max(skipna=True)) * 1.15]
    ax.plot(lims, lims, "--", color=GRID_COLOR, lw=1.0, label="y = x (no overfit)")
    ax.set_xlim(0, lims[1]); ax.set_ylim(0, lims[1])
    dark_ax(ax, xlabel="Final Train Loss", ylabel="Final Val Loss"); legend(ax)
    fig.tight_layout(rect=[0,0,1,0.93]); pdf.savefig(fig, facecolor=BG_COLOR); plt.close(fig)

    # ── 1C: val-train gap (overfitting indicator) ─────────────
    fig = new_fig(figsize=(18, 6), title="Train-Val Loss Gap  (Val − Train)  — Overfitting Indicator")
    ax  = fig.add_subplot(111); ax.set_facecolor(BG_COLOR)
    for mt, color, offset in [("lstm", LSTM_COLOR, -w/2), ("transformer", TR_COLOR, w/2)]:
        gaps = []
        sub  = df[df["model"] == mt].set_index("symbol")
        for s in syms:
            try:    gaps.append(sub.loc[s,"final_val_loss"] - sub.loc[s,"final_train_loss"])
            except: gaps.append(0.0)
        bar_colors = [color if g >= 0 else "#EF5350" for g in gaps]
        ax.bar(x + offset, gaps, w, color=bar_colors, alpha=0.85,
               edgecolor=BG_COLOR, label=mt.upper())
    ax.axhline(0, color=TEXT_COLOR, lw=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8, color=TEXT_COLOR)
    dark_ax(ax, ylabel="Val − Train Loss  (red = >0 → overfitting)"); legend(ax)
    fig.tight_layout(rect=[0,0,1,0.93]); pdf.savefig(fig, facecolor=BG_COLOR); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# SECTION 2 – METRIC COMPARISON BARS per symbol
# ──────────────────────────────────────────────────────────────
def metric_comparison_bars(pdf, metrics_df):
    print("[VIZ] Per-metric comparison bars …")
    syms  = list(dict.fromkeys(metrics_df["symbol"]))
    short = [s.replace(".NS","").replace("^","") for s in syms]
    n     = len(syms)
    x     = np.arange(n)
    w     = 0.28

    for mkey, (label, hib) in METRIC_META.items():
        lstm_v  = []; tr_v = []; naive_v = []
        for s in syms:
            sub = metrics_df[metrics_df["symbol"] == s]
            lv = sub[sub["model"]=="lstm"][mkey].values
            tv = sub[sub["model"]=="transformer"][mkey].values
            nk = f"naive_{mkey}"
            nv = sub[sub["model"]=="lstm"][nk].values if nk in sub.columns else [np.nan]
            lstm_v.append(float(lv[0]) if len(lv) else np.nan)
            tr_v.append(  float(tv[0]) if len(tv) else np.nan)
            naive_v.append(float(nv[0]) if len(nv) else np.nan)

        fig = new_fig(figsize=(18, 6), title=f"Model Comparison — {label}")
        ax  = fig.add_subplot(111); ax.set_facecolor(BG_COLOR)
        ax.bar(x - w,   lstm_v,  w, color=LSTM_COLOR,  alpha=0.85, edgecolor=BG_COLOR, label="LSTM")
        ax.bar(x,       tr_v,    w, color=TR_COLOR,    alpha=0.85, edgecolor=BG_COLOR, label="Transformer")
        has_naive = not all(np.isnan(v) for v in naive_v)
        if has_naive:
            ax.bar(x + w, naive_v, w, color=NAIVE_COLOR, alpha=0.6,  edgecolor=BG_COLOR, label="Naïve baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8, color=TEXT_COLOR)
        dark_ax(ax, ylabel=label)
        legend(ax)
        fig.tight_layout(rect=[0,0,1,0.93]); pdf.savefig(fig, facecolor=BG_COLOR); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# SECTION 3 – WIN HEATMAP
# ──────────────────────────────────────────────────────────────
def win_heatmap(pdf, report):
    print("[VIZ] Win heatmap …")
    ok_rows = [r for r in report if r.get("status") == "ok"
               and "mae_winner" in r]
    if not ok_rows: return
    syms = [r["symbol"] for r in ok_rows]
    metric_cols = ["mae_winner","rmse_winner","mape_winner","r2_winner","dir_acc_winner"]
    headers     = ["MAE\n(↓)", "RMSE\n(↓)", "MAPE\n(↓)", "R²\n(↑)", "DirAcc\n(↑)"]

    mat = np.full((len(syms), 5), 0.5)
    for i, row in enumerate(ok_rows):
        for j, col in enumerate(metric_cols):
            w = row.get(col, "")
            mat[i,j] = 0.0 if w == "lstm" else (1.0 if w == "transformer" else 0.5)

    cmap = LinearSegmentedColormap.from_list("ltr", [LSTM_COLOR, TIE_COLOR, TR_COLOR])
    fig  = new_fig(figsize=(11, max(6, len(syms)*0.55+2)),
                   title="Win Heatmap — LSTM vs Transformer")
    ax   = fig.add_subplot(111); ax.set_facecolor(BG_COLOR)
    ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(headers, color=TEXT_COLOR, fontsize=9)
    ax.set_yticks(range(len(syms)))
    ax.set_yticklabels([s.replace(".NS","").replace("^","") for s in syms],
                       color=TEXT_COLOR, fontsize=9)
    for i in range(len(syms)):
        for j in range(5):
            v   = mat[i,j]
            txt = "LSTM" if v < 0.4 else ("Transf." if v > 0.6 else "Tie")
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="white")
    for sp in ax.spines.values(): sp.set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    patches = [mpatches.Patch(color=LSTM_COLOR, label="LSTM wins"),
               mpatches.Patch(color=TIE_COLOR,  label="Tie"),
               mpatches.Patch(color=TR_COLOR,    label="Transformer wins")]
    ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.35,1.0),
              facecolor="#1C2128", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    fig.tight_layout(rect=[0,0,0.87,0.93])
    pdf.savefig(fig, facecolor=BG_COLOR, bbox_inches="tight"); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# SECTION 4 – AGGREGATE SUMMARY
# ──────────────────────────────────────────────────────────────
def aggregate_summary(pdf, metrics_df, report):
    print("[VIZ] Aggregate summary …")
    lstm_df = metrics_df[metrics_df["model"]=="lstm"]
    tr_df   = metrics_df[metrics_df["model"]=="transformer"]
    common  = set(lstm_df["symbol"]) & set(tr_df["symbol"])
    lc = lstm_df[lstm_df["symbol"].isin(common)]
    tc = tr_df[  tr_df["symbol"].isin(common)]

    fig = new_fig(figsize=(16, 10), title="Aggregate Performance Summary")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # ── average bars ──
    ax_avg = fig.add_subplot(gs[0,:2]); ax_avg.set_facecolor(BG_COLOR)
    mkeys  = list(METRIC_META.keys())
    labels = [METRIC_META[k][0] for k in mkeys]
    x = np.arange(len(mkeys)); w = 0.3
    lavg = [lc[k].mean() for k in mkeys]
    tavg = [tc[k].mean() for k in mkeys]
    ax_avg.bar(x-w/2, lavg, w, color=LSTM_COLOR, alpha=0.85, label="LSTM (avg)")
    ax_avg.bar(x+w/2, tavg, w, color=TR_COLOR,   alpha=0.85, label="Transformer (avg)")
    for xi, (lv, tv, mk) in enumerate(zip(lavg, tavg, mkeys)):
        hib   = METRIC_META[mk][1]
        better = "LSTM" if (lv > tv if hib else lv < tv) else "TF"
        col    = LSTM_COLOR if better == "LSTM" else TR_COLOR
        ax_avg.annotate(f"{better}✓", xy=(xi, max(abs(lv),abs(tv))),
                        xytext=(0,5), textcoords="offset points",
                        ha="center", fontsize=7, color=col, fontweight="bold")
    ax_avg.set_xticks(x); ax_avg.set_xticklabels(labels, color=TEXT_COLOR, fontsize=8)
    dark_ax(ax_avg, ylabel="Mean across all symbols"); legend(ax_avg)

    # ── win-share pie ──
    if report:
        ax_pie = fig.add_subplot(gs[0,2]); ax_pie.set_facecolor(BG_COLOR)
        lw = tw = 0
        cols = ["mae_winner","rmse_winner","mape_winner","r2_winner","dir_acc_winner"]
        for row in report:
            if row.get("status") != "ok": continue
            for c in cols:
                w_ = row.get(c,"")
                if   w_ == "lstm":        lw += 1
                elif w_ == "transformer": tw += 1
        sizes   = [lw, tw]
        colors  = [LSTM_COLOR, TR_COLOR]
        explode = (0.05, 0.05)
        wedges, texts, autotexts = ax_pie.pie(
            sizes, labels=["LSTM","Transformer"], colors=colors,
            autopct="%1.0f%%", startangle=90, explode=explode,
            textprops={"color":TEXT_COLOR,"fontsize":9})
        for at in autotexts: at.set_fontsize(9); at.set_color("white")
        ax_pie.set_title("Win Share\n(all metrics × symbols)",
                         color=TEXT_COLOR, fontsize=9, fontweight="bold")

    # ── MAPE violin ──
    ax_vio = fig.add_subplot(gs[1,:]); ax_vio.set_facecolor(BG_COLOR)
    lstm_mape = lc["mape"].dropna().values
    tr_mape   = tc["mape"].dropna().values
    vp = ax_vio.violinplot([lstm_mape, tr_mape], showmedians=True,
                            positions=[1,2], widths=0.6)
    for body, c in zip(vp["bodies"], [LSTM_COLOR, TR_COLOR]):
        body.set_facecolor(c); body.set_alpha(0.7)
    for part in ("cmedians","cmins","cmaxes","cbars"):
        vp[part].set_color(TEXT_COLOR)
    rng = np.random.default_rng(42)
    ax_vio.scatter(rng.uniform(0.82,1.18,len(lstm_mape)), lstm_mape,
                   c=LSTM_COLOR, alpha=0.65, s=40, zorder=4)
    ax_vio.scatter(rng.uniform(1.82,2.18,len(tr_mape)), tr_mape,
                   c=TR_COLOR,   alpha=0.65, s=40, zorder=4)
    ax_vio.set_xticks([1,2])
    ax_vio.set_xticklabels(["LSTM","Transformer"], color=TEXT_COLOR, fontsize=10)
    dark_ax(ax_vio, title="MAPE % Distribution (Test Split)", ylabel="MAPE (%)")

    fig.patch.set_facecolor(BG_COLOR)
    pdf.savefig(fig, facecolor=BG_COLOR); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# SECTION 5 – STOCKS vs INDICES SUB-GROUP
# ──────────────────────────────────────────────────────────────
def stocks_vs_indices(pdf, metrics_df):
    print("[VIZ] Stocks vs Indices breakdown …")
    mkeys  = list(METRIC_META.keys())
    labels = [METRIC_META[k][0] for k in mkeys]
    x = np.arange(len(mkeys)); w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(18,6), facecolor=BG_COLOR)
    fig.suptitle("Average Metrics — Stocks vs Indices",
                 color=TEXT_COLOR, fontsize=13, fontweight="bold")

    for ax, group in zip(axes, ["stocks","indices"]):
        sub = metrics_df[metrics_df["asset_type"]==group]
        lc  = sub[sub["model"]=="lstm"]
        tc  = sub[sub["model"]=="transformer"]
        ax.set_facecolor(BG_COLOR)
        ax.bar(x-w/2, [lc[k].mean() for k in mkeys], w,
               color=LSTM_COLOR, alpha=0.85, label="LSTM")
        ax.bar(x+w/2, [tc[k].mean() for k in mkeys], w,
               color=TR_COLOR,   alpha=0.85, label="Transformer")
        ax.set_xticks(x); ax.set_xticklabels(labels, color=TEXT_COLOR, fontsize=8)
        dark_ax(ax, title=group.capitalize(), ylabel="Average Metric Value")
        legend(ax)

    plt.tight_layout(rect=[0,0,1,0.93])
    pdf.savefig(fig, facecolor=BG_COLOR); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# SECTION 6 – ERROR DISTRIBUTION HISTOGRAMS
# ──────────────────────────────────────────────────────────────
def error_distributions(pdf, all_syms):
    print("[VIZ] Error distribution histograms …")
    lstm_pct = []; tr_pct = []
    for sym in all_syms:
        atyp = "indices" if sym.startswith("^") else "stocks"
        df = load_fresh_eval_csv(sym, atyp)
        if df is None: continue
        for col, store in [("lstm_abs_err", lstm_pct), ("transformer_abs_err", tr_pct)]:
            if col in df.columns and "actual" in df.columns:
                pct = (df[col] / df["actual"].replace(0,np.nan) * 100).dropna()
                store.extend(pct.tolist())

    fig = new_fig(figsize=(14,6), title="Absolute % Error Distribution Across All Symbols (Test Split)")
    ax  = fig.add_subplot(111); ax.set_facecolor(BG_COLOR)
    bins = np.linspace(0, 30, 60)
    ax.hist(lstm_pct, bins=bins, color=LSTM_COLOR, alpha=0.65, density=True, edgecolor="none", label="LSTM")
    ax.hist(tr_pct,   bins=bins, color=TR_COLOR,   alpha=0.65, density=True, edgecolor="none", label="Transformer")
    for data, color, lbl in [(lstm_pct, LSTM_COLOR, "LSTM"), (tr_pct, TR_COLOR, "Transformer")]:
        if data:
            med = np.median(data)
            ax.axvline(med, color=color, lw=1.5, linestyle="--",
                       label=f"{lbl} median={med:.1f}%")
    dark_ax(ax, xlabel="Abs % Error (%)", ylabel="Density"); legend(ax)
    fig.tight_layout(rect=[0,0,1,0.93]); pdf.savefig(fig, facecolor=BG_COLOR); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# SECTION 7 – FULL METRICS TABLE
# ──────────────────────────────────────────────────────────────
def metrics_table(pdf, report):
    print("[VIZ] Metrics table …")
    ok  = [r for r in report if r.get("status") == "ok"]
    if not ok: return

    col_labels = [
        "Symbol",
        "MAE\nLSTM","MAE\nTransf.",
        "RMSE\nLSTM","RMSE\nTransf.",
        "MAPE\nLSTM","MAPE\nTransf.",
        "R²\nLSTM","R²\nTransf.",
        "DirAcc\nLSTM","DirAcc\nTransf.",
        "Naive\nMAE",
    ]
    rows = []
    for r in ok:
        sym = r["symbol"].replace(".NS","").replace("^","")
        rows.append([
            sym,
            f"{r.get('lstm_mae',0):.1f}", f"{r.get('transformer_mae',0):.1f}",
            f"{r.get('lstm_rmse',0):.1f}", f"{r.get('transformer_rmse',0):.1f}",
            f"{r.get('lstm_mape',0):.2f}%", f"{r.get('transformer_mape',0):.2f}%",
            f"{r.get('lstm_r2',0):.3f}",  f"{r.get('transformer_r2',0):.3f}",
            f"{r.get('lstm_dir_acc',0):.1f}%", f"{r.get('transformer_dir_acc',0):.1f}%",
            f"{r.get('naive_mae',0):.1f}",
        ])

    fig = new_fig(figsize=(22, max(5, len(rows)*0.5+2)),
                  title="Full Metrics Table — Fresh Inference (Test Split)")
    ax  = fig.add_subplot(111); ax.axis("off"); ax.set_facecolor(BG_COLOR)
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="upper center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.55)
    for (i,j), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor("#1C2128")
            cell.set_text_props(color=ACCENT_COLOR, fontweight="bold")
        elif i % 2 == 0:
            cell.set_facecolor("#161B22"); cell.set_text_props(color=TEXT_COLOR)
        else:
            cell.set_facecolor("#0D1117");  cell.set_text_props(color=TEXT_COLOR)
        cell.set_edgecolor(GRID_COLOR)
    fig.tight_layout(rect=[0,0,1,0.93])
    pdf.savefig(fig, facecolor=BG_COLOR, bbox_inches="tight"); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# SECTION 8 – PER-SYMBOL PREDICTION CURVES
# ──────────────────────────────────────────────────────────────
def per_symbol_curves(pdf, all_syms):
    print("[VIZ] Per-symbol prediction curves …")
    for sym in all_syms:
        atyp = "indices" if sym.startswith("^") else "stocks"
        df   = load_fresh_eval_csv(sym, atyp)
        if df is None: continue

        fig = new_fig(figsize=(16, 6), title=f"Test-Split Predictions — {sym}")
        ax  = fig.add_subplot(111); ax.set_facecolor(BG_COLOR)

        if "actual" in df.columns:
            ax.plot(df["date"], df["actual"], color=ACTUAL_COLOR, lw=1.5,
                    label="Actual", zorder=3)

        if "lstm_pred" in df.columns and df["lstm_pred"].notna().any():
            ax.plot(df["date"], df["lstm_pred"], color=LSTM_COLOR, lw=1.2,
                    alpha=0.85, label="LSTM")
            ax.fill_between(df["date"], df["actual"], df["lstm_pred"],
                            alpha=0.07, color=LSTM_COLOR)

        if "transformer_pred" in df.columns and df["transformer_pred"].notna().any():
            ax.plot(df["date"], df["transformer_pred"], color=TR_COLOR, lw=1.2,
                    alpha=0.85, label="Transformer", linestyle="--")
            ax.fill_between(df["date"], df["actual"], df["transformer_pred"],
                            alpha=0.05, color=TR_COLOR)

        if "naive" in df.columns:
            ax.plot(df["date"], df["naive"], color=NAIVE_COLOR, lw=0.8,
                    alpha=0.45, label="Naïve (prev-close)", linestyle=":")

        dark_ax(ax, xlabel="Date", ylabel="Close Price (₹)")
        ax.tick_params(axis="x", rotation=28)
        legend(ax, loc="upper left")
        fig.tight_layout(rect=[0,0,1,0.93])
        pdf.savefig(fig, facecolor=BG_COLOR); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# CSV EXPORT
# ──────────────────────────────────────────────────────────────
def export_summary_csv(report, path):
    rows = []
    for r in report:
        if r.get("status") != "ok": continue
        row = {
            "symbol":         r["symbol"],
            "asset_type":     r.get("asset_type",""),
            "lstm_mae":       r.get("lstm_mae",   np.nan),
            "lstm_rmse":      r.get("lstm_rmse",  np.nan),
            "lstm_mape":      r.get("lstm_mape",  np.nan),
            "lstm_r2":        r.get("lstm_r2",    np.nan),
            "lstm_dir_acc":   r.get("lstm_dir_acc",np.nan),
            "tr_mae":         r.get("transformer_mae",   np.nan),
            "tr_rmse":        r.get("transformer_rmse",  np.nan),
            "tr_mape":        r.get("transformer_mape",  np.nan),
            "tr_r2":          r.get("transformer_r2",    np.nan),
            "tr_dir_acc":     r.get("transformer_dir_acc",np.nan),
            "naive_mae":      r.get("naive_mae",  np.nan),
            "naive_mape":     r.get("naive_mape", np.nan),
            "mae_winner":     r.get("mae_winner",""),
            "rmse_winner":    r.get("rmse_winner",""),
            "mape_winner":    r.get("mape_winner",""),
            "r2_winner":      r.get("r2_winner",""),
            "dir_acc_winner": r.get("dir_acc_winner",""),
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    # Accept explicit JSON path or auto-find latest
    json_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_inference_json()
    if json_path is None:
        print("[ERROR] No fresh_inference_*.json found. Run fresh_inference.py first.")
        sys.exit(1)

    print("=" * 65)
    print("  Model Evaluation & Visualization Pipeline")
    print(f"  Source: {json_path}")
    print("=" * 65)

    report     = load_inference(json_path)
    metrics_df = to_metrics_df(report)
    all_syms   = STOCK_SYMS + INDEX_SYMS

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_pdf = os.path.join(VIZ_DIR, f"model_comparison_report_{ts}.pdf")
    out_csv = os.path.join(VIZ_DIR, f"metrics_summary_{ts}.csv")

    print(f"[INFO] {len([r for r in report if r.get('status')=='ok'])} symbols with OK status")
    print(f"[INFO] Output PDF → {out_pdf}")

    with PdfPages(out_pdf) as pdf:
        d = pdf.infodict()
        d["Title"]  = "LSTM vs Transformer — Fresh Inference Comparison"
        d["Author"] = "model_evaluation_and_viz.py"

        cover_page(pdf, report, metrics_df)
        training_loss_section(pdf, all_syms)
        metric_comparison_bars(pdf, metrics_df)
        win_heatmap(pdf, report)
        aggregate_summary(pdf, metrics_df, report)
        stocks_vs_indices(pdf, metrics_df)
        error_distributions(pdf, all_syms)
        metrics_table(pdf, report)
        per_symbol_curves(pdf, all_syms)

    export_summary_csv(report, out_csv)
    print(f"\n[✅] Report PDF → {out_pdf}")
    print(f"[✅] Metrics CSV → {out_csv}")


if __name__ == "__main__":
    main()
