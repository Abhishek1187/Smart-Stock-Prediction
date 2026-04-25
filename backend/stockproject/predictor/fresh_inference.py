"""
fresh_inference.py
==================
Runs BOTH saved LSTM and Transformer models on freshly-fetched stock data.
Uses the SAME preprocessing / split / scaler logic as asset_aware_trainer.py
so the comparison is guaranteed apples-to-apples.

Outputs (written to models/  directory):
  fresh_eval_{model}_{asset_type}_{symbol_key}.csv
      columns: date, actual, lstm_pred, transformer_pred, naive,
               lstm_abs_err, transformer_abs_err, naive_abs_err

  fresh_metrics_summary_{timestamp}.json   -- full per-symbol metrics
  fresh_metrics_summary_{timestamp}.csv    -- flat CSV version

Run from:
  d:\\major-project\\backend\\stockproject\\predictor\\
  python fresh_inference.py
"""

import os
import sys
import json
import glob
import datetime
import warnings
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")

# ── local imports ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asset_aware_trainer as aat
from asset_aware_trainer import (
    AssetAwareTrainer, STOCK_SYMBOLS, INDEX_SYMBOLS, FEATURES
)

# Silence the network-bound sentiment provider during evaluation
aat.get_daily_sentiment_series = lambda *a, **kw: {}

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

PERIOD       = "5y"
SEED         = 42
EXTENDED_FTS = FEATURES + [f"pct_change_{c}" for c in ["open", "high", "low", "close"]]


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def symbol_key(sym: str) -> str:
    return sym.replace(".", "_").replace("^", "INDEX_")


def safe_mape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.clip(np.abs(actual), 1e-8, None)
    return float(np.mean(np.abs((actual - pred) / denom)) * 100.0)


def directional_accuracy(actual, pred, previous, nonflat=False):
    actual_dir = np.sign(actual - previous)
    pred_dir   = np.sign(pred   - previous)
    if nonflat:
        mask = actual_dir != 0
        if not np.any(mask):
            return float("nan")
        actual_dir, pred_dir = actual_dir[mask], pred_dir[mask]
    return float(np.mean(actual_dir == pred_dir) * 100.0)


def model_artifact_paths(symbol: str, model_type: str, asset_type: str) -> dict:
    sk = symbol_key(symbol)
    return {
        "model":          os.path.join(MODEL_DIR, f"{model_type}_{asset_type}_{sk}_model.keras"),
        "best":           os.path.join(MODEL_DIR, f"{model_type}_{asset_type}_{sk}_best.keras"),
        "feature_scaler": os.path.join(MODEL_DIR, f"feature_scaler_{asset_type}_{sk}.pkl"),
        "target_scaler":  os.path.join(MODEL_DIR, f"target_scaler_{asset_type}_{sk}.pkl"),
        "metadata":       os.path.join(MODEL_DIR, f"metadata_{model_type}_{asset_type}_{sk}.json"),
    }


def artifacts_exist(paths: dict) -> bool:
    for key in ("model", "feature_scaler", "target_scaler"):
        p = paths[key]
        # prefer best checkpoint if it exists
        if key == "model" and os.path.exists(paths.get("best", "")):
            continue
        if not os.path.exists(p):
            return False
    return True


def _strip_renorm_from_config(cfg):
    """Recursively remove version-specific keys from layer configs."""
    if isinstance(cfg, dict):
        inner = cfg.get("config", {})
        cls   = cfg.get("class_name", "")
        if cls == "BatchNormalization":
            for k in ("renorm", "renorm_clipping", "renorm_momentum"):
                inner.pop(k, None)
        if cls == "MultiHeadAttention":
            for k in ("use_gate",):
                inner.pop(k, None)
        for v in cfg.values():
            _strip_renorm_from_config(v)
    elif isinstance(cfg, list):
        for item in cfg:
            _strip_renorm_from_config(item)


def _load_keras_model(path: str):
    """
    Load a .keras (zip) model robustly across Keras 2/3 version differences.
    Strategy:
      1. Fast-path: load directly with compile=False.
      2. Fallback: patch the config.json inside the zip to strip unsupported
         BatchNormalization kwargs (renorm*), then load from patched bytes.
    """
    import zipfile, io, tempfile, shutil

    # ── Fast path ──────────────────────────────────────────────
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        pass  # fall through to patching

    # ── Patching path ──────────────────────────────────────────
    # Read original zip, patch config.json, write to a temp file, load that.
    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(path, "r") as zin, \
             zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                fname = item.filename   # ZipInfo uses .filename, not .name
                data = zin.read(fname)
                if fname == "config.json":
                    cfg = json.loads(data.decode("utf-8"))
                    _strip_renorm_from_config(cfg)
                    data = json.dumps(cfg).encode("utf-8")
                zout.writestr(item, data)

        buf.seek(0)
        # Write patched zip to a temp file (Keras needs a real path)
        tmp = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
        tmp.write(buf.read())
        tmp.close()

        try:
            model = tf.keras.models.load_model(tmp.name, compile=False)
        finally:
            os.unlink(tmp.name)
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model {path}: {e}") from e


def load_model_safe(paths: dict):
    """Load best checkpoint if available, else _model.keras."""
    best = paths.get("best", "")
    if os.path.exists(best):
        return _load_keras_model(best)
    return _load_keras_model(paths["model"])


def load_metadata_seq_len(paths: dict, default: int = 120) -> int:
    mp = paths.get("metadata", "")
    if os.path.exists(mp):
        with open(mp) as f:
            return int(json.load(f).get("sequence_length", default))
    return default


# ──────────────────────────────────────────────────────────────
# Core: infer one symbol, both models
# ──────────────────────────────────────────────────────────────
def infer_symbol(trainer: AssetAwareTrainer, symbol: str) -> dict:
    asset_config, asset_type = trainer.get_asset_config(symbol)
    sk = symbol_key(symbol)

    # Load artifact paths for LSTM and Transformer
    lstm_paths = model_artifact_paths(symbol, "lstm", asset_type)
    tr_paths   = model_artifact_paths(symbol, "transformer", asset_type)

    lstm_ok = artifacts_exist(lstm_paths)
    tr_ok   = artifacts_exist(tr_paths)

    if not lstm_ok and not tr_ok:
        print(f"  [SKIP] No saved artifacts for {symbol}")
        return {"symbol": symbol, "status": "no_artifacts"}

    # ── Fetch & preprocess data ────────────────────────────────
    print(f"  [FETCH] {symbol} …", end="", flush=True)
    df = trainer.nse_fetcher.fetch_data(symbol, period=PERIOD)
    if df is None or df.empty:
        print(" NO DATA")
        return {"symbol": symbol, "status": "no_data"}

    df = trainer.preprocess_data_consistent(df, asset_type, symbol=symbol)
    if df is None or df.empty:
        print(" PREPROCESS FAILED")
        return {"symbol": symbol, "status": "preprocess_failed"}

    missing = [c for c in EXTENDED_FTS + ["close"] if c not in df.columns]
    if missing:
        print(f" MISSING COLS: {missing}")
        return {"symbol": symbol, "status": f"missing_cols:{missing}"}

    print(f" {len(df)} rows", flush=True)

    X_raw = df[EXTENDED_FTS].values
    y_raw = df["close"].values.reshape(-1, 1)

    # Chronological split (70/15/15) mirroring training
    split_cfg = asset_config.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    n        = len(X_raw)
    train_end = int(n * split_cfg["train"])
    val_end   = int(n * (split_cfg["train"] + split_cfg["val"]))

    # Dates
    dates = trainer._extract_dates(df)
    date_strs = (
        pd.to_datetime(dates).dt.strftime("%Y-%m-%d")
        if dates is not None
        else pd.Series(np.arange(n)).astype(str)
    )

    result = {
        "symbol":     symbol,
        "asset_type": asset_type,
        "n_total":    int(n),
        "train_end":  int(train_end),
        "val_end":    int(val_end),
        "status":     "ok",
    }

    # ── Run inference per model ────────────────────────────────
    preds = {}   # model_type -> (pred_prices, test_idx)

    for model_type, paths in [("lstm", lstm_paths), ("transformer", tr_paths)]:
        if not artifacts_exist(paths):
            print(f"    [{model_type.upper()} SKIP] Missing artifacts")
            preds[model_type] = None
            continue

        seq_len = load_metadata_seq_len(paths, asset_config["sequence_length"])

        feature_scaler = joblib.load(paths["feature_scaler"])
        target_scaler  = joblib.load(paths["target_scaler"])
        model          = load_model_safe(paths)

        X_scaled = feature_scaler.transform(X_raw)
        y_scaled = target_scaler.transform(y_raw)

        X_seq, y_seq, target_idx = trainer.create_sequences_with_index(
            X_scaled, y_scaled, seq_len
        )
        if X_seq.size == 0:
            print(f"    [{model_type.upper()} SKIP] Not enough sequences")
            preds[model_type] = None
            continue

        test_mask = target_idx >= val_end
        X_test    = X_seq[test_mask]
        test_idx  = target_idx[test_mask]

        if X_test.size == 0:
            print(f"    [{model_type.upper()} SKIP] Empty test split")
            preds[model_type] = None
            continue

        pred_scaled = model.predict(X_test, verbose=0)
        pred_prices = target_scaler.inverse_transform(pred_scaled).flatten()

        preds[model_type] = (pred_prices, test_idx)
        print(f"    [{model_type.upper()}] inferred {len(pred_prices)} test points")

    # ── Build unified evaluation DataFrame ────────────────────
    # Align on test indices common to both models (or just one if only one ran)
    lstm_res = preds.get("lstm")
    tr_res   = preds.get("transformer")

    if lstm_res is None and tr_res is None:
        return {**result, "status": "inference_failed_both"}

    # Use whichever test index set is available; prefer intersection
    if lstm_res is not None and tr_res is not None:
        lstm_idx_set = set(lstm_res[1].tolist())
        tr_idx_set   = set(tr_res[1].tolist())
        common_idx   = sorted(lstm_idx_set & tr_idx_set)
        if not common_idx:
            result["status"] = "no_common_test_indices"

        common_arr    = np.array(common_idx)
        lstm_mask_c   = np.isin(lstm_res[1], common_arr)
        tr_mask_c     = np.isin(tr_res[1], common_arr)

        actual   = y_raw[common_arr].flatten()
        previous = y_raw[common_arr - 1].flatten()
        lstm_p   = lstm_res[0][lstm_mask_c]
        tr_p     = tr_res[0][tr_mask_c]
        dates_c  = date_strs.iloc[common_arr].values
    elif lstm_res is not None:
        test_idx = lstm_res[1]
        actual   = y_raw[test_idx].flatten()
        previous = y_raw[test_idx - 1].flatten()
        lstm_p   = lstm_res[0]
        tr_p     = np.full_like(lstm_p, np.nan)
        dates_c  = date_strs.iloc[test_idx].values
        common_arr = test_idx
    else:
        test_idx = tr_res[1]
        actual   = y_raw[test_idx].flatten()
        previous = y_raw[test_idx - 1].flatten()
        tr_p     = tr_res[0]
        lstm_p   = np.full_like(tr_p, np.nan)
        dates_c  = date_strs.iloc[test_idx].values
        common_arr = test_idx

    naive = previous  # previous-close naive baseline

    # ── Save merged evaluation CSV ─────────────────────────────
    eval_df = pd.DataFrame({
        "date":                dates_c,
        "actual":              actual,
        "lstm_pred":           lstm_p,
        "transformer_pred":    tr_p,
        "naive":               naive,
        "lstm_abs_err":        np.abs(actual - lstm_p),
        "transformer_abs_err": np.abs(actual - tr_p),
        "naive_abs_err":       np.abs(actual - naive),
    })
    eval_csv = os.path.join(
        MODEL_DIR, f"fresh_eval_{asset_type}_{sk}.csv"
    )
    eval_df.to_csv(eval_csv, index=False)

    # ── Compute metrics ────────────────────────────────────────
    def metrics_block(pred_arr, label):
        if np.all(np.isnan(pred_arr)):
            return {}
        valid = ~np.isnan(pred_arr)
        a, p, pv = actual[valid], pred_arr[valid], previous[valid]
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae  = float(mean_absolute_error(a, p))
        rmse = float(np.sqrt(mean_squared_error(a, p)))
        mape = safe_mape(a, p)
        r2   = float(r2_score(a, p))
        da   = directional_accuracy(a, p, pv)
        da_nf= directional_accuracy(a, p, pv, nonflat=True)
        flat = float(np.mean(np.sign(a - pv) == 0) * 100)
        return {
            f"{label}_mae":                   mae,
            f"{label}_rmse":                  rmse,
            f"{label}_mape":                  mape,
            f"{label}_r2":                    r2,
            f"{label}_dir_acc":               da,
            f"{label}_dir_acc_nonflat":       da_nf,
            f"{label}_flat_day_ratio":        flat,
            f"{label}_test_n":                int(valid.sum()),
        }

    naive_mae  = float(np.mean(np.abs(actual - naive)))
    naive_rmse = float(np.sqrt(np.mean((actual - naive) ** 2)))
    naive_mape = safe_mape(actual, naive)
    naive_da   = directional_accuracy(actual, naive, previous)

    result.update(metrics_block(lstm_p, "lstm"))
    result.update(metrics_block(tr_p,   "transformer"))
    result["naive_mae"]      = naive_mae
    result["naive_rmse"]     = naive_rmse
    result["naive_mape"]     = naive_mape
    result["naive_dir_acc"]  = naive_da
    result["eval_csv"]       = eval_csv
    result["test_rows"]      = int(len(eval_df))

    # Winner flags (lower is better for error metrics, higher for R2/DirAcc)
    if lstm_res and tr_res:
        result["mae_winner"]     = "lstm" if result.get("lstm_mae",  1e9) < result.get("transformer_mae",  1e9) else "transformer"
        result["rmse_winner"]    = "lstm" if result.get("lstm_rmse", 1e9) < result.get("transformer_rmse", 1e9) else "transformer"
        result["mape_winner"]    = "lstm" if result.get("lstm_mape", 1e9) < result.get("transformer_mape", 1e9) else "transformer"
        result["r2_winner"]      = "lstm" if result.get("lstm_r2",  -1e9) > result.get("transformer_r2",  -1e9) else "transformer"
        result["dir_acc_winner"] = "lstm" if result.get("lstm_dir_acc", 0) > result.get("transformer_dir_acc", 0) else "transformer"

    return result


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 65)
    print("  Fresh Inference Pipeline — LSTM vs Transformer")
    print("=" * 65)

    trainer = AssetAwareTrainer(seed=SEED)

    all_symbols = STOCK_SYMBOLS + INDEX_SYMBOLS
    results = []

    for sym in all_symbols:
        print(f"\n[{sym}]")
        res = infer_symbol(trainer, sym)
        results.append(res)

    # ── Save full results ──────────────────────────────────────
    out_json = os.path.join(MODEL_DIR, f"fresh_inference_{ts}.json")
    out_csv  = os.path.join(MODEL_DIR, f"fresh_inference_{ts}.csv")

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    flat_rows = []
    for r in results:
        flat = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)

    # ── Print summary ──────────────────────────────────────────
    ok = [r for r in results if r.get("status") == "ok"]
    print("\n" + "=" * 65)
    print(f"  Inference complete: {len(ok)}/{len(all_symbols)} symbols OK")
    print("=" * 65)
    if ok:
        lstm_mapes = [r["lstm_mape"] for r in ok if "lstm_mape" in r]
        tr_mapes   = [r["transformer_mape"] for r in ok if "transformer_mape" in r]
        lstm_r2s   = [r["lstm_r2"]   for r in ok if "lstm_r2"   in r]
        tr_r2s     = [r["transformer_r2"] for r in ok if "transformer_r2" in r]

        print(f"  Avg MAPE  → LSTM: {np.mean(lstm_mapes):.2f}%  |  Transformer: {np.mean(tr_mapes):.2f}%")
        print(f"  Avg R²    → LSTM: {np.mean(lstm_r2s):.3f}   |  Transformer: {np.mean(tr_r2s):.3f}")

        # Per-metric win count
        for metric, col in [("MAE","mae_winner"),("RMSE","rmse_winner"),
                             ("MAPE","mape_winner"),("R2","r2_winner"),
                             ("DirAcc","dir_acc_winner")]:
            winners = [r.get(col) for r in ok if col in r]
            lw = winners.count("lstm")
            tw = winners.count("transformer")
            print(f"  {metric:8s} wins → LSTM: {lw}  |  Transformer: {tw}")

    print(f"\n  JSON: {out_json}")
    print(f"  CSV:  {out_csv}")
    return out_json


if __name__ == "__main__":
    main()
