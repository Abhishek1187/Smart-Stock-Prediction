import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import asset_aware_trainer as aat
from asset_aware_trainer import AssetAwareTrainer, STOCK_SYMBOLS, FEATURES


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# Make evaluation robust/offline by avoiding network-bound sentiment providers.
aat.get_daily_sentiment_series = lambda *args, **kwargs: {}


def safe_mape(actual, pred):
    denom = np.clip(np.abs(actual), 1e-8, None)
    return float(np.mean(np.abs((actual - pred) / denom)) * 100.0)


def directional_accuracy(actual, pred, previous):
    actual_dir = np.sign(actual - previous)
    pred_dir = np.sign(pred - previous)
    return float(np.mean(actual_dir == pred_dir) * 100.0)


def model_paths(symbol, asset_type, model_type):
    symbol_key = symbol.replace(".", "_").replace("^", "INDEX_")
    return {
        "model": os.path.join(MODEL_DIR, f"{model_type}_{asset_type}_{symbol_key}_model.keras"),
        "feature_scaler": os.path.join(MODEL_DIR, f"feature_scaler_{asset_type}_{symbol_key}.pkl"),
        "target_scaler": os.path.join(MODEL_DIR, f"target_scaler_{asset_type}_{symbol_key}.pkl"),
        "metadata_new": os.path.join(MODEL_DIR, f"metadata_{model_type}_{asset_type}_{symbol_key}.json"),
        "metadata_legacy": os.path.join(MODEL_DIR, f"metadata_{asset_type}_{symbol_key}.json"),
    }


def load_metadata(paths):
    if os.path.exists(paths["metadata_new"]):
        with open(paths["metadata_new"], "r", encoding="utf-8") as f:
            return json.load(f), "new"
    if os.path.exists(paths["metadata_legacy"]):
        with open(paths["metadata_legacy"], "r", encoding="utf-8") as f:
            return json.load(f), "legacy"
    return {}, "missing"


def load_predictions_for_model(trainer, symbol, model_type, period, default_seq_len):
    asset_config, asset_type = trainer.get_asset_config(symbol)
    paths = model_paths(symbol, asset_type, model_type)

    required = [paths["model"], paths["feature_scaler"], paths["target_scaler"]]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        return None, {
            "ok": False,
            "reason": "missing_artifacts",
            "missing": missing,
            "metadata_source": "missing",
            "sequence_length": None,
        }

    metadata, metadata_source = load_metadata(paths)
    seq_len = int(metadata.get("sequence_length", default_seq_len))

    df = trainer.nse_fetcher.fetch_data(symbol, period=period)
    if df is None or df.empty:
        return None, {
            "ok": False,
            "reason": "no_data",
            "metadata_source": metadata_source,
            "sequence_length": seq_len,
        }

    processed = trainer.preprocess_data_consistent(df, asset_type, symbol=symbol)
    if processed is None or processed.empty:
        return None, {
            "ok": False,
            "reason": "preprocess_failed",
            "metadata_source": metadata_source,
            "sequence_length": seq_len,
        }

    extended_features = FEATURES + [f"pct_change_{col}" for col in ["open", "high", "low", "close"]]
    if any(col not in processed.columns for col in extended_features + ["close"]):
        return None, {
            "ok": False,
            "reason": "missing_columns",
            "metadata_source": metadata_source,
            "sequence_length": seq_len,
        }

    X = processed[extended_features].values
    y = processed["close"].values.reshape(-1, 1)

    split_cfg = asset_config.get("split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    n_points = len(X)
    train_end = int(n_points * split_cfg["train"])
    val_end = int(n_points * (split_cfg["train"] + split_cfg["val"]))

    feature_scaler = joblib.load(paths["feature_scaler"])
    target_scaler = joblib.load(paths["target_scaler"])
    model = tf.keras.models.load_model(paths["model"])

    X_scaled = feature_scaler.transform(X)
    y_scaled = target_scaler.transform(y)

    X_seq, y_seq, target_idx = trainer.create_sequences_with_index(X_scaled, y_scaled, seq_len)
    if X_seq.size == 0 or target_idx.size == 0:
        return None, {
            "ok": False,
            "reason": "not_enough_sequence_data",
            "metadata_source": metadata_source,
            "sequence_length": seq_len,
        }

    test_mask = target_idx >= val_end
    X_test = X_seq[test_mask]
    test_idx = target_idx[test_mask]

    if X_test.size == 0:
        return None, {
            "ok": False,
            "reason": "empty_test_split",
            "metadata_source": metadata_source,
            "sequence_length": seq_len,
        }

    pred_scaled = model.predict(X_test, verbose=0)
    pred = target_scaler.inverse_transform(pred_scaled).flatten()
    actual = y[test_idx].flatten()
    previous = y[test_idx - 1].flatten()

    dates = trainer._extract_dates(processed)
    if dates is None:
        date_values = pd.Series(np.arange(len(processed)))
    else:
        date_values = pd.to_datetime(dates).dt.strftime("%Y-%m-%d")

    out = pd.DataFrame(
        {
            "date": date_values.iloc[test_idx].values,
            "actual": actual,
            "pred": pred,
            "previous": previous,
        }
    )

    info = {
        "ok": True,
        "metadata_source": metadata_source,
        "sequence_length": seq_len,
        "test_rows": int(len(out)),
    }
    return out, info


def compare_symbol(trainer, symbol, period):
    default_seq = trainer.asset_configs["stocks"]["sequence_length"]
    lstm_df, lstm_info = load_predictions_for_model(trainer, symbol, "lstm", period, default_seq)
    tr_df, tr_info = load_predictions_for_model(trainer, symbol, "transformer", period, default_seq)

    result = {
        "symbol": symbol,
        "lstm_ok": lstm_info.get("ok", False),
        "transformer_ok": tr_info.get("ok", False),
        "lstm_metadata_source": lstm_info.get("metadata_source"),
        "transformer_metadata_source": tr_info.get("metadata_source"),
        "lstm_sequence_length": lstm_info.get("sequence_length"),
        "transformer_sequence_length": tr_info.get("sequence_length"),
    }

    if not lstm_info.get("ok"):
        result["failure_reason"] = f"lstm:{lstm_info.get('reason')}"
        return result
    if not tr_info.get("ok"):
        result["failure_reason"] = f"transformer:{tr_info.get('reason')}"
        return result

    merged = lstm_df.merge(
        tr_df[["date", "pred"]].rename(columns={"pred": "pred_transformer"}),
        on="date",
        how="inner",
    )

    if merged.empty:
        result["failure_reason"] = "no_common_test_dates"
        return result

    actual = merged["actual"].values
    previous = merged["previous"].values
    pred_lstm = merged["pred"].values
    pred_tr = merged["pred_transformer"].values

    metrics = {
        "test_rows_common": int(len(merged)),
        "mae_lstm": float(mean_absolute_error(actual, pred_lstm)),
        "mae_transformer": float(mean_absolute_error(actual, pred_tr)),
        "rmse_lstm": float(np.sqrt(mean_squared_error(actual, pred_lstm))),
        "rmse_transformer": float(np.sqrt(mean_squared_error(actual, pred_tr))),
        "mape_lstm": safe_mape(actual, pred_lstm),
        "mape_transformer": safe_mape(actual, pred_tr),
        "r2_lstm": float(r2_score(actual, pred_lstm)),
        "r2_transformer": float(r2_score(actual, pred_tr)),
        "dir_acc_lstm": directional_accuracy(actual, pred_lstm, previous),
        "dir_acc_transformer": directional_accuracy(actual, pred_tr, previous),
    }

    winners = {
        "mae_winner": "lstm" if metrics["mae_lstm"] < metrics["mae_transformer"] else "transformer",
        "rmse_winner": "lstm" if metrics["rmse_lstm"] < metrics["rmse_transformer"] else "transformer",
        "mape_winner": "lstm" if metrics["mape_lstm"] < metrics["mape_transformer"] else "transformer",
        "r2_winner": "lstm" if metrics["r2_lstm"] > metrics["r2_transformer"] else "transformer",
        "dir_acc_winner": "lstm" if metrics["dir_acc_lstm"] > metrics["dir_acc_transformer"] else "transformer",
    }

    result.update(metrics)
    result.update(winners)
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare saved LSTM and Transformer models fairly per stock")
    parser.add_argument("--period", type=str, default="5y", help="History period for fresh data fetch")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic preprocessing")
    parser.add_argument("--symbols", nargs="*", default=None, help="Optional subset of stock symbols")
    args = parser.parse_args()

    trainer = AssetAwareTrainer(seed=args.seed)
    symbols = args.symbols if args.symbols else STOCK_SYMBOLS

    rows = []
    for symbol in symbols:
        print(f"[INFO] Comparing models for {symbol}")
        rows.append(compare_symbol(trainer, symbol, args.period))

    df = pd.DataFrame(rows)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(MODEL_DIR, f"comparison_lstm_vs_transformer_stocks_{ts}.csv")
    out_json = os.path.join(MODEL_DIR, f"comparison_lstm_vs_transformer_stocks_{ts}.json")

    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    ok_df = df[(df.get("lstm_ok", False) == True) & (df.get("transformer_ok", False) == True) & (~df.get("mae_lstm", pd.Series([np.nan] * len(df))).isna())]

    print("\n=== Comparison Summary ===")
    print(f"Symbols requested: {len(symbols)}")
    print(f"Symbols compared successfully: {len(ok_df)}")
    if len(ok_df) > 0:
        print(f"Avg MAE   | LSTM={ok_df['mae_lstm'].mean():.4f} vs Transformer={ok_df['mae_transformer'].mean():.4f}")
        print(f"Avg RMSE  | LSTM={ok_df['rmse_lstm'].mean():.4f} vs Transformer={ok_df['rmse_transformer'].mean():.4f}")
        print(f"Avg MAPE% | LSTM={ok_df['mape_lstm'].mean():.4f} vs Transformer={ok_df['mape_transformer'].mean():.4f}")
        print(f"Avg R2    | LSTM={ok_df['r2_lstm'].mean():.4f} vs Transformer={ok_df['r2_transformer'].mean():.4f}")
        print(f"Avg DirAcc% | LSTM={ok_df['dir_acc_lstm'].mean():.4f} vs Transformer={ok_df['dir_acc_transformer'].mean():.4f}")

        def winner_count(col):
            return ok_df[col].value_counts().to_dict()

        print(f"Winner counts (MAE lower): {winner_count('mae_winner')}")
        print(f"Winner counts (RMSE lower): {winner_count('rmse_winner')}")
        print(f"Winner counts (MAPE lower): {winner_count('mape_winner')}")
        print(f"Winner counts (R2 higher): {winner_count('r2_winner')}")
        print(f"Winner counts (DirAcc higher): {winner_count('dir_acc_winner')}")

    print(f"\n[INFO] Detailed CSV: {out_csv}")
    print(f"[INFO] Detailed JSON: {out_json}")


if __name__ == "__main__":
    main()
