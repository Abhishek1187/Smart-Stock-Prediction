import argparse
import json
import os
from datetime import datetime

import pandas as pd

from asset_aware_trainer import AssetAwareTrainer, STOCK_SYMBOLS, INDEX_SYMBOLS
from run_model_comparison import compare_symbol


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def _metadata_path(symbol, model_type, asset_type):
    symbol_key = symbol.replace(".", "_").replace("^", "INDEX_")
    return os.path.join(MODEL_DIR, f"metadata_{model_type}_{asset_type}_{symbol_key}.json")


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fairness_checks(symbol, trainer):
    asset_config, asset_type = trainer.get_asset_config(symbol)

    lstm_meta = _load_json(_metadata_path(symbol, "lstm", asset_type))
    tr_meta = _load_json(_metadata_path(symbol, "transformer", asset_type))

    checks = {
        "asset_type": asset_type,
        "same_sequence_length": None,
        "same_history_period": None,
        "same_seed": None,
        "same_train_split_size": None,
        "same_val_split_size": None,
        "same_test_split_size": None,
    }

    if not lstm_meta or not tr_meta:
        checks["missing_metadata"] = True
        checks["message"] = "Missing LSTM or Transformer metadata for strict fairness checks"
        return checks

    checks["missing_metadata"] = False
    checks["same_sequence_length"] = lstm_meta.get("sequence_length") == tr_meta.get("sequence_length")
    checks["same_history_period"] = lstm_meta.get("history_period") == tr_meta.get("history_period")
    checks["same_seed"] = lstm_meta.get("seed") == tr_meta.get("seed")

    lstm_split = lstm_meta.get("split", {})
    tr_split = tr_meta.get("split", {})
    checks["same_train_split_size"] = lstm_split.get("train_sequences") == tr_split.get("train_sequences")
    checks["same_val_split_size"] = lstm_split.get("val_sequences") == tr_split.get("val_sequences")
    checks["same_test_split_size"] = lstm_split.get("test_sequences") == tr_split.get("test_sequences")

    checks["expected_sequence_length"] = int(asset_config.get("sequence_length", 120))
    checks["lstm_sequence_length"] = lstm_meta.get("sequence_length")
    checks["transformer_sequence_length"] = tr_meta.get("sequence_length")
    return checks


def run_protocol(symbols, period, seed, seq_len, require_gpu=False):
    trainer = AssetAwareTrainer(seed=seed)
    trainer.ensure_gpu_ready(require_gpu=False)
    rows = []

    for symbol in symbols:
        print(f"\n{'=' * 72}")
        print(f"[INFO] Retraining TRANSFORMER for {symbol}")
        print(f"{'=' * 72}")

        trained = trainer.train_asset_specific_model(
            symbol=symbol,
            model_type="transformer",
            history_period=period,
            sequence_length=seq_len,
            init_from_global=False,
            require_gpu=require_gpu,
        )

        row = {
            "symbol": symbol,
            "transformer_retrained": bool(trained),
        }

        if trained:
            comparison = compare_symbol(trainer, symbol, period)
            fairness = _fairness_checks(symbol, trainer)
            row.update(comparison)
            row["fairness_checks"] = fairness

            if comparison.get("lstm_ok") and comparison.get("transformer_ok"):
                print(
                    "[RESULT] "
                    f"MAE LSTM={comparison['mae_lstm']:.4f} vs Transformer={comparison['mae_transformer']:.4f} | "
                    f"RMSE LSTM={comparison['rmse_lstm']:.4f} vs Transformer={comparison['rmse_transformer']:.4f} | "
                    f"DirAcc LSTM={comparison['dir_acc_lstm']:.2f}% vs Transformer={comparison['dir_acc_transformer']:.2f}%"
                )
            else:
                print(f"[WARNING] Comparison incomplete for {symbol}: {comparison.get('failure_reason')}")
        else:
            row["failure_reason"] = "transformer_training_failed"
            print(f"[ERROR] Transformer retraining failed for {symbol}")

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Retrain Transformer with LSTM-matched protocol and run fair comparison"
    )
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol to retrain")
    parser.add_argument("--all", action="store_true", help="Run on all configured stocks and indices")
    parser.add_argument("--period", type=str, default="5y", help="History period (default: 5y)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--seq-len", type=int, default=None, help="Optional sequence length override")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow run without GPU (not recommended for this protocol)")
    args = parser.parse_args()

    if args.all:
        symbols = STOCK_SYMBOLS + INDEX_SYMBOLS
    elif args.symbol:
        symbols = [args.symbol]
    else:
        parser.error("Provide --symbol SYMBOL or --all")

    rows = run_protocol(
        symbols=symbols,
        period=args.period,
        seed=args.seed,
        seq_len=args.seq_len,
        require_gpu=not args.allow_cpu,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(MODEL_DIR, f"transformer_fair_retrain_report_{ts}.json")
    out_csv = os.path.join(MODEL_DIR, f"transformer_fair_retrain_report_{ts}.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k != "fairness_checks"}
        fairness = row.get("fairness_checks", {})
        for k, v in fairness.items():
            flat[f"fairness_{k}"] = v
        flat_rows.append(flat)

    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)

    print("\n=== Protocol Summary ===")
    print(f"Symbols requested: {len(symbols)}")
    ok_rows = [r for r in rows if r.get("lstm_ok") and r.get("transformer_ok")]
    print(f"Symbols compared successfully: {len(ok_rows)}")
    print(f"[INFO] JSON report: {out_json}")
    print(f"[INFO] CSV report: {out_csv}")


if __name__ == "__main__":
    main()
