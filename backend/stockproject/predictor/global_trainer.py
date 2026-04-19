import os
import sys
import re
import json
import random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, LayerNormalization, MultiHeadAttention, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from .utils import add_technical_indicators
    from .nse_data_fetcher import NSEDataFetcher
except ImportError:
    # Fallback for standalone execution
    from utils import add_technical_indicators
    from nse_data_fetcher import NSEDataFetcher

# Directory and file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    'open', 'high', 'low', 'volume',
    'sma_10', 'ema_9', 'ema_21', 'rsi_14', 'ma_20',
    'bb_upper', 'bb_lower', 'ema_12', 'ema_26',
    'macd', 'signal_line', '%k', '%d', 'atr_14',
    'news_sentiment'
]

# Asset type definitions for global training
STOCK_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "SBIN.NS", "LT.NS", "WIPRO.NS", "MARUTI.NS"
]

INDEX_SYMBOLS = [
    "^NSEI", "^NSEBANK", "^NSEMDCP50", "^NSEFIN", "^CNXAUTO"
]

class GlobalTrainer:
    def __init__(self, seed=42, history_period="5y", sequence_length=120):
        self.seed = seed
        self.set_random_seeds(seed)
        self.nse_fetcher = NSEDataFetcher()
        
        # Global model configurations - same for all assets
        self.model_configs = {
            'lstm': {
                'units': [128, 64, 32],  # Deeper architecture
                'dropout': 0.1,
                'epochs': 100,
                'batch_size': 128,
                'learning_rate': 0.0005,
                'patience': 10
            },
            'transformer': {
                'head_size': 32,
                'num_heads': 4,
                'ff_dim': 128,
                'dropout': 0.1,
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.0005,
                'patience': 10
            }
        }
        
        self.sequence_length = int(sequence_length)
        self.history_period = history_period

    def set_random_seeds(self, seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass

    def _safe_mape(self, y_true, y_pred):
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        denom = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
        out = np.abs((y_true - y_pred) / denom) * 100.0
        return float(np.nanmean(out)) if np.isfinite(np.nanmean(out)) else float("nan")

    def _directional_accuracy(self, actual_prices, pred_prices, prev_prices):
        actual_direction = np.sign(np.array(actual_prices) - np.array(prev_prices))
        pred_direction = np.sign(np.array(pred_prices) - np.array(prev_prices))
        return float(np.mean(actual_direction == pred_direction) * 100.0)

    def detect_asset_type(self, symbol):
        """Detect if symbol is a stock or index"""
        if symbol.startswith("^"):
            return "indices"
        elif symbol in INDEX_SYMBOLS:
            return "indices"
        else:
            return "stocks"

    def preprocess_data_consistent(self, df, symbol):
        """Consistent preprocessing for all assets"""
        # Convert MultiIndex or ticker-suffixed columns to simple names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
        
        # Remove ticker suffix
        df.columns = [re.sub(r'_[A-Z^]+[A-Z0-9]*\.?[A-Z]*$', '', col) for col in df.columns]
        
        # Ensure lowercase columns
        df.columns = df.columns.str.lower()
        
        # Add technical indicators
        df = add_technical_indicators(df)
        if df is None or df.empty:
            print(f"[ERROR] Failed to add technical indicators for {symbol}")
            return None
            
        df = df.ffill().bfill()
        
        # Add news sentiment if missing
        if 'news_sentiment' not in df.columns:
            df['news_sentiment'] = 0.0

        # Universal volume normalization (same for all assets)
        df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
        df['volume_pct_change'] = df['volume_pct_change'].clip(-0.3, 0.3)  # Conservative clipping
        df['volume'] = df['volume_pct_change']

        # Universal price change normalization (same for all assets)
        for col in ['open', 'high', 'low', 'close']:
            df[f'pct_change_{col}'] = df[col].pct_change().fillna(0)
            df[f'pct_change_{col}'] = df[f'pct_change_{col}'].clip(-0.15, 0.15)  # Conservative clipping

        df.fillna(0, inplace=True)
        return df

    def collect_global_data(self, symbols):
        """Collect data from multiple symbols to create global scalers"""
        print(f"[INFO] Collecting data from {len(symbols)} symbols for global scaling...")
        
        all_features = []
        all_targets = []
        symbol_data = {}
        
        for symbol in symbols:
            print(f"[INFO] Fetching data for {symbol}...")
            
            try:
                # Fetch data
                df = self.nse_fetcher.fetch_data(symbol, period=self.history_period)
                if df is None or df.empty:
                    print(f"[WARNING] No data for {symbol}, skipping...")
                    continue

                # Preprocess data
                df = self.preprocess_data_consistent(df, symbol)
                if df is None or df.empty:
                    print(f"[WARNING] Preprocessing failed for {symbol}, skipping...")
                    continue

                # Prepare features and targets
                extended_features = FEATURES + [f'pct_change_{col}' for col in ['open', 'high', 'low', 'close']]
                X = df[extended_features].values
                y = df['close'].values.reshape(-1, 1)

                if len(X) < self.sequence_length:
                    print(f"[WARNING] Not enough data for {symbol}, skipping...")
                    continue

                # Store for global scaling
                all_features.append(X)
                all_targets.append(y)
                symbol_data[symbol] = {'X': X, 'y': y, 'df': df}
                
                print(f"[INFO] ✅ {symbol}: {len(X)} data points, price range: {y.min():.2f} - {y.max():.2f}")
                
            except Exception as e:
                print(f"[ERROR] Failed to process {symbol}: {str(e)}")
                continue

        if not all_features:
            print("[ERROR] No valid data collected for global scaling!")
            return None, None, None

        # Combine all features and targets for global scaling
        global_features = np.vstack(all_features)
        global_targets = np.vstack(all_targets)
        
        print(f"[INFO] ✅ Global dataset created:")
        print(f"[INFO] Features shape: {global_features.shape}")
        print(f"[INFO] Targets shape: {global_targets.shape}")
        print(f"[INFO] Target range: {global_targets.min():.2f} - {global_targets.max():.2f}")
        
        return global_features, global_targets, symbol_data

    def create_global_scalers(self, global_features, global_targets):
        """Create universal scalers trained on all symbols"""
        print("[INFO] Creating global scalers...")
        
        # Create universal scalers
        global_feature_scaler = MinMaxScaler(feature_range=(0, 1))
        global_target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit on combined data from all symbols
        global_feature_scaler.fit(global_features)
        global_target_scaler.fit(global_targets)
        
        print(f"[INFO] ✅ Global feature scaler range: {global_feature_scaler.data_range_}")
        print(f"[INFO] ✅ Global target scaler range: {global_target_scaler.data_range_}")
        
        return global_feature_scaler, global_target_scaler

    def create_sequences(self, X, y, seq_length=60):
        """Create sequences for LSTM/Transformer training"""
        X_seq, y_seq = [], []
        if len(X) <= seq_length:
            return np.array([]), np.array([])
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length])
        return np.array(X_seq), np.array(y_seq)

    def build_lstm_model(self, input_shape):
        """Build improved LSTM model"""
        config = self.model_configs['lstm']
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        
        model = Sequential([
            LSTM(config['units'][0], return_sequences=True, input_shape=input_shape, recurrent_dropout=0.1),
            BatchNormalization(),
            LSTM(config['units'][1], return_sequences=True, recurrent_dropout=0.1),
            BatchNormalization(),
            LSTM(config['units'][2], recurrent_dropout=0.1),
            BatchNormalization(),
            Dropout(config['dropout']),
            Dense(64, activation='relu'),
            Dropout(config['dropout'] / 2),
            Dense(32, activation='relu'),
            Dropout(config['dropout'] / 2),
            Dense(1)
        ])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def transformer_encoder(self, inputs, head_size=32, num_heads=4, ff_dim=128, dropout=0.1):
        """Transformer encoder block"""
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        x = Dense(ff_dim, activation="relu")(res)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def build_transformer_model(self, input_shape):
        """Build improved Transformer model"""
        config = self.model_configs['transformer']
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Multiple transformer layers for better learning
        x = self.transformer_encoder(x, head_size=config['head_size'], 
                                   num_heads=config['num_heads'], 
                                   ff_dim=config['ff_dim'], 
                                   dropout=config['dropout'])
        x = self.transformer_encoder(x, head_size=config['head_size'], 
                                   num_heads=config['num_heads'], 
                                   ff_dim=config['ff_dim'], 
                                   dropout=config['dropout'])
        
        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(config['dropout'])(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(config['dropout'] / 2)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def train_global_model(self, model_type="lstm"):
        """Train global model using data from all symbols"""
        print(f"\n{'='*60}")
        print(f"Training Global {model_type.upper()} Model")
        print(f"{'='*60}")
        
        # Collect data from all symbols
        all_symbols = STOCK_SYMBOLS + INDEX_SYMBOLS
        global_features, global_targets, symbol_data = self.collect_global_data(all_symbols)
        
        if global_features is None:
            print("[ERROR] Failed to collect global data!")
            return False

        # Build train-only scaler fit pool to avoid leakage from validation/test windows.
        scaler_train_X_parts = []
        scaler_train_y_parts = []
        filtered_symbol_data = {}

        for symbol, data in symbol_data.items():
            X, y = data['X'], data['y']
            n_rows = len(X)
            row_train_end = int(n_rows * 0.70)
            row_val_end = int(n_rows * 0.85)

            if row_train_end <= self.sequence_length or row_val_end <= row_train_end or row_val_end >= n_rows:
                print(f"[WARNING] Not enough rows for split ({symbol}), skipping...")
                continue

            scaler_train_X_parts.append(X[:row_train_end])
            scaler_train_y_parts.append(y[:row_train_end])
            filtered_symbol_data[symbol] = data

        if not scaler_train_X_parts:
            print("[ERROR] No valid training rows available to fit scalers!")
            return False

        scaler_train_X = np.vstack(scaler_train_X_parts)
        scaler_train_y = np.vstack(scaler_train_y_parts)
        global_feature_scaler, global_target_scaler = self.create_global_scalers(scaler_train_X, scaler_train_y)

        # Prepare leakage-safe chronological splits for each symbol, then combine.
        train_X_parts, train_y_parts = [], []
        val_X_parts, val_y_parts = [], []
        test_X_parts, test_y_parts = [], []
        test_prev_price_parts = []
        test_symbol_parts = []
        
        print(f"[INFO] Creating sequences from all symbols...")
        for symbol, data in filtered_symbol_data.items():
            X, y = data['X'], data['y']
            
            # Scale using global scalers
            X_scaled = global_feature_scaler.transform(X)
            y_scaled = global_target_scaler.transform(y)
            
            # Create sequences
            X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, self.sequence_length)
            
            if X_seq.size > 0:
                n = len(X_seq)
                train_end = int(n * 0.70)
                val_end = int(n * 0.85)
                if train_end < 1 or val_end <= train_end or val_end >= n:
                    print(f"[WARNING] Not enough sequences for split ({symbol}), skipping...")
                    continue

                train_X_parts.append(X_seq[:train_end])
                train_y_parts.append(y_seq[:train_end])
                val_X_parts.append(X_seq[train_end:val_end])
                val_y_parts.append(y_seq[train_end:val_end])
                test_X_parts.append(X_seq[val_end:])
                test_y_parts.append(y_seq[val_end:])

                # Next-day target starts at index sequence_length, so align previous close as index-1.
                seq_target_idx = np.arange(self.sequence_length, self.sequence_length + n)
                prev_prices = y[seq_target_idx - 1].flatten()
                test_prev_price_parts.append(prev_prices[val_end:])
                test_symbol_parts.extend([symbol] * len(X_seq[val_end:]))

                print(
                    f"[INFO] {symbol}: total={n}, train={train_end}, "
                    f"val={val_end - train_end}, test={n - val_end}"
                )

        if not train_X_parts or not val_X_parts or not test_X_parts:
            print("[ERROR] No valid split data created!")
            return False

        # Combine splits across symbols
        X_train = np.vstack(train_X_parts)
        y_train = np.vstack(train_y_parts)
        X_val = np.vstack(val_X_parts)
        y_val = np.vstack(val_y_parts)
        X_test = np.vstack(test_X_parts)
        y_test = np.vstack(test_y_parts)
        prev_prices_test = np.concatenate(test_prev_price_parts)
        test_symbols = np.array(test_symbol_parts)
        
        print(f"[INFO] ✅ Global training/validation/test data prepared:")
        print(f"[INFO] X_train shape: {X_train.shape}")
        print(f"[INFO] y_train shape: {y_train.shape}")
        print(f"[INFO] X_val shape: {X_val.shape}")
        print(f"[INFO] X_test shape: {X_test.shape}")

        # Build model
        if model_type.lower() == "lstm":
            model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        else:  # transformer
            model = self.build_transformer_model((X_train.shape[1], X_train.shape[2]))

        # Training callbacks
        config = self.model_configs[model_type]
        early_stop = EarlyStopping(monitor='val_loss', patience=config['patience'], restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        best_checkpoint_path = os.path.join(MODEL_DIR, f"global_{model_type}_best.keras")
        checkpoint = ModelCheckpoint(
            filepath=best_checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )

        # Train model
        print(f"[INFO] Starting global {model_type} training...")
        history = model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr, checkpoint],
            shuffle=False,
            verbose=1
        )

        # Strict holdout test evaluation
        print(f"\n[INFO] Evaluating holdout global {model_type} model...")
        test_pred_scaled = model.predict(X_test, verbose=0)
        test_pred = global_target_scaler.inverse_transform(test_pred_scaled).flatten()
        actual_prices = global_target_scaler.inverse_transform(y_test).flatten()
        naive_pred = prev_prices_test

        mae = float(mean_absolute_error(actual_prices, test_pred))
        rmse = float(np.sqrt(mean_squared_error(actual_prices, test_pred)))
        r2 = float(r2_score(actual_prices, test_pred))
        mape = self._safe_mape(actual_prices, test_pred)
        directional_accuracy = self._directional_accuracy(actual_prices, test_pred, prev_prices_test)

        naive_mae = float(mean_absolute_error(actual_prices, naive_pred))
        naive_rmse = float(np.sqrt(mean_squared_error(actual_prices, naive_pred)))
        naive_directional_accuracy = self._directional_accuracy(actual_prices, naive_pred, prev_prices_test)

        print(
            f"[INFO] Holdout metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, "
            f"MAPE={mape:.2f}%, R2={r2:.4f}, DirAcc={directional_accuracy:.2f}%"
        )
        print(
            f"[INFO] Naive baseline: MAE={naive_mae:.4f}, RMSE={naive_rmse:.4f}, "
            f"DirAcc={naive_directional_accuracy:.2f}%"
        )

        # Holdout metrics by symbol for generalized model inspection.
        holdout_by_symbol = {}
        for symbol in np.unique(test_symbols):
            mask = test_symbols == symbol
            if int(np.sum(mask)) < 2:
                continue
            sym_actual = actual_prices[mask]
            sym_pred = test_pred[mask]
            sym_prev = prev_prices_test[mask]
            holdout_by_symbol[symbol] = {
                'samples': int(np.sum(mask)),
                'mae': float(mean_absolute_error(sym_actual, sym_pred)),
                'rmse': float(np.sqrt(mean_squared_error(sym_actual, sym_pred))),
                'mape': self._safe_mape(sym_actual, sym_pred),
                'directional_accuracy': self._directional_accuracy(sym_actual, sym_pred, sym_prev),
            }

        # Test on multiple symbols
        test_results = {}
        
        for symbol in ["RELIANCE.NS", "TCS.NS", "^NSEI"]:  # Test on key symbols
            if symbol in symbol_data:
                X, y = symbol_data[symbol]['X'], symbol_data[symbol]['y']
                X_scaled = global_feature_scaler.transform(X)
                y_scaled = global_target_scaler.transform(y)
                
                X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, self.sequence_length)
                if X_seq.size > 0:
                    pred_scaled = model.predict(X_seq[-5:])  # Last 5 predictions
                    pred = global_target_scaler.inverse_transform(pred_scaled)
                    actual = global_target_scaler.inverse_transform(y_seq[-5:])
                    
                    mae = mean_absolute_error(actual, pred)
                    error_pct = mae / np.mean(actual) * 100
                    
                    test_results[symbol] = {
                        'mae': mae,
                        'error_pct': error_pct,
                        'actual_avg': np.mean(actual),
                        'pred_avg': np.mean(pred)
                    }
                    
                    print(f"[INFO] {symbol}: MAE={mae:.2f}, Error={error_pct:.2f}%, Actual={np.mean(actual):.2f}, Pred={np.mean(pred):.2f}")

        # Save global model and scalers
        model_filename = f"global_{model_type}_model.keras"
        feature_scaler_filename = f"global_feature_scaler.pkl"
        target_scaler_filename = f"global_target_scaler.pkl"
        
        model_path = os.path.join(MODEL_DIR, model_filename)
        feature_scaler_path = os.path.join(MODEL_DIR, feature_scaler_filename)
        target_scaler_path = os.path.join(MODEL_DIR, target_scaler_filename)
        
        model.save(model_path)
        joblib.dump(global_feature_scaler, feature_scaler_path)
        joblib.dump(global_target_scaler, target_scaler_path)
        
        print(f"[INFO] ✅ Global {model_type} model saved: {model_filename}")
        print(f"[INFO] ✅ Global scalers saved")

        # Save metadata
        # Convert any numpy float32 to native float for JSON serialization
        def convert_floats(obj):
            if isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats(i) for i in obj]
            elif isinstance(obj, np.float32):
                return float(obj)
            else:
                return obj

        metadata = {
            'model_type': f'global_{model_type}',
            'training_symbols': list(filtered_symbol_data.keys()),
            'total_sequences': int(len(X_train) + len(X_val) + len(X_test)),
            'split_counts': {
                'train': int(len(X_train)),
                'val': int(len(X_val)),
                'test': int(len(X_test)),
            },
            'scaler_fit_rows': int(len(scaler_train_X)),
            'sequence_length': self.sequence_length,
            'history_period': self.history_period,
            'global_feature_range': [float(global_features.min()), float(global_features.max())],
            'global_target_range': [float(global_targets.min()), float(global_targets.max())],
            'holdout_metrics': {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
            },
            'naive_baseline': {
                'mae': naive_mae,
                'rmse': naive_rmse,
                'directional_accuracy': naive_directional_accuracy,
            },
            'holdout_by_symbol': convert_floats(holdout_by_symbol),
            'test_results': convert_floats(test_results),
            'training_history': {
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1])
            }
        }
        
        metadata_path = os.path.join(MODEL_DIR, f"global_{model_type}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[INFO] ✅ Global {model_type} metadata saved")
        return True

    def train_all_global_models(self):
        """Train both LSTM and Transformer global models"""
        print(f"\n{'='*80}")
        print("TRAINING GLOBAL MODELS FOR ALL ASSETS")
        print(f"{'='*80}")
        
        # Train LSTM
        lstm_success = self.train_global_model("lstm")
        
        # Train Transformer
        transformer_success = self.train_global_model("transformer")
        
        if lstm_success and transformer_success:
            print(f"\n[SUCCESS] ✅ Both global models trained successfully!")
            print(f"[INFO] Models can now predict any stock or index using universal scalers")
        else:
            print(f"\n[WARNING] ⚠️ Some global models failed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["lstm", "transformer", "both"], default="both", help="Model type to train")
    parser.add_argument("--period", type=str, default="5y", help="History period for market data (e.g., 1y, 2y, 5y)")
    parser.add_argument("--seq-len", type=int, default=120, help="Sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic training")
    
    args = parser.parse_args()
    
    trainer = GlobalTrainer(seed=args.seed, history_period=args.period, sequence_length=args.seq_len)
    
    if args.model == "both":
        trainer.train_all_global_models()
    else:
        trainer.train_global_model(args.model)
