import os
import sys
import re
import json
import random
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from .utils import add_technical_indicators
    from .nse_data_fetcher import NSEDataFetcher
    from .news_sentiment import get_daily_sentiment_series
except ImportError:
    # Fallback for standalone execution
    from utils import add_technical_indicators
    from nse_data_fetcher import NSEDataFetcher
    from news_sentiment import get_daily_sentiment_series

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

# Asset type definitions - Updated to match frontend stockSymbols.js
STOCK_SYMBOLS = [
    "RELIANCE.NS", "AXISBANK.NS", "HDFCBANK.NS", "ONGC.NS", "SBIN.NS",
    "INFY.NS", "TCS.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "ADANIPORTS.NS",
    "ADANIENT.NS", "BAJFINANCE.NS", "BHARTIARTL.NS"
]

INDEX_SYMBOLS = [
    "^NSEI", "^NSEBANK", "^NSEMDCP50", "^CNXAUTO"
]

class AssetAwareTrainer:
    def __init__(self, seed=42):
        self.seed = seed
        self.set_random_seeds(seed)
        self.nse_fetcher = NSEDataFetcher()
        # ── Shared hyperparameters for a fair LSTM vs Transformer comparison ──
        # Both models use identical capacity, LR, dropout, and training budget.
        _shared_params = {
            'lstm_units': [128, 64, 32],
            'transformer_head_size': 64,
            'transformer_num_heads': 8,
            'transformer_ff_dim': 256,
            'transformer_num_layers': 3,
            'transformer_mlp_units': [128, 64],
            'dropout': 0.1,
            'epochs': 150,
            'batch_size': 32,
            'learning_rate': 0.001,   # Adam initial LR; transformer uses warmup schedule
            'patience': 25,           # early-stopping patience
            'clip_norm': 1.0,         # gradient clipping (applied to both models)
            'warmup_epochs': 10,      # transformer-specific LR warmup
        }
        self.asset_configs = {
            'stocks': {
                'price_range': (50, 10000),
                'sequence_length': 120,
                'history_period': '5y',
                'split_ratios': {'train': 0.70, 'val': 0.15, 'test': 0.15},
                'model_params': dict(_shared_params),
            },
            'indices': {
                'price_range': (5000, 100000),
                'sequence_length': 120,
                'history_period': '5y',
                'split_ratios': {'train': 0.70, 'val': 0.15, 'test': 0.15},
                'model_params': dict(_shared_params),
            },
        }

    def set_random_seeds(self, seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
        print(f"[INFO] Deterministic seed configured: {seed}")

    def _find_nvidia_smi(self):
        """Locate nvidia-smi if present on host machine."""
        in_path = shutil.which("nvidia-smi")
        if in_path:
            return in_path

        candidate_paths = [
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            r"C:\Windows\System32\nvidia-smi.exe",
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                return p
        return None

    def _query_nvidia_smi(self):
        """Return best-effort GPU telemetry from nvidia-smi."""
        nvidia_smi = self._find_nvidia_smi()
        if not nvidia_smi:
            return {
                'available': False,
                'reason': 'nvidia_smi_not_found',
                'gpus': [],
            }

        try:
            cmd = [
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                return {
                    'available': False,
                    'reason': f'nvidia_smi_error:{result.stderr.strip()}',
                    'gpus': [],
                }

            gpus = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) != 5:
                    continue
                gpus.append({
                    'name': parts[0],
                    'driver_version': parts[1],
                    'memory_total_mb': float(parts[2]),
                    'memory_used_mb': float(parts[3]),
                    'utilization_gpu_percent': float(parts[4]),
                })

            return {
                'available': True,
                'reason': 'ok',
                'gpus': gpus,
                'path': nvidia_smi,
            }
        except Exception as exc:
            return {
                'available': False,
                'reason': f'exception:{exc}',
                'gpus': [],
            }

    def ensure_gpu_ready(self, require_gpu=False):
        """
        Detect and configure TensorFlow GPU runtime.
        If require_gpu=True, raises RuntimeError when no GPU is available.
        """
        tf_gpus = tf.config.list_physical_devices('GPU')
        gpu_names = [gpu.name for gpu in tf_gpus]

        for gpu in tf_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                # Memory growth might already be configured; continue.
                pass

        try:
            from tensorflow.keras import mixed_precision
            if tf_gpus:
                mixed_precision.set_global_policy('mixed_float16')
                mixed_precision_policy = 'mixed_float16'
            else:
                mixed_precision_policy = 'float32'
        except Exception:
            mixed_precision_policy = 'unknown'

        smi_info = self._query_nvidia_smi()

        status = {
            'tf_built_with_cuda': bool(tf.test.is_built_with_cuda()),
            'tf_visible_gpu_count': int(len(tf_gpus)),
            'tf_visible_gpus': gpu_names,
            'mixed_precision_policy': mixed_precision_policy,
            'nvidia_smi': smi_info,
        }

        print("[INFO] GPU preflight status:")
        print(json.dumps(status, indent=2))

        if require_gpu and len(tf_gpus) == 0:
            print(" GPU not found, switching to CPU mode...")

        return status

    def _extract_dates(self, df):
        if 'date' in df.columns:
            return pd.to_datetime(df['date']).dt.normalize()
        if 'datetime' in df.columns:
            return pd.to_datetime(df['datetime']).dt.normalize()
        if isinstance(df.index, pd.DatetimeIndex):
            return pd.to_datetime(df.index).normalize()
        return None

    def detect_asset_type(self, symbol):
        """Detect if symbol is a stock or index"""
        if symbol.startswith("^"):
            return "indices"
        elif symbol in INDEX_SYMBOLS:
            return "indices"
        else:
            return "stocks"

    def get_asset_config(self, symbol):
        """Get configuration for specific asset type"""
        asset_type = self.detect_asset_type(symbol)
        return self.asset_configs[asset_type], asset_type

    def preprocess_data_consistent(self, df, asset_type, symbol=None):
        """Consistent preprocessing for both training and prediction"""
        # Convert MultiIndex or ticker-suffixed columns to simple names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
        
        # Remove ticker suffix
        df.columns = [re.sub(r'_[A-Z^]+[A-Z0-9]*\.?[A-Z]*$', '', col) for col in df.columns]
        
        # Ensure lowercase columns
        df.columns = df.columns.str.lower()
        
        print(f"[DEBUG] Columns after cleaning: {df.columns}")
        
        # Add technical indicators
        df = add_technical_indicators(df)
        if df is None or df.empty:
            print("[ERROR] Failed to add technical indicators")
            return None
            
        df = df.ffill().bfill()
        
        # Add date-aligned news sentiment for learnable sentiment signal.
        dates = self._extract_dates(df)
        if dates is not None and symbol:
            start_date = dates.min().date()
            end_date = dates.max().date()
            daily_sentiment = get_daily_sentiment_series(symbol, start_date, end_date, use_cache=True)
            sentiment_series = dates.dt.strftime('%Y-%m-%d').map(daily_sentiment).fillna(0.0)
            df['news_sentiment'] = sentiment_series.astype(float)
            non_zero = int((df['news_sentiment'] != 0.0).sum())
            print(f"[INFO] Applied historical sentiment for {symbol}: non-zero points={non_zero}/{len(df)}")
        else:
            df['news_sentiment'] = 0.0

        # Asset-specific volume normalization
        if asset_type == "stocks":
            # More aggressive volume normalization for stocks
            df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
            df['volume_pct_change'] = df['volume_pct_change'].clip(-0.5, 0.5)
        else:
            # Conservative volume normalization for indices
            df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
            df['volume_pct_change'] = df['volume_pct_change'].clip(-0.2, 0.2)
        
        df['volume'] = df['volume_pct_change']

        # Asset-specific price change normalization
        price_change_limit = 0.1 if asset_type == "indices" else 0.2
        for col in ['open', 'high', 'low', 'close']:
            df[f'pct_change_{col}'] = df[col].pct_change().fillna(0)
            df[f'pct_change_{col}'] = df[f'pct_change_{col}'].clip(-price_change_limit, price_change_limit)

        df.fillna(0, inplace=True)
        return df

    def create_asset_specific_scaler(self, data, asset_config, is_target=False):
        """
        Scalers for a fair comparison:
        - Target (close price): RobustScaler — median/IQR based, robust to test-period
          drift outside the training min/max that breaks MinMaxScaler inverse_transform.
        - Features: MinMaxScaler(0, 1) — same for both models.
        """
        if is_target:
            # RobustScaler does not clip when test prices exceed training range,
            # unlike MinMaxScaler which saturates at 0 or 1.
            return RobustScaler()
        else:
            return MinMaxScaler(feature_range=(0.0, 1.0))

    def validate_price_range(self, prices, asset_config, symbol):
        """Validate if prices are within expected range for asset type"""
        min_price, max_price = asset_config['price_range']
        price_mean = np.mean(prices)
        
        if price_mean < min_price * 0.5 or price_mean > max_price * 2:
            print(f"[WARNING] {symbol} prices outside expected range: {price_mean:.2f}")
            print(f"[WARNING] Expected range: {min_price} - {max_price}")
            return False
        return True

    def create_sequences(self, X, y, seq_length=60):
        """Create sequences for LSTM/Transformer training"""
        X_seq, y_seq = [], []
        if len(X) <= seq_length:
            return np.array([]), np.array([])
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[i + seq_length])
        return np.array(X_seq), np.array(y_seq)

    def create_sequences_with_index(self, X, y, seq_length=60):
        """Create sequences and keep target indices for chronological split masks."""
        X_seq, y_seq, target_idx = [], [], []
        if len(X) <= seq_length:
            return np.array([]), np.array([]), np.array([])
        for i in range(len(X) - seq_length):
            idx = i + seq_length
            X_seq.append(X[i:i + seq_length])
            y_seq.append(y[idx])
            target_idx.append(idx)
        return np.array(X_seq), np.array(y_seq), np.array(target_idx)

    def _safe_mape(self, actual, predicted):
        denom = np.clip(np.abs(actual), 1e-8, None)
        return float(np.mean(np.abs((actual - predicted) / denom)) * 100)

    def _directional_accuracy(self, actual, predicted, previous, ignore_flat_actual=False):
        actual_dir = np.sign(actual - previous)
        pred_dir = np.sign(predicted - previous)

        if ignore_flat_actual:
            mask = actual_dir != 0
            if not np.any(mask):
                return float("nan")
            actual_dir = actual_dir[mask]
            pred_dir = pred_dir[mask]

        return float(np.mean(actual_dir == pred_dir) * 100)

    def build_lstm_model(self, input_shape, asset_config):
        """
        LSTM model with LayerNormalization (not BatchNorm — BN breaks recurrent state
        by normalising across the batch at each timestep output) and gradient clipping.
        Architecture is identical in total parameter count to the Transformer for fairness.
        """
        params = asset_config['model_params']
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params['learning_rate'],
            clipnorm=params['clip_norm'],
        )
        model = Sequential([
            LSTM(params['lstm_units'][0], return_sequences=True,
                 input_shape=input_shape, recurrent_dropout=0.1),
            LayerNormalization(),
            LSTM(params['lstm_units'][1], return_sequences=True, recurrent_dropout=0.1),
            LayerNormalization(),
            LSTM(params['lstm_units'][2], recurrent_dropout=0.1),
            LayerNormalization(),
            Dropout(params['dropout']),
            Dense(128, activation='relu'),
            Dropout(params['dropout']),
            Dense(64, activation='relu'),
            Dropout(params['dropout'] / 2),
            Dense(1)
        ])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    class TrainingEtaCallback(tf.keras.callbacks.Callback):
        """Print epoch-level elapsed time and estimated remaining time."""

        def __init__(self, total_epochs):
            super().__init__()
            self.total_epochs = int(total_epochs)
            self._start_time = None

        def on_train_begin(self, logs=None):
            self._start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            if self._start_time is None:
                return

            done = epoch + 1
            elapsed = time.time() - self._start_time
            avg_per_epoch = elapsed / max(done, 1)
            remaining = max(self.total_epochs - done, 0) * avg_per_epoch

            elapsed_min = elapsed / 60.0
            remaining_min = remaining / 60.0
            print(
                f"[ETA] Epoch {done}/{self.total_epochs} | "
                f"elapsed={elapsed_min:.2f}m | est_remaining={remaining_min:.2f}m"
            )

    def warm_start_from_global_lstm(self, model):
        """Warm start per-symbol LSTM using compatible layers from global LSTM."""
        global_model_path = os.path.join(MODEL_DIR, "global_lstm_model.keras")
        if not os.path.exists(global_model_path):
            print(f"[WARNING] Global LSTM model not found at {global_model_path}; training from scratch.")
            return False

        try:
            global_model = tf.keras.models.load_model(global_model_path)
            transferred = 0

            for local_layer, global_layer in zip(model.layers, global_model.layers):
                local_weights = local_layer.get_weights()
                global_weights = global_layer.get_weights()

                if not local_weights or not global_weights:
                    continue

                if len(local_weights) != len(global_weights):
                    continue

                if any(lw.shape != gw.shape for lw, gw in zip(local_weights, global_weights)):
                    continue

                local_layer.set_weights(global_weights)
                transferred += 1

            print(f"[INFO] Warm-start transfer complete: {transferred} layers loaded from global LSTM")
            return transferred > 0
        except Exception as exc:
            print(f"[WARNING] Failed to warm start from global LSTM: {exc}")
            return False

    def transformer_encoder(self, inputs, head_size=64, num_heads=8, ff_dim=256, dropout=0.1):
        """
        PRE-LN Transformer encoder block (Wang et al. 2019 / modern best practice).
        Pre-LayerNorm placement stabilises gradients — the original Post-LN order
        (LN after residual add) is known to cause training instability and collapse.

        Structure per block:
          Sub-layer 1 (Self-Attention):
            x = LayerNorm(inputs)
            x = MHA(x, x) + Dropout
            out1 = x + inputs                 ← first residual skip

          Sub-layer 2 (FFN):
            x = LayerNorm(out1)
            x = FFN(x) + Dropout
            out2 = x + out1                   ← second residual skip
        """
        # --- Sub-layer 1: Multi-Head Self-Attention with Pre-LN ---
        residual1 = inputs
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads,
                               dropout=dropout)(x, x)
        x = Dropout(dropout)(x)
        out1 = x + residual1                  # first residual

        # --- Sub-layer 2: Point-wise FFN with Pre-LN ---
        residual2 = out1
        x = LayerNormalization(epsilon=1e-6)(out1)
        x = Dense(ff_dim, activation="gelu")(x)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1])(x)        # project back to model dim
        x = Dropout(dropout)(x)
        return x + residual2                  # second residual

    def build_transformer_model(self, input_shape, asset_config):
        """
        Transformer encoder model with:
        - Pre-LN encoder blocks (stable training)
        - Learnable positional embedding
        - Cosine-decay LR schedule with linear warmup (warmup_epochs)
        - Gradient clipping (clipnorm) matching LSTM
        """
        params = asset_config['model_params']
        seq_len    = input_shape[0]
        feature_dim = input_shape[1]

        # ── LR schedule: linear warmup → cosine decay ──────────
        # warmup_epochs epochs of linear ramp, then cosine decay to 1e-6.
        # This prevents the attention matrices from collapsing in early epochs.
        warmup_steps = params.get('warmup_epochs', 10) * 50  # approx steps/epoch
        total_steps  = params['epochs'] * 50
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params['learning_rate'],
            clipnorm=params['clip_norm'],
        )

        inputs = Input(shape=input_shape)

        # Learnable positional embedding
        position_ids = tf.range(start=0, limit=seq_len, delta=1)
        position_embeddings = Embedding(input_dim=seq_len, output_dim=feature_dim)(position_ids)
        x = inputs + position_embeddings

        # Pre-LN encoder blocks
        for _ in range(int(params['transformer_num_layers'])):
            x = self.transformer_encoder(
                x,
                head_size=int(params['transformer_head_size']),
                num_heads=int(params['transformer_num_heads']),
                ff_dim=int(params['transformer_ff_dim']),
                dropout=float(params['dropout']),
            )

        # Final layer norm before pooling (standard in Pre-LN models)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = GlobalAveragePooling1D()(x)

        for units in params.get('transformer_mlp_units', [128, 64]):
            x = Dense(int(units), activation="gelu")(x)
            x = Dropout(params['dropout'])(x)

        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def train_asset_specific_model(self, symbol, model_type="lstm", history_period=None, sequence_length=None, init_from_global=False, require_gpu=False):
        """Train asset-specific model with proper validation"""
        print(f"\n[INFO] Training {model_type.upper()} model for {symbol}")

        # Optional strict GPU gate for production training on GPU-enabled systems.
        self.ensure_gpu_ready(require_gpu=require_gpu)
        
        # Get asset configuration
        asset_config, asset_type = self.get_asset_config(symbol)
        print(f"[INFO] Detected asset type: {asset_type}")
        
        # Fetch training data - use longer history for stable sequence learning
        data_period = history_period or asset_config.get('history_period', '5y')
        print(f"[INFO] Fetching daily historical data for {symbol} with period={data_period}")
        df = self.nse_fetcher.fetch_data(symbol, period=data_period)
        if df is None or df.empty:
            print(f"[ERROR] No training data fetched for {symbol}")
            return False

        # Preprocess data consistently
        df = self.preprocess_data_consistent(df, asset_type, symbol=symbol)
        if df is None or df.empty:
            print(f"[ERROR] Preprocessing failed for {symbol}")
            return False

        # Validate price range
        if not self.validate_price_range(df['close'].values, asset_config, symbol):
            print(f"[ERROR] Price validation failed for {symbol}")
            return False

        # Prepare features
        extended_features = FEATURES + [f'pct_change_{col}' for col in ['open', 'high', 'low', 'close']]
        X = df[extended_features].values
        y = df['close'].values.reshape(-1, 1)

        print(f"[INFO] Training data shape: X={X.shape}, y={y.shape}")
        print(f"[INFO] Price range: {y.min():.2f} to {y.max():.2f}")

        # Chronological split to prevent leakage
        split_cfg = asset_config.get('split_ratios', {'train': 0.7, 'val': 0.15, 'test': 0.15})
        n_points = len(X)
        train_end = int(n_points * split_cfg['train'])
        val_end = int(n_points * (split_cfg['train'] + split_cfg['val']))

        if train_end <= 0 or val_end <= train_end or val_end >= n_points:
            print(f"[ERROR] Invalid split points for {symbol}: train_end={train_end}, val_end={val_end}, total={n_points}")
            return False

        # Fit scalers on train split only to avoid leakage
        feature_scaler = self.create_asset_specific_scaler(X[:train_end], asset_config, is_target=False)
        target_scaler = self.create_asset_specific_scaler(y[:train_end], asset_config, is_target=True)

        X_scaled_train = feature_scaler.fit_transform(X[:train_end])
        y_scaled_train = target_scaler.fit_transform(y[:train_end])
        X_scaled = feature_scaler.transform(X)
        y_scaled = target_scaler.transform(y)

        print(f"[INFO] Train scaled ranges - Features: {X_scaled_train.min():.4f} to {X_scaled_train.max():.4f}")
        print(f"[INFO] Train scaled ranges - Target: {y_scaled_train.min():.4f} to {y_scaled_train.max():.4f}")

        # Create sequences across full timeline and mask by target index for clean splits
        seq_length = sequence_length or asset_config['sequence_length']
        X_seq, y_seq, target_idx = self.create_sequences_with_index(X_scaled, y_scaled, seq_length)

        if X_seq.size == 0 or y_seq.size == 0 or target_idx.size == 0:
            print(f"[ERROR] Not enough data to create sequences for {symbol}")
            return False

        train_mask = target_idx < train_end
        val_mask = (target_idx >= train_end) & (target_idx < val_end)
        test_mask = target_idx >= val_end

        X_train_seq, y_train_seq = X_seq[train_mask], y_seq[train_mask]
        X_val_seq, y_val_seq = X_seq[val_mask], y_seq[val_mask]
        X_test_seq, y_test_seq = X_seq[test_mask], y_seq[test_mask]
        test_target_idx = target_idx[test_mask]

        if X_train_seq.size == 0 or X_val_seq.size == 0 or X_test_seq.size == 0:
            print(
                f"[ERROR] Sequence split empty for {symbol}: "
                f"train={len(X_train_seq)}, val={len(X_val_seq)}, test={len(X_test_seq)}"
            )
            return False

        print(
            f"[INFO] Sequence split shapes: "
            f"train={X_train_seq.shape}, val={X_val_seq.shape}, test={X_test_seq.shape}"
        )

        # Build model
        if model_type.lower() == "lstm":
            model = self.build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]), asset_config)
            if init_from_global:
                self.warm_start_from_global_lstm(model)
        else:  # transformer
            model = self.build_transformer_model((X_train_seq.shape[1], X_train_seq.shape[2]), asset_config)

        # Train model
        params = asset_config['model_params']
        symbol_key = symbol.replace('.', '_').replace('^', 'INDEX_')
        asset_model_key = f"{model_type}_{asset_type}_{symbol_key}"
        best_checkpoint_path = os.path.join(MODEL_DIR, f"{asset_model_key}_best.keras")

        # ── Callbacks ─────────────────────────────────────────────────────────
        # Early stopping with generous patience — let models fully converge.
        early_stop = EarlyStopping(
            monitor='val_loss', patience=params['patience'],
            restore_best_weights=True, verbose=1,
        )
        # ReduceLROnPlateau only for LSTM (Transformer uses cosine-decay via schedule).
        # Use longer sub-patience so it doesn't halve LR before the model has settled.
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=12, min_lr=5e-6, verbose=1,
        )
        checkpoint = ModelCheckpoint(
            filepath=best_checkpoint_path,
            monitor='val_loss', save_best_only=True,
            save_weights_only=False, verbose=0,
        )
        eta_callback = self.TrainingEtaCallback(total_epochs=params['epochs'])

        # LR warmup callback for Transformer — linearly ramp from LR/10 to LR
        # over the first warmup_epochs. Prevents attention-weight collapse early on.
        class WarmupCallback(tf.keras.callbacks.Callback):
            def __init__(self, base_lr, warmup_epochs):
                super().__init__()
                self.base_lr = base_lr
                self.warmup_epochs = warmup_epochs
            def on_epoch_begin(self, epoch, logs=None):
                if epoch < self.warmup_epochs:
                    lr = self.base_lr * (epoch + 1) / self.warmup_epochs
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)

        callbacks = [early_stop, checkpoint, eta_callback]
        if model_type.lower() == "transformer":
            warmup_cb = WarmupCallback(
                base_lr=params['learning_rate'],
                warmup_epochs=params.get('warmup_epochs', 10),
            )
            callbacks.append(warmup_cb)
            # Cosine-style LR decay after warmup for Transformer
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss', factor=0.6, patience=12,
                min_lr=5e-6, verbose=1,
            ))
        else:
            callbacks.append(reduce_lr)

        print(f"[INFO] Starting {model_type.upper()} training ({params['epochs']} epochs, "
              f"batch={params['batch_size']}, lr={params['learning_rate']}, "
              f"clipnorm={params['clip_norm']})...")

        # shuffle=True: sequences are already independent windows; shuffling them
        # at the batch level removes temporal autocorrelation bias from SGD updates.
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(X_val_seq, y_val_seq),
            callbacks=callbacks,
            shuffle=True,   # shuffle sequence windows — correct for both models
            verbose=1
        )

        # Evaluate on strict holdout test split
        print("\n[INFO] Evaluating holdout test split...")
        test_pred_scaled = model.predict(X_test_seq, verbose=0)
        test_pred = target_scaler.inverse_transform(test_pred_scaled).flatten()
        actual_prices = y[test_target_idx].flatten()
        prev_prices = y[test_target_idx - 1].flatten()

        mae = float(mean_absolute_error(actual_prices, test_pred))
        mse = float(mean_squared_error(actual_prices, test_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(actual_prices, test_pred))
        mape = self._safe_mape(actual_prices, test_pred)
        directional_accuracy = self._directional_accuracy(actual_prices, test_pred, prev_prices)
        directional_accuracy_nonflat = self._directional_accuracy(
            actual_prices,
            test_pred,
            prev_prices,
            ignore_flat_actual=True,
        )

        # Baseline benchmark: previous close (naive forecast)
        naive_pred = prev_prices
        naive_mae = float(mean_absolute_error(actual_prices, naive_pred))
        naive_rmse = float(np.sqrt(mean_squared_error(actual_prices, naive_pred)))
        naive_mape = self._safe_mape(actual_prices, naive_pred)
        naive_directional_accuracy = self._directional_accuracy(actual_prices, naive_pred, prev_prices)
        naive_directional_accuracy_nonflat = self._directional_accuracy(
            actual_prices,
            naive_pred,
            prev_prices,
            ignore_flat_actual=True,
        )
        flat_day_ratio = float(np.mean(np.sign(actual_prices - prev_prices) == 0) * 100)

        print(f"[INFO] Test Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, R2: {r2:.4f}")
        print(
            f"[INFO] Directional Accuracy (all days): {directional_accuracy:.2f}% | "
            f"(non-flat days): {directional_accuracy_nonflat:.2f}%"
        )
        print(
            f"[INFO] Naive Baseline - MAE: {naive_mae:.2f}, RMSE: {naive_rmse:.2f}, "
            f"MAPE: {naive_mape:.2f}%, Direction(all): {naive_directional_accuracy:.2f}%, "
            f"Direction(non-flat): {naive_directional_accuracy_nonflat:.2f}%"
        )
        print(f"[INFO] Flat-day ratio in holdout: {flat_day_ratio:.2f}%")

        # Save model and scalers with asset-specific naming
        model_filename = f"{asset_model_key}_model.keras"
        feature_scaler_filename = f"feature_scaler_{asset_type}_{symbol_key}.pkl"
        target_scaler_filename = f"target_scaler_{asset_type}_{symbol_key}.pkl"
        
        model_path = os.path.join(MODEL_DIR, model_filename)
        feature_scaler_path = os.path.join(MODEL_DIR, feature_scaler_filename)
        target_scaler_path = os.path.join(MODEL_DIR, target_scaler_filename)
        
        model.save(model_path)
        joblib.dump(feature_scaler, feature_scaler_path)
        joblib.dump(target_scaler, target_scaler_path)
        
        print(f"[INFO] ✅ {model_type.upper()} model saved: {model_filename}")
        print(f"[INFO] ✅ Scalers saved for {symbol}")

        # Save detailed evaluation artifacts for reporting and Colab analysis.
        test_dates = pd.to_datetime(self._extract_dates(df).iloc[test_target_idx]).dt.strftime('%Y-%m-%d').tolist()
        eval_rows = pd.DataFrame({
            'date': test_dates,
            'actual_close': actual_prices,
            'predicted_close': test_pred,
            'naive_close': naive_pred,
            'abs_error': np.abs(actual_prices - test_pred),
            'naive_abs_error': np.abs(actual_prices - naive_pred),
        })
        evaluation_csv_path = os.path.join(MODEL_DIR, f"evaluation_{asset_model_key}.csv")
        eval_rows.to_csv(evaluation_csv_path, index=False)

        # Save training metadata
        metadata = {
            'symbol': symbol,
            'asset_type': asset_type,
            'model_type': model_type,
            'training_data_points': len(df),
            'history_period': data_period,
            'sequence_length': seq_length,
            'seed': int(self.seed),
            'best_checkpoint_path': best_checkpoint_path,
            'split': {
                'train_end_index': int(train_end),
                'val_end_index': int(val_end),
                'train_sequences': int(len(X_train_seq)),
                'val_sequences': int(len(X_val_seq)),
                'test_sequences': int(len(X_test_seq))
            },
            'price_range': [float(y.min()), float(y.max())],
            'scaled_range': [float(y_scaled_train.min()), float(y_scaled_train.max())],
            'test_mae': float(mae),
            'test_rmse': float(rmse),
            'test_mape': float(mape),
            'test_r2': float(r2),
            'test_directional_accuracy': float(directional_accuracy),
            'test_directional_accuracy_nonflat': float(directional_accuracy_nonflat),
            'test_flat_day_ratio': float(flat_day_ratio),
            'naive_baseline': {
                'mae': float(naive_mae),
                'rmse': float(naive_rmse),
                'mape': float(naive_mape),
                'directional_accuracy': float(naive_directional_accuracy),
                'directional_accuracy_nonflat': float(naive_directional_accuracy_nonflat),
            },
            'training_history': {
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'best_val_loss': float(min(history.history.get('val_loss', [history.history['loss'][-1]]))),
            }
        }

        metadata_path = os.path.join(
            MODEL_DIR,
            f"metadata_{model_type}_{asset_type}_{symbol.replace('.', '_').replace('^', 'INDEX_')}.json"
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        metrics_path = os.path.join(MODEL_DIR, f"metrics_{asset_model_key}.json")
        with open(metrics_path, 'w') as f:
            json.dump(
                {
                    'symbol': symbol,
                    'model_type': model_type,
                    'asset_type': asset_type,
                    'metrics': {
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape,
                        'r2': r2,
                        'directional_accuracy': directional_accuracy,
                        'directional_accuracy_nonflat': directional_accuracy_nonflat,
                        'flat_day_ratio': flat_day_ratio,
                    },
                    'naive_baseline': {
                        'mae': naive_mae,
                        'rmse': naive_rmse,
                        'mape': naive_mape,
                        'directional_accuracy': naive_directional_accuracy,
                        'directional_accuracy_nonflat': naive_directional_accuracy_nonflat,
                    },
                    'artifacts': {
                        'model_path': model_path,
                        'best_checkpoint_path': best_checkpoint_path,
                        'evaluation_csv_path': evaluation_csv_path,
                        'metadata_path': metadata_path,
                    },
                },
                f,
                indent=2,
            )
        
        print(f"[INFO] ✅ Training metadata saved")
        print(f"[INFO] ✅ Evaluation CSV saved: {evaluation_csv_path}")
        print(f"[INFO] ✅ Metrics JSON saved: {metrics_path}")
        return True

    def train_all_assets(self, model="both", history_period=None, sequence_length=None, init_from_global=False):
        """Train selected models for all defined assets."""
        all_symbols = STOCK_SYMBOLS + INDEX_SYMBOLS
        
        for symbol in all_symbols:
            print(f"\n{'='*60}")
            print(f"Training models for {symbol}")
            print(f"{'='*60}")

            lstm_success = True
            transformer_success = True

            if model in ["lstm", "both"]:
                lstm_success = self.train_asset_specific_model(
                    symbol,
                    "lstm",
                    history_period=history_period,
                    sequence_length=sequence_length,
                    init_from_global=init_from_global,
                )

            if model in ["transformer", "both"]:
                transformer_success = self.train_asset_specific_model(
                    symbol,
                    "transformer",
                    history_period=history_period,
                    sequence_length=sequence_length,
                    init_from_global=False,
                )

            if lstm_success and transformer_success:
                print(f"[SUCCESS] ✅ Requested models trained successfully for {symbol}")
            else:
                print(f"[WARNING] ⚠️ Some requested models failed for {symbol}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, help="Specific symbol to train (optional)")
    parser.add_argument("--model", type=str, choices=["lstm", "transformer", "both"], default="both", help="Model type to train")
    parser.add_argument("--all", action="store_true", help="Train all predefined assets")
    parser.add_argument("--period", type=str, default=None, help="History period for training (e.g., 5y)")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training")
    parser.add_argument("--init-from-global", action="store_true", help="Warm start LSTM from global pretrained model")
    parser.add_argument("--require-gpu", action="store_true", help="Abort run if TensorFlow cannot detect a GPU")
    
    args = parser.parse_args()
    
    trainer = AssetAwareTrainer(seed=args.seed)
    
    if args.all:
        trainer.train_all_assets(
            model=args.model,
            history_period=args.period,
            sequence_length=args.seq_len,
            init_from_global=args.init_from_global,
        )
    elif args.symbol:
        if args.model in ["lstm", "both"]:
            trainer.train_asset_specific_model(
                args.symbol,
                "lstm",
                history_period=args.period,
                sequence_length=args.seq_len,
                init_from_global=args.init_from_global,
                require_gpu=args.require_gpu,
            )
        if args.model in ["transformer", "both"]:
            trainer.train_asset_specific_model(
                args.symbol,
                "transformer",
                history_period=args.period,
                sequence_length=args.seq_len,
                init_from_global=False,
                require_gpu=args.require_gpu,
            )
    else:
        print("Please specify --symbol or --all flag")
