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
        self.asset_configs = {
            'stocks': {
                'price_range': (100, 5000),  # Typical stock price range
                'scaler_range': (0, 1),  # Standard scaling range for features
                'target_scaler_range': (0, 1),  # Standard range for better generalization
                'sequence_length': 120,
                'history_period': '5y',
                'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
                'model_params': {
                    'lstm_units': [128, 64, 32],  # Deeper architecture for better learning
                    'dropout': 0.1,  # Lower dropout for stocks
                    'epochs': 100,  # More epochs for better convergence
                    'batch_size': 32,  # Larger batch for stability
                    'learning_rate': 0.0005,  # Lower learning rate for precision
                    'patience': 20  # More patience for early stopping
                }
            },
            'indices': {
                'price_range': (10000, 60000),  # Extended range to include Bank Nifty (~52k)
                'scaler_range': (0, 1),  # Standard scaling range for features
                'target_scaler_range': (0, 1),  # Standard range for consistency
                'sequence_length': 120,
                'history_period': '5y',
                'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
                'model_params': {
                    'lstm_units': [128, 64, 32],  # Consistent architecture
                    'dropout': 0.15,  # Slightly higher dropout for indices
                    'epochs': 100,  # More epochs for indices
                    'batch_size': 32,  # Larger batch for indices
                    'learning_rate': 0.0005,  # Consistent learning rate
                    'patience': 20
                }
            }
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
        """Create scaler with asset-specific range optimized for stocks vs indices"""
        if is_target:
            # Different target scaling strategies for stocks vs indices
            if asset_config == self.asset_configs['stocks']:
                # For stocks: Use tighter range for better precision
                return MinMaxScaler(feature_range=(0.1, 0.9))
            else:
                # For indices: Use full range due to larger price movements
                return MinMaxScaler(feature_range=(0.0, 1.0))
        else:
            # Feature scaling remains consistent
            scaler_min, scaler_max = asset_config['scaler_range']
            return MinMaxScaler(feature_range=(scaler_min, scaler_max))

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
        """Build improved LSTM model with deeper architecture"""
        params = asset_config['model_params']
        
        # Create optimizer with asset-specific learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
        
        model = Sequential([
            LSTM(params['lstm_units'][0], return_sequences=True, input_shape=input_shape, recurrent_dropout=0.1),
            BatchNormalization(),
            LSTM(params['lstm_units'][1], return_sequences=True, recurrent_dropout=0.1),
            BatchNormalization(),
            LSTM(params['lstm_units'][2], recurrent_dropout=0.1),
            BatchNormalization(),
            Dropout(params['dropout']),
            Dense(32, activation='relu'),
            Dropout(params['dropout'] / 2),
            Dense(16, activation='relu'),
            Dropout(params['dropout'] / 2),
            Dense(1)
        ])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

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

    def transformer_encoder(self, inputs, head_size=32, num_heads=2, ff_dim=64, dropout=0.1):
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

    def build_transformer_model(self, input_shape, asset_config):
        """Build Transformer model with asset-specific parameters"""
        params = asset_config['model_params']
        
        # Create optimizer with asset-specific learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
        
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Single transformer layer for simplicity
        x = self.transformer_encoder(x, dropout=params['dropout'])
        
        x = Flatten()(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(params['dropout'])(x)
        x = Dense(16, activation="relu")(x)
        x = Dropout(params['dropout'] / 2)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def train_asset_specific_model(self, symbol, model_type="lstm", history_period=None, sequence_length=None, init_from_global=False):
        """Train asset-specific model with proper validation"""
        print(f"\n[INFO] Training {model_type.upper()} model for {symbol}")
        
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
        
        # Training callbacks with improved patience
        early_stop = EarlyStopping(monitor='val_loss', patience=params['patience'], restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7)
        symbol_key = symbol.replace('.', '_').replace('^', 'INDEX_')
        asset_model_key = f"{model_type}_{asset_type}_{symbol_key}"
        best_checkpoint_path = os.path.join(MODEL_DIR, f"{asset_model_key}_best.keras")
        checkpoint = ModelCheckpoint(
            filepath=best_checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
        print(f"[INFO] Starting training with {params['epochs']} epochs...")
        
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(X_val_seq, y_val_seq),
            callbacks=[early_stop, reduce_lr, checkpoint],
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
            )
        if args.model in ["transformer", "both"]:
            trainer.train_asset_specific_model(
                args.symbol,
                "transformer",
                history_period=args.period,
                sequence_length=args.seq_len,
                init_from_global=False,
            )
    else:
        print("Please specify --symbol or --all flag")
