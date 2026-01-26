"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    DEEP LEARNING PER TRADING                                 ║
║                                                                              ║
║                 Da NumPy a Neural Networks per Crypto                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PREREQUISITI:
pip install numpy pandas scikit-learn tensorflow

OPZIONALE:
pip install torch  # PyTorch alternativo

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
#                    PARTE 1: NUMPY FOUNDATIONS
# ══════════════════════════════════════════════════════════════════════════════

"""
Prima di usare TensorFlow/PyTorch, devi capire NumPy profondamente.
Le neural networks sono essenzialmente operazioni su array.
"""

class NumpyFoundations:
    """
    Fondamenti NumPy per Deep Learning.
    """
    
    @staticmethod
    def demo_arrays():
        """Array basics."""
        print("=" * 60)
        print("NUMPY ARRAYS")
        print("=" * 60)
        
        # Creazione array
        a = np.array([1, 2, 3, 4, 5])
        print(f"1D array: {a}")
        print(f"Shape: {a.shape}")  # (5,)
        
        # 2D array (matrice)
        b = np.array([[1, 2, 3], [4, 5, 6]])
        print(f"\n2D array:\n{b}")
        print(f"Shape: {b.shape}")  # (2, 3)
        
        # 3D array (batch di immagini, sequenze)
        c = np.zeros((10, 28, 28))  # 10 immagini 28x28
        print(f"\n3D array shape: {c.shape}")
        
        # Reshape (FONDAMENTALE per DL)
        d = np.arange(12).reshape(3, 4)
        print(f"\nReshaped (3, 4):\n{d}")
        
        # Flatten
        print(f"Flattened: {d.flatten()}")
    
    @staticmethod
    def demo_operations():
        """Operazioni vettoriali."""
        print("\n" + "=" * 60)
        print("VECTORIZED OPERATIONS")
        print("=" * 60)
        
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])
        
        # Element-wise operations
        print(f"a + b = {a + b}")
        print(f"a * b = {a * b}")  # Element-wise, NON dot product
        print(f"a ** 2 = {a ** 2}")
        
        # Dot product (FONDAMENTALE per neural networks)
        print(f"\nDot product a·b = {np.dot(a, b)}")
        
        # Matrix multiplication
        A = np.random.randn(3, 4)  # 3x4 matrix
        B = np.random.randn(4, 2)  # 4x2 matrix
        C = A @ B  # 3x2 matrix (oppure np.matmul(A, B))
        print(f"\nMatrix mult: (3,4) @ (4,2) = {C.shape}")
        
        # Broadcasting
        X = np.random.randn(3, 4)  # 3 samples, 4 features
        bias = np.array([1, 2, 3, 4])  # 4 biases
        result = X + bias  # Broadcast bias to each row
        print(f"\nBroadcasting: (3,4) + (4,) = {result.shape}")
    
    @staticmethod
    def demo_neural_forward():
        """Forward pass manuale di neural network."""
        print("\n" + "=" * 60)
        print("NEURAL NETWORK FORWARD PASS (Manual)")
        print("=" * 60)
        
        # Input: 5 samples, 3 features
        X = np.random.randn(5, 3)
        
        # Layer 1: 3 input → 4 neurons
        W1 = np.random.randn(3, 4) * 0.01
        b1 = np.zeros(4)
        
        # Layer 2: 4 → 2 neurons (output)
        W2 = np.random.randn(4, 2) * 0.01
        b2 = np.zeros(2)
        
        # Forward pass
        Z1 = X @ W1 + b1  # Linear
        A1 = np.maximum(0, Z1)  # ReLU activation
        
        Z2 = A1 @ W2 + b2  # Linear
        # Softmax for classification
        exp_Z2 = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
        A2 = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)
        
        print(f"Input shape: {X.shape}")
        print(f"After Layer 1 (ReLU): {A1.shape}")
        print(f"Output (Softmax): {A2.shape}")
        print(f"\nPredictions (probabilities):\n{A2}")


# ══════════════════════════════════════════════════════════════════════════════
#                    PARTE 2: DATA PREPARATION FOR TRADING
# ══════════════════════════════════════════════════════════════════════════════

class TradingDataPrep:
    """
    Preparazione dati per modelli di trading.
    """
    
    @staticmethod
    def create_sequences(
        data: np.ndarray,
        seq_length: int,
        target_steps: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea sequenze per LSTM/Transformer.
        
        Args:
            data: Array di prezzi/features
            seq_length: Lunghezza sequenza input (es. 60 = ultime 60 candele)
            target_steps: Quanti step avanti predire
            
        Returns:
            X: (samples, seq_length, features)
            y: (samples, target_steps) o (samples,) se target_steps=1
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length - target_steps + 1):
            X.append(data[i:i+seq_length])
            
            if target_steps == 1:
                y.append(data[i+seq_length])
            else:
                y.append(data[i+seq_length:i+seq_length+target_steps])
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def normalize_data(
        train_data: np.ndarray,
        test_data: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Normalizza dati (fit su train, transform su test).
        
        IMPORTANTE: MAI fittare su test data!
        """
        # Fit on train
        mean = train_data.mean(axis=0)
        std = train_data.std(axis=0)
        
        # Transform
        train_normalized = (train_data - mean) / std
        
        if test_data is not None:
            test_normalized = (test_data - mean) / std
        else:
            test_normalized = None
        
        params = {'mean': mean, 'std': std}
        
        return train_normalized, test_normalized, params
    
    @staticmethod
    def create_labels_classification(
        prices: np.ndarray,
        threshold: float = 0.001
    ) -> np.ndarray:
        """
        Crea labels per classificazione (UP/DOWN/NEUTRAL).
        
        Args:
            prices: Array di prezzi close
            threshold: Soglia per movimento significativo (0.1%)
            
        Returns:
            labels: 0=down, 1=neutral, 2=up
        """
        returns = np.diff(prices) / prices[:-1]
        
        labels = np.ones(len(returns), dtype=int)  # neutral
        labels[returns > threshold] = 2  # up
        labels[returns < -threshold] = 0  # down
        
        return labels
    
    @staticmethod
    def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggiunge features tecniche per ML.
        """
        result = df.copy()
        
        # Returns
        result['return'] = result['close'].pct_change()
        result['return_5'] = result['close'].pct_change(5)
        result['return_10'] = result['close'].pct_change(10)
        
        # Volatility
        result['volatility'] = result['return'].rolling(20).std()
        
        # Moving averages
        result['sma_10'] = result['close'].rolling(10).mean()
        result['sma_20'] = result['close'].rolling(20).mean()
        result['sma_50'] = result['close'].rolling(50).mean()
        
        # Relative position
        result['close_to_sma20'] = result['close'] / result['sma_20'] - 1
        
        # RSI
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = result['close'].ewm(span=12).mean()
        ema26 = result['close'].ewm(span=26).mean()
        result['macd'] = ema12 - ema26
        result['macd_signal'] = result['macd'].ewm(span=9).mean()
        
        # Volume features
        result['volume_sma'] = result['volume'].rolling(20).mean()
        result['volume_ratio'] = result['volume'] / result['volume_sma']
        
        # Drop NaN
        result = result.dropna()
        
        return result


# ══════════════════════════════════════════════════════════════════════════════
#                    PARTE 3: TENSORFLOW MODELS
# ══════════════════════════════════════════════════════════════════════════════

def check_tensorflow():
    """Verifica installazione TensorFlow."""
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        return True
    except ImportError:
        print("TensorFlow non installato. Esegui: pip install tensorflow")
        return False


class TensorFlowModels:
    """
    Modelli TensorFlow per trading.
    """
    
    @staticmethod
    def create_mlp_classifier(
        input_shape: int,
        num_classes: int = 3
    ):
        """
        Multi-Layer Perceptron per classificazione.
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            print("Installa TensorFlow: pip install tensorflow")
            return None
        
        model = keras.Sequential([
            keras.layers.Input(shape=(input_shape,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def create_lstm_model(
        seq_length: int,
        n_features: int,
        output_size: int = 1
    ):
        """
        LSTM per previsione prezzo.
        
        Args:
            seq_length: Lunghezza sequenza (es. 60)
            n_features: Numero features per timestep
            output_size: 1 per regressione, n_classes per classificazione
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            return None
        
        model = keras.Sequential([
            keras.layers.Input(shape=(seq_length, n_features)),
            
            # First LSTM layer
            keras.layers.LSTM(128, return_sequences=True),
            keras.layers.Dropout(0.2),
            
            # Second LSTM layer
            keras.layers.LSTM(64, return_sequences=False),
            keras.layers.Dropout(0.2),
            
            # Dense layers
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(output_size)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    @staticmethod
    def create_transformer_model(
        seq_length: int,
        n_features: int,
        num_heads: int = 4,
        ff_dim: int = 64,
        num_transformer_blocks: int = 2
    ):
        """
        Transformer per time series.
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            return None
        
        inputs = keras.layers.Input(shape=(seq_length, n_features))
        x = inputs
        
        for _ in range(num_transformer_blocks):
            # Multi-head attention
            attention_output = keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=n_features
            )(x, x)
            x = keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Feed-forward network
            ffn = keras.Sequential([
                keras.layers.Dense(ff_dim, activation='relu'),
                keras.layers.Dense(n_features)
            ])
            ffn_output = ffn(x)
            x = keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        # Global pooling and output
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(1)(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model


# ══════════════════════════════════════════════════════════════════════════════
#                    PARTE 4: TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class TradingMLPipeline:
    """
    Pipeline completa per training modelli trading.
    """
    
    def __init__(
        self,
        seq_length: int = 60,
        prediction_horizon: int = 1,
        train_split: float = 0.8
    ):
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.train_split = train_split
        self.model = None
        self.scaler_params = None
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        feature_cols: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepara dati per training.
        """
        # Add features
        df_features = TradingDataPrep.add_technical_features(df)
        
        # Select features
        if feature_cols is None:
            feature_cols = ['close', 'return', 'volatility', 'rsi', 
                          'macd', 'close_to_sma20', 'volume_ratio']
        
        # Filter available columns
        feature_cols = [c for c in feature_cols if c in df_features.columns]
        
        data = df_features[feature_cols].values
        
        # Split train/test BEFORE creating sequences
        split_idx = int(len(data) * self.train_split)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        # Normalize
        train_norm, test_norm, self.scaler_params = TradingDataPrep.normalize_data(
            train_data, test_data
        )
        
        # Create sequences
        X_train, y_train = TradingDataPrep.create_sequences(
            train_norm, self.seq_length, self.prediction_horizon
        )
        X_test, y_test = TradingDataPrep.create_sequences(
            test_norm, self.seq_length, self.prediction_horizon
        )
        
        # Per regressione, target è solo il close (prima colonna)
        if self.prediction_horizon == 1:
            y_train = y_train[:, 0]
            y_test = y_test[:, 0]
        
        print(f"Training data: X={X_train.shape}, y={y_train.shape}")
        print(f"Test data: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        model_type: str = 'lstm',
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Allena il modello.
        """
        n_features = X_train.shape[2]
        
        # Create model
        if model_type == 'lstm':
            self.model = TensorFlowModels.create_lstm_model(
                self.seq_length, n_features
            )
        elif model_type == 'transformer':
            self.model = TensorFlowModels.create_transformer_model(
                self.seq_length, n_features
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if self.model is None:
            print("TensorFlow non disponibile")
            return None
        
        # Callbacks
        try:
            from tensorflow import keras
            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.5, patience=5
                )
            ]
        except ImportError:
            callbacks = []
        
        # Validation data
        validation_data = None
        if X_val is not None:
            validation_data = (X_val, y_val)
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Genera predizioni."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict(X)
    
    def evaluate_trading(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prices: np.ndarray
    ) -> dict:
        """
        Valuta performance trading.
        """
        # Direction accuracy
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred.flatten()))
        direction_accuracy = np.mean(true_direction == pred_direction)
        
        # Simulated returns
        returns = np.diff(prices) / prices[:-1]
        strategy_returns = returns[self.seq_length:len(pred_direction)+self.seq_length] * pred_direction
        
        cumulative_return = np.prod(1 + strategy_returns) - 1
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        
        return {
            'direction_accuracy': direction_accuracy,
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe,
            'mse': np.mean((y_true - y_pred.flatten()) ** 2)
        }


# ══════════════════════════════════════════════════════════════════════════════
#                    PARTE 5: ESEMPIO COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def run_complete_example():
    """Esempio completo end-to-end."""
    print("=" * 70)
    print("          DEEP LEARNING TRADING EXAMPLE")
    print("=" * 70)
    
    # 1. Generate sample data
    print("\n1. GENERATING SAMPLE DATA...")
    np.random.seed(42)
    
    n_samples = 1000
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='1h')
    
    # Simulated price with trend and noise
    trend = np.linspace(0, 10, n_samples)
    seasonality = 5 * np.sin(np.linspace(0, 20*np.pi, n_samples))
    noise = np.random.randn(n_samples) * 2
    prices = 50000 + trend * 100 + seasonality * 100 + np.cumsum(noise * 50)
    
    df = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.randint(100, 1000, n_samples)
    }, index=dates)
    
    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # 2. Prepare data
    print("\n2. PREPARING DATA...")
    pipeline = TradingMLPipeline(seq_length=60, prediction_horizon=1)
    X_train, y_train, X_test, y_test = pipeline.prepare_data(df)
    
    # 3. NumPy foundations demo
    print("\n3. NUMPY FOUNDATIONS...")
    NumpyFoundations.demo_arrays()
    NumpyFoundations.demo_operations()
    NumpyFoundations.demo_neural_forward()
    
    # 4. Check TensorFlow
    print("\n4. CHECKING TENSORFLOW...")
    if check_tensorflow():
        print("\n5. TRAINING MODEL...")
        
        # Split validation from train
        val_size = int(len(X_train) * 0.1)
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train_final, y_train_final = X_train[:-val_size], y_train[:-val_size]
        
        # Train (reduced epochs for demo)
        history = pipeline.train(
            X_train_final, y_train_final,
            X_val, y_val,
            model_type='lstm',
            epochs=5,  # More for real training
            batch_size=32
        )
        
        if history:
            print("\n6. EVALUATING...")
            predictions = pipeline.predict(X_test)
            
            metrics = pipeline.evaluate_trading(
                y_test, predictions, df['close'].values
            )
            
            print(f"   Direction Accuracy: {metrics['direction_accuracy']:.2%}")
            print(f"   Cumulative Return: {metrics['cumulative_return']:.2%}")
            print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"   MSE: {metrics['mse']:.6f}")
    else:
        print("\n   Skipping training (TensorFlow not available)")
        print("   Install with: pip install tensorflow")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    Hai imparato:
    
    1. NumPy Foundations
       - Array operations
       - Matrix multiplication
       - Manual forward pass
    
    2. Data Preparation
       - Sequence creation for LSTM
       - Normalization (fit on train only!)
       - Technical features
    
    3. Model Architecture
       - MLP for classification
       - LSTM for sequences
       - Transformer attention
    
    4. Training Pipeline
       - Train/val/test split
       - Early stopping
       - Learning rate scheduling
    
    5. Trading Evaluation
       - Direction accuracy
       - Cumulative returns
       - Sharpe ratio
    
    NEXT STEPS:
    - Prova con dati reali (ccxt)
    - Ottimizza iperparametri
    - Aggiungi più features
    - Implementa walk-forward validation
    - Considera ensemble di modelli
    """)


if __name__ == "__main__":
    run_complete_example()
