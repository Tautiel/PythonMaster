"""
🚀 SESSIONE 3 - PARTE 5: DEEP LEARNING INTRODUCTION
===================================================
Neural Networks, TensorFlow/Keras, Computer Vision, NLP
Durata: 60 minuti di deep learning fundamentals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Note: TensorFlow/Keras would be imported as:
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, models

print("="*80)
print("🧠 SESSIONE 3 PARTE 5: DEEP LEARNING INTRODUCTION")
print("="*80)

# ==============================================================================
# SEZIONE 1: NEURAL NETWORKS FROM SCRATCH
# ==============================================================================

class NeuralNetworkFromScratch:
    """Simple Neural Network implementation from scratch"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        # Store training history
        self.loss_history = []
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        return x * (1 - x)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation for output"""
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """Forward pass through network"""
        # Input to hidden
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Hidden to output
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward_propagation(self, X, y, output):
        """Backward pass (backpropagation)"""
        m = X.shape[0]
        
        # Output layer gradients
        self.dz2 = output - y
        self.dW2 = (1/m) * np.dot(self.a1.T, self.dz2)
        self.db2 = (1/m) * np.sum(self.dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(self.dz2, self.W2.T)
        self.dz1 = da1 * self.relu_derivative(self.a1)
        self.dW1 = (1/m) * np.dot(X.T, self.dz1)
        self.db1 = (1/m) * np.sum(self.dz1, axis=0, keepdims=True)
        
    def update_weights(self):
        """Update weights using gradient descent"""
        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1
        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2
        
    def calculate_loss(self, y_true, y_pred):
        """Calculate binary cross-entropy loss"""
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + 
                        (1 - y_true) * np.log(1 - y_pred))
    
    def train(self, X, y, epochs=1000, verbose=True):
        """Train the neural network"""
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward_propagation(X)
            
            # Calculate loss
            loss = self.calculate_loss(y, output)
            self.loss_history.append(loss)
            
            # Backward propagation
            self.backward_propagation(X, y, output)
            
            # Update weights
            self.update_weights()
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward_propagation(X)
        return (output > 0.5).astype(int)
    
    def visualize_training(self):
        """Visualize training progress"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.show()

def section1_neural_network_basics():
    """Neural Network basics from scratch"""
    
    print("\n" + "="*60)
    print("🧠 SEZIONE 1: NEURAL NETWORK FROM SCRATCH")
    print("="*60)
    
    # Generate XOR problem data
    print("\n📊 1.1 XOR PROBLEM")
    print("-"*40)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR output
    
    print("XOR Truth Table:")
    for i in range(len(X)):
        print(f"  {X[i]} → {y[i][0]}")
    
    # Create and train network
    print("\n🤖 Training Neural Network...")
    nn = NeuralNetworkFromScratch(
        input_size=2,
        hidden_size=4,
        output_size=1,
        learning_rate=0.5
    )
    
    nn.train(X, y, epochs=1000, verbose=False)
    
    # Test predictions
    predictions = nn.predict(X)
    
    print("\nPredictions:")
    for i in range(len(X)):
        print(f"  {X[i]} → Predicted: {predictions[i][0]}, Actual: {y[i][0]}")
    
    accuracy = accuracy_score(y, predictions)
    print(f"\nAccuracy: {accuracy:.2%}")
    
    # Visualize training
    nn.visualize_training()
    
    # Visualize decision boundary
    print("\n📈 1.2 DECISION BOUNDARY")
    print("-"*40)
    
    # Create mesh
    h = 0.02
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='coolwarm', 
                edgecolors='black', s=100, linewidth=2)
    plt.title('Neural Network Decision Boundary - XOR Problem')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.grid(True, alpha=0.3)
    plt.show()

# ==============================================================================
# SEZIONE 2: KERAS/TENSORFLOW MODELS (Simulated)
# ==============================================================================

def section2_keras_models():
    """Deep Learning with Keras/TensorFlow (simulated)"""
    
    print("\n" + "="*60)
    print("🎯 SEZIONE 2: KERAS/TENSORFLOW MODELS")
    print("="*60)
    
    print("""
    Note: This is a simulation of Keras/TensorFlow code.
    In real implementation, you would use:
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    """)
    
    # 2.1 SEQUENTIAL MODEL
    print("\n📋 2.1 SEQUENTIAL MODEL ARCHITECTURE")
    print("-"*40)
    
    model_code = """
    # Create Sequential model
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model summary
    model.summary()
    """
    
    print("Model Architecture:")
    print(model_code)
    
    # Simulate model summary
    print("\nModel Summary:")
    print("_" * 65)
    print("Layer (type)                Output Shape              Param #")
    print("=" * 65)
    print("dense (Dense)               (None, 128)               100,480")
    print("dropout (Dropout)           (None, 128)               0")
    print("dense_1 (Dense)             (None, 64)                8,256")
    print("dropout_1 (Dropout)         (None, 64)                0")
    print("dense_2 (Dense)             (None, 32)                2,080")
    print("dense_3 (Dense)             (None, 10)                330")
    print("=" * 65)
    print("Total params: 111,146")
    print("Trainable params: 111,146")
    print("Non-trainable params: 0")
    print("_" * 65)
    
    # 2.2 FUNCTIONAL API
    print("\n🔧 2.2 FUNCTIONAL API MODEL")
    print("-"*40)
    
    functional_code = """
    # Input layer
    inputs = keras.Input(shape=(784,))
    
    # Hidden layers
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Branch 1
    branch1 = layers.Dense(64, activation='relu')(x)
    branch1 = layers.Dense(32, activation='relu')(branch1)
    
    # Branch 2
    branch2 = layers.Dense(64, activation='tanh')(x)
    branch2 = layers.Dense(32, activation='tanh')(branch2)
    
    # Concatenate branches
    concatenated = layers.concatenate([branch1, branch2])
    
    # Output layer
    outputs = layers.Dense(10, activation='softmax')(concatenated)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    """
    
    print("Functional API Model:")
    print(functional_code)
    
    # 2.3 CUSTOM LAYERS
    print("\n🎨 2.3 CUSTOM LAYERS")
    print("-"*40)
    
    custom_layer_code = """
    class CustomAttention(layers.Layer):
        def __init__(self, units):
            super(CustomAttention, self).__init__()
            self.units = units
            
        def build(self, input_shape):
            self.W = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='random_normal',
                trainable=True
            )
            self.b = self.add_weight(
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )
            
        def call(self, inputs):
            # Attention mechanism
            score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
            attention_weights = tf.nn.softmax(score, axis=1)
            context = attention_weights * inputs
            return context
    """
    
    print("Custom Attention Layer:")
    print(custom_layer_code)

# ==============================================================================
# SEZIONE 3: CONVOLUTIONAL NEURAL NETWORKS
# ==============================================================================

def section3_cnn():
    """Convolutional Neural Networks for Computer Vision"""
    
    print("\n" + "="*60)
    print("🖼️ SEZIONE 3: CONVOLUTIONAL NEURAL NETWORKS")
    print("="*60)
    
    # 3.1 CNN ARCHITECTURE
    print("\n📸 3.1 CNN FOR IMAGE CLASSIFICATION")
    print("-"*40)
    
    cnn_code = """
    # CNN Model for CIFAR-10
    model = keras.Sequential([
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Convolutional Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile with advanced optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    """
    
    print("CNN Architecture for CIFAR-10:")
    print(cnn_code)
    
    # 3.2 DATA AUGMENTATION
    print("\n🎭 3.2 DATA AUGMENTATION")
    print("-"*40)
    
    augmentation_code = """
    # Data Augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ])
    
    # Preprocessing
    preprocessing = keras.Sequential([
        layers.Rescaling(1./255),
        data_augmentation
    ])
    
    # Training with augmentation
    train_dataset = train_dataset.map(
        lambda x, y: (preprocessing(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    """
    
    print("Data Augmentation Pipeline:")
    print(augmentation_code)
    
    # 3.3 TRANSFER LEARNING
    print("\n🔄 3.3 TRANSFER LEARNING")
    print("-"*40)
    
    transfer_code = """
    # Load pre-trained model
    base_model = keras.applications.VGG16(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Create new model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Fine-tuning (after initial training)
    base_model.trainable = True
    
    # Freeze all layers except last 4
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    # Compile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    """
    
    print("Transfer Learning with VGG16:")
    print(transfer_code)

# ==============================================================================
# SEZIONE 4: RECURRENT NEURAL NETWORKS
# ==============================================================================

def section4_rnn():
    """Recurrent Neural Networks for Sequential Data"""
    
    print("\n" + "="*60)
    print("📝 SEZIONE 4: RECURRENT NEURAL NETWORKS")
    print("="*60)
    
    # 4.1 LSTM FOR TIME SERIES
    print("\n📈 4.1 LSTM FOR TIME SERIES")
    print("-"*40)
    
    lstm_code = """
    # LSTM Model for Time Series
    model = keras.Sequential([
        # LSTM layers
        layers.LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # Regression output
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    """
    
    print("LSTM for Time Series Prediction:")
    print(lstm_code)
    
    # 4.2 BIDIRECTIONAL RNN
    print("\n↔️ 4.2 BIDIRECTIONAL RNN")
    print("-"*40)
    
    bidirectional_code = """
    # Bidirectional LSTM for Text Classification
    model = keras.Sequential([
        # Embedding layer
        layers.Embedding(vocab_size, 128, input_length=max_length),
        
        # Bidirectional LSTM
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.5),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.5),
        
        # Output
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    """
    
    print("Bidirectional LSTM Model:")
    print(bidirectional_code)
    
    # 4.3 GRU MODEL
    print("\n🔄 4.3 GRU (GATED RECURRENT UNIT)")
    print("-"*40)
    
    gru_code = """
    # GRU Model
    model = keras.Sequential([
        layers.GRU(128, return_sequences=True, input_shape=(timesteps, features)),
        layers.BatchNormalization(),
        layers.GRU(64, return_sequences=True),
        layers.BatchNormalization(),
        layers.GRU(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(output_dim)
    ])
    """
    
    print("GRU Model Architecture:")
    print(gru_code)

# ==============================================================================
# SEZIONE 5: ADVANCED DEEP LEARNING
# ==============================================================================

def section5_advanced_dl():
    """Advanced Deep Learning Concepts"""
    
    print("\n" + "="*60)
    print("🚀 SEZIONE 5: ADVANCED DEEP LEARNING")
    print("="*60)
    
    # 5.1 AUTOENCODER
    print("\n🔐 5.1 AUTOENCODER")
    print("-"*40)
    
    autoencoder_code = """
    # Encoder
    encoder = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(latent_dim)  # Latent space
    ])
    
    # Decoder
    decoder = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(latent_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid')
    ])
    
    # Autoencoder
    autoencoder = keras.Sequential([encoder, decoder])
    
    autoencoder.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )
    """
    
    print("Autoencoder Architecture:")
    print(autoencoder_code)
    
    # 5.2 GAN (GENERATIVE ADVERSARIAL NETWORK)
    print("\n🎨 5.2 GAN - GENERATIVE ADVERSARIAL NETWORK")
    print("-"*40)
    
    gan_code = """
    # Generator
    generator = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(784, activation='tanh')
    ])
    
    # Discriminator
    discriminator = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile discriminator
    discriminator.compile(
        optimizer=keras.optimizers.Adam(0.0002, 0.5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Combined model (Generator + Discriminator)
    discriminator.trainable = False
    gan_input = keras.Input(shape=(latent_dim,))
    generated = generator(gan_input)
    gan_output = discriminator(generated)
    
    gan = keras.Model(gan_input, gan_output)
    gan.compile(
        optimizer=keras.optimizers.Adam(0.0002, 0.5),
        loss='binary_crossentropy'
    )
    """
    
    print("GAN Architecture:")
    print(gan_code)
    
    # 5.3 ATTENTION MECHANISM
    print("\n👁️ 5.3 ATTENTION MECHANISM")
    print("-"*40)
    
    attention_code = """
    # Self-Attention Layer
    class SelfAttention(layers.Layer):
        def __init__(self, units):
            super(SelfAttention, self).__init__()
            self.units = units
            self.W1 = layers.Dense(units)
            self.W2 = layers.Dense(units)
            self.V = layers.Dense(1)
            
        def call(self, inputs):
            # inputs shape: (batch_size, seq_len, features)
            
            # Score calculation
            score = self.V(tf.nn.tanh(
                self.W1(inputs) + self.W2(inputs)
            ))
            
            # Attention weights
            attention_weights = tf.nn.softmax(score, axis=1)
            
            # Context vector
            context_vector = attention_weights * inputs
            context_vector = tf.reduce_sum(context_vector, axis=1)
            
            return context_vector
    
    # Transformer-like model
    model = keras.Sequential([
        layers.Embedding(vocab_size, embed_dim),
        SelfAttention(128),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    """
    
    print("Attention Mechanism Implementation:")
    print(attention_code)
    
    # 5.4 CALLBACKS AND TRAINING STRATEGIES
    print("\n⚙️ 5.4 TRAINING STRATEGIES")
    print("-"*40)
    
    training_code = """
    # Callbacks
    callbacks = [
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        
        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        ),
        
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    
    # Training with callbacks
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )
    """
    
    print("Advanced Training Strategies:")
    print(training_code)
    
    # 5.5 MODEL DEPLOYMENT
    print("\n🚀 5.5 MODEL DEPLOYMENT")
    print("-"*40)
    
    deployment_code = """
    # Save model
    model.save('my_model.h5')
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    # Save as SavedModel format
    model.save('saved_model/')
    
    # Convert to TensorFlow.js
    # tensorflowjs_converter --input_format keras model.h5 tfjs_model/
    
    # ONNX conversion
    # tf2onnx.convert --keras model.h5 --output model.onnx
    
    # Serving with TensorFlow Serving
    # docker run -p 8501:8501 --mount type=bind,source=/path/to/model,target=/models/my_model -e MODEL_NAME=my_model -t tensorflow/serving
    """
    
    print("Model Deployment Options:")
    print(deployment_code)

# ==============================================================================
# MAIN - Menu per le sezioni
# ==============================================================================

def main():
    """Menu principale per Deep Learning"""
    
    print("\n" + "="*60)
    print("🧠 DEEP LEARNING - SCEGLI SEZIONE")
    print("="*60)
    
    sections = [
        ("Neural Network from Scratch", section1_neural_network_basics),
        ("Keras/TensorFlow Models", section2_keras_models),
        ("Convolutional Neural Networks", section3_cnn),
        ("Recurrent Neural Networks", section4_rnn),
        ("Advanced Deep Learning", section5_advanced_dl)
    ]
    
    print("\n0. Esegui TUTTO")
    for i, (name, _) in enumerate(sections, 1):
        print(f"{i}. {name}")
    
    choice = input("\nScegli (0-5): ")
    
    try:
        choice = int(choice)
        if choice == 0:
            for name, func in sections:
                input(f"\n➡️ Press ENTER for: {name}")
                func()
        elif 1 <= choice <= len(sections):
            sections[choice-1][1]()
        else:
            print("Scelta non valida")
    except (ValueError, IndexError):
        print("Scelta non valida")
    
    print("\n" + "="*60)
    print("✅ PARTE 5 COMPLETATA!")
    print("Deep Learning foundations complete!")
    print("="*60)

if __name__ == "__main__":
    main()
