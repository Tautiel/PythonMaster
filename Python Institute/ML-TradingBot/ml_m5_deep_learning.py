#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MACHINE LEARNING - MODULE 5                               ║
║                    DEEP LEARNING BASICS                                       ║
║                    ML-M5: 10% del percorso ML                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS ML MODULE 5:
├── ML 5.1 - Neural Networks Fundamentals (neurons, layers, activation)
├── ML 5.2 - Keras/TensorFlow Basics (Sequential API)
├── ML 5.3 - Training Process (loss, optimizer, epochs, batch)
├── ML 5.4 - Regularization (dropout, early stopping, batch norm)
├── ML 5.5 - Common Architectures (MLP, CNN intro, RNN intro)
└── ML 5.6 - Model Saving & Loading

PREREQUISITI: ML Module 1-4 completati
TEMPO STIMATO: 2 settimane (2-3 ore/giorno)
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# ML 5.1 - NEURAL NETWORKS FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("ML 5.1 - NEURAL NETWORKS FUNDAMENTALS")
print("=" * 70)

print("""
📋 STRUTTURA NEURAL NETWORK

NEURON (Perceptron):
  output = activation(Σ(weights * inputs) + bias)

LAYERS:
  - Input Layer: riceve i dati
  - Hidden Layers: trasformazioni intermedie
  - Output Layer: predizione finale

DEEP LEARNING = molti hidden layers

┌─────────────────────────────────────────────────────────────────┐
│  INPUT       HIDDEN 1      HIDDEN 2      OUTPUT                │
│                                                                 │
│   O────────────O────────────O                                  │
│                 ╲          ╱                                    │
│   O──────────────O────────O───────────────O                    │
│                 ╱          ╲              │                    │
│   O────────────O────────────O             │                    │
│                                                                 │
│  (features)   (neurons)   (neurons)    (prediction)            │
└─────────────────────────────────────────────────────────────────┘

📋 ACTIVATION FUNCTIONS

┌─────────────────────┬─────────────────────────────────────────┐
│ ReLU                │ max(0, x) - Default hidden layers       │
│ Sigmoid             │ 1/(1+e^-x) - Binary output (0-1)        │
│ Softmax             │ e^xi/Σe^xj - Multiclass probabilities   │
│ Tanh                │ Range (-1, 1)                           │
│ Linear              │ x - Regression output                   │
└─────────────────────┴─────────────────────────────────────────┘
""")

# ══════════════════════════════════════════════════════════════════════════════
# ML 5.2 - KERAS/TENSORFLOW BASICS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 5.2 - KERAS/TENSORFLOW BASICS")
print("=" * 70)

print("""
📋 KERAS SEQUENTIAL API

from tensorflow import keras
from tensorflow.keras import layers

🔹 BINARY CLASSIFICATION:

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

🔹 MULTICLASS CLASSIFICATION:

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(5, activation='softmax')  # 5 classi
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

🔹 REGRESSION:

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
""")

# ══════════════════════════════════════════════════════════════════════════════
# ML 5.3 - TRAINING PROCESS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 5.3 - TRAINING PROCESS")
print("=" * 70)

print("""
📋 CONCETTI CHIAVE

EPOCH: una passata completa su TUTTO il training set
BATCH: subset di dati processato insieme
ITERATION: batches per completare un epoch

Esempio: 1000 campioni, batch_size=100 → 10 iterations/epoch

🔹 TRAINING:

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Accedere alla storia
history.history['loss']      # Training loss
history.history['val_loss']  # Validation loss

📋 LOSS FUNCTIONS:
┌─────────────────────────────┬─────────────────────────────┐
│ binary_crossentropy         │ Binary classification       │
│ sparse_categorical_cross.   │ Multiclass (int labels)     │
│ categorical_crossentropy    │ Multiclass (one-hot)        │
│ mse                         │ Regression                  │
└─────────────────────────────┴─────────────────────────────┘

📋 OPTIMIZERS:
┌─────────────────────────────┬─────────────────────────────┐
│ Adam                        │ Default consigliato         │
│ SGD                         │ Base, con momentum          │
│ RMSprop                     │ Buono per RNN               │
└─────────────────────────────┴─────────────────────────────┘
""")

# ══════════════════════════════════════════════════════════════════════════════
# ML 5.4 - REGULARIZATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 5.4 - REGULARIZATION IN DEEP LEARNING")
print("=" * 70)

print("""
📋 TECNICHE ANTI-OVERFITTING

🔹 DROPOUT:
layers.Dropout(0.3)  # 30% neuroni disattivati random

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

🔹 EARLY STOPPING:

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.fit(X, y, epochs=100, callbacks=[early_stop], validation_split=0.2)

🔹 BATCH NORMALIZATION:

model = keras.Sequential([
    layers.Dense(64, input_shape=(10,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(1, activation='sigmoid')
])

🔹 L2 REGULARIZATION:

from tensorflow.keras import regularizers

layers.Dense(64, activation='relu',
             kernel_regularizer=regularizers.l2(0.01))
""")

# ══════════════════════════════════════════════════════════════════════════════
# ML 5.5 - COMMON ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 5.5 - COMMON ARCHITECTURES")
print("=" * 70)

print("""
📋 ARCHITETTURE PRINCIPALI

┌─────────────────────┬─────────────────────────────────────────┐
│ MLP                 │ Fully connected layers                  │
│ (Multi-Layer        │ Per dati tabulari                       │
│  Perceptron)        │ Quello che abbiamo visto                │
├─────────────────────┼─────────────────────────────────────────┤
│ CNN                 │ Convolutional layers                    │
│ (Convolutional      │ Per immagini, pattern locali            │
│  Neural Network)    │ Conv2D, MaxPooling2D                    │
├─────────────────────┼─────────────────────────────────────────┤
│ RNN                 │ Recurrent connections                   │
│ (Recurrent          │ Per sequenze, time series               │
│  Neural Network)    │ LSTM, GRU                               │
├─────────────────────┼─────────────────────────────────────────┤
│ Transformer         │ Attention mechanism                     │
│                     │ NLP, state-of-the-art                   │
└─────────────────────┴─────────────────────────────────────────┘

🔹 CNN ESEMPIO (per immagini):

model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

🔹 LSTM ESEMPIO (per sequenze/time series):

model = keras.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(timesteps, features)),
    layers.LSTM(50),
    layers.Dense(1)
])

# Per trading: previsione prezzo futuro basato su sequenza storica
""")

# ══════════════════════════════════════════════════════════════════════════════
# ML 5.6 - MODEL SAVING & LOADING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ML 5.6 - MODEL SAVING & LOADING")
print("=" * 70)

print("""
📋 SALVARE E CARICARE MODELLI

🔹 FORMATO KERAS (.keras):

# Salva tutto (architettura + pesi + optimizer)
model.save('my_model.keras')

# Carica
model = keras.models.load_model('my_model.keras')

🔹 SOLO PESI:

# Salva solo i pesi
model.save_weights('weights.h5')

# Carica (richiede stessa architettura)
model.load_weights('weights.h5')

🔹 CHECKPOINT DURANTE TRAINING:

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True
)

model.fit(X, y, callbacks=[checkpoint, early_stop])

🔹 EXPORT PER PRODUZIONE:

# TensorFlow SavedModel format
model.save('saved_model/', save_format='tf')

# Conversione a TensorFlow Lite (mobile)
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/')
tflite_model = converter.convert()
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ DI VERIFICA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA - ML MODULE 5")
print("=" * 70)

print("""
Q1. ReLU activation è?
    A) 1/(1+e^-x)  B) max(0,x)  C) e^x/Σe^x  D) tanh(x)
    → RISPOSTA: B

Q2. Per binary classification, output activation è?
    A) ReLU  B) Softmax  C) Sigmoid  D) Linear
    → RISPOSTA: C

Q3. Per multiclass con 5 classi, ultimo layer ha?
    A) 1 neuron sigmoid  B) 5 neurons softmax  C) 5 neurons sigmoid  D) 1 neuron
    → RISPOSTA: B

Q4. 1000 samples, batch_size=100, quante iterations per epoch?
    A) 100  B) 1000  C) 10  D) 10000
    → RISPOSTA: C

Q5. Dropout disattiva neuroni durante?
    A) Solo training  B) Solo prediction  C) Entrambi  D) Mai
    → RISPOSTA: A

Q6. EarlyStopping monitora tipicamente?
    A) Training loss  B) Validation loss  C) Accuracy  D) Epochs
    → RISPOSTA: B

Q7. CNN è usato principalmente per?
    A) Testo  B) Immagini  C) Dati tabulari  D) Audio
    → RISPOSTA: B

Q8. LSTM è un tipo di?
    A) CNN  B) MLP  C) RNN  D) Transformer
    → RISPOSTA: C

Q9. Adam è un?
    A) Loss function  B) Optimizer  C) Layer  D) Metric
    → RISPOSTA: B

Q10. model.save() salva?
    A) Solo pesi  B) Solo architettura  C) Tutto  D) Nulla
    → RISPOSTA: C
""")

# ══════════════════════════════════════════════════════════════════════════════
# ESERCIZI PRATICI
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ESERCIZI PRATICI")
print("=" * 70)

print("""
📝 ESERCIZIO 1: MLP Classification
Crea un MLP per classificazione binaria:
- Input: 20 features
- 2 hidden layers (64, 32 neurons)
- Dropout 0.3 dopo ogni hidden layer
- Compila con Adam e binary_crossentropy

📝 ESERCIZIO 2: Regression Network
Crea una rete per previsione prezzo:
- Input: 10 features
- 3 hidden layers con BatchNormalization
- Output lineare
- Early stopping con patience=15

📝 ESERCIZIO 3: Learning Curves
Allena un modello e:
- Plotta training vs validation loss
- Identifica se c'è overfitting
- Aggiungi regularization se necessario

📝 ESERCIZIO 4: Model Checkpoint
Implementa training con:
- EarlyStopping
- ModelCheckpoint (salva solo il migliore)
- Carica il modello salvato e fai predizioni
""")

print("\n" + "=" * 70)
print("ML MODULE 5 - DEEP LEARNING COMPLETATO!")
print("=" * 70)
print("""
PERCORSO ML COMPLETATO! 🎉

RIEPILOGO MODULI:
  ML-M1: Data Foundations (25%)
  ML-M2: Preprocessing (20%)
  ML-M3: Supervised Learning (30%)
  ML-M4: Model Tuning (15%)
  ML-M5: Deep Learning (10%)

Prossimo: TRADING BOT MODULES
""")
