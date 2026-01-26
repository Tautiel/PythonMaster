#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ML & TRADING BOT - EXAM SIMULATIONS                       ║
║                    50 Domande di Verifica                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

STRUTTURA:
├── ML Exam (25 domande) - Copre ML-M1 to ML-M5
└── Trading Exam (25 domande) - Copre TB-M1 to TB-M4
"""

# ══════════════════════════════════════════════════════════════════════════════
#                    ML EXAM SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

ML_EXAM = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MACHINE LEARNING EXAM - 25 DOMANDE                        ║
║                    Tempo: 45 minuti | Passing: 70%                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

SECTION 1: DATA FOUNDATIONS (ML-M1) - 5 domande

Q1. np.array([[1,2],[3,4]]).shape restituisce?
    A) (4,)
    B) (2, 2)
    C) (2,)
    D) 4
    
Q2. Per selezionare righe dove 'age' > 25 in un DataFrame df?
    A) df['age' > 25]
    B) df[df['age'] > 25]
    C) df.select(age > 25)
    D) df.query['age > 25']
    
Q3. df.isnull().sum() restituisce?
    A) Numero totale di null nel DataFrame
    B) Series con conteggio null per colonna
    C) True/False
    D) Lista di colonne con null
    
Q4. Per calcolare la correlazione tra colonne numeriche?
    A) df.correlation()
    B) df.corr()
    C) df.cor()
    D) df.pearson()
    
Q5. df['close'].shift(1) equivale a Pine Script?
    A) close
    B) close[-1]
    C) close[1]
    D) close.prev

SECTION 2: PREPROCESSING (ML-M2) - 5 domande

Q6. StandardScaler produce dati con?
    A) Range 0-1
    B) Media=0, Std=1
    C) Mediana=0
    D) Range -1 a 1
    
Q7. Per variabili categoriche NOMINALI (es: colori), usare?
    A) LabelEncoder
    B) StandardScaler
    C) OneHotEncoder / get_dummies
    D) MinMaxScaler
    
Q8. train_test_split con stratify=y serve per?
    A) Velocizzare lo split
    B) Mantenere proporzioni classi
    C) Shuffle random
    D) Cross-validation
    
Q9. fit_transform() su TEST data è?
    A) Corretto
    B) Data leakage (ERRORE)
    C) Opzionale
    D) Necessario
    
Q10. SMOTE serve per?
    A) Feature scaling
    B) Feature selection
    C) Dati sbilanciati (oversampling)
    D) Encoding

SECTION 3: SUPERVISED LEARNING (ML-M3) - 5 domande

Q11. Logistic Regression è per?
    A) Regressione
    B) Classification
    C) Clustering
    D) Dimensionality reduction
    
Q12. Random Forest è un esempio di?
    A) Boosting
    B) Bagging
    C) Stacking
    D) Single model
    
Q13. SVM richiede feature scaling?
    A) Sì, sempre
    B) No, mai
    C) Solo per kernel lineare
    D) Solo per kernel RBF
    
Q14. Precision = TP / (TP + ?)?
    A) TN
    B) FP
    C) FN
    D) Total
    
Q15. R² = 0 significa?
    A) Modello perfetto
    B) Modello peggiore della media
    C) Modello come la media
    D) Errore

SECTION 4: MODEL TUNING (ML-M4) - 5 domande

Q16. 5-Fold Cross Validation usa quante iterazioni?
    A) 1
    B) 5
    C) 10
    D) Dipende dai dati
    
Q17. GridSearchCV prova?
    A) Combinazioni random
    B) TUTTE le combinazioni
    C) Solo le migliori
    D) Una combinazione
    
Q18. Train error BASSO, Test error ALTO indica?
    A) Underfitting
    B) Overfitting
    C) Good fit
    D) Data leakage
    
Q19. Pipeline in sklearn evita?
    A) Overfitting
    B) Data leakage
    C) Underfitting
    D) Tutte
    
Q20. RandomizedSearchCV è preferibile quando?
    A) Pochi hyperparameters
    B) Molti hyperparameters
    C) Mai
    D) Sempre

SECTION 5: DEEP LEARNING (ML-M5) - 5 domande

Q21. ReLU activation è?
    A) 1/(1+e^-x)
    B) max(0, x)
    C) e^x / Σe^x
    D) tanh(x)
    
Q22. Per multiclass classification con 5 classi, output layer ha?
    A) 1 neuron, sigmoid
    B) 5 neurons, softmax
    C) 5 neurons, sigmoid
    D) 1 neuron, linear
    
Q23. Dropout disattiva neuroni durante?
    A) Solo training
    B) Solo prediction
    C) Entrambi
    D) Mai
    
Q24. EarlyStopping monitora tipicamente?
    A) Training loss
    B) Validation loss
    C) Accuracy
    D) Numero epochs
    
Q25. LSTM è usato principalmente per?
    A) Immagini
    B) Sequenze / Time series
    C) Dati tabulari
    D) Clustering

═══════════════════════════════════════════════════════════════════════════════
                              RISPOSTE ML EXAM
═══════════════════════════════════════════════════════════════════════════════

Q1: B (2,2)          Q6: B               Q11: B              Q16: B
Q2: B                Q7: C               Q12: B              Q17: B
Q3: B                Q8: B               Q13: A              Q18: B
Q4: B                Q9: B               Q14: B              Q19: B
Q5: C                Q10: C              Q15: C              Q20: B
Q21: B               Q22: B              Q23: A              Q24: B              Q25: B
"""

# ══════════════════════════════════════════════════════════════════════════════
#                    TRADING BOT EXAM SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

TRADING_EXAM = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TRADING BOT EXAM - 25 DOMANDE                             ║
║                    Tempo: 45 minuti | Passing: 70%                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

SECTION 1: FOUNDATIONS (TB-M1) - 6 domande

Q1. OHLCV: la 'V' sta per?
    A) Value
    B) Volume
    C) Volatility
    D) Variance
    
Q2. Spread è?
    A) High - Low
    B) Ask - Bid
    C) Close - Open
    D) Return giornaliero
    
Q3. Un Market Order garantisce?
    A) Prezzo
    B) Esecuzione
    C) Entrambi
    D) Nessuno
    
Q4. SMA in Python si calcola con?
    A) df['close'].ewm(span=20).mean()
    B) df['close'].rolling(20).mean()
    C) df['close'].cumsum()
    D) df['close'].diff()
    
Q5. EMA in Python si calcola con?
    A) df['close'].ewm(span=20).mean()
    B) df['close'].rolling(20).mean()
    C) df['close'].cumsum()
    D) df['close'].shift(20)
    
Q6. Max Drawdown misura?
    A) Profitto massimo
    B) Perdita massima da picco
    C) Volatilità
    D) Sharpe ratio

SECTION 2: STRATEGY (TB-M2) - 7 domande

Q7. RSI oversold è tipicamente?
    A) < 30
    B) > 70
    C) = 50
    D) < 0
    
Q8. MACD è la differenza tra?
    A) Due SMA
    B) Due EMA
    C) High e Low
    D) Open e Close
    
Q9. Risk per trade consigliato è?
    A) 10%
    B) 50%
    C) 1-2%
    D) 100%
    
Q10. ATR misura?
    A) Trend direction
    B) Volatilità
    C) Volume
    D) Momentum
    
Q11. Bollinger Upper Band è?
    A) SMA + n*std
    B) SMA - n*std
    C) EMA + n*std
    D) High rolling
    
Q12. crossover(fast, slow) è True quando?
    A) fast > slow
    B) fast < slow
    C) fast incrocia SOPRA slow
    D) fast incrocia SOTTO slow
    
Q13. Risk/Reward 1:2 significa?
    A) Rischio 2x il reward
    B) Reward 2x il rischio
    C) Uguali
    D) Nessun rapporto

SECTION 3: BACKTESTING (TB-M3) - 6 domande

Q14. Look-ahead bias è?
    A) Usare dati futuri nella decisione
    B) Ignorare commissioni
    C) Overfitting
    D) Slippage
    
Q15. Profit Factor > 1 significa?
    A) Perdita
    B) Break-even
    C) Profitto
    D) Errore
    
Q16. Sharpe Ratio misura?
    A) Solo return
    B) Risk-adjusted return
    C) Solo risk
    D) Numero trades
    
Q17. Win rate 40% può essere profittevole?
    A) Mai
    B) Sì, se avg_win >> avg_loss
    C) Sempre
    D) Solo con leverage
    
Q18. Walk-forward optimization serve per?
    A) Velocità
    B) Evitare overfitting
    C) Più trades
    D) Meno codice
    
Q19. Out-of-sample data è?
    A) Dati di training
    B) Dati mai usati per ottimizzare
    C) Dati futuri reali
    D) Dati random

SECTION 4: EXCHANGE (TB-M4) - 6 domande

Q20. CCXT supporta quanti exchange?
    A) 10
    B) 50
    C) 100+
    D) Solo Binance
    
Q21. fetch_ohlcv() restituisce?
    A) Prezzo corrente
    B) Candele storiche
    C) Order book
    D) Balance
    
Q22. sandbox=True in CCXT significa?
    A) Produzione
    B) Testnet
    C) Debug mode
    D) Errore
    
Q23. API keys devono essere?
    A) Nel codice sorgente
    B) In .env (mai committare)
    C) Pubbliche su GitHub
    D) Condivise con altri
    
Q24. create_market_buy_order garantisce?
    A) Prezzo esatto
    B) Esecuzione immediata
    C) Entrambi
    D) Nessuno
    
Q25. Prima di live trading, devi fare?
    A) Solo backtest
    B) Backtest + paper trading
    C) Nulla
    D) Solo paper trading

═══════════════════════════════════════════════════════════════════════════════
                           RISPOSTE TRADING EXAM
═══════════════════════════════════════════════════════════════════════════════

Q1: B              Q6: B              Q11: A             Q16: B             Q21: B
Q2: B              Q7: A              Q12: C             Q17: B             Q22: B
Q3: B              Q8: B              Q13: B             Q18: B             Q23: B
Q4: B              Q9: C              Q14: A             Q19: B             Q24: B
Q5: A              Q10: B             Q15: C             Q20: C             Q25: B
"""

def run_exam(exam_text, exam_name):
    print(exam_text)
    print(f"\n{'='*70}")
    print(f"FINE {exam_name}")
    print(f"{'='*70}")
    print(f"\nPassing score: 70% (18/25 corrette)")
    print(f"Obiettivo: 85%+ prima di procedere")

if __name__ == "__main__":
    print("SCEGLI ESAME:")
    print("1. ML Exam (25 domande)")
    print("2. Trading Exam (25 domande)")
    print("3. Entrambi")
    
    # Default: mostra entrambi
    run_exam(ML_EXAM, "ML EXAM")
    print("\n" + "="*70 + "\n")
    run_exam(TRADING_EXAM, "TRADING EXAM")
