#!/usr/bin/env python3
"""
📐 MATEMATICA AVANZATA PER TRADING
==================================
Questo modulo spiega i concetti STATISTICI che userai in PARTE 8-10.
LEGGI QUESTO FILE PRIMA di iniziare PARTE8 (NumPy)!

PREREQUISITO: Aver completato MODULO_0_MATEMATICA_BASE.py
"""

print("=" * 70)
print("📐 MATEMATICA AVANZATA PER TRADING")
print("Da leggere PRIMA di PARTE 8 (NumPy)")
print("=" * 70)

# ==============================================================================
# CONCETTO 1: LA MEDIA (MEAN)
# ==============================================================================

def concetto_1_media():
    """La media aritmetica spiegata."""
    
    print("\n" + "=" * 50)
    print("CONCETTO 1: LA MEDIA (MEAN)")
    print("=" * 50)
    
    print("\n📌 COS'È LA MEDIA?")
    print("-" * 40)
    print("La media è il 'valore tipico' di un gruppo di numeri.")
    print("Si calcola: somma di tutti i numeri / quanti numeri sono")
    print()
    
    # Esempio semplice
    numeri = [10, 20, 30, 40, 50]
    somma = sum(numeri)  # 150
    quantita = len(numeri)  # 5
    media = somma / quantita  # 30
    
    print(f"Numeri: {numeri}")
    print(f"Somma:  {' + '.join(map(str, numeri))} = {somma}")
    print(f"Quanti: {quantita}")
    print(f"Media:  {somma} / {quantita} = {media}")
    
    # Esempio trading
    print("\n📌 ESEMPIO TRADING")
    print("-" * 40)
    
    rendimenti_giornalieri = [0.02, -0.01, 0.03, 0.01, -0.02]  # +2%, -1%, +3%, +1%, -2%
    media_rendimento = sum(rendimenti_giornalieri) / len(rendimenti_giornalieri)
    
    print(f"Rendimenti giornalieri: {rendimenti_giornalieri}")
    print(f"(cioè: +2%, -1%, +3%, +1%, -2%)")
    print(f"Media rendimento: {media_rendimento:.4f} = {media_rendimento*100:.2f}%")
    print()
    print("Interpretazione: In media, ogni giorno guadagni 0.6%")

# Esegui
concetto_1_media()


# ==============================================================================
# CONCETTO 2: LA DEVIAZIONE STANDARD (STD)
# ==============================================================================

def concetto_2_deviazione_standard():
    """La deviazione standard spiegata."""
    
    print("\n" + "=" * 50)
    print("CONCETTO 2: DEVIAZIONE STANDARD (STD)")
    print("=" * 50)
    
    print("\n📌 COS'È LA DEVIAZIONE STANDARD?")
    print("-" * 40)
    print("Misura QUANTO i valori si allontanano dalla media.")
    print()
    print("STD bassa = valori vicini alla media = STABILE")
    print("STD alta  = valori lontani dalla media = VOLATILE")
    
    # Esempio visivo
    print("\n📌 ESEMPIO VISIVO")
    print("-" * 40)
    
    # Stesso media (30), diversa volatilità
    stabile = [28, 29, 30, 31, 32]     # Tutti vicini a 30
    volatile = [10, 20, 30, 40, 50]    # Sparsi lontano da 30
    
    media_stabile = sum(stabile) / len(stabile)  # 30
    media_volatile = sum(volatile) / len(volatile)  # 30
    
    print(f"Serie STABILE:  {stabile}")
    print(f"  Media: {media_stabile}")
    print(f"  Valori tutti vicini a 30 → STD bassa")
    print()
    print(f"Serie VOLATILE: {volatile}")
    print(f"  Media: {media_volatile}")
    print(f"  Valori sparsi lontano da 30 → STD alta")
    
    # Calcolo manuale (semplificato)
    print("\n📌 COME SI CALCOLA (semplificato)")
    print("-" * 40)
    print("1. Calcola la media")
    print("2. Per ogni numero: (numero - media)²")
    print("3. Fai la media di questi quadrati")
    print("4. Fai la radice quadrata")
    print()
    print("Non preoccuparti: NumPy lo fa automaticamente!")
    print("  np.std(numeri)  →  deviazione standard")
    
    # Esempio trading
    print("\n📌 NEL TRADING")
    print("-" * 40)
    print("La deviazione standard dei rendimenti = VOLATILITÀ")
    print()
    print("Azione A: STD = 1%  → Poco volatile, prezzo stabile")
    print("Azione B: STD = 5%  → Molto volatile, prezzo salta")
    print()
    print("La volatilità è RISCHIO:")
    print("  → Alta volatilità = più possibilità di guadagno MA anche di perdita")
    print("  → Bassa volatilità = più stabile, meno rischio")

# Esegui
concetto_2_deviazione_standard()


# ==============================================================================
# CONCETTO 3: LO SHARPE RATIO
# ==============================================================================

def concetto_3_sharpe_ratio():
    """Lo Sharpe Ratio spiegato."""
    
    print("\n" + "=" * 50)
    print("CONCETTO 3: SHARPE RATIO")
    print("=" * 50)
    
    print("\n📌 COS'È LO SHARPE RATIO?")
    print("-" * 40)
    print("Misura il RENDIMENTO rispetto al RISCHIO.")
    print()
    print("Formula semplificata:")
    print("  Sharpe = Rendimento Medio / Volatilità")
    print()
    print("Interpretazione:")
    print("  Sharpe > 1  = Buono (guadagno rispetto al rischio)")
    print("  Sharpe > 2  = Ottimo")
    print("  Sharpe < 0  = Perdita")
    
    # Esempio
    print("\n📌 ESEMPIO")
    print("-" * 40)
    
    # Strategia A: guadagna 10% con 20% volatilità
    # Strategia B: guadagna 5% con 5% volatilità
    
    sharpe_a = 0.10 / 0.20  # = 0.5
    sharpe_b = 0.05 / 0.05  # = 1.0
    
    print("Strategia A:")
    print(f"  Rendimento: 10%")
    print(f"  Volatilità: 20%")
    print(f"  Sharpe: 10% / 20% = {sharpe_a}")
    print()
    print("Strategia B:")
    print(f"  Rendimento: 5%")
    print(f"  Volatilità: 5%")
    print(f"  Sharpe: 5% / 5% = {sharpe_b}")
    print()
    print("Strategia B è MIGLIORE!")
    print("Anche se guadagna meno, il rapporto rischio/rendimento è superiore.")
    
    # Formula completa
    print("\n📌 FORMULA COMPLETA (per annualizzare)")
    print("-" * 40)
    print("Sharpe_annuale = (rendimento_medio_giornaliero / std_giornaliera) × √252")
    print()
    print("252 = giorni di trading in un anno")
    print("√252 ≈ 15.87")
    print()
    print("In Python:")
    print("  import numpy as np")
    print("  sharpe = (returns.mean() / returns.std()) * np.sqrt(252)")

# Esegui
concetto_3_sharpe_ratio()


# ==============================================================================
# CONCETTO 4: DRAWDOWN
# ==============================================================================

def concetto_4_drawdown():
    """Il Drawdown spiegato."""
    
    print("\n" + "=" * 50)
    print("CONCETTO 4: DRAWDOWN")
    print("=" * 50)
    
    print("\n📌 COS'È IL DRAWDOWN?")
    print("-" * 40)
    print("Il drawdown è la PERDITA dal punto più alto.")
    print()
    print("Immagina:")
    print("  - Il tuo portafoglio arriva a 10.000€ (massimo)")
    print("  - Poi scende a 8.000€")
    print("  - Drawdown = -2.000€ = -20%")
    
    # Esempio visivo
    print("\n📌 ESEMPIO VISIVO")
    print("-" * 40)
    
    equity = [100, 110, 120, 115, 105, 125, 130, 110]
    
    print("Equity curve: 100 → 110 → 120 → 115 → 105 → 125 → 130 → 110")
    print()
    print("Analisi:")
    print("  100 → 120: Massimo a 120")
    print("  120 → 105: Drawdown! -15 (dal massimo 120)")
    print("            Drawdown % = -15/120 = -12.5%")
    print("  105 → 130: Nuovo massimo a 130")
    print("  130 → 110: Drawdown! -20 (dal massimo 130)")
    print("            Drawdown % = -20/130 = -15.4%")
    print()
    print("MAX DRAWDOWN = -15.4% (il peggiore)")
    
    # Importanza
    print("\n📌 PERCHÉ È IMPORTANTE")
    print("-" * 40)
    print("Il Max Drawdown ti dice:")
    print("  → La PEGGIORE perdita che puoi aspettarti")
    print("  → Quanto capitale 'a rischio' serve")
    print()
    print("Esempio:")
    print("  Max Drawdown atteso: -30%")
    print("  Se inizi con 10.000€, potresti scendere a 7.000€")
    print("  Devi essere psicologicamente preparato!")

# Esegui
concetto_4_drawdown()


# ==============================================================================
# CONCETTO 5: CORRELAZIONE
# ==============================================================================

def concetto_5_correlazione():
    """La correlazione spiegata."""
    
    print("\n" + "=" * 50)
    print("CONCETTO 5: CORRELAZIONE")
    print("=" * 50)
    
    print("\n📌 COS'È LA CORRELAZIONE?")
    print("-" * 40)
    print("Misura se due cose si MUOVONO INSIEME.")
    print()
    print("Correlazione va da -1 a +1:")
    print("  +1  = Si muovono INSIEME (uno sale, anche l'altro)")
    print("   0  = NON correlati (movimenti indipendenti)")
    print("  -1  = Si muovono OPPOSTI (uno sale, l'altro scende)")
    
    # Esempi
    print("\n📌 ESEMPI")
    print("-" * 40)
    
    print("Correlazione +1 (forte positiva):")
    print("  Bitcoin e Ethereum: quando BTC sale, ETH tende a salire")
    print()
    print("Correlazione 0 (nessuna):")
    print("  Apple e prezzo del grano: non c'è relazione")
    print()
    print("Correlazione -1 (forte negativa):")
    print("  Azioni e oro: quando azioni scendono, oro tende a salire")
    print("  (è un rifugio sicuro)")
    
    # Uso nel trading
    print("\n📌 USO NEL TRADING")
    print("-" * 40)
    print("Diversificazione: compra asset NON correlati")
    print()
    print("Se hai solo BTC e ETH (corr. +0.9):")
    print("  → Se uno crolla, probabilmente anche l'altro")
    print("  → Rischio alto!")
    print()
    print("Se hai BTC e Oro (corr. -0.3):")
    print("  → Se uno crolla, l'altro potrebbe salire")
    print("  → Rischio ridotto!")

# Esegui
concetto_5_correlazione()


# ==============================================================================
# RIEPILOGO
# ==============================================================================

def riepilogo_avanzato():
    """Riepilogo concetti avanzati."""
    
    print("\n" + "=" * 70)
    print("📋 RIEPILOGO: FORMULE STATISTICHE CHIAVE")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ MEDIA (MEAN)                                                        │
├─────────────────────────────────────────────────────────────────────┤
│ Formula:    somma(valori) / numero_valori                          │
│ Python:     np.mean(array) o array.mean()                          │
│ Trading:    Rendimento medio giornaliero/mensile/annuale           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ DEVIAZIONE STANDARD (STD) = VOLATILITÀ                             │
├─────────────────────────────────────────────────────────────────────┤
│ Significato: Quanto i valori si allontanano dalla media            │
│ Python:      np.std(array) o array.std()                           │
│ Trading:     STD alta = volatile/rischioso, STD bassa = stabile    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ SHARPE RATIO                                                        │
├─────────────────────────────────────────────────────────────────────┤
│ Formula:    (rendimento_medio / volatilità) × √252                 │
│ Python:     (returns.mean() / returns.std()) * np.sqrt(252)        │
│ Trading:    > 1 = buono, > 2 = ottimo, < 0 = perdita               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ DRAWDOWN                                                            │
├─────────────────────────────────────────────────────────────────────┤
│ Significato: Perdita dal massimo raggiunto                         │
│ Max DD:      La peggiore perdita storica                           │
│ Trading:     Ti dice quanto capitale puoi perdere                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ CORRELAZIONE                                                        │
├─────────────────────────────────────────────────────────────────────┤
│ Range:       -1 (opposti) a +1 (insieme)                           │
│ Python:      np.corrcoef(a, b) o df.corr()                         │
│ Trading:     Diversifica con asset non correlati                   │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print("✅ Ora sei pronto per PARTE 8 (NumPy) e PARTE 9 (Pandas)!")

# Esegui
riepilogo_avanzato()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("HAI COMPLETATO IL MODULO DI MATEMATICA AVANZATA!")
    print("=" * 70)
    print()
    print("Sequenza di studio:")
    print("  1. MODULO_0_MATEMATICA_BASE.py ✓ (già fatto)")
    print("  2. PARTE 1-7 ✓ (fondamenta Python)")
    print("  3. MODULO_MATEMATICA_AVANZATA.py ✓ (questo file)")
    print("  4. PARTE 8-10 → Ora sei pronto!")
