#!/usr/bin/env python3
"""
📐 MATEMATICA BASE PER TRADING
==============================
Questo modulo spiega i concetti matematici che userai negli esercizi Python.
LEGGI QUESTO FILE PRIMA di iniziare PARTE1!

Non è un corso di matematica avanzata.
Sono solo i concetti BASE che servono per capire gli esercizi.
"""

print("=" * 70)
print("📐 MATEMATICA BASE PER TRADING")
print("Leggi con calma, non c'è fretta!")
print("=" * 70)

# ==============================================================================
# CONCETTO 1: LE QUATTRO OPERAZIONI
# ==============================================================================
"""
🔢 LE QUATTRO OPERAZIONI BASE

Non dare per scontato di ricordarle perfettamente.
Rivediamole con esempi pratici.
"""

def concetto_1_operazioni_base():
    """Le quattro operazioni con esempi trading."""
    
    print("\n" + "=" * 50)
    print("CONCETTO 1: LE QUATTRO OPERAZIONI")
    print("=" * 50)
    
    # ADDIZIONE (+)
    # -------------
    # "Sommare" = mettere insieme
    print("\n📌 ADDIZIONE (+) = Mettere insieme")
    print("-" * 40)
    
    capitale_iniziale = 1000   # Hai 1000€
    deposito = 500             # Depositi altri 500€
    capitale_totale = capitale_iniziale + deposito  # Quanto hai ora?
    
    print(f"Capitale iniziale: {capitale_iniziale}€")
    print(f"Deposito:          {deposito}€")
    print(f"Totale:            {capitale_iniziale} + {deposito} = {capitale_totale}€")
    
    # SOTTRAZIONE (-)
    # ---------------
    # "Sottrarre" = togliere
    print("\n📌 SOTTRAZIONE (-) = Togliere")
    print("-" * 40)
    
    prezzo_acquisto = 100      # Compri a 100€
    prezzo_vendita = 120       # Vendi a 120€
    guadagno = prezzo_vendita - prezzo_acquisto  # Quanto hai guadagnato?
    
    print(f"Prezzo acquisto: {prezzo_acquisto}€")
    print(f"Prezzo vendita:  {prezzo_vendita}€")
    print(f"Guadagno:        {prezzo_vendita} - {prezzo_acquisto} = {guadagno}€")
    
    # Se vendi a meno di quanto hai comprato = PERDITA
    prezzo_vendita_male = 80
    perdita = prezzo_vendita_male - prezzo_acquisto
    print(f"\nSe vendi a {prezzo_vendita_male}€:")
    print(f"Risultato: {prezzo_vendita_male} - {prezzo_acquisto} = {perdita}€ (PERDITA!)")
    
    # MOLTIPLICAZIONE (× o *)
    # -----------------------
    # "Moltiplicare" = ripetere tante volte
    print("\n📌 MOLTIPLICAZIONE (*) = Ripetere")
    print("-" * 40)
    
    prezzo_azione = 50         # Un'azione costa 50€
    numero_azioni = 10         # Compri 10 azioni
    costo_totale = prezzo_azione * numero_azioni  # Quanto spendi?
    
    print(f"Prezzo per azione: {prezzo_azione}€")
    print(f"Numero azioni:     {numero_azioni}")
    print(f"Costo totale:      {prezzo_azione} × {numero_azioni} = {costo_totale}€")
    print("  (È come sommare 50+50+50+50+50+50+50+50+50+50)")
    
    # DIVISIONE (÷ o /)
    # -----------------
    # "Dividere" = spartire in parti uguali
    print("\n📌 DIVISIONE (/) = Spartire")
    print("-" * 40)
    
    capitale = 1000            # Hai 1000€
    prezzo_azione = 50         # Un'azione costa 50€
    quante_azioni = capitale / prezzo_azione  # Quante ne puoi comprare?
    
    print(f"Capitale:          {capitale}€")
    print(f"Prezzo per azione: {prezzo_azione}€")
    print(f"Azioni comprabil:  {capitale} ÷ {prezzo_azione} = {quante_azioni} azioni")
    
    # DIVISIONE INTERA (//) - Solo la parte intera
    print("\n📌 DIVISIONE INTERA (//) = Solo numeri interi")
    print("-" * 40)
    
    capitale = 1000
    prezzo_azione = 156        # Prezzo "scomodo"
    
    # Divisione normale
    risultato_normale = capitale / prezzo_azione
    print(f"{capitale} / {prezzo_azione} = {risultato_normale:.4f}")
    print("  Ma non puoi comprare 6.41 azioni! Solo numeri interi.")
    
    # Divisione intera - butta via la parte decimale
    risultato_intero = capitale // prezzo_azione
    print(f"{capitale} // {prezzo_azione} = {risultato_intero}")
    print("  Puoi comprare 6 azioni intere.")
    
    # RESTO (%) - Quanto avanza
    resto = capitale % prezzo_azione
    print(f"\n📌 RESTO (%) = Quanto avanza")
    print(f"{capitale} % {prezzo_azione} = {resto}€ che ti rimangono")
    
    # Verifica: 6 azioni × 156€ = 936€, ti rimangono 1000-936 = 64€
    print(f"Verifica: {risultato_intero} × {prezzo_azione} = {risultato_intero * prezzo_azione}€")
    print(f"          {capitale} - {risultato_intero * prezzo_azione} = {resto}€ ✓")

# Esegui
concetto_1_operazioni_base()


# ==============================================================================
# CONCETTO 2: LE PERCENTUALI (%)
# ==============================================================================
"""
🔢 LE PERCENTUALI

La percentuale è un MODO DI ESPRIMERE una proporzione.
"Per cento" significa "su 100".

50% = 50 su 100 = metà
25% = 25 su 100 = un quarto
10% = 10 su 100 = un decimo
1%  = 1 su 100  = un centesimo
"""

def concetto_2_percentuali():
    """Le percentuali spiegate da zero."""
    
    print("\n" + "=" * 50)
    print("CONCETTO 2: LE PERCENTUALI")
    print("=" * 50)
    
    # COS'È UNA PERCENTUALE?
    print("\n📌 COS'È UNA PERCENTUALE?")
    print("-" * 40)
    
    print("'Per cento' = 'su cento' = 'diviso 100'")
    print()
    print("50%  significa  50 su 100  =  50/100  =  0.50  =  metà")
    print("25%  significa  25 su 100  =  25/100  =  0.25  =  un quarto")
    print("10%  significa  10 su 100  =  10/100  =  0.10  =  un decimo")
    print("1%   significa  1 su 100   =  1/100   =  0.01  =  un centesimo")
    print("100% significa  100 su 100 =  100/100 =  1.00  =  tutto")
    
    # DA PERCENTUALE A DECIMALE
    print("\n📌 DA PERCENTUALE A NUMERO DECIMALE")
    print("-" * 40)
    print("Regola: dividi per 100 (o sposta la virgola di 2 posti a sinistra)")
    print()
    print("50%  → 50 ÷ 100  = 0.50")
    print("25%  → 25 ÷ 100  = 0.25")
    print("10%  → 10 ÷ 100  = 0.10")
    print("5%   → 5 ÷ 100   = 0.05")
    print("1%   → 1 ÷ 100   = 0.01")
    print("0.5% → 0.5 ÷ 100 = 0.005")
    print("0.1% → 0.1 ÷ 100 = 0.001")
    
    # DA DECIMALE A PERCENTUALE
    print("\n📌 DA NUMERO DECIMALE A PERCENTUALE")
    print("-" * 40)
    print("Regola: moltiplica per 100 (o sposta la virgola di 2 posti a destra)")
    print()
    print("0.50  → 0.50 × 100 = 50%")
    print("0.25  → 0.25 × 100 = 25%")
    print("0.10  → 0.10 × 100 = 10%")
    print("0.05  → 0.05 × 100 = 5%")
    print("0.01  → 0.01 × 100 = 1%")
    print("0.001 → 0.001 × 100 = 0.1%")
    
    # CALCOLARE LA PERCENTUALE DI UN NUMERO
    print("\n📌 CALCOLARE LA PERCENTUALE DI UN NUMERO")
    print("-" * 40)
    print("Domanda: 'Quanto è il 10% di 500?'")
    print()
    print("Metodo 1: Trasforma in decimale e moltiplica")
    print("  10% = 0.10")
    print("  0.10 × 500 = 50")
    print("  Risposta: 50")
    print()
    
    # Esempi pratici
    capitale = 1000
    percentuale = 10  # 10%
    
    # Passo 1: trasforma percentuale in decimale
    decimale = percentuale / 100  # 10 / 100 = 0.10
    
    # Passo 2: moltiplica
    risultato = capitale * decimale  # 1000 × 0.10 = 100
    
    print("Esempio: Quanto è il 10% di 1000€?")
    print(f"  Passo 1: {percentuale}% = {percentuale}/100 = {decimale}")
    print(f"  Passo 2: {capitale} × {decimale} = {risultato}€")
    
    # In una sola formula
    print(f"\n  Formula unica: {capitale} × ({percentuale}/100) = {capitale * percentuale / 100}€")
    
    # ALTRI ESEMPI
    print("\n📌 ALTRI ESEMPI PRATICI")
    print("-" * 40)
    
    esempi = [
        (1000, 5, "Commissione broker"),
        (1000, 2, "Rischio per trade"),
        (500, 20, "Guadagno"),
        (500, 15, "Perdita"),
    ]
    
    for capitale, perc, descrizione in esempi:
        risultato = capitale * perc / 100
        print(f"{perc}% di {capitale}€ ({descrizione}) = {risultato}€")
    
    # CALCOLARE CHE PERCENTUALE È
    print("\n📌 CALCOLARE CHE PERCENTUALE È UN NUMERO")
    print("-" * 40)
    print("Domanda: '50 è che percentuale di 200?'")
    print()
    print("Formula: (parte / totale) × 100")
    print()
    
    parte = 50
    totale = 200
    percentuale = (parte / totale) * 100
    
    print(f"  ({parte} / {totale}) × 100 = {percentuale}%")
    print(f"  50 è il 25% di 200")
    
    # Esempio trading
    print("\nEsempio trading:")
    prezzo_acquisto = 100
    prezzo_vendita = 115
    guadagno = prezzo_vendita - prezzo_acquisto  # 15€
    percentuale_guadagno = (guadagno / prezzo_acquisto) * 100
    
    print(f"  Comprato a: {prezzo_acquisto}€")
    print(f"  Venduto a:  {prezzo_vendita}€")
    print(f"  Guadagno:   {guadagno}€")
    print(f"  Percentuale: ({guadagno} / {prezzo_acquisto}) × 100 = {percentuale_guadagno}%")

# Esegui
concetto_2_percentuali()


# ==============================================================================
# CONCETTO 3: COMMISSIONI
# ==============================================================================
"""
🔢 LE COMMISSIONI

Una commissione è una PERCENTUALE che paghi a qualcuno per un servizio.
Il broker (chi ti permette di comprare azioni) prende una commissione.
"""

def concetto_3_commissioni():
    """Cosa sono le commissioni."""
    
    print("\n" + "=" * 50)
    print("CONCETTO 3: LE COMMISSIONI")
    print("=" * 50)
    
    print("\n📌 COS'È UNA COMMISSIONE?")
    print("-" * 40)
    print("È una percentuale che PAGHI per usare un servizio.")
    print("Il broker (Binance, eToro, etc.) prende una commissione")
    print("ogni volta che compri o vendi.")
    print()
    
    # Esempio semplice
    print("📌 ESEMPIO SEMPLICE")
    print("-" * 40)
    
    valore_acquisto = 1000  # Compri per 1000€
    commissione_percentuale = 0.1  # 0.1% (comune in crypto)
    
    # Calcolo commissione
    commissione = valore_acquisto * commissione_percentuale / 100
    
    print(f"Compri per:     {valore_acquisto}€")
    print(f"Commissione:    {commissione_percentuale}%")
    print(f"")
    print(f"Calcolo: {valore_acquisto} × ({commissione_percentuale}/100)")
    print(f"       = {valore_acquisto} × {commissione_percentuale/100}")
    print(f"       = {commissione}€")
    print(f"")
    print(f"Paghi al broker: {commissione}€")
    print(f"Costo totale:    {valore_acquisto + commissione}€")
    
    # Commissioni comuni
    print("\n📌 COMMISSIONI TIPICHE")
    print("-" * 40)
    print("Binance crypto:  0.1% (molto basso)")
    print("eToro azioni:    0% (ma spread nascosto)")
    print("Banca italiana:  0.5-1% (alto!)")
    print("Trading pro:     0.01-0.05%")
    
    # Impatto su guadagno
    print("\n📌 PERCHÉ LE COMMISSIONI CONTANO")
    print("-" * 40)
    
    capitale = 1000
    commissione_perc = 0.1  # 0.1%
    
    # Compri e vendi (2 commissioni!)
    comm_acquisto = capitale * commissione_perc / 100
    comm_vendita = capitale * commissione_perc / 100
    totale_commissioni = comm_acquisto + comm_vendita
    
    print(f"Capitale: {capitale}€")
    print(f"Commissione per operazione: {commissione_perc}%")
    print()
    print(f"Commissione acquisto: {comm_acquisto}€")
    print(f"Commissione vendita:  {comm_vendita}€")
    print(f"TOTALE commissioni:   {totale_commissioni}€")
    print()
    print(f"Per guadagnare, il prezzo deve salire ALMENO del")
    print(f"{commissione_perc * 2}% solo per coprire le commissioni!")

# Esegui
concetto_3_commissioni()


# ==============================================================================
# CONCETTO 4: STOP LOSS E TAKE PROFIT
# ==============================================================================
"""
🔢 STOP LOSS E TAKE PROFIT

Sono LIVELLI DI PREZZO a cui vendere automaticamente.
- Stop Loss: limite di perdita (vendi se scende troppo)
- Take Profit: obiettivo di guadagno (vendi se sale abbastanza)
"""

def concetto_4_stop_loss_take_profit():
    """Stop Loss e Take Profit spiegati semplicemente."""
    
    print("\n" + "=" * 50)
    print("CONCETTO 4: STOP LOSS E TAKE PROFIT")
    print("=" * 50)
    
    print("\n📌 COSA SONO?")
    print("-" * 40)
    print("Sono PREZZI a cui decidi IN ANTICIPO di vendere.")
    print()
    print("STOP LOSS = 'Fermati alle perdite'")
    print("  → Se il prezzo SCENDE a questo livello, VENDI")
    print("  → Limita quanto puoi perdere")
    print()
    print("TAKE PROFIT = 'Prendi il profitto'")
    print("  → Se il prezzo SALE a questo livello, VENDI")
    print("  → Assicura il guadagno")
    
    # Esempio visuale
    print("\n📌 ESEMPIO VISUALE")
    print("-" * 40)
    
    prezzo_acquisto = 100
    stop_loss = 95        # -5% dal prezzo di acquisto
    take_profit = 110     # +10% dal prezzo di acquisto
    
    print(f"""
    TAKE PROFIT -----> {take_profit}€  (+10%)  🎯 Vendi qui se sale!
                         ↑
                         |
                         |
    PREZZO ACQUISTO --> {prezzo_acquisto}€  (qui hai comprato)
                         |
                         |
                         ↓
    STOP LOSS -------> {stop_loss}€   (-5%)   🛑 Vendi qui se scende!
    """)
    
    # Calcolo percentuali
    print("📌 COME CALCOLARE STOP LOSS E TAKE PROFIT")
    print("-" * 40)
    
    prezzo_acquisto = 100
    rischio_percentuale = 5    # Vuoi rischiare max 5%
    guadagno_percentuale = 10  # Vuoi guadagnare almeno 10%
    
    # Stop Loss = prezzo - (prezzo × percentuale / 100)
    stop_loss = prezzo_acquisto - (prezzo_acquisto * rischio_percentuale / 100)
    
    # Take Profit = prezzo + (prezzo × percentuale / 100)
    take_profit = prezzo_acquisto + (prezzo_acquisto * guadagno_percentuale / 100)
    
    print(f"Prezzo acquisto: {prezzo_acquisto}€")
    print(f"Rischio massimo: {rischio_percentuale}%")
    print(f"Obiettivo:       {guadagno_percentuale}%")
    print()
    print(f"Stop Loss:")
    print(f"  {prezzo_acquisto} - ({prezzo_acquisto} × {rischio_percentuale}/100)")
    print(f"  = {prezzo_acquisto} - {prezzo_acquisto * rischio_percentuale / 100}")
    print(f"  = {stop_loss}€")
    print()
    print(f"Take Profit:")
    print(f"  {prezzo_acquisto} + ({prezzo_acquisto} × {guadagno_percentuale}/100)")
    print(f"  = {prezzo_acquisto} + {prezzo_acquisto * guadagno_percentuale / 100}")
    print(f"  = {take_profit}€")
    
    # Scenario
    print("\n📌 SCENARI POSSIBILI")
    print("-" * 40)
    
    print("Scenario A: Prezzo sale a 110€")
    print(f"  → Raggiunge Take Profit → VENDI → Guadagno: +{guadagno_percentuale}%")
    print()
    print("Scenario B: Prezzo scende a 95€")
    print(f"  → Raggiunge Stop Loss → VENDI → Perdita: -{rischio_percentuale}%")
    print()
    print("Scenario C: Prezzo oscilla tra 96-109€")
    print("  → Né SL né TP raggiunti → Aspetti")

# Esegui
concetto_4_stop_loss_take_profit()


# ==============================================================================
# CONCETTO 5: POTENZE E RADICI
# ==============================================================================
"""
🔢 POTENZE E RADICI

Servono per calcolare la crescita nel tempo (interesse composto)
e per alcune formule di rischio.
"""

def concetto_5_potenze():
    """Potenze spiegate semplicemente."""
    
    print("\n" + "=" * 50)
    print("CONCETTO 5: POTENZE")
    print("=" * 50)
    
    print("\n📌 COS'È UNA POTENZA?")
    print("-" * 40)
    print("Una potenza è una moltiplicazione ripetuta.")
    print()
    print("2³ = 2 × 2 × 2 = 8")
    print("   (2 moltiplicato per se stesso 3 volte)")
    print()
    print("10² = 10 × 10 = 100")
    print("5² = 5 × 5 = 25")
    print("2⁴ = 2 × 2 × 2 × 2 = 16")
    
    print("\n📌 IN PYTHON")
    print("-" * 40)
    print("Si usa ** (due asterischi)")
    print()
    
    print(f"2 ** 3 = {2 ** 3}  (2³)")
    print(f"10 ** 2 = {10 ** 2}  (10²)")
    print(f"5 ** 2 = {5 ** 2}  (5²)")
    print(f"2 ** 4 = {2 ** 4}  (2⁴)")
    
    print("\n📌 A COSA SERVE NEL TRADING?")
    print("-" * 40)
    print("Per calcolare la CRESCITA COMPOSTA nel tempo.")
    print()
    
    # Esempio: Se guadagni 10% all'anno per 3 anni
    capitale_iniziale = 1000
    rendimento = 1.10  # 10% in più = ×1.10
    anni = 3
    
    print("Esempio: 1000€ con rendimento 10% annuo per 3 anni")
    print()
    print("Anno 1: 1000 × 1.10 = 1100€")
    print("Anno 2: 1100 × 1.10 = 1210€")
    print("Anno 3: 1210 × 1.10 = 1331€")
    print()
    print("Oppure in un colpo solo:")
    capitale_finale = capitale_iniziale * (rendimento ** anni)
    print(f"1000 × (1.10)³ = 1000 × {rendimento ** anni} = {capitale_finale}€")

# Esegui
concetto_5_potenze()


# ==============================================================================
# RIEPILOGO
# ==============================================================================

def riepilogo():
    """Riepilogo di tutti i concetti."""
    
    print("\n" + "=" * 70)
    print("📋 RIEPILOGO: FORMULE CHIAVE")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ PERCENTUALI                                                         │
├─────────────────────────────────────────────────────────────────────┤
│ Da percentuale a decimale:  X% = X / 100                           │
│ Da decimale a percentuale:  0.X = 0.X × 100 = X%                   │
│ X% di N:                    N × (X / 100)                          │
│ Che % è A di B:             (A / B) × 100                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ TRADING BASE                                                        │
├─────────────────────────────────────────────────────────────────────┤
│ Quante azioni comprare:     capitale // prezzo_azione              │
│ Costo totale:               prezzo × quantità                       │
│ Commissione:                valore × (commissione% / 100)          │
│ Guadagno/Perdita:           prezzo_vendita - prezzo_acquisto       │
│ Guadagno %:                 (guadagno / prezzo_acquisto) × 100     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ STOP LOSS E TAKE PROFIT                                            │
├─────────────────────────────────────────────────────────────────────┤
│ Stop Loss:    prezzo - (prezzo × rischio% / 100)                   │
│ Take Profit:  prezzo + (prezzo × obiettivo% / 100)                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ POTENZE (in Python)                                                 │
├─────────────────────────────────────────────────────────────────────┤
│ X elevato a N:  X ** N                                             │
│ Crescita composta: capitale × (1 + rendimento%) ** anni            │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print("✅ Ora sei pronto per iniziare ESERCIZI_COMPLETI_PARTE1!")
    print("   Quando incontri matematica, torna qui a rivedere.")

# Esegui
riepilogo()


# ==============================================================================
# ESERCIZI DI PRATICA (opzionali)
# ==============================================================================
"""
🎯 ESERCIZI DI PRATICA

Prova a fare questi calcoli A MANO (con carta e penna),
poi verifica con Python.
"""

def esercizi_pratica():
    """Esercizi per verificare comprensione."""
    
    print("\n" + "=" * 70)
    print("🎯 ESERCIZI DI PRATICA (falli a mano, poi verifica)")
    print("=" * 70)
    
    print("""
ESERCIZIO 1: Percentuali
------------------------
a) Quanto è il 15% di 200?
b) 30 è che percentuale di 150?
c) Se un prezzo aumenta del 20%, da 50€ quanto diventa?

ESERCIZIO 2: Commissioni
------------------------
a) Commissione 0.1% su acquisto di 5000€?
b) Commissione totale (acquisto + vendita) con 0.1% su 2000€?

ESERCIZIO 3: Stop Loss / Take Profit
------------------------------------
a) Compri a 80€, vuoi rischiare max 10%. Qual è lo Stop Loss?
b) Compri a 80€, vuoi guadagnare almeno 25%. Qual è il Take Profit?

ESERCIZIO 4: Quante azioni
--------------------------
a) Hai 5000€, un'azione costa 73€. Quante ne puoi comprare?
b) Quanto ti rimane dopo l'acquisto?
""")
    
    input("\nPremi INVIO per vedere le soluzioni...")
    
    print("""
SOLUZIONI:
----------
1a) 200 × 0.15 = 30
1b) (30/150) × 100 = 20%
1c) 50 + (50 × 0.20) = 50 + 10 = 60€

2a) 5000 × 0.001 = 5€
2b) 2000 × 0.001 × 2 = 4€

3a) 80 - (80 × 0.10) = 80 - 8 = 72€
3b) 80 + (80 × 0.25) = 80 + 20 = 100€

4a) 5000 // 73 = 68 azioni
4b) 5000 - (68 × 73) = 5000 - 4964 = 36€
""")

# Esegui esercizi
# esercizi_pratica()  # Decommenta per fare pratica


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("HAI COMPLETATO IL MODULO DI MATEMATICA BASE!")
    print("=" * 70)
    print()
    print("Ora puoi iniziare con:")
    print("  → ESERCIZI_COMPLETI_PARTE1_FUNDAMENTALS.py")
    print()
    print("Quando incontri matematica che non capisci,")
    print("torna qui a rivedere la sezione corrispondente.")
    print()
    print("Buono studio! 🚀")
