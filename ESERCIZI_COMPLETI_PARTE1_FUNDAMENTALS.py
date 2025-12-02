"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🐍 PYTHON MASTER - SCHEDA ESERCIZI COMPLETA               ║
║                                                                              ║
║                    PARTE 1: FUNDAMENTALS (Variabili, Tipi, Memory)           ║
║                                                                              ║
║                    Da Zero a Master - 600+ Esercizi con Soluzioni            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

STRUTTURA DEL DOCUMENTO:
========================
Ogni esercizio contiene:
1. 📋 CONSEGNA: Cosa devi fare
2. 💡 TEORIA: Concetto da imparare
3. 🎯 SKILLS: Competenze sviluppate
4. ⏱️ TEMPO: Tempo stimato
5. 🔢 LIVELLO: Principiante/Intermedio/Avanzato/Expert
6. ✅ SOLUZIONE: Codice completo commentato
7. 🧪 TEST: Come verificare la soluzione
8. 📚 APPROFONDIMENTO: Note aggiuntive

COME USARE QUESTO FILE:
=======================
1. Leggi la CONSEGNA
2. Prova a risolvere SENZA guardare la soluzione
3. Confronta con la SOLUZIONE
4. Esegui i TEST
5. Leggi l'APPROFONDIMENTO
"""

# ==============================================================================
# SEZIONE 1: VARIABILI E ASSEGNAZIONE
# ==============================================================================

print("=" * 70)
print("SEZIONE 1: VARIABILI E ASSEGNAZIONE")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 1.1: Hello Variables
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Crea tre variabili:
- name: il tuo nome (stringa)
- age: la tua età (intero)
- height: la tua altezza in metri (float)
Stampa una frase che le contenga tutte.

💡 TEORIA:
In Python le variabili sono "etichette" che puntano a oggetti in memoria.
Non serve dichiarare il tipo: Python lo inferisce automaticamente.
Questa caratteristica si chiama "dynamic typing".

🎯 SKILLS: Assegnazione variabili, tipi base, f-strings
⏱️ TEMPO: 5 minuti
🔢 LIVELLO: Principiante
"""

# ✅ SOLUZIONE:
def esercizio_1_1():
    """Hello Variables - Prima assegnazione"""
    
    # Assegnazione variabili
    name = "Marco"      # str: stringa di testo
    age = 30            # int: numero intero
    height = 1.75       # float: numero decimale
    
    # Stampa con f-string (formatted string literal)
    # Le f-string permettono di inserire variabili direttamente nella stringa
    print(f"Mi chiamo {name}, ho {age} anni e sono alto {height}m")
    
    # Verifica i tipi
    print(f"Tipo di name: {type(name)}")    # <class 'str'>
    print(f"Tipo di age: {type(age)}")      # <class 'int'>
    print(f"Tipo di height: {type(height)}")  # <class 'float'>
    
    return name, age, height

# 🧪 TEST:
if __name__ == "__main__":
    result = esercizio_1_1()
    assert isinstance(result[0], str), "name deve essere una stringa"
    assert isinstance(result[1], int), "age deve essere un intero"
    assert isinstance(result[2], float), "height deve essere un float"
    print("✅ Esercizio 1.1 completato!\n")

"""
📚 APPROFONDIMENTO:
- Le f-string (f"...") sono disponibili da Python 3.6+
- Sono più leggibili e performanti di .format() e %
- Puoi inserire espressioni: f"Tra 10 anni avrò {age + 10} anni"
"""


# ------------------------------------------------------------------------------
# ESERCIZIO 1.2: Multiple Assignment
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Usa l'assegnazione multipla per:
1. Assegnare tre valori a tre variabili in una riga
2. Scambiare i valori di due variabili SENZA usare una variabile temporanea
3. Assegnare lo stesso valore a tre variabili

💡 TEORIA:
Python supporta l'unpacking: puoi assegnare più valori contemporaneamente.
Lo swap senza temp è possibile grazie all'evaluazione simultanea del lato destro.

🎯 SKILLS: Tuple unpacking, swap idiomatico, assegnazione multipla
⏱️ TEMPO: 5 minuti
🔢 LIVELLO: Principiante
"""

# ✅ SOLUZIONE:
def esercizio_1_2():
    """Multiple Assignment - Assegnazioni multiple"""
    
    # 1. Assegnazione multipla (tuple unpacking)
    # Il lato destro crea una tupla (1, 2, 3) che viene "spacchettata"
    a, b, c = 1, 2, 3
    print(f"a={a}, b={b}, c={c}")
    
    # 2. Swap senza variabile temporanea
    # Python valuta TUTTO il lato destro prima di assegnare
    x, y = 10, 20
    print(f"Prima dello swap: x={x}, y={y}")
    
    x, y = y, x  # Magia! Equivale a: temp = (y, x); x, y = temp
    
    print(f"Dopo lo swap: x={x}, y={y}")
    
    # 3. Stesso valore a più variabili (chained assignment)
    alpha = beta = gamma = 100
    print(f"alpha={alpha}, beta={beta}, gamma={gamma}")
    
    # ATTENZIONE con oggetti mutabili!
    # Questo crea UN SOLO oggetto, tutti puntano allo stesso
    list1 = list2 = [1, 2, 3]
    list1.append(4)
    print(f"list1={list1}, list2={list2}")  # ENTRAMBE hanno 4!
    
    return a, b, c, x, y

# 🧪 TEST:
if __name__ == "__main__":
    a, b, c, x, y = esercizio_1_2()
    assert (a, b, c) == (1, 2, 3), "Assegnazione multipla fallita"
    assert (x, y) == (20, 10), "Swap fallito"
    print("✅ Esercizio 1.2 completato!\n")

"""
📚 APPROFONDIMENTO:
- Lo swap x, y = y, x funziona perché Python:
  1. Valuta il lato destro → crea tupla (y, x) = (20, 10)
  2. Assegna la tupla al lato sinistro → x=20, y=10
- Con oggetti mutabili (liste, dict), l'assegnazione multipla
  crea alias, non copie! Usa list.copy() o list[:] per copiare.
"""


# ------------------------------------------------------------------------------
# ESERCIZIO 1.3: Variable Naming Convention
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Crea variabili che seguono le convenzioni Python (PEP 8):
1. Una variabile normale
2. Una "costante"
3. Una variabile "privata"
4. Una variabile per un conteggio in loop
5. Una variabile per un valore inutilizzato

💡 TEORIA:
Python usa convenzioni di naming per comunicare l'intento:
- snake_case: variabili e funzioni
- SCREAMING_SNAKE_CASE: costanti
- _prefisso: uso interno (convenzione)
- __doppio_prefisso: name mangling in classi
- _: valore ignorato

🎯 SKILLS: PEP 8, naming conventions, codice leggibile
⏱️ TEMPO: 5 minuti
🔢 LIVELLO: Principiante
"""

# ✅ SOLUZIONE:
def esercizio_1_3():
    """Naming Conventions - Convenzioni di naming"""
    
    # 1. Variabile normale: snake_case
    user_name = "marco_rossi"
    total_count = 42
    is_active = True  # Booleani: is_, has_, can_
    
    # 2. Costante: SCREAMING_SNAKE_CASE
    # Python non ha vere costanti, è solo convenzione!
    MAX_RETRIES = 3
    API_BASE_URL = "https://api.example.com"
    PI = 3.14159
    
    # 3. Variabile "privata": _prefisso
    # Indica "uso interno, non toccare dall'esterno"
    _internal_counter = 0
    _cache = {}
    
    # 4. Variabile loop: spesso i, j, k o nomi descrittivi
    for i in range(3):
        print(f"Iterazione {i}")
    
    for index, value in enumerate([10, 20, 30]):
        print(f"index={index}, value={value}")
    
    # 5. Valore ignorato: underscore _
    # Quando non ti serve un valore
    first, _, third = (1, 2, 3)  # Ignoro il secondo
    print(f"first={first}, third={third}")
    
    # Ignoro il valore in un loop
    for _ in range(3):
        print("Eseguo 3 volte, non mi serve l'indice")
    
    return user_name, MAX_RETRIES

# 🧪 TEST:
if __name__ == "__main__":
    result = esercizio_1_3()
    print("✅ Esercizio 1.3 completato!\n")

"""
📚 APPROFONDIMENTO:
Nomi da EVITARE:
- l, O, I (confondibili con 1 e 0)
- Nomi builtin: list, str, type, id, sum, max, min
- Nomi troppo generici: data, value, temp, x

Nomi BUONI per trading:
- entry_price, exit_price (non ep, xp)
- position_size (non ps)
- stop_loss_percent (non slp)
"""


# ==============================================================================
# SEZIONE 2: TIPI DI DATI BASE
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 2: TIPI DI DATI BASE")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 2.1: Integer Operations
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Esplora le operazioni sugli interi:
1. Divisione intera (//) vs divisione normale (/)
2. Modulo (%) per il resto
3. Potenza (**)
4. Operatori bitwise (opzionale)

💡 TEORIA:
Gli interi Python hanno precisione arbitraria (no overflow!).
La divisione / restituisce sempre float, // tronca al intero inferiore.

🎯 SKILLS: Operazioni aritmetiche, divisione intera, modulo
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Principiante
"""

# ✅ SOLUZIONE:
def esercizio_2_1():
    """Integer Operations - Operazioni su interi"""
    
    # Divisione normale vs intera
    a, b = 17, 5
    
    print(f"{a} / {b} = {a / b}")      # 3.4 (float!)
    print(f"{a} // {b} = {a // b}")    # 3 (troncato)
    print(f"{a} % {b} = {a % b}")      # 2 (resto)
    
    # Relazione fondamentale: a = (a // b) * b + (a % b)
    assert a == (a // b) * b + (a % b), "Relazione violata!"
    print(f"Verifica: {a} = {a // b} * {b} + {a % b}")
    
    # divmod() restituisce entrambi
    quoziente, resto = divmod(a, b)
    print(f"divmod({a}, {b}) = ({quoziente}, {resto})")
    
    # Potenza
    print(f"2 ** 10 = {2 ** 10}")      # 1024
    print(f"2 ** 100 = {2 ** 100}")    # Numero ENORME, no overflow!
    
    # pow() con modulo (utile in crittografia)
    print(f"pow(2, 10, 1000) = {pow(2, 10, 1000)}")  # (2^10) % 1000 = 24
    
    # APPLICAZIONE TRADING: Calcolo lotti
    capital = 10000
    price_per_share = 156.78
    
    shares_can_buy = capital // price_per_share  # Arrotondo per difetto
    remaining_cash = capital % price_per_share
    
    print(f"\nCapitale: ${capital}")
    print(f"Prezzo azione: ${price_per_share}")
    print(f"Azioni acquistabili: {int(shares_can_buy)}")
    print(f"Cash rimanente: ${remaining_cash:.2f}")
    
    return shares_can_buy

# 🧪 TEST:
if __name__ == "__main__":
    result = esercizio_2_1()
    assert result == 63, "Calcolo lotti errato"
    print("✅ Esercizio 2.1 completato!\n")

"""
📚 APPROFONDIMENTO:
- // con numeri negativi arrotonda verso -∞
  -17 // 5 = -4 (non -3!)
- Per arrotondare verso zero: int(-17 / 5) = -3
- In trading, usa sempre Decimal per i soldi (vedi esercizio successivo)
"""


# ------------------------------------------------------------------------------
# ESERCIZIO 2.2: Float Precision
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Dimostra i problemi di precisione dei float e come risolverli:
1. Mostra che 0.1 + 0.2 != 0.3
2. Usa math.isclose() per confronti
3. Usa Decimal per precisione finanziaria

💡 TEORIA:
I float seguono IEEE 754 e hanno precisione limitata (circa 15-17 cifre).
0.1 in binario è un numero periodico, quindi viene troncato.
Per soldi e finance: SEMPRE usare Decimal!

🎯 SKILLS: Precisione floating-point, Decimal, confronti
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_2_2():
    """Float Precision - Precisione dei float"""
    
    import math
    from decimal import Decimal, ROUND_HALF_UP
    
    # 1. Il problema classico
    result = 0.1 + 0.2
    print(f"0.1 + 0.2 = {result}")
    print(f"0.1 + 0.2 == 0.3? {result == 0.3}")  # False!
    print(f"Rappresentazione reale: {result:.20f}")
    
    # 2. Confronto corretto con isclose()
    print(f"\nmath.isclose(0.1 + 0.2, 0.3)? {math.isclose(0.1 + 0.2, 0.3)}")
    
    # isclose con tolleranza personalizzata
    a, b = 1.0000001, 1.0000002
    print(f"isclose con rel_tol=1e-9: {math.isclose(a, b, rel_tol=1e-9)}")
    print(f"isclose con rel_tol=1e-6: {math.isclose(a, b, rel_tol=1e-6)}")
    
    # 3. Decimal per finanza
    print("\n--- DECIMAL PER TRADING ---")
    
    # MAI creare Decimal da float!
    bad_decimal = Decimal(0.1)  # Eredita l'errore del float
    print(f"Decimal(0.1) = {bad_decimal}")  # Brutto!
    
    # SEMPRE creare da stringa
    good_decimal = Decimal('0.1')
    print(f"Decimal('0.1') = {good_decimal}")  # Perfetto!
    
    # Operazioni precise
    price = Decimal('156.78')
    quantity = Decimal('100')
    commission_rate = Decimal('0.001')  # 0.1%
    
    total = price * quantity
    commission = total * commission_rate
    net_total = total + commission
    
    print(f"\nCalcolo ordine:")
    print(f"Prezzo: ${price}")
    print(f"Quantità: {quantity}")
    print(f"Totale: ${total}")
    print(f"Commissione (0.1%): ${commission}")
    print(f"Totale netto: ${net_total}")
    
    # Arrotondamento controllato
    rounded = commission.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    print(f"Commissione arrotondata: ${rounded}")
    
    return float(net_total)

# 🧪 TEST:
if __name__ == "__main__":
    result = esercizio_2_2()
    print("✅ Esercizio 2.2 completato!\n")

"""
📚 APPROFONDIMENTO:
Nel trading, errori di arrotondamento possono accumularsi:
- 1 milione di trade con errore 0.0001 = errore di $100
- Usa SEMPRE Decimal per:
  - Prezzi
  - Quantità
  - Commissioni
  - P&L calculations

Configurazione consigliata:
from decimal import getcontext
getcontext().prec = 28  # Precisione alta
"""


# ------------------------------------------------------------------------------
# ESERCIZIO 2.3: String Mastery
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Padroneggia le operazioni sulle stringhe:
1. Slicing con indici positivi e negativi
2. Metodi principali: upper, lower, strip, split, join
3. Formattazione avanzata con f-strings
4. String methods per validazione

💡 TEORIA:
Le stringhe Python sono immutabili: ogni operazione crea una nuova stringa.
Lo slicing usa la sintassi [start:stop:step].
Gli indici negativi contano dalla fine.

🎯 SKILLS: Slicing, metodi stringa, formattazione
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Principiante-Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_2_3():
    """String Mastery - Padronanza delle stringhe"""
    
    # Stringa di esempio
    text = "Python Trading Bot"
    
    # 1. SLICING
    print("--- SLICING ---")
    print(f"Stringa: '{text}'")
    print(f"Lunghezza: {len(text)}")
    
    # Indici positivi (da sinistra, partono da 0)
    print(f"text[0] = '{text[0]}'")        # 'P'
    print(f"text[7] = '{text[7]}'")        # 'T'
    print(f"text[0:6] = '{text[0:6]}'")    # 'Python'
    print(f"text[:6] = '{text[:6]}'")      # 'Python' (start default = 0)
    print(f"text[7:] = '{text[7:]}'")      # 'Trading Bot' (end default = fine)
    
    # Indici negativi (da destra, partono da -1)
    print(f"text[-1] = '{text[-1]}'")      # 't'
    print(f"text[-3:] = '{text[-3:]}'")    # 'Bot'
    print(f"text[:-4] = '{text[:-4]}'")    # 'Python Trading'
    
    # Step
    print(f"text[::2] = '{text[::2]}'")    # 'Pto rdn o' (ogni 2)
    print(f"text[::-1] = '{text[::-1]}'")  # Stringa invertita!
    
    # 2. METODI COMUNI
    print("\n--- METODI ---")
    
    messy = "  Hello World  \n"
    print(f"strip(): '{messy.strip()}'")
    print(f"lower(): '{text.lower()}'")
    print(f"upper(): '{text.upper()}'")
    print(f"title(): '{'hello world'.title()}'")
    print(f"capitalize(): '{'hello WORLD'.capitalize()}'")
    
    # Split e Join
    words = text.split()  # Default: spazi
    print(f"split(): {words}")
    
    csv_line = "AAPL,156.78,100,BUY"
    fields = csv_line.split(',')
    print(f"split(','): {fields}")
    
    rejoined = " | ".join(fields)
    print(f"join(): '{rejoined}'")
    
    # Replace
    new_text = text.replace("Python", "AI")
    print(f"replace(): '{new_text}'")
    
    # 3. FORMATTAZIONE AVANZATA
    print("\n--- F-STRINGS AVANZATE ---")
    
    symbol = "AAPL"
    price = 156.789
    change = -2.5
    volume = 1234567
    
    # Formattazione numeri
    print(f"Prezzo: ${price:.2f}")           # 2 decimali
    print(f"Cambio: {change:+.2f}%")         # Con segno
    print(f"Volume: {volume:,}")             # Separatore migliaia
    print(f"Volume: {volume:_}")             # Separatore underscore
    
    # Allineamento
    print(f"{'Symbol':<10} {'Price':>10} {'Change':>10}")
    print(f"{symbol:<10} {price:>10.2f} {change:>+10.2f}%")
    
    # Padding con zeri
    order_id = 42
    print(f"Order ID: {order_id:05d}")       # 00042
    
    # Notazione scientifica
    big_num = 1234567890
    print(f"Scientifico: {big_num:.2e}")     # 1.23e+09
    
    # Percentuale
    ratio = 0.156
    print(f"Percentuale: {ratio:.1%}")       # 15.6%
    
    # 4. VALIDAZIONE
    print("\n--- VALIDAZIONE ---")
    
    print(f"'123'.isdigit(): {'123'.isdigit()}")
    print(f"'abc'.isalpha(): {'abc'.isalpha()}")
    print(f"'abc123'.isalnum(): {'abc123'.isalnum()}")
    print(f"'  '.isspace(): {'  '.isspace()}")
    print(f"'Hello'.startswith('He'): {'Hello'.startswith('He')}")
    print(f"'Hello'.endswith('lo'): {'Hello'.endswith('lo')}")
    print(f"'Python' in text: {'Python' in text}")
    
    return text[::-1]  # Ritorna stringa invertita

# 🧪 TEST:
if __name__ == "__main__":
    result = esercizio_2_3()
    assert result == "toB gnidarT nohtyP", "Inversione stringa fallita"
    print("✅ Esercizio 2.3 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 2.4: Boolean Logic
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Esplora la logica booleana:
1. Truthy e Falsy values
2. Operatori and, or, not
3. Short-circuit evaluation
4. Confronti concatenati

💡 TEORIA:
In Python, ogni oggetto ha un valore booleano.
Falsy: False, None, 0, 0.0, '', [], {}, set()
Truthy: tutto il resto

🎯 SKILLS: Logica booleana, truthy/falsy, short-circuit
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Principiante-Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_2_4():
    """Boolean Logic - Logica booleana"""
    
    # 1. TRUTHY e FALSY
    print("--- TRUTHY e FALSY ---")
    
    falsy_values = [False, None, 0, 0.0, '', [], {}, set()]
    truthy_values = [True, 1, -1, 0.1, 'hello', [1], {'a': 1}, {1}]
    
    print("Valori Falsy:")
    for val in falsy_values:
        print(f"  bool({val!r:10}) = {bool(val)}")
    
    print("\nValori Truthy:")
    for val in truthy_values:
        print(f"  bool({val!r:10}) = {bool(val)}")
    
    # 2. OPERATORI LOGICI
    print("\n--- OPERATORI LOGICI ---")
    
    # and: ritorna il primo falsy, o l'ultimo truthy
    print(f"True and False = {True and False}")
    print(f"1 and 2 and 3 = {1 and 2 and 3}")  # 3 (ultimo truthy)
    print(f"1 and 0 and 3 = {1 and 0 and 3}")  # 0 (primo falsy)
    
    # or: ritorna il primo truthy, o l'ultimo falsy
    print(f"True or False = {True or False}")
    print(f"0 or '' or 'hello' = {0 or '' or 'hello'}")  # 'hello'
    print(f"0 or '' or [] = {0 or '' or []}")  # [] (ultimo falsy)
    
    # not: inverte
    print(f"not True = {not True}")
    print(f"not [] = {not []}")  # True (lista vuota è falsy)
    
    # 3. SHORT-CIRCUIT (valutazione pigra)
    print("\n--- SHORT-CIRCUIT ---")
    
    def check_positive(n):
        print(f"  Checking {n}...")
        return n > 0
    
    print("Test: False and check_positive(5)")
    result = False and check_positive(5)  # check_positive NON viene chiamata!
    print(f"Risultato: {result}")
    
    print("\nTest: True or check_positive(5)")
    result = True or check_positive(5)  # check_positive NON viene chiamata!
    print(f"Risultato: {result}")
    
    # APPLICAZIONE: Default values
    user_name = None
    display_name = user_name or "Guest"
    print(f"\nuser_name = {user_name}")
    print(f"display_name = {display_name}")
    
    # APPLICAZIONE: Guard clause
    data = {'price': 100}
    price = data and data.get('price')  # Safe access
    print(f"price = {price}")
    
    # 4. CONFRONTI CONCATENATI
    print("\n--- CONFRONTI CONCATENATI ---")
    
    x = 5
    # Invece di: x > 0 and x < 10
    print(f"0 < {x} < 10: {0 < x < 10}")
    
    # Funziona con qualsiasi confronto
    print(f"1 <= 2 <= 3: {1 <= 2 <= 3}")
    print(f"1 < 2 > 0: {1 < 2 > 0}")  # True! (1 < 2 and 2 > 0)
    
    # APPLICAZIONE TRADING: Validazione range
    price = 156.50
    stop_loss = 150.00
    take_profit = 165.00
    
    valid_order = stop_loss < price < take_profit
    print(f"\nOrdine valido (SL < Price < TP): {valid_order}")
    
    return valid_order

# 🧪 TEST:
if __name__ == "__main__":
    result = esercizio_2_4()
    assert result == True, "Validazione range fallita"
    print("✅ Esercizio 2.4 completato!\n")


# ==============================================================================
# SEZIONE 3: MEMORY MODEL E REFERENCE
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 3: MEMORY MODEL E REFERENCE")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 3.1: Identity vs Equality
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Dimostra la differenza tra:
1. == (equality): i valori sono uguali?
2. is (identity): è lo STESSO oggetto in memoria?

💡 TEORIA:
Ogni oggetto Python ha un id() univoco (indirizzo memoria).
"is" confronta gli id, "==" confronta i valori.
Python fa "interning" di small integers (-5 to 256) e alcune stringhe.

🎯 SKILLS: Identity, equality, id(), interning
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_3_1():
    """Identity vs Equality - Identità vs Uguaglianza"""
    
    print("--- INTERI: INTERNING ---")
    
    # Small integers: Python riusa gli stessi oggetti
    a = 256
    b = 256
    print(f"a = {a}, b = {b}")
    print(f"a == b: {a == b}")  # True (valori uguali)
    print(f"a is b: {a is b}")  # True (stesso oggetto!)
    print(f"id(a) = {id(a)}, id(b) = {id(b)}")
    
    # Large integers: oggetti diversi
    x = 257
    y = 257
    print(f"\nx = {x}, y = {y}")
    print(f"x == y: {x == y}")  # True
    print(f"x is y: {x is y}")  # False (oggetti diversi!)
    print(f"id(x) = {id(x)}, id(y) = {id(y)}")
    
    print("\n--- STRINGHE: INTERNING ---")
    
    # Stringhe semplici: interned
    s1 = "hello"
    s2 = "hello"
    print(f"s1 = '{s1}', s2 = '{s2}'")
    print(f"s1 is s2: {s1 is s2}")  # True
    
    # Stringhe con spazi: potrebbe non essere interned
    s3 = "hello world"
    s4 = "hello world"
    print(f"\ns3 = '{s3}', s4 = '{s4}'")
    print(f"s3 is s4: {s3 is s4}")  # Dipende dall'implementazione
    print(f"s3 == s4: {s3 == s4}")  # Sempre True
    
    print("\n--- LISTE: MAI INTERNED ---")
    
    # Liste: sempre oggetti diversi
    list1 = [1, 2, 3]
    list2 = [1, 2, 3]
    print(f"list1 = {list1}, list2 = {list2}")
    print(f"list1 == list2: {list1 == list2}")  # True (stessi valori)
    print(f"list1 is list2: {list1 is list2}")  # False (oggetti diversi)
    
    # Stesso riferimento
    list3 = list1
    print(f"\nlist3 = list1")
    print(f"list3 is list1: {list3 is list1}")  # True (stesso oggetto)
    
    print("\n--- CONFRONTO CON NONE ---")
    
    # SEMPRE usare "is" con None
    value = None
    
    # CORRETTO:
    if value is None:
        print("value is None: CORRETTO!")
    
    # Tecnicamente funziona ma NON idiomatico:
    if value == None:
        print("value == None: funziona ma evitare")
    
    return True

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_3_1()
    print("✅ Esercizio 3.1 completato!\n")

"""
📚 APPROFONDIMENTO:
REGOLE D'ORO:
1. Usa "is" SOLO per: None, True, False, singleton
2. Usa "==" per confrontare valori
3. Non fare affidamento sull'interning (comportamento implementazione)
4. Per debug, usa id() per capire cosa sta succedendo
"""


# ------------------------------------------------------------------------------
# ESERCIZIO 3.2: Mutable vs Immutable
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Dimostra la differenza tra oggetti mutabili e immutabili:
1. Modifica una lista (mutabile) e osserva l'id
2. "Modifica" una stringa (immutabile) e osserva l'id
3. Mostra il problema degli argomenti mutabili di default

💡 TEORIA:
Immutabili: int, float, str, tuple, frozenset, bytes
Mutabili: list, dict, set, oggetti custom

Quando "modifichi" un immutabile, crei un nuovo oggetto.
Quando modifichi un mutabile, cambi l'oggetto esistente.

🎯 SKILLS: Mutabilità, side effects, defensive programming
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_3_2():
    """Mutable vs Immutable - Mutabilità"""
    
    print("--- IMMUTABILE: STRINGA ---")
    
    s = "Hello"
    print(f"s = '{s}', id = {id(s)}")
    
    original_id = id(s)
    s = s + " World"  # Crea NUOVO oggetto!
    
    print(f"Dopo s = s + ' World':")
    print(f"s = '{s}', id = {id(s)}")
    print(f"ID cambiato: {id(s) != original_id}")
    
    print("\n--- MUTABILE: LISTA ---")
    
    lst = [1, 2, 3]
    print(f"lst = {lst}, id = {id(lst)}")
    
    original_id = id(lst)
    lst.append(4)  # Modifica l'oggetto ESISTENTE
    
    print(f"Dopo lst.append(4):")
    print(f"lst = {lst}, id = {id(lst)}")
    print(f"ID cambiato: {id(lst) != original_id}")  # False!
    
    print("\n--- ALIAS E SIDE EFFECTS ---")
    
    original = [1, 2, 3]
    alias = original  # NON è una copia!
    
    print(f"original = {original}")
    print(f"alias = {alias}")
    print(f"alias is original: {alias is original}")
    
    alias.append(4)  # Modifica ENTRAMBI!
    
    print(f"\nDopo alias.append(4):")
    print(f"original = {original}")  # Anche original ha 4!
    print(f"alias = {alias}")
    
    print("\n--- IL PROBLEMA DEL DEFAULT MUTABILE ---")
    
    # SBAGLIATO: default mutabile
    def bad_append(item, lst=[]):
        lst.append(item)
        return lst
    
    result1 = bad_append(1)
    print(f"bad_append(1) = {result1}")
    
    result2 = bad_append(2)  # La stessa lista!
    print(f"bad_append(2) = {result2}")  # [1, 2] invece di [2]!
    
    result3 = bad_append(3)
    print(f"bad_append(3) = {result3}")  # [1, 2, 3]!
    
    print("\n--- SOLUZIONE: None come default ---")
    
    # CORRETTO: None come default
    def good_append(item, lst=None):
        if lst is None:
            lst = []
        lst.append(item)
        return lst
    
    result1 = good_append(1)
    print(f"good_append(1) = {result1}")
    
    result2 = good_append(2)
    print(f"good_append(2) = {result2}")  # [2] come atteso!
    
    return True

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_3_2()
    print("✅ Esercizio 3.2 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 3.3: Shallow vs Deep Copy
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Dimostra le differenze tra:
1. Assegnazione (alias)
2. Shallow copy (copia superficiale)
3. Deep copy (copia profonda)

💡 TEORIA:
- Assegnazione: crea un alias, stesso oggetto
- Shallow copy: copia l'oggetto esterno, ma non gli oggetti interni
- Deep copy: copia tutto ricorsivamente

🎯 SKILLS: copy module, liste nested, defensive copying
⏱️ TEMPO: 15 minuti
🔢 LIVELLO: Intermedio-Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_3_3():
    """Shallow vs Deep Copy - Tipi di copia"""
    
    import copy
    
    # Lista nested (lista di liste)
    original = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    print("--- ORIGINAL ---")
    print(f"original = {original}")
    print(f"id(original) = {id(original)}")
    print(f"id(original[0]) = {id(original[0])}")
    
    # 1. ASSEGNAZIONE (alias)
    print("\n--- ASSEGNAZIONE (ALIAS) ---")
    
    alias = original
    
    print(f"alias is original: {alias is original}")  # True
    print(f"alias[0] is original[0]: {alias[0] is original[0]}")  # True
    
    alias[0][0] = 999
    print(f"\nDopo alias[0][0] = 999:")
    print(f"original = {original}")  # Modificato!
    print(f"alias = {alias}")
    
    # Reset
    original[0][0] = 1
    
    # 2. SHALLOW COPY
    print("\n--- SHALLOW COPY ---")
    
    # Tre modi equivalenti:
    shallow1 = original.copy()
    shallow2 = list(original)
    shallow3 = original[:]
    shallow4 = copy.copy(original)
    
    print(f"shallow1 is original: {shallow1 is original}")  # False
    print(f"shallow1[0] is original[0]: {shallow1[0] is original[0]}")  # True!
    
    # La lista esterna è copiata, ma le liste interne sono condivise!
    shallow1[0][0] = 888
    print(f"\nDopo shallow1[0][0] = 888:")
    print(f"original = {original}")  # Modificato anche original!
    print(f"shallow1 = {shallow1}")
    
    # Reset
    original[0][0] = 1
    
    # 3. DEEP COPY
    print("\n--- DEEP COPY ---")
    
    deep = copy.deepcopy(original)
    
    print(f"deep is original: {deep is original}")  # False
    print(f"deep[0] is original[0]: {deep[0] is original[0]}")  # False!
    
    deep[0][0] = 777
    print(f"\nDopo deep[0][0] = 777:")
    print(f"original = {original}")  # NON modificato!
    print(f"deep = {deep}")
    
    print("\n--- APPLICAZIONE TRADING: Portfolio Copy ---")
    
    portfolio = {
        'positions': [
            {'symbol': 'AAPL', 'qty': 100, 'entry': 150.0},
            {'symbol': 'GOOGL', 'qty': 50, 'entry': 140.0}
        ],
        'cash': 10000.0
    }
    
    # Per simulazioni, SEMPRE deep copy!
    simulation = copy.deepcopy(portfolio)
    simulation['positions'][0]['qty'] = 200  # Non modifica l'originale
    
    print(f"Portfolio originale: {portfolio['positions'][0]['qty']} AAPL")
    print(f"Simulazione: {simulation['positions'][0]['qty']} AAPL")
    
    return True

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_3_3()
    print("✅ Esercizio 3.3 completato!\n")

"""
📚 APPROFONDIMENTO:
QUANDO USARE COSA:
- Alias: quando VUOI che le modifiche si propaghino
- Shallow copy: oggetti semplici (liste di primitivi)
- Deep copy: oggetti nested, quando vuoi isolamento totale

PERFORMANCE:
- Alias: O(1)
- Shallow copy: O(n) dove n è la lunghezza
- Deep copy: O(n*m) dove m è la profondità

Per trading: SEMPRE deep copy per simulazioni e backtesting!
"""


# ------------------------------------------------------------------------------
# ESERCIZIO 3.4: Reference Counter
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Esplora il reference counting di Python:
1. Usa sys.getrefcount() per vedere i riferimenti
2. Dimostra quando gli oggetti vengono deallocati
3. Comprendi il ciclo di vita degli oggetti

💡 TEORIA:
Python usa reference counting + garbage collector.
Quando refcount arriva a 0, l'oggetto viene deallocato.
getrefcount() mostra sempre +1 (il riferimento del parametro).

🎯 SKILLS: Memory management, reference counting, gc
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Avanzato
"""

# ✅ SOLUZIONE:
def esercizio_3_4():
    """Reference Counter - Conteggio riferimenti"""
    
    import sys
    
    print("--- REFERENCE COUNTING ---")
    
    # Crea un oggetto
    my_list = [1, 2, 3]
    
    # getrefcount() aggiunge 1 per il parametro passato
    print(f"Riferimenti a my_list: {sys.getrefcount(my_list)}")
    # Tipicamente 2: my_list + parametro di getrefcount
    
    # Aggiungi un riferimento
    another_ref = my_list
    print(f"Dopo another_ref = my_list: {sys.getrefcount(my_list)}")  # 3
    
    # Aggiungi in una lista
    container = [my_list]
    print(f"Dopo container = [my_list]: {sys.getrefcount(my_list)}")  # 4
    
    # Rimuovi riferimenti
    del another_ref
    print(f"Dopo del another_ref: {sys.getrefcount(my_list)}")  # 3
    
    container.clear()
    print(f"Dopo container.clear(): {sys.getrefcount(my_list)}")  # 2
    
    print("\n--- INTERNING E REFCOUNT ---")
    
    # Small integers hanno refcount altissimo (sono condivisi)
    small = 1
    print(f"Riferimenti a 1: {sys.getrefcount(1)}")  # Molto alto!
    
    # Large integers hanno refcount basso
    large = 99999
    print(f"Riferimenti a 99999: {sys.getrefcount(large)}")  # Basso
    
    print("\n--- DIMENSIONE OGGETTI ---")
    
    print(f"int: {sys.getsizeof(1)} bytes")
    print(f"float: {sys.getsizeof(1.0)} bytes")
    print(f"str vuota: {sys.getsizeof('')} bytes")
    print(f"str 'hello': {sys.getsizeof('hello')} bytes")
    print(f"list vuota: {sys.getsizeof([])} bytes")
    print(f"list [1,2,3]: {sys.getsizeof([1,2,3])} bytes")
    print(f"dict vuoto: {sys.getsizeof({})} bytes")
    
    return True

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_3_4()
    print("✅ Esercizio 3.4 completato!\n")


# ==============================================================================
# SEZIONE 4: TYPE CHECKING E CONVERSIONI
# ==============================================================================

print("\n" + "=" * 70)
print("SEZIONE 4: TYPE CHECKING E CONVERSIONI")
print("=" * 70)

# ------------------------------------------------------------------------------
# ESERCIZIO 4.1: Type Checking
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Impara a verificare i tipi in Python:
1. Usa type() per ottenere il tipo
2. Usa isinstance() per verifiche (preferito)
3. Gestisci tipi multipli

💡 TEORIA:
- type() ritorna il tipo esatto
- isinstance() verifica anche l'ereditarietà
- Usa isinstance() per il duck typing

🎯 SKILLS: type(), isinstance(), duck typing
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_4_1():
    """Type Checking - Verifica dei tipi"""
    
    print("--- type() vs isinstance() ---")
    
    x = 42
    
    # type() - tipo esatto
    print(f"type(42) = {type(x)}")
    print(f"type(42) == int: {type(x) == int}")
    
    # isinstance() - include sottoclassi
    print(f"isinstance(42, int): {isinstance(x, int)}")
    
    # bool è sottoclasse di int!
    b = True
    print(f"\ntype(True) = {type(b)}")
    print(f"type(True) == int: {type(b) == int}")  # False
    print(f"isinstance(True, int): {isinstance(b, int)}")  # True!
    
    print("\n--- TIPI MULTIPLI ---")
    
    # Verifica multipli tipi con tupla
    def process_number(n):
        if isinstance(n, (int, float)):
            return n * 2
        else:
            raise TypeError(f"Expected number, got {type(n)}")
    
    print(f"process_number(5) = {process_number(5)}")
    print(f"process_number(5.5) = {process_number(5.5)}")
    
    try:
        process_number("5")
    except TypeError as e:
        print(f"Errore: {e}")
    
    print("\n--- NUMBERS ABC ---")
    
    from numbers import Number, Real, Integral
    from decimal import Decimal
    
    values = [42, 3.14, Decimal('1.5'), True, 2+3j]
    
    for v in values:
        print(f"{v:>15} - Number: {isinstance(v, Number):>5}, "
              f"Real: {isinstance(v, Real):>5}, "
              f"Integral: {isinstance(v, Integral):>5}")
    
    print("\n--- APPLICAZIONE TRADING ---")
    
    def validate_order(symbol, quantity, price):
        """Valida parametri ordine"""
        errors = []
        
        if not isinstance(symbol, str):
            errors.append(f"symbol deve essere str, ricevuto {type(symbol)}")
        
        if not isinstance(quantity, (int, float)):
            errors.append(f"quantity deve essere numero, ricevuto {type(quantity)}")
        elif quantity <= 0:
            errors.append("quantity deve essere positivo")
        
        if not isinstance(price, (int, float, Decimal)):
            errors.append(f"price deve essere numero, ricevuto {type(price)}")
        elif price <= 0:
            errors.append("price deve essere positivo")
        
        if errors:
            raise ValueError("; ".join(errors))
        
        return True
    
    # Test
    print(validate_order("AAPL", 100, 156.78))
    
    try:
        validate_order(123, "cento", -50)
    except ValueError as e:
        print(f"Errori: {e}")
    
    return True

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_4_1()
    print("✅ Esercizio 4.1 completato!\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 4.2: Type Conversions
# ------------------------------------------------------------------------------
"""
📋 CONSEGNA:
Padroneggia le conversioni di tipo:
1. Conversioni esplicite: int(), float(), str(), bool()
2. Gestisci gli errori di conversione
3. Parsing di stringhe numeriche

💡 TEORIA:
Le conversioni possono fallire (ValueError, TypeError).
Usa try/except per gestire input malformati.

🎯 SKILLS: Type casting, error handling, parsing
⏱️ TEMPO: 10 minuti
🔢 LIVELLO: Intermedio
"""

# ✅ SOLUZIONE:
def esercizio_4_2():
    """Type Conversions - Conversioni di tipo"""
    
    print("--- CONVERSIONI BASE ---")
    
    # Da stringa a numero
    s = "42"
    i = int(s)
    f = float(s)
    print(f"int('{s}') = {i}")
    print(f"float('{s}') = {f}")
    
    # Da numero a stringa
    n = 42
    s = str(n)
    print(f"str({n}) = '{s}'")
    
    # Da float a int (tronca, non arrotonda!)
    f = 3.7
    print(f"int({f}) = {int(f)}")  # 3, non 4!
    print(f"round({f}) = {round(f)}")  # 4
    
    print("\n--- CASI SPECIALI ---")
    
    # Float con notazione scientifica
    print(f"float('1e-5') = {float('1e-5')}")
    
    # Int con base
    print(f"int('1010', 2) = {int('1010', 2)}")   # Binario → 10
    print(f"int('ff', 16) = {int('ff', 16)}")     # Hex → 255
    print(f"int('77', 8) = {int('77', 8)}")       # Ottale → 63
    
    # Conversioni inverse
    print(f"bin(10) = {bin(10)}")    # '0b1010'
    print(f"hex(255) = {hex(255)}")  # '0xff'
    print(f"oct(63) = {oct(63)}")    # '0o77'
    
    print("\n--- GESTIONE ERRORI ---")
    
    def safe_int(value, default=0):
        """Conversione sicura a int"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def safe_float(value, default=0.0):
        """Conversione sicura a float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    print(f"safe_int('42') = {safe_int('42')}")
    print(f"safe_int('abc') = {safe_int('abc')}")
    print(f"safe_int(None) = {safe_int(None)}")
    print(f"safe_float('3.14') = {safe_float('3.14')}")
    print(f"safe_float('invalid') = {safe_float('invalid')}")
    
    print("\n--- PARSING TRADING DATA ---")
    
    def parse_price(price_str):
        """Parse prezzo da stringa, gestendo formati diversi"""
        if not price_str:
            return None
        
        # Rimuovi simboli valuta e spazi
        cleaned = price_str.strip().replace('$', '').replace('€', '').replace(',', '')
        
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    test_prices = ["$156.78", "€99.99", "1,234.56", "invalid", "", None]
    
    for p in test_prices:
        print(f"parse_price({p!r}) = {parse_price(p)}")
    
    return True

# 🧪 TEST:
if __name__ == "__main__":
    esercizio_4_2()
    print("✅ Esercizio 4.2 completato!\n")


# ==============================================================================
# RIEPILOGO SEZIONE 1-4: FUNDAMENTALS
# ==============================================================================

print("\n" + "=" * 70)
print("RIEPILOGO: ESERCIZI FUNDAMENTALS COMPLETATI")
print("=" * 70)

print("""
ESERCIZI COMPLETATI IN QUESTA PARTE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEZIONE 1 - Variabili e Assegnazione:
  ✅ 1.1 Hello Variables
  ✅ 1.2 Multiple Assignment
  ✅ 1.3 Variable Naming Convention

SEZIONE 2 - Tipi di Dati Base:
  ✅ 2.1 Integer Operations
  ✅ 2.2 Float Precision (Decimal!)
  ✅ 2.3 String Mastery
  ✅ 2.4 Boolean Logic

SEZIONE 3 - Memory Model e Reference:
  ✅ 3.1 Identity vs Equality
  ✅ 3.2 Mutable vs Immutable
  ✅ 3.3 Shallow vs Deep Copy
  ✅ 3.4 Reference Counter

SEZIONE 4 - Type Checking e Conversioni:
  ✅ 4.1 Type Checking
  ✅ 4.2 Type Conversions

TOTALE: 13 esercizi

PROSSIMA PARTE: Control Flow (if, loops, comprehensions)
""")

# Esegui tutti gli esercizi se eseguito come script principale
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ESECUZIONE TUTTI GLI ESERCIZI")
    print("=" * 70 + "\n")
    
    # Esegui in sequenza
    esercizio_1_1()
    esercizio_1_2()
    esercizio_1_3()
    esercizio_2_1()
    esercizio_2_2()
    esercizio_2_3()
    esercizio_2_4()
    esercizio_3_1()
    esercizio_3_2()
    esercizio_3_3()
    esercizio_3_4()
    esercizio_4_1()
    esercizio_4_2()
    
    print("\n🎉 TUTTI GLI ESERCIZI DELLA PARTE 1 COMPLETATI!")
