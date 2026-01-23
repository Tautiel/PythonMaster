"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PYTHON ESSENTIALS 1 - MODULE 2                            ║
║              Data Types, Variables, Operators, Basic I/O                     ║
║                                                                              ║
║                    Allineato al Syllabus PCEP-30-02                          ║
║                    BLOCK 1: Fundamentals (18% dell'esame)                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

STRUTTURA MODULO:
├── Section 2.1: Literals (integers, floats, strings, booleans)
├── Section 2.2: Operators (arithmetic, string)
├── Section 2.3: Variables and naming conventions
├── Section 2.4: Comments
├── Section 2.5: input() and type conversion
├── Section 2.6: String and numeric operators in detail
├── Section 2.7: Bitwise operators
├── Quiz e Labs (30+ esercizi)

TEMPO STIMATO: 6-8 ore
QUESTO È IL MODULO PIÙ DENSO PER LE BASI!

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2.1: LITERALS
# ══════════════════════════════════════════════════════════════════════════════
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  2.1 LITERALS - Valori scritti direttamente nel codice                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

Un LITERAL è un valore fisso scritto direttamente nel source code.
Python riconosce il tipo dal modo in cui è scritto.

───────────────────────────────────────────────────────────────────────────────
2.1.1 - INTEGER LITERALS (int)
───────────────────────────────────────────────────────────────────────────────

Numeri interi, positivi o negativi, senza punto decimale.
"""

# DECIMALE (base 10) - il modo normale
a = 42
b = -17
c = 0

# BINARIO (base 2) - prefisso 0b o 0B
binary = 0b1010     # = 10 in decimale
binary2 = 0B1111    # = 15 in decimale

# OTTALE (base 8) - prefisso 0o o 0O
octal = 0o17        # = 15 in decimale (1×8 + 7 = 15)
octal2 = 0O777      # = 511 in decimale

# ESADECIMALE (base 16) - prefisso 0x o 0X
hexa = 0xFF         # = 255 in decimale (15×16 + 15 = 255)
hexa2 = 0x1A        # = 26 in decimale

# UNDERSCORE per leggibilità (Python 3.6+)
million = 1_000_000     # = 1000000
binary3 = 0b_1111_0000  # = 240


"""
⚠️ IMPORTANTE PER L'ESAME:
print() stampa SEMPRE in DECIMALE, indipendentemente da come hai scritto il numero!
"""
print(0b1010)   # Output: 10 (non "0b1010"!)
print(0o17)     # Output: 15 (non "0o17"!)
print(0xFF)     # Output: 255 (non "0xFF"!)


"""
───────────────────────────────────────────────────────────────────────────────
2.1.2 - FLOAT LITERALS (floating-point)
───────────────────────────────────────────────────────────────────────────────

Numeri con parte decimale.
"""

# Notazione normale
pi = 3.14159
negative = -2.5

# Il punto è OBBLIGATORIO per float
f1 = 4.0        # float
f2 = 4.         # float (equivalente a 4.0) - il punto basta!
f3 = .5         # float (equivalente a 0.5) - zero opzionale

# NOTAZIONE SCIENTIFICA (E notation)
# xEy significa: x × 10^y
light = 3e8         # 3 × 10^8 = 300,000,000.0
small = 6.62e-34    # 6.62 × 10^-34
also = 2.5E3        # 2.5 × 10^3 = 2500.0

print(3e8)      # Output: 300000000.0
print(1e-2)     # Output: 0.01
print(2.5e3)    # Output: 2500.0


"""
⚠️ IMPORTANTE PER L'ESAME:
- Notazione E produce SEMPRE un float
- Python stampa float in formato più leggibile quando possibile
"""


"""
───────────────────────────────────────────────────────────────────────────────
2.1.3 - STRING LITERALS (str)
───────────────────────────────────────────────────────────────────────────────

Sequenze di caratteri racchiuse in apici.
"""

# Singoli o doppi apici - equivalenti
s1 = 'Hello'
s2 = "Hello"

# Stringhe con apici interni
s3 = "It's Python"         # Doppi fuori, singoli dentro
s4 = 'Say "Hello"'         # Singoli fuori, doppi dentro
s5 = "It\'s Python"        # Escape del singolo apice
s6 = 'Say \"Hello\"'       # Escape dei doppi apici

# Stringa VUOTA
empty = ""
empty2 = ''

# Stringhe MULTILINEA con triple quotes
multiline = """Questa è una stringa
su più righe.
Mantiene i newline."""

multiline2 = '''Anche questo
funziona.'''


"""
───────────────────────────────────────────────────────────────────────────────
2.1.4 - BOOLEAN LITERALS (bool)
───────────────────────────────────────────────────────────────────────────────

Solo DUE valori possibili: True e False
CASE-SENSITIVE: True ≠ true ≠ TRUE
"""

flag = True
is_valid = False

# ERRORE COMUNE:
# x = true    # NameError! Deve essere True con la T maiuscola

# Booleans sono anche numeri!
print(True + True)      # Output: 2 (True = 1)
print(False + 1)        # Output: 1 (False = 0)
print(True * 10)        # Output: 10


"""
───────────────────────────────────────────────────────────────────────────────
2.1.5 - NONE LITERAL
───────────────────────────────────────────────────────────────────────────────

None rappresenta l'ASSENZA di valore.
"""

nothing = None
print(None)         # Output: None
print(type(None))   # Output: <class 'NoneType'>

# None NON è:
# - 0
# - False
# - "" (stringa vuota)
# È un tipo a sé stante!


"""
───────────────────────────────────────────────────────────────────────────────
2.1.6 - LA FUNZIONE type()
───────────────────────────────────────────────────────────────────────────────

type() restituisce il tipo di un oggetto.
"""
print(type(42))         # <class 'int'>
print(type(3.14))       # <class 'float'>
print(type("hello"))    # <class 'str'>
print(type(True))       # <class 'bool'>
print(type(None))       # <class 'NoneType'>


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2.2: OPERATORS
# ══════════════════════════════════════════════════════════════════════════════
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  2.2 OPERATORS - Operatori aritmetici                                      ║
╚═══════════════════════════════════════════════════════════════════════════╝

───────────────────────────────────────────────────────────────────────────────
2.2.1 - OPERATORI ARITMETICI
───────────────────────────────────────────────────────────────────────────────
"""

# ADDIZIONE (+)
print(5 + 3)        # 8
print(2.5 + 1.5)    # 4.0

# SOTTRAZIONE (-)
print(5 - 3)        # 2

# MOLTIPLICAZIONE (*)
print(5 * 3)        # 15

# DIVISIONE (/) - TRUE DIVISION
# ⚠️ SEMPRE restituisce FLOAT in Python 3!
print(5 / 2)        # 2.5
print(6 / 2)        # 3.0 (anche se divisibile, è float!)
print(10 / 5)       # 2.0

# FLOOR DIVISION (//) - INTEGER DIVISION
# Arrotonda verso MENO INFINITO (non verso zero!)
print(5 // 2)       # 2
print(7 // 3)       # 2
print(-7 // 2)      # -4 ⚠️ NON -3! Arrotonda verso -infinito
print(7 // -2)      # -4

# MODULO (%) - Resto della divisione
# Il risultato ha il SEGNO DEL DIVISORE
print(7 % 2)        # 1
print(-7 % 2)       # 1 ⚠️ NON -1!
print(7 % -2)       # -1
print(-7 % -2)      # -1

# ESPONENTE (**)
print(2 ** 3)       # 8 (2 elevato alla 3)
print(2 ** 0.5)     # 1.414... (radice quadrata)
print(9 ** 0.5)     # 3.0

# NEGAZIONE UNARIA (-)
print(-5)           # -5
print(--5)          # 5 (doppia negazione)


"""
───────────────────────────────────────────────────────────────────────────────
2.2.2 - PRECEDENZA DEGLI OPERATORI
───────────────────────────────────────────────────────────────────────────────

Dalla PRIORITÀ PIÙ ALTA alla più bassa:

    1. ()           Parentesi
    2. **           Esponente (associa a DESTRA!)
    3. +x, -x       Unario positivo/negativo
    4. *, /, //, %  Moltiplicazione, divisione, floor division, modulo
    5. +, -         Addizione, sottrazione

REGOLA: Stessa precedenza → associa a SINISTRA
ECCEZIONE: ** associa a DESTRA!
"""

# Esempi precedenza
print(2 + 3 * 4)        # 14 (non 20! Prima *, poi +)
print((2 + 3) * 4)      # 20 (parentesi forzano ordine)

# ** ASSOCIA A DESTRA! ⚠️ CRITICO PER ESAME!
print(2 ** 3 ** 2)      # 512 = 2^(3^2) = 2^9, NON (2^3)^2 = 64
print((2 ** 3) ** 2)    # 64 = (2^3)^2 = 8^2

# - unario ha MENO precedenza di **
print(-3 ** 2)          # -9 = -(3^2), NON (-3)^2 = 9
print((-3) ** 2)        # 9 = (-3)^2


"""
───────────────────────────────────────────────────────────────────────────────
2.2.3 - OPERATORI SU STRINGHE
───────────────────────────────────────────────────────────────────────────────
"""

# CONCATENAZIONE (+)
print("Hello" + " " + "World")  # "Hello World"

# REPLICA (*)
print("Ha" * 3)                 # "HaHaHa"
print(3 * "Ho")                 # "HoHoHo"

# ERRORI:
# print("Hello" + 5)            # TypeError! Non puoi sommare str e int
# print("Hello" - "H")          # TypeError! - non funziona con stringhe


"""
───────────────────────────────────────────────────────────────────────────────
2.2.4 - OPERATORI DI CONFRONTO
───────────────────────────────────────────────────────────────────────────────
"""

print(5 == 5)       # True (uguale)
print(5 != 3)       # True (diverso)
print(5 > 3)        # True (maggiore)
print(5 < 3)        # False (minore)
print(5 >= 5)       # True (maggiore o uguale)
print(5 <= 4)       # False (minore o uguale)

# CONFRONTI A CATENA (chained comparisons)
x = 5
print(1 < x < 10)       # True (equivale a: 1 < x and x < 10)
print(1 < 2 < 3 < 4)    # True


"""
───────────────────────────────────────────────────────────────────────────────
2.2.5 - OPERATORI DI ASSEGNAZIONE COMPOSTI
───────────────────────────────────────────────────────────────────────────────
"""

x = 10
x += 5      # x = x + 5 → x = 15
x -= 3      # x = x - 3 → x = 12
x *= 2      # x = x * 2 → x = 24
x /= 4      # x = x / 4 → x = 6.0 (diventa float!)
x //= 2     # x = x // 2 → x = 3.0
x %= 2      # x = x % 2 → x = 1.0
x **= 3     # x = x ** 3 → x = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2.3: VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  2.3 VARIABLES - Variabili e naming                                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

Una VARIABILE è un nome che fa riferimento a un valore in memoria.

───────────────────────────────────────────────────────────────────────────────
2.3.1 - REGOLE PER I NOMI DI VARIABILI
───────────────────────────────────────────────────────────────────────────────

✅ Può contenere: lettere, cifre, underscore (_)
✅ Deve INIZIARE con: lettera o underscore (MAI una cifra!)
✅ È CASE-SENSITIVE: age, Age, AGE sono tre variabili DIVERSE
❌ Non può essere una KEYWORD riservata

NOMI VALIDI:
    name, _name, name1, my_name, myName, __init__, _

NOMI NON VALIDI:
    1name       # inizia con cifra
    my-name     # contiene trattino
    my name     # contiene spazio
    for         # keyword riservata
"""

# Assegnazione base
x = 10
name = "Marco"
pi = 3.14159

# Python è DINAMICAMENTE TIPIZZATO
x = 10          # x è int
x = "hello"     # ora x è str - nessun errore!
x = 3.14        # ora x è float

# Assegnazione multipla
a = b = c = 0               # Tutte valgono 0
x, y, z = 1, 2, 3           # Unpacking

# SWAP elegante in Python
a, b = 10, 20
a, b = b, a     # Ora a=20, b=10 - senza variabile temp!


"""
───────────────────────────────────────────────────────────────────────────────
2.3.2 - KEYWORDS RISERVATE (Python 3)
───────────────────────────────────────────────────────────────────────────────

Queste parole NON possono essere usate come nomi di variabili:

False    await    else      import    pass
None     break    except    in        raise
True     class    finally   is        return
and      continue for       lambda    try
as       def      from      nonlocal  while
assert   del      global    not       with
async    elif     if        or        yield

Per vedere la lista completa in Python:
>>> import keyword
>>> print(keyword.kwlist)
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2.4: COMMENTS
# ══════════════════════════════════════════════════════════════════════════════
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  2.4 COMMENTS - Commenti nel codice                                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

I commenti sono ignorati dall'interprete.
Servono per documentare il codice.
"""

# Questo è un commento singola linea
x = 5  # Commento a fine riga

# Python NON ha commenti multi-linea ufficiali
# Puoi usare # su ogni riga:
# Questo è un commento
# su più righe
# usando # ripetuto

# Oppure stringhe triple (non assegnate) come convenzione:
"""
Questo non è tecnicamente un commento,
ma una stringa non assegnata.
L'interprete la ignora effettivamente.
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2.5: INPUT E TYPE CONVERSION
# ══════════════════════════════════════════════════════════════════════════════
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  2.5 input() E TYPE CONVERSION                                             ║
╚═══════════════════════════════════════════════════════════════════════════╝

───────────────────────────────────────────────────────────────────────────────
2.5.1 - LA FUNZIONE input()
───────────────────────────────────────────────────────────────────────────────

input() legge una riga dalla console e restituisce SEMPRE una STRINGA!
"""

# Sintassi base
# name = input()                      # Senza prompt
# name = input("Come ti chiami? ")    # Con prompt

# ⚠️ CRITICO: input() restituisce SEMPRE str!
# Se l'utente digita "42", il risultato è la STRINGA "42", non il numero 42


"""
───────────────────────────────────────────────────────────────────────────────
2.5.2 - TYPE CONVERSION (Type Casting)
───────────────────────────────────────────────────────────────────────────────

Funzioni per convertire tra tipi:
"""

# str → int
x = int("42")           # x = 42 (int)
# x = int("42.5")       # ⚠️ ValueError! Non converte str con punto!
# x = int("hello")      # ⚠️ ValueError!

# str → float
y = float("3.14")       # y = 3.14
y = float("42")         # y = 42.0 (funziona anche con interi)

# int/float → str
s = str(42)             # s = "42"
s = str(3.14)           # s = "3.14"

# int ↔ float
n = int(3.7)            # n = 3 ⚠️ TRONCA, non arrotonda!
n = int(-3.7)           # n = -3 ⚠️ TRONCA verso ZERO!
f = float(42)           # f = 42.0


"""
⚠️ DIFFERENZA int() su float vs floor division:

int(-3.7) = -3   (tronca verso ZERO)
-7 // 2   = -4   (arrotonda verso -INFINITO)
"""

# Conversione con input
# age = int(input("Età: "))           # Converte direttamente in int
# price = float(input("Prezzo: "))    # Converte in float


"""
───────────────────────────────────────────────────────────────────────────────
2.5.3 - ALTRE FUNZIONI BUILT-IN UTILI
───────────────────────────────────────────────────────────────────────────────
"""

# len() - lunghezza di sequenze
print(len("hello"))     # 5
print(len([1, 2, 3]))   # 3

# abs() - valore assoluto
print(abs(-5))          # 5
print(abs(3.14))        # 3.14

# round() - arrotondamento
print(round(3.7))       # 4
print(round(3.14159, 2)) # 3.14

# min(), max()
print(min(1, 5, 3))     # 1
print(max(1, 5, 3))     # 5

# pow() - potenza (equivalente a **)
print(pow(2, 3))        # 8


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2.6: FLOATING-POINT PRECISION
# ══════════════════════════════════════════════════════════════════════════════
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  2.6 PRECISIONE DEI FLOAT                                                  ║
╚═══════════════════════════════════════════════════════════════════════════╝

I float NON sono precisi! Sono rappresentati in binario (IEEE 754).
"""

print(0.1 + 0.2)            # 0.30000000000000004 ⚠️
print(0.1 + 0.2 == 0.3)     # False! ⚠️

# Per confronti, usa una tolleranza:
# abs(a - b) < 0.0001


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2.7: BITWISE OPERATORS
# ══════════════════════════════════════════════════════════════════════════════
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  2.7 BITWISE OPERATORS - Operatori bit a bit                               ║
╚═══════════════════════════════════════════════════════════════════════════╝

Operano sui singoli bit della rappresentazione binaria.

    & (AND)     - 1 se entrambi i bit sono 1
    | (OR)      - 1 se almeno un bit è 1
    ^ (XOR)     - 1 se i bit sono diversi
    ~ (NOT)     - inverte tutti i bit
    << (LEFT)   - shift a sinistra (moltiplica per 2^n)
    >> (RIGHT)  - shift a destra (divide per 2^n)
"""

# AND bit a bit
print(5 & 3)        # 1
# 5 = 101
# 3 = 011
# & = 001 = 1

# OR bit a bit
print(5 | 3)        # 7
# 5 = 101
# 3 = 011
# | = 111 = 7

# XOR bit a bit
print(5 ^ 3)        # 6
# 5 = 101
# 3 = 011
# ^ = 110 = 6

# NOT bit a bit (complemento)
print(~5)           # -6
# ~x = -(x+1)

# LEFT SHIFT (equivale a moltiplicare per 2^n)
print(5 << 1)       # 10 (5 * 2^1 = 10)
print(5 << 2)       # 20 (5 * 2^2 = 20)

# RIGHT SHIFT (equivale a dividere per 2^n)
print(8 >> 1)       # 4 (8 / 2^1 = 4)
print(8 >> 2)       # 2 (8 / 2^2 = 2)


# ══════════════════════════════════════════════════════════════════════════════
#                              QUIZ MODULE 2
# ══════════════════════════════════════════════════════════════════════════════
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                         QUIZ MODULE 2                                      ║
║                                                                            ║
║  ISTRUZIONI: Rispondi PRIMA di verificare!                                 ║
║  Target: 25/30 per considerarsi pronti                                     ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

QUIZ_MODULE_2 = """
═══════════════════════════════════════════════════════════════════════════════
Q1. print(0o17) stampa:
    A) 017    B) 0o17    C) 15    D) 17

Q2. print(0b1010) stampa:
    A) 1010   B) 0b1010  C) 10    D) 2

Q3. print(0xFF) stampa:
    A) FF     B) 0xFF    C) 255   D) 16

Q4. print(2e3) stampa:
    A) 2000   B) 2000.0  C) 2e3   D) 6

Q5. print(True + True + True) stampa:
    A) True   B) 3       C) TrueTrueTrue   D) Error

Q6. print(7 / 2) stampa:
    A) 3      B) 3.5     C) 3.0   D) Error

Q7. print(7 // 2) stampa:
    A) 3      B) 3.5     C) 3.0   D) 4

Q8. print(-7 // 2) stampa:
    A) -3     B) -4      C) -3.5  D) 3

Q9. print(-7 % 2) stampa:
    A) -1     B) 1       C) 0     D) -3

Q10. print(2 ** 3 ** 2) stampa:
     A) 64    B) 512     C) 12    D) Error

Q11. print(-3 ** 2) stampa:
     A) 9     B) -9      C) 6     D) Error

Q12. print((-3) ** 2) stampa:
     A) 9     B) -9      C) 6     D) Error

Q13. print(10 / 5) stampa:
     A) 2     B) 2.0     C) 2.00  D) Error

Q14. Quale nome variabile è VALIDO?
     A) 2name  B) my-var  C) _value  D) for

Q15. x = 10; x /= 4; print(x) stampa:
     A) 2     B) 2.5     C) 2.0   D) Error

Q16. a, b = 5, 10; a, b = b, a; print(a) stampa:
     A) 5     B) 10      C) Error D) (5, 10)

Q17. print(int(3.9)) stampa:
     A) 4     B) 3       C) 3.0   D) Error

Q18. print(int(-3.9)) stampa:
     A) -4    B) -3      C) -3.0  D) Error

Q19. Se input è "5": x = input(); print(x * 2) stampa:
     A) 10    B) 55      C) Error D) 52

Q20. print(int("3.14")) causa:
     A) 3     B) 3.14    C) ValueError    D) 3.0

Q21. print(type(4.)) stampa:
     A) int   B) float   C) Error D) double

Q22. print("Ha" * 3) stampa:
     A) HaHaHa  B) Ha3   C) Ha * 3   D) Error

Q23. print(5 == 5.0) stampa:
     A) True  B) False   C) Error D) None

Q24. print(1 < 2 < 3) stampa:
     A) True  B) False   C) Error D) 1

Q25. print(2 + 3 * 4) stampa:
     A) 20    B) 14      C) 24    D) 9

Q26. print(5 << 2) stampa:
     A) 10    B) 20      C) 7     D) 3

Q27. print(~5) stampa:
     A) -5    B) -6      C) 5     D) Error

Q28. x = 10; x %= 3; print(x) stampa:
     A) 3     B) 1       C) 0     D) 3.33

Q29. print(bool(0), bool("")) stampa:
     A) True True   B) False False   C) True False   D) False True

Q30. print(float("5") + int("5")) stampa:
     A) 10    B) 10.0    C) "55"  D) Error
"""

ANSWERS_MODULE_2 = """
═══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ MODULE 2
═══════════════════════════════════════════════════════════════════════════════

Q1:  C) 15 - ottale: 1×8 + 7 = 15
Q2:  C) 10 - binario: 8+0+2+0 = 10
Q3:  C) 255 - esadecimale: 15×16 + 15 = 255
Q4:  B) 2000.0 - notazione E dà sempre float
Q5:  B) 3 - True = 1, quindi 1+1+1 = 3
Q6:  B) 3.5 - true division SEMPRE float
Q7:  A) 3 - floor division con int dà int
Q8:  B) -4 - floor verso -infinito! ⚠️
Q9:  B) 1 - modulo ha segno del divisore ⚠️
Q10: B) 512 - ** associa a DESTRA: 2^(3^2) = 2^9 ⚠️
Q11: B) -9 - ** ha precedenza su -: -(3^2) ⚠️
Q12: A) 9 - parentesi: (-3)^2
Q13: B) 2.0 - / SEMPRE float!
Q14: C) _value
Q15: B) 2.5 - /= usa true division
Q16: B) 10 - swap elegante
Q17: B) 3 - int() TRONCA, non arrotonda
Q18: B) -3 - int() tronca verso ZERO ⚠️
Q19: B) 55 - input è str, * replica!
Q20: C) ValueError - int() non accetta "."
Q21: B) float - 4. = 4.0
Q22: A) HaHaHa
Q23: A) True - 5 == 5.0 per valore
Q24: A) True - chained comparison
Q25: B) 14 - * prima di +
Q26: B) 20 - 5 × 2^2 = 5 × 4
Q27: B) -6 - ~x = -(x+1)
Q28: B) 1 - 10 % 3 = 1
Q29: B) False False - 0 e "" sono falsy
Q30: B) 10.0 - float + int = float

PUNTEGGIO:
27-30: Eccellente! Pronto per Module 3
24-26: Buono, rivedi gli errori
20-23: Discreto, studia ancora le trappole (⚠️)
<20: Riguarda la teoria con attenzione
"""


# ══════════════════════════════════════════════════════════════════════════════
#                              LABS MODULE 2
# ══════════════════════════════════════════════════════════════════════════════

LABS_MODULE_2 = """
═══════════════════════════════════════════════════════════════════════════════
LAB 2.1 - Conversione temperatura
═══════════════════════════════════════════════════════════════════════════════
Scrivi un programma che converta Fahrenheit in Celsius.
Formula: C = (F - 32) × 5/9

Input: 98.6
Output atteso: 37.0

# Soluzione:
# f = float(input("Fahrenheit: "))
# c = (f - 32) * 5 / 9
# print("Celsius:", c)


═══════════════════════════════════════════════════════════════════════════════
LAB 2.2 - Secondi in ore/minuti/secondi
═══════════════════════════════════════════════════════════════════════════════
L'utente inserisce un numero di secondi.
Converti in ore, minuti, secondi.

Input: 3661
Output atteso: 1 ore, 1 minuti, 1 secondi

# Soluzione:
# total = int(input("Secondi: "))
# hours = total // 3600
# minutes = (total % 3600) // 60
# seconds = total % 60
# print(hours, "ore,", minutes, "minuti,", seconds, "secondi")


═══════════════════════════════════════════════════════════════════════════════
LAB 2.3 - Ultimo digit
═══════════════════════════════════════════════════════════════════════════════
L'utente inserisce un numero.
Stampa l'ultima cifra.

Input: 1234
Output atteso: 4

# Soluzione:
# n = int(input("Numero: "))
# print("Ultima cifra:", n % 10)


═══════════════════════════════════════════════════════════════════════════════
LAB 2.4 - Swap variabili
═══════════════════════════════════════════════════════════════════════════════
Date due variabili a=5 e b=10, scambiale SENZA usare una terza variabile.

# Soluzione:
# a, b = 5, 10
# a, b = b, a
# print(a, b)  # 10 5


═══════════════════════════════════════════════════════════════════════════════
LAB 2.5 - Calcolo ipotenusa
═══════════════════════════════════════════════════════════════════════════════
Dati i cateti a e b, calcola l'ipotenusa.
Formula: c = √(a² + b²)

Input: a=3, b=4
Output atteso: 5.0

# Soluzione:
# a = float(input("Cateto a: "))
# b = float(input("Cateto b: "))
# c = (a**2 + b**2) ** 0.5
# print("Ipotenusa:", c)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                         CHECKLIST MODULO 2
# ══════════════════════════════════════════════════════════════════════════════
"""
Prima di procedere al Modulo 3, assicurati di:

[ ] Saper riconoscere literal int in binario (0b), ottale (0o), esadecimale (0x)
[ ] Sapere che print() stampa SEMPRE in decimale
[ ] Ricordare che / restituisce SEMPRE float, // restituisce int (se operandi int)
[ ] Sapere che // arrotonda verso -infinito (non verso zero!)
[ ] Sapere che % ha il segno del divisore
[ ] Ricordare che ** associa a DESTRA
[ ] Ricordare che - unario ha MENO precedenza di **
[ ] Sapere che input() restituisce SEMPRE str
[ ] Sapere che int() TRONCA verso zero (diverso da //)
[ ] Conoscere gli operatori bitwise base (&, |, ^, ~, <<, >>)
"""


if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("PE1 MODULE 2 - Quiz e Labs")
    print("═" * 70)
    print("""
    Per vedere i quiz: print(QUIZ_MODULE_2)
    Per vedere le risposte: print(ANSWERS_MODULE_2)
    Per vedere i labs: print(LABS_MODULE_2)
    """)
