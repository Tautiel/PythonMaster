"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PYTHON ESSENTIALS 1 - MODULE 2                            ║
║             Data Types, Variables, Operators, Basic I/O                      ║
║                                                                              ║
║                     Allineato al Syllabus PCEP-30-02                         ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCEP-30-02 Exam Block 1: Computer Programming and Python Fundamentals (18%)
Questo modulo copre la MAGGIOR PARTE del Block 1

STRUTTURA MODULO:
├── Section 2.1: The print() function
├── Section 2.2: Python Literals
├── Section 2.3: Operators
├── Section 2.4: Variables
├── Section 2.5: Comments
├── Section 2.6: The input() function
├── Labs (10 esercizi pratici)
└── Module 2 Quiz (30 domande)

TEMPO STIMATO: 6-8 ore

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.1: THE PRINT() FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                         2.1 TEORIA: print()                                  │
└──────────────────────────────────────────────────────────────────────────────┘

LA FUNZIONE print():
────────────────────
print() è una FUNZIONE BUILT-IN che stampa output sulla console.

Sintassi base:
    print(value1, value2, ..., sep=' ', end='\\n')

COMPONENTI:
───────────
1. ARGOMENTI POSIZIONALI: I valori da stampare (separati da virgola)
2. sep: Separatore tra i valori (default: spazio ' ')
3. end: Cosa stampare alla fine (default: newline '\\n')


ESEMPI FONDAMENTALI:
────────────────────
"""

# Stampa semplice
print("Hello, World!")  # Output: Hello, World!

# Più argomenti (separati da spazio di default)
print("Hello", "World")  # Output: Hello World

# Argomento sep (separator)
print("a", "b", "c", sep="-")  # Output: a-b-c
print("a", "b", "c", sep="")   # Output: abc
print("a", "b", "c", sep="***")  # Output: a***b***c

# Argomento end (cosa mettere alla fine)
print("Hello", end=" ")
print("World")  # Output: Hello World (sulla stessa riga)

print("A", end="")
print("B", end="")
print("C")  # Output: ABC

# Combinazione sep e end
print("1", "2", "3", sep="-", end="!\n")  # Output: 1-2-3!


"""
ESCAPE SEQUENCES:
─────────────────
Caratteri speciali preceduti da backslash \\

\\n  - Newline (vai a capo)
\\t  - Tab (tabulazione)
\\\\  - Backslash letterale
\\'  - Apostrofo/quote singolo
\\"  - Quote doppio

ESEMPI:
"""
print("Riga 1\nRiga 2")     # Newline
print("Col1\tCol2\tCol3")   # Tab
print("C:\\Users\\Marco")    # Backslash letterale
print('It\'s Python')        # Quote dentro quote
print("Disse \"Ciao\"")      # Double quote dentro double quote


"""
PRINT SENZA ARGOMENTI:
──────────────────────
print() senza argomenti stampa una riga vuota (solo newline)
"""
print()  # Stampa riga vuota


"""
STRINGHE RAW (r""):
───────────────────
Il prefisso r ignora le escape sequences
"""
print(r"C:\new\folder")  # Output: C:\new\folder (non interpreta \n)


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 2.1 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_2_1 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 2.1.1
══════════════════════════════════════════════════════════════════════════════
Cosa stampa questo codice?

print("a", "b", "c")

A) abc
B) a b c
C) a, b, c
D) "a" "b" "c"

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.1.2
══════════════════════════════════════════════════════════════════════════════
Cosa stampa questo codice?

print("a", "b", "c", sep="")

A) abc
B) a b c
C) a""b""c
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.1.3
══════════════════════════════════════════════════════════════════════════════
Cosa stampa questo codice?

print("X", end="")
print("Y", end="")
print("Z")

A) X
   Y
   Z
B) XYZ
C) X Y Z
D) X""Y""Z

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.1.4
══════════════════════════════════════════════════════════════════════════════
Cosa stampa questo codice?

print("1", "2", "3", sep="-", end="!")
print("done")

A) 1-2-3!
   done
B) 1-2-3!done
C) 1-2-3-!done
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.1.5
══════════════════════════════════════════════════════════════════════════════
Cosa stampa questo codice?

print("Hello\\nWorld")

A) Hello\\nWorld
B) HellonWorld
C) Hello
   World
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.1.6
══════════════════════════════════════════════════════════════════════════════
Cosa stampa questo codice?

print("A\\tB\\tC")

A) A\\tB\\tC
B) AtBtC
C) A	B	C (con tab)
D) ABC

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.1.7
══════════════════════════════════════════════════════════════════════════════
Qual è il valore DEFAULT di sep in print()?

A) Nessuno (None)
B) Virgola ","
C) Spazio " "
D) Niente ""

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.1.8
══════════════════════════════════════════════════════════════════════════════
Qual è il valore DEFAULT di end in print()?

A) Nessuno (None)
B) Spazio " "
C) Newline "\\n"
D) Niente ""

Tua risposta: ___
"""


RISPOSTE_2_1 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 2.1
══════════════════════════════════════════════════════════════════════════════

2.1.1: B) a b c
       Default sep è spazio, quindi gli argomenti sono separati da spazio.

2.1.2: A) abc
       sep="" significa nessun separatore tra gli argomenti.

2.1.3: B) XYZ
       end="" significa che print non va a capo, tutto sulla stessa riga.

2.1.4: B) 1-2-3!done
       Il primo print termina con "!" invece di newline,
       quindi "done" continua sulla stessa riga.

2.1.5: C) Hello
          World
       \\n è l'escape sequence per newline (vai a capo).

2.1.6: C) A	B	C (con tab)
       \\t è l'escape sequence per tabulazione.

2.1.7: C) Spazio " "
       Di default, print separa gli argomenti con uno spazio.

2.1.8: C) Newline "\\n"
       Di default, print termina con un newline (vai a capo).
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.2: PYTHON LITERALS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                         2.2 TEORIA: LITERALS                                 │
└──────────────────────────────────────────────────────────────────────────────┘

Un LITERAL è un valore scritto direttamente nel codice.
Ad esempio: 42, 3.14, "hello", True sono tutti literals.


TIPI DI LITERALS IN PYTHON:
───────────────────────────

1. INTEGERS (int) - Numeri interi
───────────────────────────────────
"""
# Decimale (base 10) - il modo normale
a = 42
b = -17
c = 0

# Binario (base 2) - prefisso 0b o 0B
binary = 0b1010      # = 10 in decimale
binary2 = 0B1111     # = 15 in decimale

# Ottale (base 8) - prefisso 0o o 0O
octal = 0o17         # = 15 in decimale
octal2 = 0O777       # = 511 in decimale

# Esadecimale (base 16) - prefisso 0x o 0X
hexa = 0xFF          # = 255 in decimale
hexa2 = 0x1A         # = 26 in decimale

# Underscore per leggibilità (Python 3.6+)
big_number = 1_000_000  # = 1000000

print(0b1010)   # Output: 10
print(0o17)     # Output: 15
print(0xFF)     # Output: 255


"""
2. FLOATS (float) - Numeri decimali
───────────────────────────────────
"""
# Notazione normale
pi = 3.14159
negative = -2.5
zero_point = 0.0

# ATTENZIONE: Il punto è OBBLIGATORIO per i float
f1 = 4.0    # float
f2 = 4.     # float (equivalente a 4.0)
f3 = .5     # float (equivalente a 0.5)

# Notazione scientifica (E = *10^)
light_speed = 3e8       # 3 * 10^8 = 300000000.0
small = 1e-4            # 1 * 10^-4 = 0.0001
also_small = 6.62e-34   # Costante di Planck

print(3e8)    # Output: 300000000.0
print(1e-2)   # Output: 0.01
print(2.5e3)  # Output: 2500.0


"""
3. STRINGS (str) - Testo
────────────────────────
"""
# Single quotes
s1 = 'Hello'

# Double quotes (equivalenti)
s2 = "Hello"

# Stringhe con quote all'interno
s3 = "It's Python"      # Double fuori, single dentro
s4 = 'Say "Hello"'      # Single fuori, double dentro
s5 = "It\'s Python"     # Escape con backslash

# Stringhe multilinea con triple quotes
s6 = """Questa è una
stringa su più
righe"""

s7 = '''Anche questo
funziona'''

# Stringhe vuote
empty1 = ""
empty2 = ''


"""
4. BOOLEANS (bool) - Valori logici
──────────────────────────────────
"""
# SOLO due valori possibili (CASE-SENSITIVE!)
true_val = True     # Nota la T maiuscola
false_val = False   # Nota la F maiuscola

# true, TRUE, false, FALSE sono ERRORI!
# true = True  # NameError: name 'true' is not defined

# I booleani sono anche numeri!
print(True + True)   # Output: 2 (True = 1)
print(False + 1)     # Output: 1 (False = 0)
print(True * 10)     # Output: 10


"""
5. NONE - L'assenza di valore
─────────────────────────────
"""
nothing = None

# None non è 0, non è "", non è False
# È l'assenza di qualsiasi valore

print(None)          # Output: None
print(type(None))    # Output: <class 'NoneType'>


"""
FUNZIONE type():
────────────────
Restituisce il tipo di un oggetto
"""
print(type(42))        # <class 'int'>
print(type(3.14))      # <class 'float'>
print(type("hello"))   # <class 'str'>
print(type(True))      # <class 'bool'>
print(type(None))      # <class 'NoneType'>


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 2.2 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_2_2 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.1 - CRITICO PER L'ESAME!
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(0o17)?

A) 017
B) 0o17
C) 15
D) 17

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.2 - CRITICO PER L'ESAME!
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(0b1010)?

A) 1010
B) 0b1010
C) 10
D) 2

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.3 - CRITICO PER L'ESAME!
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(0xFF)?

A) FF
B) 0xFF
C) 255
D) 16

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.4
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(1e-2)?

A) 1e-2
B) 0.01
C) -2
D) 100

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.5
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(2e3)?

A) 2000
B) 2000.0
C) 2e3
D) 6

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.6
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(True + True + False)?

A) True
B) 2
C) TrueTrueFalse
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.7
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(type(4.0))?

A) <class 'int'>
B) <class 'float'>
C) <class 'double'>
D) <class 'number'>

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.8
══════════════════════════════════════════════════════════════════════════════
Quale di questi è un literal VALIDO in Python?

A) true
B) TRUE
C) True
D) true

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.9
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(.5 + .5)?

A) Error
B) 1
C) 1.0
D) .5.5

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.2.10
══════════════════════════════════════════════════════════════════════════════
Quale prefix indica un numero BINARIO?

A) 0x
B) 0o
C) 0b
D) 0d

Tua risposta: ___
"""


RISPOSTE_2_2 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 2.2
══════════════════════════════════════════════════════════════════════════════

2.2.1: C) 15
       0o17 è ottale. 1*8 + 7 = 15 in decimale.
       Python stampa SEMPRE in decimale!

2.2.2: C) 10
       0b1010 è binario. 1*8 + 0*4 + 1*2 + 0*1 = 10 in decimale.

2.2.3: C) 255
       0xFF è esadecimale. 15*16 + 15 = 255 in decimale.

2.2.4: B) 0.01
       1e-2 = 1 * 10^(-2) = 1/100 = 0.01

2.2.5: B) 2000.0
       2e3 = 2 * 10^3 = 2000.0 (è un float!)

2.2.6: B) 2
       True = 1, False = 0. Quindi 1 + 1 + 0 = 2

2.2.7: B) <class 'float'>
       4.0 ha il punto decimale, quindi è float.

2.2.8: C) True
       I booleani sono CASE-SENSITIVE: True e False (maiuscola).

2.2.9: C) 1.0
       .5 = 0.5 (float). 0.5 + 0.5 = 1.0 (ancora float!)

2.2.10: C) 0b
        0b = binario, 0o = ottale, 0x = esadecimale
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.3: OPERATORS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                         2.3 TEORIA: OPERATORI                                │
└──────────────────────────────────────────────────────────────────────────────┘

OPERATORI ARITMETICI:
─────────────────────
"""
# Addizione
print(5 + 3)      # 8

# Sottrazione
print(5 - 3)      # 2

# Moltiplicazione
print(5 * 3)      # 15

# Divisione (TRUE DIVISION - sempre float in Python 3!)
print(5 / 2)      # 2.5
print(6 / 2)      # 3.0 (sempre float!)

# Floor Division (INTEGER DIVISION - arrotonda verso -infinito)
print(5 // 2)     # 2
print(-5 // 2)    # -3 (NON -2! Arrotonda verso -infinito)

# Modulo (resto della divisione)
print(5 % 2)      # 1
print(-5 % 2)     # 1 (segno del DIVISORE in Python!)

# Esponente
print(2 ** 3)     # 8
print(2 ** 0.5)   # 1.414... (radice quadrata)

# Negazione unaria
print(-5)         # -5
print(--5)        # 5 (doppia negazione)


"""
PRECEDENZA DEGLI OPERATORI (dalla più alta alla più bassa):
───────────────────────────────────────────────────────────

1. ()         Parentesi
2. **         Esponente (associa a DESTRA!)
3. +x, -x     Unario positivo/negativo
4. *, /, //, % Moltiplicazione, divisioni, modulo
5. +, -       Addizione, sottrazione

REGOLA FONDAMENTALE:
- Stessa precedenza: associa a SINISTRA (da sinistra a destra)
- ECCEZIONE: ** associa a DESTRA!
"""

# Esempi precedenza
print(2 + 3 * 4)      # 14 (non 20! * prima di +)
print((2 + 3) * 4)    # 20 (parentesi forzano ordine)
print(2 ** 3 ** 2)    # 512 (non 64! ** associa a DESTRA: 2^(3^2) = 2^9)
print((2 ** 3) ** 2)  # 64 (parentesi: (2^3)^2 = 8^2)


"""
DIVISIONE CON NUMERI NEGATIVI - ATTENZIONE!
───────────────────────────────────────────
Python usa "floor division" che arrotonda verso -INFINITO (non verso zero!)
"""
print(7 // 2)       # 3
print(-7 // 2)      # -4 (NON -3!)
print(7 // -2)      # -4 (NON -3!)
print(-7 // -2)     # 3

# Il modulo ha il segno del DIVISORE
print(7 % 2)        # 1
print(-7 % 2)       # 1 (NON -1!)
print(7 % -2)       # -1
print(-7 % -2)      # -1


"""
OPERATORI DI CONFRONTO:
───────────────────────
"""
print(5 == 5)    # True (uguaglianza)
print(5 != 3)    # True (diverso)
print(5 > 3)     # True (maggiore)
print(5 < 3)     # False (minore)
print(5 >= 5)    # True (maggiore o uguale)
print(5 <= 5)    # True (minore o uguale)

# Confronti a catena (chained comparisons)
x = 5
print(1 < x < 10)       # True (equivale a: 1 < x and x < 10)
print(1 < x < 3)        # False
print(1 < 2 < 3 < 4)    # True


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 2.3 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_2_3 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.1 - FONDAMENTALE!
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(5 / 2)?

A) 2
B) 2.5
C) 2.0
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.2 - FONDAMENTALE!
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(5 // 2)?

A) 2
B) 2.5
C) 2.0
D) 3

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.3 - TRAPPOLA ESAME!
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(-7 // 2)?

A) -3
B) -3.5
C) -4
D) 3

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.4 - TRAPPOLA ESAME!
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(-7 % 2)?

A) -1
B) 1
C) -3
D) 0

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.5 - TRAPPOLA ESAME!
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(2 ** 3 ** 2)?

A) 64
B) 512
C) 12
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.6 - TRAPPOLA ESAME!
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(-3 ** 2)?

A) 9
B) -9
C) 6
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.7
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print((-3) ** 2)?

A) 9
B) -9
C) 6
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.8
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(2 + 3 * 4)?

A) 20
B) 14
C) 24
D) 9

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.9
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(10 - 5 - 2)?

A) 3
B) 7
C) -3
D) 13

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.10
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(6 / 3)?

A) 2
B) 2.0
C) 2.00
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.11
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(1 < 2 < 3)?

A) True
B) False
C) Error
D) 1

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.3.12
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(5.0 // 2)?

A) 2
B) 2.0
C) 2.5
D) Error

Tua risposta: ___
"""


RISPOSTE_2_3 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 2.3
══════════════════════════════════════════════════════════════════════════════

2.3.1: B) 2.5
       / è "true division" in Python 3, restituisce SEMPRE float.

2.3.2: A) 2
       // è "floor division", restituisce int se entrambi int.

2.3.3: C) -4
       // arrotonda verso -INFINITO, non verso zero!
       -7 / 2 = -3.5, floor(-3.5) = -4

2.3.4: B) 1
       In Python, il modulo ha il segno del DIVISORE (non del dividendo).
       -7 = 2 * (-4) + 1, quindi -7 % 2 = 1

2.3.5: B) 512
       ** associa a DESTRA! Quindi: 2 ** (3 ** 2) = 2 ** 9 = 512

2.3.6: B) -9
       ** ha precedenza su - unario!
       -3 ** 2 = -(3 ** 2) = -(9) = -9

2.3.7: A) 9
       Con parentesi: (-3) ** 2 = 9

2.3.8: B) 14
       * ha precedenza su +: 2 + (3 * 4) = 2 + 12 = 14

2.3.9: A) 3
       - associa a sinistra: (10 - 5) - 2 = 5 - 2 = 3

2.3.10: B) 2.0
        / restituisce SEMPRE float in Python 3, anche se divisibile esatto!

2.3.11: A) True
        Chained comparison: (1 < 2) and (2 < 3) = True and True = True

2.3.12: B) 2.0
        Se uno degli operandi è float, // restituisce float.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.4: VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                         2.4 TEORIA: VARIABILI                                │
└──────────────────────────────────────────────────────────────────────────────┘

Una VARIABILE è un nome che riferisce a un valore in memoria.

REGOLE PER I NOMI (IDENTIFIERS):
────────────────────────────────
✅ Può contenere: lettere, cifre, underscore
✅ Deve iniziare con: lettera o underscore (NON cifra)
✅ È case-sensitive: 'age', 'Age', 'AGE' sono DIVERSE
❌ Non può essere una keyword riservata

NOMI VALIDI:
    name, _name, name1, my_name, MyName, __init__

NOMI NON VALIDI:
    1name    (inizia con cifra)
    my-name  (contiene trattino)
    my name  (contiene spazio)
    for      (keyword riservata)
"""

# Assegnazione
x = 10
name = "Marco"
pi = 3.14159

# Python è DINAMICAMENTE TIPIZZATO
x = 10        # x è int
x = "hello"   # ora x è str (nessun errore!)
x = 3.14      # ora x è float

# Assegnazione multipla
a = b = c = 0           # Tutte valgono 0
x, y, z = 1, 2, 3       # Unpacking

# Swap di variabili (elegante in Python!)
a, b = 10, 20
a, b = b, a    # Ora a=20, b=10


"""
OPERATORI DI ASSEGNAZIONE COMPOSTI:
───────────────────────────────────
"""
x = 10
x += 5    # x = x + 5  → x = 15
x -= 3    # x = x - 3  → x = 12
x *= 2    # x = x * 2  → x = 24
x /= 4    # x = x / 4  → x = 6.0 (diventa float!)
x //= 2   # x = x // 2 → x = 3.0
x %= 2    # x = x % 2  → x = 1.0
x **= 3   # x = x ** 3 → x = 1.0


"""
KEYWORDS RISERVATE (Python 3):
──────────────────────────────
False, None, True, and, as, assert, async, await, break, class,
continue, def, del, elif, else, except, finally, for, from, global,
if, import, in, is, lambda, nonlocal, not, or, pass, raise, return,
try, while, with, yield
"""


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 2.4 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_2_4 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 2.4.1
══════════════════════════════════════════════════════════════════════════════
Quale di questi è un nome di variabile VALIDO?

A) 2name
B) my-var
C) _private
D) class

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.4.2
══════════════════════════════════════════════════════════════════════════════
Cosa stampa questo codice?

x = 5
x += 3
x *= 2
print(x)

A) 10
B) 13
C) 16
D) 11

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.4.3
══════════════════════════════════════════════════════════════════════════════
Cosa stampa questo codice?

a = b = 5
a += 1
print(b)

A) 5
B) 6
C) Error
D) None

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.4.4
══════════════════════════════════════════════════════════════════════════════
Cosa stampa questo codice?

x, y = 10, 20
x, y = y, x
print(x, y)

A) 10 20
B) 20 10
C) Error
D) 20 20

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.4.5
══════════════════════════════════════════════════════════════════════════════
Cosa stampa questo codice?

x = 10
x /= 5
print(x, type(x))

A) 2 <class 'int'>
B) 2.0 <class 'float'>
C) 2 <class 'float'>
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.4.6
══════════════════════════════════════════════════════════════════════════════
Le variabili 'Name' e 'name' sono:

A) La stessa variabile
B) Due variabili diverse
C) Un errore di sintassi
D) Dipende dal contesto

Tua risposta: ___
"""


RISPOSTE_2_4 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 2.4
══════════════════════════════════════════════════════════════════════════════

2.4.1: C) _private
       - 2name: inizia con cifra ❌
       - my-var: contiene trattino ❌
       - _private: valido ✅
       - class: keyword riservata ❌

2.4.2: C) 16
       x = 5 → x += 3 → x = 8 → x *= 2 → x = 16

2.4.3: A) 5
       Interi sono IMMUTABILI. a += 1 crea un nuovo oggetto per a,
       ma b punta ancora al 5 originale.

2.4.4: B) 20 10
       Questo è lo swap idiomatico in Python.
       Prima valuta il lato destro (y, x) = (20, 10),
       poi assegna a x e y.

2.4.5: B) 2.0 <class 'float'>
       /= usa true division, che restituisce sempre float!

2.4.6: B) Due variabili diverse
       Python è case-sensitive.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.5: COMMENTS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                         2.5 TEORIA: COMMENTI                                 │
└──────────────────────────────────────────────────────────────────────────────┘

I commenti sono ignorati dall'interprete.
Servono per documentare il codice.

COMMENTO SINGOLA LINEA: #
─────────────────────────
"""
# Questo è un commento
x = 5  # Questo è un commento a fine riga

"""
COMMENTI MULTI-LINEA:
─────────────────────
Python NON ha un vero commento multi-linea.
Si usano stringhe triple (che non vengono assegnate):
"""

"""
Questo è un commento
su più righe usando
triple double quotes
"""

'''
Anche questo funziona
con triple single quotes
'''


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 2.6: THE INPUT() FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                         2.6 TEORIA: input()                                  │
└──────────────────────────────────────────────────────────────────────────────┘

input() legge una riga dalla console e restituisce una STRINGA.

IMPORTANTE: input() restituisce SEMPRE una stringa!
"""

# Sintassi base
# name = input()                    # Aspetta input senza prompt
# name = input("Come ti chiami? ") # Con prompt

# ATTENZIONE: Il risultato è SEMPRE stringa!
# age = input("Quanti anni hai? ")  # Se digiti 25, age = "25" (stringa!)

# Per avere un numero, devi CONVERTIRE:
# age = int(input("Quanti anni hai? "))    # Converti in int
# price = float(input("Prezzo? "))         # Converti in float


"""
CONVERSIONI DI TIPO (Type Casting):
───────────────────────────────────
"""
# str → int
x = int("42")       # x = 42
# x = int("42.5")   # ERROR! Non può convertire float string in int

# str → float
y = float("3.14")   # y = 3.14
y = float("42")     # y = 42.0 (funziona anche con int string)

# int/float → str
s = str(42)         # s = "42"
s = str(3.14)       # s = "3.14"

# int ↔ float
n = int(3.7)        # n = 3 (TRONCA, non arrotonda!)
n = int(-3.7)       # n = -3 (TRONCA verso zero!)
f = float(42)       # f = 42.0


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 2.6 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_2_6 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 2.6.1
══════════════════════════════════════════════════════════════════════════════
Se l'utente digita "5" e "3", cosa stampa?

x = input()
y = input()
print(x + y)

A) 8
B) 53
C) 5 3
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.6.2
══════════════════════════════════════════════════════════════════════════════
Se l'utente digita "5" e "3", cosa stampa?

x = int(input())
y = int(input())
print(x + y)

A) 8
B) 53
C) 5 3
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.6.3
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(int(3.9))?

A) 4
B) 3
C) 3.0
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.6.4
══════════════════════════════════════════════════════════════════════════════
Cosa stampa print(int(-3.9))?

A) -4
B) -3
C) -3.0
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.6.5
══════════════════════════════════════════════════════════════════════════════
Quale riga causa un ERRORE?

A) int("42")
B) int("42.5")
C) float("42")
D) float("42.5")

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 2.6.6
══════════════════════════════════════════════════════════════════════════════
Cosa restituisce type(input()) se l'utente digita 123?

A) <class 'int'>
B) <class 'str'>
C) <class 'float'>
D) Dipende dall'input

Tua risposta: ___
"""


RISPOSTE_2_6 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 2.6
══════════════════════════════════════════════════════════════════════════════

2.6.1: B) 53
       input() restituisce SEMPRE stringhe.
       "5" + "3" = "53" (concatenazione stringhe)

2.6.2: A) 8
       int() converte le stringhe in interi.
       5 + 3 = 8

2.6.3: B) 3
       int() TRONCA verso zero, non arrotonda.
       int(3.9) = 3

2.6.4: B) -3
       int() TRONCA verso zero anche per negativi!
       int(-3.9) = -3 (NON -4!)

2.6.5: B) int("42.5")
       int() non può convertire una stringa con punto decimale.
       Devi prima usare float(): int(float("42.5"))

2.6.6: B) <class 'str'>
       input() restituisce SEMPRE una stringa,
       indipendentemente da cosa digita l'utente!
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 2 - LABS (ESERCIZI PRATICI)
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                         MODULE 2 - LABS                                      │
│                     Esercizi pratici da completare                           │
└──────────────────────────────────────────────────────────────────────────────┘
"""

LABS_MODULE_2 = """
══════════════════════════════════════════════════════════════════════════════
LAB 2.1: print() formattato
══════════════════════════════════════════════════════════════════════════════
Scrivi un programma che stampi:
    1***2***3
    ABC
(usando solo UNA print per riga, con sep e end appropriati)

# Il tuo codice:



══════════════════════════════════════════════════════════════════════════════
LAB 2.2: Conversione basi numeriche
══════════════════════════════════════════════════════════════════════════════
Assegna a tre variabili il numero 255 in:
- Decimale
- Binario
- Esadecimale

Poi stampa tutte e tre (dovrebbero dare lo stesso risultato).

# Il tuo codice:



══════════════════════════════════════════════════════════════════════════════
LAB 2.3: Operazioni con divisione
══════════════════════════════════════════════════════════════════════════════
Dato x = 17 e y = 5, stampa:
- Il risultato della divisione vera
- Il risultato della divisione intera
- Il resto della divisione

# Il tuo codice:



══════════════════════════════════════════════════════════════════════════════
LAB 2.4: Swap senza variabile temporanea
══════════════════════════════════════════════════════════════════════════════
Scrivi codice che scambia i valori di a e b SENZA usare una terza variabile.
a = 10
b = 20
# Dopo lo swap: a = 20, b = 10

# Il tuo codice:



══════════════════════════════════════════════════════════════════════════════
LAB 2.5: Calcolo con input
══════════════════════════════════════════════════════════════════════════════
Chiedi all'utente due numeri e stampa:
- La somma
- Il prodotto
- La media

# Il tuo codice:



══════════════════════════════════════════════════════════════════════════════
LAB 2.6: Secondi → Ore/Minuti/Secondi
══════════════════════════════════════════════════════════════════════════════
L'utente inserisce un numero di secondi totali.
Converti in ore, minuti e secondi.
Esempio: 3661 secondi = 1 ora, 1 minuto, 1 secondo

# Il tuo codice:



══════════════════════════════════════════════════════════════════════════════
LAB 2.7: Espressione complessa
══════════════════════════════════════════════════════════════════════════════
Calcola e stampa il risultato di:
    (3 ** 2 + 4 ** 2) ** 0.5

Cosa rappresenta matematicamente?

# Il tuo codice:



══════════════════════════════════════════════════════════════════════════════
LAB 2.8: Escape sequences
══════════════════════════════════════════════════════════════════════════════
Stampa questo output ESATTO usando escape sequences:

    Riga 1	Col 1	Col 2
    Riga 2	"test"	'test'
    Path: C:\\Users\\Marco

# Il tuo codice:



══════════════════════════════════════════════════════════════════════════════
LAB 2.9: Operatori composti
══════════════════════════════════════════════════════════════════════════════
Partendo da x = 100, applica in sequenza:
1. Aggiungi 50
2. Dividi per 3 (floor division)
3. Eleva al quadrato
4. Sottrai 1

Usa SOLO operatori composti (+=, //=, **=, -=).
Qual è il risultato finale?

# Il tuo codice:



══════════════════════════════════════════════════════════════════════════════
LAB 2.10: Ultimo digit
══════════════════════════════════════════════════════════════════════════════
L'utente inserisce un numero intero.
Stampa l'ultima cifra del numero.
(Suggerimento: usa l'operatore modulo)

# Il tuo codice:


"""


LABS_SOLUTIONS = """
══════════════════════════════════════════════════════════════════════════════
SOLUZIONI LABS MODULE 2
══════════════════════════════════════════════════════════════════════════════

LAB 2.1:
print(1, 2, 3, sep="***")
print("A", "B", "C", sep="")

LAB 2.2:
decimal = 255
binary = 0b11111111
hexa = 0xFF
print(decimal, binary, hexa)  # 255 255 255

LAB 2.3:
x, y = 17, 5
print(x / y)   # 3.4
print(x // y)  # 3
print(x % y)   # 2

LAB 2.4:
a = 10
b = 20
a, b = b, a
print(a, b)  # 20 10

LAB 2.5:
x = float(input("Primo numero: "))
y = float(input("Secondo numero: "))
print("Somma:", x + y)
print("Prodotto:", x * y)
print("Media:", (x + y) / 2)

LAB 2.6:
total_seconds = int(input("Secondi: "))
hours = total_seconds // 3600
minutes = (total_seconds % 3600) // 60
seconds = total_seconds % 60
print(hours, "ore,", minutes, "minuti,", seconds, "secondi")

LAB 2.7:
print((3 ** 2 + 4 ** 2) ** 0.5)  # 5.0
# È il teorema di Pitagora! sqrt(9 + 16) = sqrt(25) = 5

LAB 2.8:
print("Riga 1\\tCol 1\\tCol 2")
print("Riga 2\\t\\"test\\"\\t'test'")
print("Path: C:\\\\Users\\\\Marco")

LAB 2.9:
x = 100
x += 50   # 150
x //= 3   # 50
x **= 2   # 2500
x -= 1    # 2499
print(x)

LAB 2.10:
n = int(input("Numero: "))
print("Ultima cifra:", n % 10)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 2 - TEST FINALE
# ══════════════════════════════════════════════════════════════════════════════

MODULE_2_TEST = """
══════════════════════════════════════════════════════════════════════════════
                         MODULE 2 - TEST FINALE
                    30 domande - Target: 70% (21/30)
                     Tempo consigliato: 30 minuti
══════════════════════════════════════════════════════════════════════════════

Q1. print("A", "B", "C", sep="-", end="!")  stampa:
    A) A-B-C!    B) A B C!    C) ABC!    D) A-B-C!\\n

Q2. print(0o10) stampa:
    A) 10    B) 8    C) 0o10    D) Error

Q3. print(0b1111) stampa:
    A) 15    B) 1111    C) 0b1111    D) 4

Q4. print(0xA) stampa:
    A) A    B) 10    C) 0xA    D) Error

Q5. print(2e3) stampa:
    A) 2000    B) 2000.0    C) 2e3    D) 6

Q6. print(True + True + True) stampa:
    A) True    B) 3    C) TrueTrueTrue    D) Error

Q7. print(7 / 2) stampa:
    A) 3    B) 3.5    C) 3.0    D) Error

Q8. print(7 // 2) stampa:
    A) 3    B) 3.5    C) 3.0    D) 4

Q9. print(-7 // 2) stampa:
    A) -3    B) -4    C) -3.5    D) 3

Q10. print(7 % -2) stampa:
     A) 1    B) -1    C) 0    D) Error

Q11. print(2 ** 3 ** 2) stampa:
     A) 64    B) 512    C) 12    D) Error

Q12. print(-2 ** 4) stampa:
     A) 16    B) -16    C) -8    D) Error

Q13. print(10 // 3 * 3 + 10 % 3) stampa:
     A) 10    B) 9    C) 11    D) 12

Q14. Quale nome è INVALIDO?
     A) _var    B) var1    C) 1var    D) VAR

Q15. x = 10; x /= 4; print(x) stampa:
     A) 2    B) 2.5    C) 2.0    D) Error

Q16. a, b = 5, 10; a, b = b, a; print(a) stampa:
     A) 5    B) 10    C) Error    D) None

Q17. print(int(3.9)) stampa:
     A) 4    B) 3    C) 3.0    D) Error

Q18. print(int(-3.9)) stampa:
     A) -4    B) -3    C) -3.0    D) Error

Q19. Se input è "5": x = input(); print(x * 2) stampa:
     A) 10    B) 55    C) Error    D) 52

Q20. print(int("3.14")) causa:
     A) 3    B) 3.14    C) ValueError    D) 3.0

Q21. print("A\\nB") stampa:
     A) A\\nB    B) AnB    C) A (newline) B    D) AB

Q22. print(1 < 2 < 3) stampa:
     A) True    B) False    C) Error    D) 1

Q23. print(1 < 2 > 0) stampa:
     A) True    B) False    C) Error    D) 1

Q24. print(float(5)) stampa:
     A) 5    B) 5.0    C) "5.0"    D) Error

Q25. print(type(4.)) stampa:
     A) int    B) float    C) Error    D) double

Q26. x = y = 5; x += 1; print(y) stampa:
     A) 5    B) 6    C) Error    D) None

Q27. print(10 - 5 - 2) stampa:
     A) 3    B) 7    C) -3    D) 13

Q28. print(2 ** 2 ** 3) stampa:
     A) 64    B) 256    C) 16    D) Error

Q29. print(5 == 5.0) stampa:
     A) True    B) False    C) Error    D) None

Q30. print(bool(""), bool(" ")) stampa:
     A) False False    B) True True    C) False True    D) True False


══════════════════════════════════════════════════════════════════════════════
                              FINE TEST
══════════════════════════════════════════════════════════════════════════════
"""


MODULE_2_TEST_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                    MODULE 2 - RISPOSTE TEST FINALE
══════════════════════════════════════════════════════════════════════════════

Q1:  A) A-B-C!
Q2:  B) 8 (ottale: 1*8 + 0 = 8)
Q3:  A) 15 (binario: 8+4+2+1 = 15)
Q4:  B) 10 (esadecimale: A = 10)
Q5:  B) 2000.0 (notazione scientifica, è float)
Q6:  B) 3 (True = 1)
Q7:  B) 3.5 (true division, sempre float)
Q8:  A) 3 (floor division, int se entrambi int)
Q9:  B) -4 (floor verso -infinito!)
Q10: B) -1 (modulo ha segno del divisore)
Q11: B) 512 (** associa a destra: 2^9)
Q12: B) -16 (** ha precedenza: -(2^4))
Q13: A) 10 (10//3=3, 3*3=9, 10%3=1, 9+1=10)
Q14: C) 1var (non può iniziare con cifra)
Q15: B) 2.5 (/= usa true division)
Q16: B) 10 (swap idiomatico)
Q17: B) 3 (tronca verso zero)
Q18: B) -3 (tronca verso zero, NON -4!)
Q19: B) 55 (input è stringa, * ripete)
Q20: C) ValueError (int non accetta string con punto)
Q21: C) A (newline) B
Q22: A) True (chained comparison)
Q23: A) True (1<2 AND 2>0)
Q24: B) 5.0
Q25: B) float (4. = 4.0)
Q26: A) 5 (int immutabile, y non cambia)
Q27: A) 3 (associa a sinistra)
Q28: B) 256 (** a destra: 2^8)
Q29: A) True (5 e 5.0 sono uguali in valore)
Q30: C) False True ("" è falsy, " " è truthy)


PUNTEGGIO:
──────────
28-30: Eccellente! 
24-27: Ottimo!
21-23: Buono, target PCEP raggiunto
<21:   Rivedi la teoria prima di procedere
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    ESECUZIONE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 78)
    print("PYTHON ESSENTIALS 1 - MODULE 2")
    print("Data Types, Variables, Operators, Basic I/O")
    print("=" * 78)
    print("""
    
    CONTENUTO:
    ──────────
    - Section 2.1: print() - 8 quiz
    - Section 2.2: Literals - 10 quiz
    - Section 2.3: Operators - 12 quiz
    - Section 2.4: Variables - 6 quiz
    - Section 2.5: Comments
    - Section 2.6: input() - 6 quiz
    - Labs: 10 esercizi pratici
    - Test finale: 30 domande
    
    COMANDI:
    ────────
    print(QUIZ_2_1)         # Quiz print()
    print(QUIZ_2_2)         # Quiz literals
    print(QUIZ_2_3)         # Quiz operators
    print(QUIZ_2_4)         # Quiz variables
    print(QUIZ_2_6)         # Quiz input()
    print(LABS_MODULE_2)    # Esercizi pratici
    print(MODULE_2_TEST)    # Test finale 30 domande
    
    """)
    print("\n" + "=" * 78)
    print("MODULE 2 COMPLETATO!")
    print("Passa a: pe1_m3_control_flow.py")
    print("=" * 78)
