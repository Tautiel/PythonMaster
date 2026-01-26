"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PCEP-30-02 CERTIFICATION EXERCISES                        ║
║              100 Esercizi Allineati al Syllabus Ufficiale                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Questi esercizi coprono TUTTO il syllabus PCEP-30-02 con la stessa
distribuzione di peso dell'esame reale.

STRUTTURA:
- Blocco 1: Computer Programming and Python Fundamentals (18%) → 20 esercizi
- Blocco 2: Control Flow (29%) → 30 esercizi
- Blocco 3: Data Collections (25%) → 25 esercizi
- Blocco 4: Functions and Exceptions (28%) → 25 esercizi

ISTRUZIONI:
1. Leggi attentamente ogni esercizio
2. Scrivi la soluzione SENZA guardare le risposte
3. Testa il codice
4. Confronta con la soluzione
5. Segna ✓ se corretto, ✗ se sbagliato
6. Ripeti gli errori dopo 24h

TRACKING:
- [ ] Blocco 1 completato: ___/20
- [ ] Blocco 2 completato: ___/30
- [ ] Blocco 3 completato: ___/25
- [ ] Blocco 4 completato: ___/25
- [ ] TOTALE: ___/100 (target: 70+)
"""


# ══════════════════════════════════════════════════════════════════════════════
# BLOCCO 1: COMPUTER PROGRAMMING AND PYTHON FUNDAMENTALS (18%)
# ══════════════════════════════════════════════════════════════════════════════

"""
PCEP 1.1 - Fundamental terms and definitions
PCEP 1.2 - Python's logic and structure
PCEP 1.3 - Literals and variables, numeral systems
PCEP 1.4 - Operators and data types
"""

# ------------------------------------------------------------------------------
# ESERCIZIO 1.1: Interpretazione vs Compilazione
# ------------------------------------------------------------------------------
"""
DOMANDA TEORICA (stile esame):

Quale delle seguenti affermazioni è VERA riguardo Python?

A) Python è un linguaggio compilato che genera file .exe
B) Python è un linguaggio interpretato che esegue il codice riga per riga
C) Python compila il codice in bytecode (.pyc) che viene poi interpretato dalla PVM
D) Python non utilizza nessuna forma di compilazione

RISPOSTA: ___

SPIEGAZIONE (scrivi la tua):
"""

# Risposta: C
# Python prima compila in bytecode, poi la Python Virtual Machine (PVM) interpreta
# il bytecode. I file .pyc nella cartella __pycache__ sono il bytecode compilato.


# ------------------------------------------------------------------------------
# ESERCIZIO 1.2: Keywords Python
# ------------------------------------------------------------------------------
"""
Quali di questi sono KEYWORDS Python riservate? (seleziona tutte le corrette)

A) print
B) if
C) True
D) class
E) function
F) elif
G) import

RISPOSTA: ___

Scrivi codice che verifica se una parola è una keyword:
"""

# Il tuo codice qui:
# import keyword
# print(keyword.kwlist)  # Lista tutte le keywords

# Risposta: B, C, D, F, G
# 'print' è una funzione built-in, non keyword
# 'function' non esiste, si usa 'def'


# ------------------------------------------------------------------------------
# ESERCIZIO 1.3: Indentazione
# ------------------------------------------------------------------------------
"""
Questo codice ha un errore di indentazione. Identificalo e correggilo:
"""

# CODICE CON ERRORE:
# x = 5
# if x > 3:
# print("maggiore di 3")
#     print("fine")

# Il tuo codice corretto:
x = 5
if x > 3:
    print("maggiore di 3")
    print("fine")


# ------------------------------------------------------------------------------
# ESERCIZIO 1.4: Literals - Tipi di dato
# ------------------------------------------------------------------------------
"""
Per ogni literal, indica il tipo di dato:

1. 42           → ___
2. 3.14         → ___
3. "hello"      → ___
4. True         → ___
5. None         → ___
6. 0o17         → ___
7. 0xFF         → ___
8. 3+4j         → ___
9. 1_000_000    → ___
10. .5          → ___

Verifica con type():
"""

# Il tuo codice di verifica:
print(type(42))         # int
print(type(3.14))       # float
print(type("hello"))    # str
print(type(True))       # bool
print(type(None))       # NoneType
print(type(0o17))       # int (ottale = 15)
print(type(0xFF))       # int (esadecimale = 255)
print(type(3+4j))       # complex
print(type(1_000_000))  # int (underscore separatore)
print(type(.5))         # float


# ------------------------------------------------------------------------------
# ESERCIZIO 1.5: Sistemi Numerali
# ------------------------------------------------------------------------------
"""
Converti:

1. Binario 0b1010 in decimale: ___
2. Ottale 0o17 in decimale: ___
3. Esadecimale 0xFF in decimale: ___
4. Decimale 42 in binario: ___
5. Decimale 255 in esadecimale: ___

Scrivi il codice per ogni conversione:
"""

# Conversioni:
print(0b1010)           # 10
print(0o17)             # 15
print(0xFF)             # 255
print(bin(42))          # 0b101010
print(hex(255))         # 0xff


# ------------------------------------------------------------------------------
# ESERCIZIO 1.6: Operatori Aritmetici
# ------------------------------------------------------------------------------
"""
Calcola il risultato di ogni espressione:

1. 17 // 5 = ___
2. 17 % 5 = ___
3. 2 ** 10 = ___
4. -17 // 5 = ___
5. -17 % 5 = ___
6. 5.0 // 2 = ___
7. 10 / 4 = ___
8. 10 // 4 = ___
"""

# Risposte:
print(17 // 5)    # 3 (floor division)
print(17 % 5)     # 2 (modulo)
print(2 ** 10)    # 1024 (esponente)
print(-17 // 5)   # -4 (floor verso -infinito!)
print(-17 % 5)    # 3 (Python: segno del divisore)
print(5.0 // 2)   # 2.0 (float result perché 5.0 è float)
print(10 / 4)     # 2.5 (true division, sempre float)
print(10 // 4)    # 2 (floor division, int)


# ------------------------------------------------------------------------------
# ESERCIZIO 1.7: Precedenza Operatori
# ------------------------------------------------------------------------------
"""
Calcola senza eseguire, poi verifica:

1. 2 + 3 * 4 = ___
2. (2 + 3) * 4 = ___
3. 2 ** 3 ** 2 = ___
4. 10 - 5 - 2 = ___
5. 2 ** 3 * 4 = ___
6. 100 // 10 % 3 = ___
"""

# Risposte:
print(2 + 3 * 4)        # 14 (* prima di +)
print((2 + 3) * 4)      # 20 (parentesi prima)
print(2 ** 3 ** 2)      # 512 (** associa a destra: 2^(3^2) = 2^9)
print(10 - 5 - 2)       # 3 (- associa a sinistra: (10-5)-2)
print(2 ** 3 * 4)       # 32 (** prima di *)
print(100 // 10 % 3)    # 1 (// e % stessa precedenza, sinistra→destra)


# ------------------------------------------------------------------------------
# ESERCIZIO 1.8: Operatori di Confronto
# ------------------------------------------------------------------------------
"""
Cosa restituiscono queste espressioni?

1. 5 == 5.0
2. 5 is 5.0
3. "abc" < "abd"
4. [1, 2] == [1, 2]
5. [1, 2] is [1, 2]
6. True == 1
7. False == 0
"""

# Risposte:
print(5 == 5.0)         # True (uguaglianza valore)
print(5 is 5.0)         # False (diversi oggetti)
print("abc" < "abd")    # True (confronto lessicografico)
print([1, 2] == [1, 2]) # True (stesso contenuto)
print([1, 2] is [1, 2]) # False (diversi oggetti)
print(True == 1)        # True (bool è sottoclasse di int)
print(False == 0)       # True


# ------------------------------------------------------------------------------
# ESERCIZIO 1.9: Operatori Logici
# ------------------------------------------------------------------------------
"""
Calcola:

1. True and False
2. True or False
3. not True
4. True and True or False
5. False or True and False
6. not (True and False)
"""

# Risposte (precedenza: not > and > or):
print(True and False)            # False
print(True or False)             # True
print(not True)                  # False
print(True and True or False)    # True (True and True = True, poi or False)
print(False or True and False)   # False (True and False = False, poi or)
print(not (True and False))      # True (not False)


# ------------------------------------------------------------------------------
# ESERCIZIO 1.10: Short-circuit Evaluation
# ------------------------------------------------------------------------------
"""
Cosa stampa questo codice? E perché?
"""

def f1():
    print("f1 called")
    return True

def f2():
    print("f2 called")
    return False

# Caso 1:
print("--- Caso 1: True or f2() ---")
result = True or f2()  # f2 NON viene chiamata (short-circuit)

# Caso 2:
print("--- Caso 2: False and f1() ---")
result = False and f1()  # f1 NON viene chiamata (short-circuit)

# Caso 3:
print("--- Caso 3: True and f2() ---")
result = True and f2()  # f2 VIENE chiamata


# ------------------------------------------------------------------------------
# ESERCIZIO 1.11: Operatori Bitwise
# ------------------------------------------------------------------------------
"""
Calcola (prima su carta, poi verifica):

1. 5 & 3 = ___  (AND bit a bit)
2. 5 | 3 = ___  (OR bit a bit)
3. 5 ^ 3 = ___  (XOR bit a bit)
4. ~5 = ___     (NOT bit a bit)
5. 5 << 1 = ___ (shift sinistra)
6. 5 >> 1 = ___ (shift destra)

Suggerimento: 5 = 0b101, 3 = 0b011
"""

# Risposte:
# 5 = 101
# 3 = 011
print(5 & 3)   # 001 = 1 (AND)
print(5 | 3)   # 111 = 7 (OR)
print(5 ^ 3)   # 110 = 6 (XOR)
print(~5)      # -6 (complemento a 2: -(5+1))
print(5 << 1)  # 1010 = 10 (shift left = *2)
print(5 >> 1)  # 010 = 2 (shift right = //2)


# ------------------------------------------------------------------------------
# ESERCIZIO 1.12: Assegnazione Composta
# ------------------------------------------------------------------------------
"""
Riscrivi usando operatori composti:
"""

x = 10
# x = x + 5  → 
x += 5

y = 20
# y = y * 2  →
y *= 2

z = 100
# z = z // 3 →
z //= 3

s = "hello"
# s = s + " world" →
s += " world"

print(x, y, z, s)  # 15, 40, 33, "hello world"


# ------------------------------------------------------------------------------
# ESERCIZIO 1.13: Type Conversion
# ------------------------------------------------------------------------------
"""
Converti e stampa:

1. Stringa "42" in intero
2. Intero 42 in stringa
3. Float 3.7 in intero
4. Intero 42 in float
5. Stringa "3.14" in float
6. Lista [1, 2, 3] in tupla
7. Intero 0 in booleano
8. Stringa vuota "" in booleano
"""

print(int("42"))       # 42
print(str(42))         # "42"
print(int(3.7))        # 3 (troncamento, non arrotondamento!)
print(float(42))       # 42.0
print(float("3.14"))   # 3.14
print(tuple([1,2,3]))  # (1, 2, 3)
print(bool(0))         # False
print(bool(""))        # False


# ------------------------------------------------------------------------------
# ESERCIZIO 1.14: Truthy e Falsy Values
# ------------------------------------------------------------------------------
"""
Quali di questi valori sono "falsy" (valutati come False in contesto booleano)?

A) 0
B) 0.0
C) ""
D) []
E) {}
F) None
G) "False"
H) [0]
I) " "  (spazio)
"""

# Test:
values = [0, 0.0, "", [], {}, None, "False", [0], " "]
for v in values:
    print(f"{repr(v):10} → {bool(v)}")

# Falsy: A, B, C, D, E, F
# "False" è truthy (stringa non vuota)
# [0] è truthy (lista non vuota)
# " " è truthy (stringa non vuota)


# ------------------------------------------------------------------------------
# ESERCIZIO 1.15: Input e Output
# ------------------------------------------------------------------------------
"""
Scrivi un programma che:
1. Chiede nome e età all'utente
2. Calcola l'anno di nascita (approssimativo)
3. Stampa un messaggio formattato
"""

# La tua soluzione:
def greet_user():
    name = input("Come ti chiami? ")
    age = int(input("Quanti anni hai? "))
    birth_year = 2025 - age
    print(f"Ciao {name}! Sei nato/a circa nel {birth_year}")

# greet_user()  # Decommentare per testare


# ------------------------------------------------------------------------------
# ESERCIZIO 1.16: Escape Sequences
# ------------------------------------------------------------------------------
"""
Cosa stampa questo codice?
"""

print("Line1\nLine2")        # newline
print("Tab\there")           # tab
print("Quote: \"hello\"")    # virgolette
print("Backslash: \\")       # backslash
print('It\'s Python')        # apostrofo
print("Bell\a")              # beep (se supportato)
print(r"Raw: \n no escape")  # raw string


# ------------------------------------------------------------------------------
# ESERCIZIO 1.17: String Formatting
# ------------------------------------------------------------------------------
"""
Formatta la stringa "Il prezzo è 42.567 euro" in 3 modi diversi:
1. Concatenazione
2. f-string
3. .format()
"""

prezzo = 42.567

# 1. Concatenazione
s1 = "Il prezzo è " + str(prezzo) + " euro"

# 2. f-string (con 2 decimali)
s2 = f"Il prezzo è {prezzo:.2f} euro"

# 3. .format()
s3 = "Il prezzo è {:.2f} euro".format(prezzo)

print(s1, s2, s3, sep="\n")


# ------------------------------------------------------------------------------
# ESERCIZIO 1.18: Variabili e Riferimenti
# ------------------------------------------------------------------------------
"""
Cosa stampa questo codice? Perché?
"""

a = [1, 2, 3]
b = a          # b riferisce allo stesso oggetto di a
c = a.copy()   # c è una copia indipendente

a.append(4)

print(f"a = {a}")  # [1, 2, 3, 4]
print(f"b = {b}")  # [1, 2, 3, 4] (stesso oggetto!)
print(f"c = {c}")  # [1, 2, 3] (copia separata)

print(f"a is b: {a is b}")  # True
print(f"a is c: {a is c}")  # False


# ------------------------------------------------------------------------------
# ESERCIZIO 1.19: Naming Conventions (PEP 8)
# ------------------------------------------------------------------------------
"""
Correggi questi nomi secondo PEP 8:

1. MyVariable    → ___
2. CONSTANT      → ___ (va bene così?)
3. myFunction    → ___
4. class myclass → ___
5. _private      → ___ (va bene così?)
"""

# Risposte PEP 8:
# 1. my_variable (snake_case per variabili)
# 2. CONSTANT (UPPER_CASE per costanti - corretto)
# 3. my_function (snake_case per funzioni)
# 4. class MyClass (PascalCase per classi)
# 5. _private (underscore prefix per "private" - corretto)


# ------------------------------------------------------------------------------
# ESERCIZIO 1.20: Mini Quiz Blocco 1
# ------------------------------------------------------------------------------
"""
QUIZ STILE ESAME - 5 domande da completare:

Q1. Qual è il risultato di: 3 ** 2 ** 1?
A) 9
B) 6
C) 8
D) 3

Q2. Quale NON è una keyword Python?
A) None
B) pass
C) elif
D) function

Q3. Qual è il tipo di: type(1 == 1)?
A) int
B) str
C) bool
D) NoneType

Q4. Cosa restituisce: -7 % 3?
A) -1
B) 2
C) 1
D) -2

Q5. Quale espressione è True?
A) "abc" > "abd"
B) 5 is 5.0
C) [] == False
D) not ""
"""

# Risposte:
# Q1: A (3 ** 2 ** 1 = 3 ** 2 = 9, ** associa a destra)
# Q2: D (function non esiste, si usa def)
# Q3: C (1 == 1 restituisce True, che è bool)
# Q4: B (Python: -7 = 3*(-3) + 2, resto ha segno del divisore)
# Q5: D (not "" = not False = True)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCCO 2: CONTROL FLOW (29%)
# ══════════════════════════════════════════════════════════════════════════════

"""
PCEP 2.1 - Conditional statements (if, if-else, if-elif-else)
PCEP 2.2 - Loops (while, for, range, break, continue, pass)
PCEP 2.3 - Logic and bit operations in control flow
"""

# ------------------------------------------------------------------------------
# ESERCIZIO 2.1: if-else Base
# ------------------------------------------------------------------------------
"""
Scrivi una funzione che:
- Riceve un numero
- Restituisce "positivo", "negativo", o "zero"
"""

def check_sign(n):
    if n > 0:
        return "positivo"
    elif n < 0:
        return "negativo"
    else:
        return "zero"

# Test:
assert check_sign(5) == "positivo"
assert check_sign(-3) == "negativo"
assert check_sign(0) == "zero"
print("✓ Esercizio 2.1 completato")


# ------------------------------------------------------------------------------
# ESERCIZIO 2.2: Condizioni Multiple
# ------------------------------------------------------------------------------
"""
Scrivi una funzione che determina il voto in lettere:
- 90-100: A
- 80-89: B
- 70-79: C
- 60-69: D
- <60: F
"""

def grade_letter(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

# Test:
assert grade_letter(95) == "A"
assert grade_letter(85) == "B"
assert grade_letter(75) == "C"
assert grade_letter(65) == "D"
assert grade_letter(55) == "F"
print("✓ Esercizio 2.2 completato")


# ------------------------------------------------------------------------------
# ESERCIZIO 2.3: Ternary Operator
# ------------------------------------------------------------------------------
"""
Riscrivi usando l'operatore ternario:

if x > 0:
    result = "positive"
else:
    result = "non-positive"
"""

x = 5
# La tua soluzione:
result = "positive" if x > 0 else "non-positive"
print(f"x={x} → {result}")


# ------------------------------------------------------------------------------
# ESERCIZIO 2.4: Nested Conditions
# ------------------------------------------------------------------------------
"""
Scrivi una funzione che classifica un anno:
- bisestile: divisibile per 4, MA non per 100, A MENO CHE divisibile per 400
"""

def is_leap_year(year):
    if year % 400 == 0:
        return True
    elif year % 100 == 0:
        return False
    elif year % 4 == 0:
        return True
    else:
        return False

# Versione compatta:
def is_leap_year_compact(year):
    return year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)

# Test:
assert is_leap_year(2000) == True   # divisibile per 400
assert is_leap_year(1900) == False  # divisibile per 100 ma non 400
assert is_leap_year(2024) == True   # divisibile per 4
assert is_leap_year(2023) == False  # non divisibile per 4
print("✓ Esercizio 2.4 completato")


# ------------------------------------------------------------------------------
# ESERCIZIO 2.5: while Loop Base
# ------------------------------------------------------------------------------
"""
Scrivi un loop while che:
- Parte da 1
- Raddoppia il valore ogni iterazione
- Si ferma quando supera 1000
- Stampa tutti i valori
"""

n = 1
print("Raddoppio:", end=" ")
while n <= 1000:
    print(n, end=" ")
    n *= 2
print()  # 1 2 4 8 16 32 64 128 256 512


# ------------------------------------------------------------------------------
# ESERCIZIO 2.6: while con Input Validation
# ------------------------------------------------------------------------------
"""
Scrivi una funzione che chiede un numero finché non è valido (1-10):
"""

def get_valid_number():
    while True:
        try:
            n = int(input("Inserisci un numero (1-10): "))
            if 1 <= n <= 10:
                return n
            else:
                print("Fuori range!")
        except ValueError:
            print("Non è un numero!")

# get_valid_number()  # Decommentare per testare


# ------------------------------------------------------------------------------
# ESERCIZIO 2.7: for con range()
# ------------------------------------------------------------------------------
"""
Completa le espressioni range() per ottenere:

1. [0, 1, 2, 3, 4]        → range(___)
2. [1, 2, 3, 4, 5]        → range(___)
3. [0, 2, 4, 6, 8]        → range(___)
4. [10, 8, 6, 4, 2]       → range(___)
5. [5, 4, 3, 2, 1, 0]     → range(___)
"""

# Risposte:
print(list(range(5)))           # [0, 1, 2, 3, 4]
print(list(range(1, 6)))        # [1, 2, 3, 4, 5]
print(list(range(0, 10, 2)))    # [0, 2, 4, 6, 8]
print(list(range(10, 0, -2)))   # [10, 8, 6, 4, 2]
print(list(range(5, -1, -1)))   # [5, 4, 3, 2, 1, 0]


# ------------------------------------------------------------------------------
# ESERCIZIO 2.8: for con enumerate()
# ------------------------------------------------------------------------------
"""
Stampa ogni elemento di una lista con il suo indice:
["apple", "banana", "cherry"]
Output:
0: apple
1: banana
2: cherry
"""

fruits = ["apple", "banana", "cherry"]

# Modo 1: con indice manuale
for i in range(len(fruits)):
    print(f"{i}: {fruits[i]}")

print("---")

# Modo 2: con enumerate (preferito)
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")


# ------------------------------------------------------------------------------
# ESERCIZIO 2.9: break
# ------------------------------------------------------------------------------
"""
Trova il primo numero divisibile per 7 e 11 partendo da 1:
"""

for n in range(1, 1000):
    if n % 7 == 0 and n % 11 == 0:
        print(f"Primo divisibile per 7 e 11: {n}")
        break  # 77


# ------------------------------------------------------------------------------
# ESERCIZIO 2.10: continue
# ------------------------------------------------------------------------------
"""
Stampa i numeri da 1 a 10, saltando i multipli di 3:
"""

for n in range(1, 11):
    if n % 3 == 0:
        continue
    print(n, end=" ")
print()  # 1 2 4 5 7 8 10


# ------------------------------------------------------------------------------
# ESERCIZIO 2.11: pass (placeholder)
# ------------------------------------------------------------------------------
"""
Quando usare pass:
"""

# Classe vuota (da implementare)
class MyFutureClass:
    pass

# Funzione vuota
def not_implemented_yet():
    pass

# Loop che ignora certi casi
for i in range(5):
    if i == 2:
        pass  # TODO: gestire caso speciale
    else:
        print(i)


# ------------------------------------------------------------------------------
# ESERCIZIO 2.12: else in Loops
# ------------------------------------------------------------------------------
"""
L'else di un loop viene eseguito se il loop completa SENZA break.
"""

# Cerca un numero primo
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False  # break implicito con return
    return True  # completato senza trovare divisori

# Versione con else:
def is_prime_with_else(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            break
    else:
        # Eseguito solo se il loop non ha fatto break
        return True
    return False

# Test:
print(is_prime(17))  # True
print(is_prime(18))  # False


# ------------------------------------------------------------------------------
# ESERCIZIO 2.13: Nested Loops
# ------------------------------------------------------------------------------
"""
Stampa una tavola pitagorica 5x5:
"""

print("Tavola Pitagorica:")
for i in range(1, 6):
    for j in range(1, 6):
        print(f"{i*j:3}", end=" ")
    print()


# ------------------------------------------------------------------------------
# ESERCIZIO 2.14: Loop con Accumulatore
# ------------------------------------------------------------------------------
"""
Calcola la somma dei numeri da 1 a 100:
"""

# Con loop:
total = 0
for i in range(1, 101):
    total += i
print(f"Somma 1-100: {total}")  # 5050

# Formula di Gauss (verifica):
print(f"Gauss: {100 * 101 // 2}")  # 5050


# ------------------------------------------------------------------------------
# ESERCIZIO 2.15: FizzBuzz
# ------------------------------------------------------------------------------
"""
Classico problema di coding interview:
- Per multipli di 3: "Fizz"
- Per multipli di 5: "Buzz"
- Per multipli di entrambi: "FizzBuzz"
- Altrimenti: il numero
"""

def fizzbuzz(n):
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:  # Multiplo di entrambi (15 = 3*5)
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result

print(fizzbuzz(15))


# ------------------------------------------------------------------------------
# ESERCIZIO 2.16: Countdown con while
# ------------------------------------------------------------------------------
"""
Scrivi un countdown da n a 0 con while:
"""

def countdown(n):
    while n >= 0:
        print(n, end=" ")
        n -= 1
    print("Boom!")

countdown(5)  # 5 4 3 2 1 0 Boom!


# ------------------------------------------------------------------------------
# ESERCIZIO 2.17: Trova Cifre di un Numero
# ------------------------------------------------------------------------------
"""
Conta le cifre di un numero usando while:
"""

def count_digits(n):
    n = abs(n)  # Gestisce negativi
    if n == 0:
        return 1
    count = 0
    while n > 0:
        n //= 10
        count += 1
    return count

assert count_digits(12345) == 5
assert count_digits(0) == 1
assert count_digits(-42) == 2
print("✓ Esercizio 2.17 completato")


# ------------------------------------------------------------------------------
# ESERCIZIO 2.18: Sequenza di Collatz
# ------------------------------------------------------------------------------
"""
La congettura di Collatz:
- Se n è pari: n = n // 2
- Se n è dispari: n = 3n + 1
- Continua finché n = 1
"""

def collatz(n):
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence

print(f"Collatz(7): {collatz(7)}")
# [7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]


# ------------------------------------------------------------------------------
# ESERCIZIO 2.19: Pattern con Loop
# ------------------------------------------------------------------------------
"""
Stampa questo pattern:
*
**
***
****
*****
"""

for i in range(1, 6):
    print("*" * i)

print("\nPattern invertito:")
for i in range(5, 0, -1):
    print("*" * i)


# ------------------------------------------------------------------------------
# ESERCIZIO 2.20: Numeri Primi fino a N
# ------------------------------------------------------------------------------
"""
Trova tutti i numeri primi fino a n:
"""

def primes_up_to(n):
    primes = []
    for num in range(2, n + 1):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes

print(f"Primi fino a 30: {primes_up_to(30)}")


# ------------------------------------------------------------------------------
# ESERCIZIO 2.21-2.30: Esercizi Pratici Control Flow
# ------------------------------------------------------------------------------

# 2.21: Verifica se una stringa è palindroma
def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]

assert is_palindrome("radar") == True
assert is_palindrome("hello") == False

# 2.22: Conta vocali in una stringa
def count_vowels(s):
    count = 0
    for char in s.lower():
        if char in "aeiou":
            count += 1
    return count

assert count_vowels("Hello World") == 3

# 2.23: Trova il massimo in una lista (senza max())
def find_max(lst):
    if not lst:
        return None
    maximum = lst[0]
    for item in lst[1:]:
        if item > maximum:
            maximum = item
    return maximum

assert find_max([3, 1, 4, 1, 5, 9, 2, 6]) == 9

# 2.24: Calcola fattoriale
def factorial(n):
    if n < 0:
        raise ValueError("n deve essere >= 0")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

assert factorial(5) == 120
assert factorial(0) == 1

# 2.25: Fibonacci fino a n termini
def fibonacci(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    fib = [0, 1]
    for _ in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib

assert fibonacci(10) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# 2.26: Inverti un numero
def reverse_number(n):
    negative = n < 0
    n = abs(n)
    reversed_n = 0
    while n > 0:
        reversed_n = reversed_n * 10 + n % 10
        n //= 10
    return -reversed_n if negative else reversed_n

assert reverse_number(12345) == 54321
assert reverse_number(-42) == -24

# 2.27: Trova tutti i divisori
def find_divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]

assert find_divisors(12) == [1, 2, 3, 4, 6, 12]

# 2.28: Somma delle cifre
def sum_of_digits(n):
    n = abs(n)
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total

assert sum_of_digits(12345) == 15

# 2.29: Verifica numero Armstrong
# (somma delle cifre elevate alla potenza del numero di cifre = numero)
def is_armstrong(n):
    digits = str(n)
    power = len(digits)
    return sum(int(d) ** power for d in digits) == n

assert is_armstrong(153) == True  # 1^3 + 5^3 + 3^3 = 153
assert is_armstrong(9474) == True

# 2.30: Binary search
def binary_search(lst, target):
    left, right = 0, len(lst) - 1
    while left <= right:
        mid = (left + right) // 2
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

assert binary_search([1, 3, 5, 7, 9, 11], 7) == 3
assert binary_search([1, 3, 5, 7, 9, 11], 4) == -1

print("✓ Esercizi 2.21-2.30 completati")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCCO 3: DATA COLLECTIONS (25%)
# ══════════════════════════════════════════════════════════════════════════════

"""
PCEP 3.1 - Lists
PCEP 3.2 - Tuples
PCEP 3.3 - Dictionaries
PCEP 3.4 - Strings (advanced operations)
"""

# ------------------------------------------------------------------------------
# ESERCIZIO 3.1: Liste - Creazione e Accesso
# ------------------------------------------------------------------------------
"""
Data la lista:
"""
numbers = [10, 20, 30, 40, 50]

# Accedi a:
print(numbers[0])      # Primo: 10
print(numbers[-1])     # Ultimo: 50
print(numbers[2])      # Terzo: 30
print(numbers[-2])     # Penultimo: 40


# ------------------------------------------------------------------------------
# ESERCIZIO 3.2: Liste - Slicing
# ------------------------------------------------------------------------------
"""
Data la lista:
"""
lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Ottieni:
print(lst[2:5])        # [2, 3, 4]
print(lst[:3])         # [0, 1, 2]
print(lst[7:])         # [7, 8, 9]
print(lst[::2])        # [0, 2, 4, 6, 8] (ogni 2)
print(lst[::-1])       # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] (invertita)
print(lst[1:8:2])      # [1, 3, 5, 7]
print(lst[-3:])        # [7, 8, 9]


# ------------------------------------------------------------------------------
# ESERCIZIO 3.3: Liste - Metodi Principali
# ------------------------------------------------------------------------------
"""
Completa usando i metodi corretti:
"""
fruits = ["apple", "banana"]

# Aggiungi "cherry" alla fine
fruits.append("cherry")

# Inserisci "apricot" all'inizio
fruits.insert(0, "apricot")

# Rimuovi "banana"
fruits.remove("banana")

# Aggiungi multipli frutti
fruits.extend(["date", "elderberry"])

# Estrai e rimuovi l'ultimo elemento
last = fruits.pop()

# Trova l'indice di "cherry"
idx = fruits.index("cherry")

# Conta quante "apple"
count = fruits.count("apple")

print(fruits)


# ------------------------------------------------------------------------------
# ESERCIZIO 3.4: Liste - Ordinamento
# ------------------------------------------------------------------------------
"""
Ordina la lista in vari modi:
"""
nums = [3, 1, 4, 1, 5, 9, 2, 6]

# Ordina in place (modifica originale)
nums.sort()
print(f"Ordinata: {nums}")

# Ordina al contrario
nums.sort(reverse=True)
print(f"Decrescente: {nums}")

# Crea nuova lista ordinata (non modifica)
original = [3, 1, 4, 1, 5]
sorted_copy = sorted(original)
print(f"Original: {original}, Sorted copy: {sorted_copy}")


# ------------------------------------------------------------------------------
# ESERCIZIO 3.5: List Comprehension Base
# ------------------------------------------------------------------------------
"""
Riscrivi usando list comprehension:
"""

# Versione con loop:
squares = []
for x in range(10):
    squares.append(x ** 2)

# List comprehension:
squares = [x ** 2 for x in range(10)]
print(squares)

# Con condizione (solo pari):
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
print(even_squares)


# ------------------------------------------------------------------------------
# ESERCIZIO 3.6: Liste Annidate
# ------------------------------------------------------------------------------
"""
Matrice 3x3:
"""
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Accedi all'elemento centrale (5):
print(matrix[1][1])

# Estrai la seconda riga:
print(matrix[1])

# Estrai la seconda colonna:
column = [row[1] for row in matrix]
print(column)

# Trasponi la matrice:
transposed = [[row[i] for row in matrix] for i in range(3)]
print(transposed)


# ------------------------------------------------------------------------------
# ESERCIZIO 3.7: Copia Superficiale vs Profonda
# ------------------------------------------------------------------------------
"""
ATTENZIONE: differenza cruciale!
"""
import copy

original = [[1, 2], [3, 4]]

# Shallow copy (copia solo primo livello)
shallow = original.copy()  # oppure: list(original) o original[:]
shallow[0][0] = 99
print(f"Original dopo shallow: {original}")  # [[99, 2], [3, 4]] - MODIFICATO!

original = [[1, 2], [3, 4]]  # Reset

# Deep copy (copia tutto)
deep = copy.deepcopy(original)
deep[0][0] = 99
print(f"Original dopo deep: {original}")  # [[1, 2], [3, 4]] - INTATTO!


# ------------------------------------------------------------------------------
# ESERCIZIO 3.8: Tuple
# ------------------------------------------------------------------------------
"""
Le tuple sono IMMUTABILI
"""

# Creazione
t1 = (1, 2, 3)
t2 = 1, 2, 3        # Parentesi opzionali
t3 = (1,)           # Tuple con un elemento (virgola necessaria!)
t4 = ()             # Tuple vuota

# Packing e unpacking
coords = (10, 20, 30)
x, y, z = coords    # Unpacking

# Swap con unpacking
a, b = 1, 2
a, b = b, a         # Swap in una riga!

# Extended unpacking
first, *rest = [1, 2, 3, 4, 5]
print(f"first={first}, rest={rest}")  # first=1, rest=[2, 3, 4, 5]

*head, last = [1, 2, 3, 4, 5]
print(f"head={head}, last={last}")    # head=[1, 2, 3, 4], last=5


# ------------------------------------------------------------------------------
# ESERCIZIO 3.9: Tuple come Chiavi di Dizionario
# ------------------------------------------------------------------------------
"""
Le tuple possono essere chiavi perché immutabili:
"""

# Coordinate come chiavi
positions = {
    (0, 0): "origin",
    (1, 0): "east",
    (0, 1): "north"
}

print(positions[(1, 0)])  # "east"

# Le liste NON possono essere chiavi:
# {[1, 2]: "value"}  # TypeError: unhashable type: 'list'


# ------------------------------------------------------------------------------
# ESERCIZIO 3.10: Dizionari - Creazione e Accesso
# ------------------------------------------------------------------------------

# Modi di creare un dizionario:
d1 = {"name": "Alice", "age": 30}
d2 = dict(name="Bob", age=25)
d3 = dict([("name", "Charlie"), ("age", 35)])

# Accesso
print(d1["name"])       # "Alice"
print(d1.get("job"))    # None (no KeyError)
print(d1.get("job", "Unknown"))  # "Unknown" (valore default)


# ------------------------------------------------------------------------------
# ESERCIZIO 3.11: Dizionari - Metodi
# ------------------------------------------------------------------------------

person = {"name": "Alice", "age": 30, "city": "Rome"}

# Keys, values, items
print(list(person.keys()))    # ['name', 'age', 'city']
print(list(person.values()))  # ['Alice', 30, 'Rome']
print(list(person.items()))   # [('name', 'Alice'), ...]

# Aggiungere/modificare
person["job"] = "Engineer"    # Aggiunge
person["age"] = 31            # Modifica

# Rimuovere
del person["city"]            # Rimuove
job = person.pop("job")       # Rimuove e restituisce

# Update (unione)
person.update({"city": "Milan", "country": "Italy"})

# setdefault (aggiungi solo se non esiste)
person.setdefault("nickname", "Ali")


# ------------------------------------------------------------------------------
# ESERCIZIO 3.12: Dict Comprehension
# ------------------------------------------------------------------------------

# Quadrati come dizionario
squares = {x: x**2 for x in range(1, 6)}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Inverti chiavi e valori
inverted = {v: k for k, v in squares.items()}
print(inverted)

# Con condizione
even_squares = {x: x**2 for x in range(1, 11) if x % 2 == 0}
print(even_squares)


# ------------------------------------------------------------------------------
# ESERCIZIO 3.13: Iterazione su Dizionari
# ------------------------------------------------------------------------------

prices = {"apple": 1.0, "banana": 0.5, "cherry": 2.0}

# Solo chiavi (default)
for fruit in prices:
    print(fruit)

# Chiavi e valori
for fruit, price in prices.items():
    print(f"{fruit}: ${price}")

# Ordinato per chiave
for fruit in sorted(prices.keys()):
    print(f"{fruit}: ${prices[fruit]}")

# Ordinato per valore
for fruit, price in sorted(prices.items(), key=lambda x: x[1]):
    print(f"{fruit}: ${price}")


# ------------------------------------------------------------------------------
# ESERCIZIO 3.14: Stringhe - Metodi
# ------------------------------------------------------------------------------

s = "  Hello, World!  "

# Case
print(s.upper())        # "  HELLO, WORLD!  "
print(s.lower())        # "  hello, world!  "
print(s.title())        # "  Hello, World!  "
print(s.capitalize())   # "  hello, world!  " (solo primo char)

# Strip (rimuovi whitespace)
print(s.strip())        # "Hello, World!"
print(s.lstrip())       # "Hello, World!  "
print(s.rstrip())       # "  Hello, World!"

# Find e replace
print(s.find("World"))  # 9
print(s.replace("World", "Python"))

# Split e join
words = "one,two,three".split(",")
print(words)            # ['one', 'two', 'three']
print("-".join(words))  # "one-two-three"


# ------------------------------------------------------------------------------
# ESERCIZIO 3.15: Stringhe - Verifica
# ------------------------------------------------------------------------------

# Metodi di verifica
print("abc123".isalnum())    # True
print("abc".isalpha())       # True
print("123".isdigit())       # True
print("   ".isspace())       # True
print("hello".islower())     # True
print("HELLO".isupper())     # True
print("Hello World".istitle())  # True

# Starts/ends
print("hello.py".endswith(".py"))     # True
print("hello.py".startswith("he"))    # True


# ------------------------------------------------------------------------------
# ESERCIZIO 3.16-3.25: Esercizi Pratici Collections
# ------------------------------------------------------------------------------

# 3.16: Rimuovi duplicati preservando ordine
def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

assert remove_duplicates([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]

# 3.17: Trova elementi comuni tra due liste
def find_common(lst1, lst2):
    return list(set(lst1) & set(lst2))

assert set(find_common([1, 2, 3], [2, 3, 4])) == {2, 3}

# 3.18: Appiattisci lista annidata (un livello)
def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result

assert flatten([[1, 2], [3, 4], 5]) == [1, 2, 3, 4, 5]

# 3.19: Conta frequenza caratteri
def char_frequency(s):
    freq = {}
    for char in s.lower():
        if char.isalpha():
            freq[char] = freq.get(char, 0) + 1
    return freq

assert char_frequency("hello") == {'h': 1, 'e': 1, 'l': 2, 'o': 1}

# 3.20: Raggruppa per lunghezza
def group_by_length(words):
    groups = {}
    for word in words:
        length = len(word)
        if length not in groups:
            groups[length] = []
        groups[length].append(word)
    return groups

result = group_by_length(["hi", "bye", "hello", "go"])
assert result == {2: ['hi', 'go'], 3: ['bye'], 5: ['hello']}

# 3.21: Trova la parola più frequente
def most_frequent_word(text):
    words = text.lower().split()
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return max(freq, key=freq.get)

# 3.22: Merge dizionari
def merge_dicts(d1, d2):
    result = d1.copy()
    result.update(d2)
    return result
# Python 3.9+: d1 | d2

# 3.23: Trova chiave per valore
def find_key_by_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None

# 3.24: Inverti dizionario (assumi valori unici)
def invert_dict(d):
    return {v: k for k, v in d.items()}

# 3.25: Ordina lista di dizionari
people = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
]

# Per età
sorted_by_age = sorted(people, key=lambda x: x["age"])
print(sorted_by_age)

print("✓ Esercizi 3.16-3.25 completati")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCCO 4: FUNCTIONS AND EXCEPTIONS (28%)
# ══════════════════════════════════════════════════════════════════════════════

"""
PCEP 4.1 - Defining and calling functions
PCEP 4.2 - Scope (local, global)
PCEP 4.3 - Recursion
PCEP 4.4 - Exceptions (try-except)
"""

# ------------------------------------------------------------------------------
# ESERCIZIO 4.1: Funzione Base
# ------------------------------------------------------------------------------

def greet(name):
    """Saluta una persona."""
    return f"Hello, {name}!"

# Chiamata
print(greet("Alice"))
print(greet.__doc__)  # Stampa docstring


# ------------------------------------------------------------------------------
# ESERCIZIO 4.2: Parametri Default
# ------------------------------------------------------------------------------

def power(base, exponent=2):
    """Calcola base^exponent, default quadrato."""
    return base ** exponent

print(power(3))      # 9 (usa default)
print(power(3, 3))   # 27
print(power(2, 10))  # 1024


# ------------------------------------------------------------------------------
# ESERCIZIO 4.3: Keyword Arguments
# ------------------------------------------------------------------------------

def describe_pet(name, animal_type="dog", age=None):
    """Descrive un animale domestico."""
    desc = f"{name} is a {animal_type}"
    if age:
        desc += f", {age} years old"
    return desc

# Vari modi di chiamare:
print(describe_pet("Rex"))
print(describe_pet("Whiskers", "cat"))
print(describe_pet("Tweety", animal_type="bird", age=2))
print(describe_pet(animal_type="fish", name="Nemo"))


# ------------------------------------------------------------------------------
# ESERCIZIO 4.4: *args (Positional Variable)
# ------------------------------------------------------------------------------

def sum_all(*numbers):
    """Somma qualsiasi numero di argomenti."""
    return sum(numbers)

print(sum_all(1, 2, 3))        # 6
print(sum_all(1, 2, 3, 4, 5))  # 15
print(sum_all())               # 0


# ------------------------------------------------------------------------------
# ESERCIZIO 4.5: **kwargs (Keyword Variable)
# ------------------------------------------------------------------------------

def print_info(**kwargs):
    """Stampa tutte le info passate."""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=30, city="Rome")


# ------------------------------------------------------------------------------
# ESERCIZIO 4.6: Combinazione Completa
# ------------------------------------------------------------------------------

def complex_function(a, b, *args, option=True, **kwargs):
    """
    a, b: posizionali obbligatori
    *args: posizionali extra
    option: keyword con default
    **kwargs: keyword extra
    """
    print(f"a={a}, b={b}")
    print(f"args={args}")
    print(f"option={option}")
    print(f"kwargs={kwargs}")

complex_function(1, 2, 3, 4, 5, option=False, x=10, y=20)


# ------------------------------------------------------------------------------
# ESERCIZIO 4.7: Return Multiple Values
# ------------------------------------------------------------------------------

def min_max(numbers):
    """Restituisce minimo e massimo."""
    return min(numbers), max(numbers)

# Unpacking del return
minimum, maximum = min_max([3, 1, 4, 1, 5, 9])
print(f"Min: {minimum}, Max: {maximum}")


# ------------------------------------------------------------------------------
# ESERCIZIO 4.8: Scope - Local vs Global
# ------------------------------------------------------------------------------

global_var = "I'm global"

def scope_demo():
    local_var = "I'm local"
    print(global_var)  # Può leggere globale
    print(local_var)

scope_demo()
# print(local_var)  # NameError: local_var non esiste qui


# ------------------------------------------------------------------------------
# ESERCIZIO 4.9: global Keyword
# ------------------------------------------------------------------------------

counter = 0

def increment():
    global counter  # Dichiara che vogliamo modificare la globale
    counter += 1

increment()
increment()
print(counter)  # 2


# ------------------------------------------------------------------------------
# ESERCIZIO 4.10: Shadowing
# ------------------------------------------------------------------------------

x = "global"

def shadow():
    x = "local"  # Crea nuova variabile locale, non modifica globale
    print(f"Inside: {x}")

shadow()           # "Inside: local"
print(f"Outside: {x}")  # "Outside: global"


# ------------------------------------------------------------------------------
# ESERCIZIO 4.11: Ricorsione - Fattoriale
# ------------------------------------------------------------------------------

def factorial_recursive(n):
    """Calcola n! ricorsivamente."""
    # Caso base
    if n <= 1:
        return 1
    # Caso ricorsivo
    return n * factorial_recursive(n - 1)

print(factorial_recursive(5))  # 120


# ------------------------------------------------------------------------------
# ESERCIZIO 4.12: Ricorsione - Fibonacci
# ------------------------------------------------------------------------------

def fib_recursive(n):
    """Restituisce l'n-esimo numero di Fibonacci."""
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)

# Nota: inefficiente per n grande (esponenziale)
print([fib_recursive(i) for i in range(10)])


# ------------------------------------------------------------------------------
# ESERCIZIO 4.13: Ricorsione - Somma Lista
# ------------------------------------------------------------------------------

def sum_recursive(lst):
    """Somma elementi di una lista ricorsivamente."""
    if not lst:  # Lista vuota = caso base
        return 0
    return lst[0] + sum_recursive(lst[1:])

print(sum_recursive([1, 2, 3, 4, 5]))  # 15


# ------------------------------------------------------------------------------
# ESERCIZIO 4.14: try-except Base
# ------------------------------------------------------------------------------

def safe_divide(a, b):
    """Divisione con gestione errori."""
    try:
        result = a / b
    except ZeroDivisionError:
        return "Errore: divisione per zero"
    except TypeError:
        return "Errore: tipi non validi"
    else:
        return result  # Eseguito solo se nessuna eccezione

print(safe_divide(10, 2))   # 5.0
print(safe_divide(10, 0))   # Errore: divisione per zero
print(safe_divide("10", 2)) # Errore: tipi non validi


# ------------------------------------------------------------------------------
# ESERCIZIO 4.15: Eccezioni Multiple
# ------------------------------------------------------------------------------

def process_data(data, index):
    try:
        value = data[index]
        result = 100 / value
        return result
    except IndexError:
        return "Indice fuori range"
    except ZeroDivisionError:
        return "Valore è zero"
    except (TypeError, KeyError) as e:
        return f"Errore: {type(e).__name__}"

print(process_data([1, 2, 3], 1))   # 50.0
print(process_data([1, 2, 3], 10))  # Indice fuori range
print(process_data([1, 0, 3], 1))   # Valore è zero


# ------------------------------------------------------------------------------
# ESERCIZIO 4.16: try-except-else-finally
# ------------------------------------------------------------------------------

def read_file(filename):
    try:
        f = open(filename, 'r')
        content = f.read()
    except FileNotFoundError:
        print("File non trovato")
        return None
    else:
        print("File letto con successo")
        return content
    finally:
        print("Cleanup (sempre eseguito)")
        # f.close() se necessario


# ------------------------------------------------------------------------------
# ESERCIZIO 4.17: raise
# ------------------------------------------------------------------------------

def validate_age(age):
    """Valida età con eccezione custom."""
    if not isinstance(age, int):
        raise TypeError("Età deve essere un intero")
    if age < 0:
        raise ValueError("Età non può essere negativa")
    if age > 150:
        raise ValueError("Età non realistica")
    return True

try:
    validate_age(-5)
except ValueError as e:
    print(f"Validazione fallita: {e}")


# ------------------------------------------------------------------------------
# ESERCIZIO 4.18: assert
# ------------------------------------------------------------------------------

def calculate_average(numbers):
    """Calcola media con precondizioni."""
    assert len(numbers) > 0, "Lista non può essere vuota"
    assert all(isinstance(n, (int, float)) for n in numbers), "Solo numeri"
    return sum(numbers) / len(numbers)

# assert viene disabilitato con python -O


# ------------------------------------------------------------------------------
# ESERCIZIO 4.19: Eccezioni Comuni
# ------------------------------------------------------------------------------
"""
Eccezioni più comuni che appaiono nell'esame:

1. ValueError - valore inappropriato
   int("abc")

2. TypeError - tipo inappropriato
   "2" + 2

3. IndexError - indice fuori range
   [1,2,3][10]

4. KeyError - chiave non esiste
   {}["key"]

5. ZeroDivisionError - divisione per zero
   1/0

6. AttributeError - attributo non esiste
   "string".foo()

7. NameError - nome non definito
   print(undefined_var)

8. FileNotFoundError - file non esiste
   open("nonexistent.txt")
"""


# ------------------------------------------------------------------------------
# ESERCIZIO 4.20-4.25: Esercizi Pratici
# ------------------------------------------------------------------------------

# 4.20: Funzione che restituisce funzione
def multiplier(n):
    """Factory function."""
    def multiply(x):
        return x * n
    return multiply

double = multiplier(2)
triple = multiplier(3)
print(double(5))  # 10
print(triple(5))  # 15

# 4.21: Decoratore semplice (preview PCAP)
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Returned {result}")
        return result
    return wrapper

@logger
def add(a, b):
    return a + b

add(2, 3)

# 4.22: Lambda
square = lambda x: x ** 2
print(square(5))  # 25

# Con map
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(squared)

# Con filter
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)

# 4.23: Validatore con eccezioni
def validate_email(email):
    if not isinstance(email, str):
        raise TypeError("Email deve essere stringa")
    if "@" not in email:
        raise ValueError("Email deve contenere @")
    if not email.endswith((".com", ".org", ".net", ".it")):
        raise ValueError("Dominio non valido")
    return True

# 4.24: Ricorsione - conteggio elementi
def count_items(nested_list):
    """Conta tutti gli elementi in lista annidata."""
    count = 0
    for item in nested_list:
        if isinstance(item, list):
            count += count_items(item)
        else:
            count += 1
    return count

assert count_items([1, [2, 3, [4, 5]], 6]) == 6

# 4.25: Gestione robusta input
def get_positive_int(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value <= 0:
                raise ValueError("Deve essere positivo")
            return value
        except ValueError as e:
            print(f"Errore: {e}. Riprova.")

print("✓ Esercizi 4.20-4.25 completati")


# ══════════════════════════════════════════════════════════════════════════════
# QUIZ FINALE - SIMULAZIONE ESAME PCEP
# ══════════════════════════════════════════════════════════════════════════════

"""
PCEP MOCK EXAM - 30 domande, 40 minuti

Rispondi senza eseguire il codice!
"""

QUIZ = """
Q1. Qual è il risultato di: print(2 ** 3 ** 2)?
A) 64
B) 512
C) 262144
D) 6561

Q2. Cosa restituisce: type(1.0) == type(1)?
A) True
B) False
C) TypeError
D) None

Q3. Qual è il valore di x dopo: x = 5; x //= 2?
A) 2.5
B) 2
C) 3
D) 2.0

Q4. Cosa stampa: print([1,2,3][10:])?
A) IndexError
B) []
C) [1,2,3]
D) None

Q5. Quale NON è un modo valido di creare un dizionario?
A) {}
B) dict()
C) {1, 2, 3}
D) dict(a=1)

Q6. Cosa restituisce: "hello".find("x")?
A) False
B) -1
C) None
D) IndexError

Q7. Qual è il risultato di: bool([0])?
A) True
B) False
C) 0
D) None

Q8. Cosa stampa questo codice?
x = [1, 2, 3]
y = x
y.append(4)
print(len(x))

A) 3
B) 4
C) Error
D) None

Q9. Quale keyword rende una variabile globale?
A) var
B) global
C) public
D) extern

Q10. Cosa restituisce: range(5, 0)?
A) [5, 4, 3, 2, 1]
B) range(5, 0)
C) []
D) Error

Q11. Qual è il risultato di: -7 % 5?
A) -2
B) 2
C) 3
D) -3

Q12. Come si crea una tupla con un solo elemento?
A) (1)
B) (1,)
C) tuple(1)
D) [1]

Q13. Cosa fa l'istruzione 'pass'?
A) Termina il loop
B) Salta all'iterazione successiva
C) Non fa nulla
D) Solleva un'eccezione

Q14. Qual è l'output di: print(10 / 3)?
A) 3
B) 3.333...
C) 3.0
D) Error

Q15. Cosa restituisce: [1,2,3].pop()?
A) 1
B) 3
C) [1,2]
D) None

Q16. Qual è il risultato di: "abc" * 2 + "d"?
A) "abcabc d"
B) "abcabcd"
C) Error
D) "abc2d"

Q17. Cosa stampa: print("a" < "b" < "c")?
A) True
B) False
C) Error
D) None

Q18. Qual è il valore di: len({1: 'a', 2: 'b', 1: 'c'})?
A) 2
B) 3
C) 4
D) Error

Q19. Cosa fa 'break' in un loop annidato?
A) Esce da tutti i loop
B) Esce solo dal loop più interno
C) Salta all'iterazione successiva
D) Termina il programma

Q20. Qual è il risultato di: print(type(None))?
A) <class 'none'>
B) <class 'NoneType'>
C) NoneType
D) None

Q21. Cosa restituisce: "hello"[1:-1]?
A) "ell"
B) "hello"
C) "llo"
D) "hel"

Q22. Quale eccezione solleva: int("abc")?
A) TypeError
B) ValueError
C) SyntaxError
D) RuntimeError

Q23. Cosa stampa: print(1, 2, 3, sep="-")?
A) 1 2 3
B) 1-2-3
C) (1, 2, 3)
D) Error

Q24. Qual è il risultato di: True + True + False?
A) True
B) 2
C) 1
D) Error

Q25. Cosa restituisce: dict.get({"a": 1}, "b", 0)?
A) KeyError
B) None
C) 0
D) 1

Q26. Qual è l'output di: print(not not True)?
A) True
B) False
C) not True
D) Error

Q27. Cosa fa: x, y = y, x?
A) Errore
B) Assegna y a x
C) Scambia i valori
D) Crea una tupla

Q28. Quale slice restituisce una lista invertita?
A) lst[::-1]
B) lst[-1::-1]
C) lst[:-1:-1]
D) A e B

Q29. Cosa restituisce: {1, 2, 3} & {2, 3, 4}?
A) {1, 2, 3, 4}
B) {2, 3}
C) {1, 4}
D) Error

Q30. Qual è il risultato di: 0.1 + 0.2 == 0.3?
A) True
B) False
C) Error
D) None
"""

ANSWERS = """
RISPOSTE:
Q1: B (512) - ** associa a destra: 2^(3^2) = 2^9 = 512
Q2: B - type(1.0) è float, type(1) è int
Q3: B (2) - floor division di 5//2
Q4: B ([]) - slicing oltre i limiti non da errore
Q5: C - {1,2,3} crea un SET, non un dict
Q6: B (-1) - find() restituisce -1 se non trova
Q7: A (True) - lista non vuota è truthy
Q8: B (4) - y e x sono lo stesso oggetto
Q9: B (global)
Q10: B/C - range(5,0) è vuoto (nessun step negativo)
Q11: C (3) - Python: resto ha segno del divisore
Q12: B ((1,)) - virgola necessaria per tupla singola
Q13: C - pass è un no-op
Q14: B (3.333...) - true division restituisce float
Q15: B (3) - pop() senza indice rimuove ultimo
Q16: B ("abcabcd")
Q17: A (True) - chained comparison
Q18: A (2) - chiave duplicata sovrascritta
Q19: B - break esce solo dal loop più interno
Q20: B (<class 'NoneType'>)
Q21: A ("ell") - indici 1,2,3
Q22: B (ValueError)
Q23: B (1-2-3)
Q24: B (2) - True=1, False=0
Q25: C (0) - get con default
Q26: A (True)
Q27: C - swap idiomatico Python
Q28: D - entrambi A e B invertono
Q29: B ({2,3}) - intersezione
Q30: B (False) - floating point precision!
"""

print(QUIZ)
print(ANSWERS)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("PCEP-30-02 CERTIFICATION EXERCISES")
    print("═" * 70)
    print("""
    Hai completato 100 esercizi che coprono l'intero syllabus PCEP!
    
    PROSSIMI PASSI:
    1. Ripeti gli esercizi sbagliati
    2. Fai practice tests su Udemy/OpenEDG
    3. Prenota l'esame quando raggiungi 80%+ nei mock
    
    Tempo medio preparazione: 4-6 settimane con 2h/giorno
    
    BUONA FORTUNA! 🍀
    """)
