"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PYTHON ESSENTIALS 1 - MODULE 3                            ║
║              Boolean Values, Conditionals, Loops, Lists                      ║
║                                                                              ║
║                     Allineato al Syllabus PCEP-30-02                         ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCEP-30-02 Exam Blocks coperti:
- Block 2: Control Flow – Conditional Blocks and Loops (29%)
- Block 3: Data Collections – Tuples, Dictionaries, Lists, Strings (25% parziale)

STRUTTURA MODULO:
├── Section 3.1: Boolean and Comparison Operators
├── Section 3.2: Conditional Statements (if/elif/else)
├── Section 3.3: Loops (while, for, range)
├── Section 3.4: Lists - Basics
├── Section 3.5: Sorting (Bubble Sort)
├── Section 3.6: Lists - Operations and Slicing
├── Section 3.7: Nested Lists (2D)
├── Labs (25 esercizi)
└── Module 3 Quiz (40 domande)

TEMPO STIMATO: 10-12 ore (modulo più lungo!)

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.1: BOOLEAN AND COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.1 TEORIA: BOOLEAN E CONFRONTI                           │
└──────────────────────────────────────────────────────────────────────────────┘

OPERATORI DI CONFRONTO:
───────────────────────
"""
print(5 == 5)    # True  (uguale)
print(5 != 3)    # True  (diverso)
print(5 > 3)     # True  (maggiore)
print(5 < 3)     # False (minore)
print(5 >= 5)    # True  (maggiore o uguale)
print(5 <= 4)    # False (minore o uguale)


"""
OPERATORI LOGICI - PRECEDENZA: not > and > or
─────────────────────────────────────────────
"""
# not (negazione)
print(not True)   # False
print(not False)  # True
print(not 0)      # True (0 è falsy)
print(not 1)      # False (1 è truthy)

# and (AND logico)
print(True and True)   # True
print(True and False)  # False
print(False and True)  # False
print(False and False) # False

# or (OR logico)
print(True or True)    # True
print(True or False)   # True
print(False or True)   # True
print(False or False)  # False


"""
PRECEDENZA: not > and > or
──────────────────────────
CRITICO PER L'ESAME!
"""
# Esempio: True or False and False
# Ordine: (False and False) = False, poi True or False = True
print(True or False and False)  # True

# Esempio: not True or True and not False
# Ordine: not True = False, not False = True
#         True and True = True
#         False or True = True
print(not True or True and not False)  # True


"""
SHORT-CIRCUIT EVALUATION:
─────────────────────────
Python smette di valutare appena conosce il risultato!
"""
# and: se il primo è False, non valuta il secondo
print(False and print("Non eseguito"))  # False, print non eseguito

# or: se il primo è True, non valuta il secondo
print(True or print("Non eseguito"))    # True, print non eseguito


"""
VALORI TRUTHY E FALSY:
──────────────────────
In contesto booleano, questi valori sono FALSY (equivalenti a False):
- False
- None
- 0 (int), 0.0 (float)
- "" (stringa vuota)
- [] (lista vuota)
- {} (dict vuoto)
- () (tuple vuoto)
- set()

TUTTO IL RESTO è TRUTHY!
"""
print(bool(0))       # False
print(bool(1))       # True
print(bool(-1))      # True (qualsiasi non-zero!)
print(bool(""))      # False
print(bool(" "))     # True (contiene uno spazio!)
print(bool([]))      # False
print(bool([0]))     # True (contiene qualcosa, anche se è 0!)


"""
and/or RESTITUISCONO VALORI, NON SEMPRE bool!
─────────────────────────────────────────────
CRITICO PER L'ESAME!

and: restituisce il primo valore falsy, o l'ultimo se tutti truthy
or:  restituisce il primo valore truthy, o l'ultimo se tutti falsy
"""
print(5 and 3)       # 3 (entrambi truthy, restituisce ultimo)
print(0 and 3)       # 0 (primo falsy, restituisce subito)
print(5 or 3)        # 5 (primo truthy, restituisce subito)
print(0 or 3)        # 3 (primo falsy, continua e restituisce secondo)
print(0 or "" or []) # [] (tutti falsy, restituisce ultimo)

# Uso pratico: default values
name = "" or "Anonymous"  # "Anonymous"


"""
CHAINED COMPARISONS:
────────────────────
Python permette confronti concatenati eleganti
"""
x = 5
print(1 < x < 10)       # True (equivale a 1 < x and x < 10)
print(1 < x < 3)        # False
print(1 < 2 < 3 < 4)    # True
print(1 < 2 > 0)        # True (1 < 2 and 2 > 0)


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 3.1 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_3_1 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 3.1.1 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
print(True or False and False) stampa:

A) True
B) False
C) Error
D) None

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.1.2 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
print(not True or True and not False) stampa:

A) True
B) False
C) Error
D) None

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.1.3 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
print(5 and 3) stampa:

A) True
B) 5
C) 3
D) False

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.1.4 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
print(0 and 3) stampa:

A) True
B) 0
C) 3
D) False

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.1.5 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
print(5 or 3) stampa:

A) True
B) 5
C) 3
D) 8

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.1.6 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
print(0 or 3) stampa:

A) True
B) 0
C) 3
D) False

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.1.7
══════════════════════════════════════════════════════════════════════════════
print("" or "default") stampa:

A) ""
B) "default"
C) True
D) False

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.1.8
══════════════════════════════════════════════════════════════════════════════
print(bool([]), bool([0])) stampa:

A) False False
B) True True
C) False True
D) True False

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.1.9
══════════════════════════════════════════════════════════════════════════════
print(1 < 2 < 3) stampa:

A) True
B) False
C) Error
D) 1

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.1.10
══════════════════════════════════════════════════════════════════════════════
print(3 > 2 > 2) stampa:

A) True
B) False
C) Error
D) 3

Tua risposta: ___
"""


RISPOSTE_3_1 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 3.1
══════════════════════════════════════════════════════════════════════════════

3.1.1: A) True
       and ha precedenza su or!
       False and False = False → True or False = True

3.1.2: A) True
       not > and > or
       not True = False, not False = True
       True and True = True
       False or True = True

3.1.3: C) 3
       and restituisce l'ultimo valore se entrambi truthy.
       5 (truthy) → continua → 3 (truthy) → restituisce 3

3.1.4: B) 0
       and restituisce il primo valore falsy.
       0 è falsy → restituisce 0 subito (short-circuit)

3.1.5: B) 5
       or restituisce il primo valore truthy.
       5 è truthy → restituisce 5 subito (short-circuit)

3.1.6: C) 3
       or: primo è falsy, continua al secondo.
       0 è falsy → continua → 3 (truthy o ultimo) → restituisce 3

3.1.7: B) "default"
       "" è falsy, quindi or continua e restituisce "default"

3.1.8: C) False True
       [] è vuota = falsy, [0] contiene qualcosa = truthy

3.1.9: A) True
       Chained: 1 < 2 AND 2 < 3 → True and True = True

3.1.10: B) False
        Chained: 3 > 2 AND 2 > 2 → True and False = False
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.2: CONDITIONAL STATEMENTS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.2 TEORIA: IF/ELIF/ELSE                                  │
└──────────────────────────────────────────────────────────────────────────────┘
"""

# if semplice
x = 10
if x > 5:
    print("x è grande")

# if-else
if x > 5:
    print("grande")
else:
    print("piccolo")

# if-elif-else
score = 75
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"
print(grade)  # C


"""
INDENTAZIONE:
─────────────
Python usa l'indentazione per definire i blocchi!
Standard: 4 spazi (non tab)
"""

# ERRORE: IndentationError
# if True:
# print("errore")  # Manca indentazione!

# Blocchi annidati
if True:
    if True:
        print("nested")


"""
OPERATORE TERNARIO (Conditional Expression):
────────────────────────────────────────────
"""
x = 10
result = "grande" if x > 5 else "piccolo"
print(result)  # grande

# Equivale a:
if x > 5:
    result = "grande"
else:
    result = "piccolo"


"""
pass STATEMENT:
───────────────
Placeholder per blocchi vuoti
"""
if True:
    pass  # Non fa nulla, ma evita errore di sintassi


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 3.2 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_3_2 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 3.2.1
══════════════════════════════════════════════════════════════════════════════
x = 10
result = "A" if x > 5 else "B"
print(result)

Stampa:
A) A
B) B
C) True
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.2.2
══════════════════════════════════════════════════════════════════════════════
x = 5
if x > 10:
    print("A")
elif x > 3:
    print("B")
elif x > 1:
    print("C")
else:
    print("D")

Stampa:
A) A
B) B
C) C
D) D

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.2.3
══════════════════════════════════════════════════════════════════════════════
x = 5
if x > 10:
    print("A")
if x > 3:
    print("B")
if x > 1:
    print("C")

Stampa:
A) A
B) B
C) BC
D) ABC

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.2.4
══════════════════════════════════════════════════════════════════════════════
Qual è il problema con questo codice?

if True:
print("hello")

A) Nessun problema
B) IndentationError
C) SyntaxError
D) NameError

Tua risposta: ___
"""


RISPOSTE_3_2 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 3.2
══════════════════════════════════════════════════════════════════════════════

3.2.1: A) A
       x > 5 è True, quindi restituisce "A"

3.2.2: B) B
       elif: esegue SOLO il primo blocco che matcha.
       x > 3 è True, stampa B e ESCE.

3.2.3: C) BC
       Sono if SEPARATI (non elif), quindi vengono tutti valutati.
       x > 10? No. x > 3? Sì → B. x > 1? Sì → C.

3.2.4: B) IndentationError
       Manca l'indentazione dopo if:
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.3: LOOPS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.3 TEORIA: WHILE e FOR                                   │
└──────────────────────────────────────────────────────────────────────────────┘

WHILE LOOP:
───────────
"""
i = 0
while i < 3:
    print(i)
    i += 1
# Output: 0, 1, 2


"""
FOR LOOP con range():
─────────────────────
range(stop)        → 0, 1, 2, ..., stop-1
range(start, stop) → start, start+1, ..., stop-1
range(start, stop, step) → start, start+step, ...
"""
for i in range(3):
    print(i)  # 0, 1, 2

for i in range(2, 5):
    print(i)  # 2, 3, 4

for i in range(0, 10, 2):
    print(i)  # 0, 2, 4, 6, 8

for i in range(5, 0, -1):
    print(i)  # 5, 4, 3, 2, 1


"""
break, continue, pass:
──────────────────────
"""
# break: esce dal loop
for i in range(10):
    if i == 5:
        break
    print(i)  # 0, 1, 2, 3, 4

# continue: salta all'iterazione successiva
for i in range(5):
    if i == 2:
        continue
    print(i)  # 0, 1, 3, 4

# pass: non fa nulla (placeholder)
for i in range(3):
    pass  # Loop valido che non fa nulla


"""
else CLAUSE nei loop - CRITICO PER ESAME!
─────────────────────────────────────────
else viene eseguito SE il loop termina NORMALMENTE (senza break)
"""
# Con else - loop completa normalmente
for i in range(3):
    print(i)
else:
    print("done")  # Viene eseguito!
# Output: 0, 1, 2, done

# Con break - else NON eseguito
for i in range(5):
    if i == 3:
        break
    print(i)
else:
    print("done")  # NON viene eseguito!
# Output: 0, 1, 2


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 3.3 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_3_3 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 3.3.1
══════════════════════════════════════════════════════════════════════════════
for i in range(3):
    print(i, end=" ")

Stampa:
A) 1 2 3
B) 0 1 2
C) 0 1 2 3
D) 1 2

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.3.2
══════════════════════════════════════════════════════════════════════════════
for i in range(2, 5):
    print(i, end=" ")

Stampa:
A) 2 3 4 5
B) 2 3 4
C) 1 2 3 4
D) 2 3 4 5

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.3.3
══════════════════════════════════════════════════════════════════════════════
for i in range(0, 10, 3):
    print(i, end=" ")

Stampa:
A) 0 3 6 9
B) 0 3 6
C) 3 6 9
D) 0 3 6 9 12

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.3.4 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
for i in range(3):
    print(i, end=" ")
else:
    print("done")

Stampa:
A) 0 1 2
B) 0 1 2 done
C) done
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.3.5 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
for i in range(5):
    if i == 3:
        break
    print(i, end=" ")
else:
    print("done")

Stampa:
A) 0 1 2 done
B) 0 1 2
C) 0 1 2 3 done
D) 0 1 2 3

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.3.6
══════════════════════════════════════════════════════════════════════════════
for i in range(5):
    if i == 2:
        continue
    print(i, end=" ")
else:
    print("done")

Stampa:
A) 0 1 3 4 done
B) 0 1 done
C) 0 1 2 3 4 done
D) 0 1 3 4

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.3.7
══════════════════════════════════════════════════════════════════════════════
i = 0
while i < 5:
    i += 1
    if i == 3:
        continue
    print(i, end=" ")

Stampa:
A) 1 2 4 5
B) 1 2 3 4 5
C) 0 1 2 4 5
D) 1 2 4

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.3.8
══════════════════════════════════════════════════════════════════════════════
for i in range(5, 0, -1):
    print(i, end=" ")

Stampa:
A) 5 4 3 2 1 0
B) 5 4 3 2 1
C) 4 3 2 1 0
D) 1 2 3 4 5

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.3.9
══════════════════════════════════════════════════════════════════════════════
print(list(range(3)))

Stampa:
A) [1, 2, 3]
B) [0, 1, 2]
C) range(0, 3)
D) (0, 1, 2)

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.3.10
══════════════════════════════════════════════════════════════════════════════
print(type(range(5)))

Stampa:
A) <class 'list'>
B) <class 'range'>
C) <class 'tuple'>
D) <class 'iterator'>

Tua risposta: ___
"""


RISPOSTE_3_3 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 3.3
══════════════════════════════════════════════════════════════════════════════

3.3.1: B) 0 1 2
       range(3) = 0, 1, 2

3.3.2: B) 2 3 4
       range(2, 5) = 2, 3, 4 (stop escluso!)

3.3.3: A) 0 3 6 9
       range(0, 10, 3) = 0, 3, 6, 9

3.3.4: B) 0 1 2 done
       Loop completa normalmente → else eseguito

3.3.5: B) 0 1 2
       break interrompe → else NON eseguito!

3.3.6: A) 0 1 3 4 done
       continue salta solo un'iterazione, loop completa → else eseguito

3.3.7: A) 1 2 4 5
       i incrementa PRIMA di continue, quindi 3 viene saltato

3.3.8: B) 5 4 3 2 1
       range(5, 0, -1) = 5, 4, 3, 2, 1 (0 escluso!)

3.3.9: B) [0, 1, 2]
       list() converte range in lista

3.3.10: B) <class 'range'>
        range è un tipo a sé stante, non una lista!
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.4: LISTS BASICS
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.4 TEORIA: LISTE - BASICS                                │
└──────────────────────────────────────────────────────────────────────────────┘
"""

# Creazione
lst = [1, 2, 3, 4, 5]
empty = []
mixed = [1, "hello", 3.14, True]

# Accesso (indexing)
print(lst[0])    # 1 (primo elemento)
print(lst[-1])   # 5 (ultimo elemento)
print(lst[-2])   # 4 (penultimo)

# Modifica
lst[0] = 10
print(lst)  # [10, 2, 3, 4, 5]

# Lunghezza
print(len(lst))  # 5


"""
METODI FONDAMENTALI:
────────────────────
"""
lst = [1, 2, 3]

# append() - aggiunge alla fine
lst.append(4)     # [1, 2, 3, 4]

# insert(index, value) - inserisce alla posizione
lst.insert(0, 0)  # [0, 1, 2, 3, 4]

# remove(value) - rimuove PRIMA occorrenza del valore
lst.remove(2)     # [0, 1, 3, 4]

# pop() - rimuove e restituisce ultimo (o all'indice specificato)
x = lst.pop()     # x = 4, lst = [0, 1, 3]
y = lst.pop(0)    # y = 0, lst = [1, 3]

# del - rimuove per indice
del lst[0]        # lst = [3]


"""
del vs remove vs pop:
─────────────────────
del lst[i]  → rimuove per INDICE, non restituisce nulla
remove(v)   → rimuove per VALORE (prima occorrenza)
pop(i)      → rimuove per INDICE e RESTITUISCE il valore
"""


"""
ITERAZIONE:
───────────
"""
lst = ["a", "b", "c"]

# Per valore
for item in lst:
    print(item)

# Per indice
for i in range(len(lst)):
    print(i, lst[i])

# Con enumerate (indice + valore)
for i, item in enumerate(lst):
    print(i, item)


"""
in e not in:
────────────
"""
lst = [1, 2, 3]
print(1 in lst)      # True
print(5 in lst)      # False
print(5 not in lst)  # True


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 3.4 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_3_4 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 3.4.1
══════════════════════════════════════════════════════════════════════════════
lst = [1, 2, 3, 4, 5]
print(lst[-2])

Stampa:
A) 2
B) 4
C) 5
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.4.2
══════════════════════════════════════════════════════════════════════════════
lst = [1, 2, 3]
lst.append(4)
lst.append(5)
print(len(lst))

Stampa:
A) 3
B) 4
C) 5
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.4.3
══════════════════════════════════════════════════════════════════════════════
lst = [1, 2, 3, 4]
lst.insert(2, 99)
print(lst)

Stampa:
A) [1, 2, 99, 3, 4]
B) [1, 99, 2, 3, 4]
C) [99, 1, 2, 3, 4]
D) [1, 2, 3, 99, 4]

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.4.4
══════════════════════════════════════════════════════════════════════════════
lst = [1, 2, 3, 2, 4]
lst.remove(2)
print(lst)

Stampa:
A) [1, 3, 4]
B) [1, 3, 2, 4]
C) [1, 2, 3, 4]
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.4.5
══════════════════════════════════════════════════════════════════════════════
lst = [1, 2, 3]
x = lst.pop()
print(x, lst)

Stampa:
A) 1 [2, 3]
B) 3 [1, 2]
C) 3 [1, 2, 3]
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.4.6
══════════════════════════════════════════════════════════════════════════════
lst = [1, 2, 3]
del lst[1]
print(lst)

Stampa:
A) [2, 3]
B) [1, 3]
C) [1, 2]
D) Error

Tua risposta: ___
"""


RISPOSTE_3_4 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 3.4
══════════════════════════════════════════════════════════════════════════════

3.4.1: B) 4
       lst[-2] = penultimo elemento

3.4.2: C) 5
       Inizia con 3, append aggiunge 2 elementi → 5

3.4.3: A) [1, 2, 99, 3, 4]
       insert(2, 99) inserisce 99 all'indice 2

3.4.4: B) [1, 3, 2, 4]
       remove() rimuove solo la PRIMA occorrenza di 2

3.4.5: B) 3 [1, 2]
       pop() rimuove e restituisce l'ultimo elemento

3.4.6: B) [1, 3]
       del lst[1] rimuove l'elemento all'indice 1
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 3.6: LISTS - OPERATIONS AND SLICING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                    3.6 TEORIA: SLICING E RIFERIMENTI                         │
└──────────────────────────────────────────────────────────────────────────────┘

SLICING [start:stop:step]:
──────────────────────────
"""
lst = [0, 1, 2, 3, 4, 5]

print(lst[1:4])    # [1, 2, 3] (da 1 a 3, 4 escluso)
print(lst[:3])     # [0, 1, 2] (dall'inizio a 2)
print(lst[3:])     # [3, 4, 5] (da 3 alla fine)
print(lst[::2])    # [0, 2, 4] (ogni 2)
print(lst[::-1])   # [5, 4, 3, 2, 1, 0] (reverse!)
print(lst[4:1:-1]) # [4, 3, 2] (al contrario)


"""
SLICING FUORI RANGE - NON DÀ ERRORE!
────────────────────────────────────
"""
lst = [0, 1, 2]
print(lst[10:])    # [] (nessun errore!)
print(lst[:10])    # [0, 1, 2] (prende tutto ciò che c'è)


"""
RIFERIMENTI VS COPIE - CRITICO PER ESAME!
─────────────────────────────────────────
"""
# RIFERIMENTO (stesso oggetto!)
a = [1, 2, 3]
b = a           # b punta allo STESSO oggetto!
b.append(4)
print(a)        # [1, 2, 3, 4] - a è cambiato!

# COPIA (nuovo oggetto)
a = [1, 2, 3]
b = a[:]        # Shallow copy con slicing
b.append(4)
print(a)        # [1, 2, 3] - a è INTATTO!

# Altri modi per copiare:
b = list(a)     # Costruttore
b = a.copy()    # Metodo copy()


"""
ATTENZIONE: += vs + per liste!
──────────────────────────────
"""
# + crea NUOVA lista
a = [1, 2]
b = a
a = a + [3]     # a è ora una NUOVA lista
print(b)        # [1, 2] - b non cambia!

# += MODIFICA in place
a = [1, 2]
b = a
a += [3]        # Estende a IN PLACE (come extend)
print(b)        # [1, 2, 3] - b vede la modifica!


"""
SHALLOW vs DEEP COPY:
─────────────────────
"""
# Shallow copy: copia solo il primo livello
original = [[1, 2], [3, 4]]
shallow = original[:]
shallow[0][0] = 99
print(original)  # [[99, 2], [3, 4]] - Modificato!

# Per deep copy, usa import copy
import copy
deep = copy.deepcopy(original)


"""
OPERAZIONI SU LISTE:
────────────────────
"""
lst1 = [1, 2, 3]
lst2 = [4, 5, 6]

# Concatenazione
print(lst1 + lst2)    # [1, 2, 3, 4, 5, 6]

# Ripetizione
print([1, 2] * 3)     # [1, 2, 1, 2, 1, 2]

# ATTENZIONE alla ripetizione con riferimenti!
lst = [[0]] * 3       # [0], [0], [0]] - STESSO oggetto!
lst[0][0] = 1
print(lst)            # [[1], [1], [1]] - Tutti modificati!


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 3.6 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_3_6 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 3.6.1
══════════════════════════════════════════════════════════════════════════════
lst = [0, 1, 2, 3, 4]
print(lst[1:4])

Stampa:
A) [0, 1, 2, 3]
B) [1, 2, 3]
C) [1, 2, 3, 4]
D) [2, 3, 4]

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.6.2
══════════════════════════════════════════════════════════════════════════════
lst = [0, 1, 2, 3, 4]
print(lst[::2])

Stampa:
A) [0, 2, 4]
B) [0, 1]
C) [2, 4]
D) [1, 3]

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.6.3
══════════════════════════════════════════════════════════════════════════════
lst = [0, 1, 2, 3, 4]
print(lst[10:])

Stampa:
A) Error (IndexError)
B) []
C) [4]
D) None

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.6.4 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
a = [1, 2, 3]
b = a
b.append(4)
print(a)

Stampa:
A) [1, 2, 3]
B) [1, 2, 3, 4]
C) [4, 1, 2, 3]
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.6.5 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
a = [1, 2, 3]
b = a[:]
b.append(4)
print(a)

Stampa:
A) [1, 2, 3]
B) [1, 2, 3, 4]
C) [4, 1, 2, 3]
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.6.6 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
a = [1, 2, 3]
b = a
a = a + [4]
print(b)

Stampa:
A) [1, 2, 3]
B) [1, 2, 3, 4]
C) [4]
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.6.7 - CRITICO!
══════════════════════════════════════════════════════════════════════════════
a = [1, 2, 3]
b = a
a += [4]
print(b)

Stampa:
A) [1, 2, 3]
B) [1, 2, 3, 4]
C) [4]
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.6.8 - TRAPPOLA!
══════════════════════════════════════════════════════════════════════════════
lst = [[0]] * 3
lst[0][0] = 1
print(lst)

Stampa:
A) [[1], [0], [0]]
B) [[1], [1], [1]]
C) [[0], [0], [1]]
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.6.9
══════════════════════════════════════════════════════════════════════════════
print([1, 2] * 3)

Stampa:
A) [3, 6]
B) [1, 2, 1, 2, 1, 2]
C) [[1, 2], [1, 2], [1, 2]]
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 3.6.10
══════════════════════════════════════════════════════════════════════════════
lst = [0, 1, 2, 3, 4]
print(lst[-2::-2])

Stampa:
A) [3, 1]
B) [4, 2, 0]
C) [3, 1, -1]
D) Error

Tua risposta: ___
"""


RISPOSTE_3_6 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 3.6
══════════════════════════════════════════════════════════════════════════════

3.6.1: B) [1, 2, 3]
       Slicing [1:4] = indici 1, 2, 3 (4 escluso)

3.6.2: A) [0, 2, 4]
       [::2] = ogni secondo elemento partendo da 0

3.6.3: B) []
       Slicing NON dà errore se fuori range, restituisce lista vuota

3.6.4: B) [1, 2, 3, 4]
       b = a crea un RIFERIMENTO. b.append modifica lo stesso oggetto!

3.6.5: A) [1, 2, 3]
       b = a[:] crea una COPIA. b.append non tocca a.

3.6.6: A) [1, 2, 3]
       a = a + [4] crea NUOVA lista. b punta ancora alla vecchia.

3.6.7: B) [1, 2, 3, 4]
       a += [4] modifica IN PLACE! Equivale a a.extend([4]).
       b vede la modifica perché punta allo stesso oggetto.

3.6.8: B) [[1], [1], [1]]
       [[0]] * 3 crea 3 RIFERIMENTI allo stesso [0]!
       Modificare uno li modifica tutti.

3.6.9: B) [1, 2, 1, 2, 1, 2]
       * ripete gli elementi (concatenazione multipla)

3.6.10: A) [3, 1]
        Parte da -2 (indice 3), va all'inizio, step -2
        → 3, 1 (0 non incluso perché step negativo)
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 3 - TEST FINALE
# ══════════════════════════════════════════════════════════════════════════════

MODULE_3_TEST = """
══════════════════════════════════════════════════════════════════════════════
                         MODULE 3 - TEST FINALE
                    40 domande - Target: 70% (28/40)
                     Tempo consigliato: 45 minuti
══════════════════════════════════════════════════════════════════════════════

Q1. print(True or False and False) = ?
    A) True    B) False    C) Error    D) None

Q2. print(5 and 3) = ?
    A) True    B) 5    C) 3    D) False

Q3. print(0 or "" or []) = ?
    A) 0    B) ""    C) []    D) False

Q4. print(1 < 2 < 3) = ?
    A) True    B) False    C) Error    D) 1

Q5. print(bool(" ")) = ?
    A) True    B) False    C) Error    D) None

Q6. x = 5; print("A" if x > 3 else "B") = ?
    A) A    B) B    C) True    D) 5

Q7. for i in range(3): print(i, end="") → output?
    A) 123    B) 012    C) 0123    D) 321

Q8. for i in range(5,2,-1): print(i, end="") → output?
    A) 543    B) 345    C) 234    D) 3210

Q9. for i in range(3): pass; else: print("X") → output?
    A) X    B) (niente)    C) Error    D) XXX

Q10. for i in range(5):
         if i==2: break
     else: print("X") → output?
    A) X    B) (niente)    C) Error    D) 2

Q11. lst=[1,2,3]; print(lst[-1]) = ?
    A) 1    B) 3    C) -1    D) Error

Q12. lst=[1,2,3]; lst.append(4); print(len(lst)) = ?
    A) 3    B) 4    C) 5    D) Error

Q13. lst=[1,2,3]; lst.insert(1,9); print(lst) = ?
    A) [9,1,2,3]    B) [1,9,2,3]    C) [1,2,9,3]    D) [1,2,3,9]

Q14. lst=[1,2,3,2]; lst.remove(2); print(lst) = ?
    A) [1,3]    B) [1,3,2]    C) [1,2,3]    D) Error

Q15. lst=[1,2,3]; x=lst.pop(); print(x,lst) = ?
    A) 1 [2,3]    B) 3 [1,2]    C) 3 [1,2,3]    D) Error

Q16. lst=[0,1,2,3,4]; print(lst[1:4]) = ?
    A) [0,1,2,3]    B) [1,2,3]    C) [1,2,3,4]    D) [2,3,4]

Q17. lst=[0,1,2,3,4]; print(lst[::2]) = ?
    A) [0,2,4]    B) [0,1]    C) [2,4]    D) [1,3]

Q18. lst=[0,1,2]; print(lst[10:]) = ?
    A) Error    B) []    C) [2]    D) None

Q19. a=[1,2]; b=a; b.append(3); print(a) = ?
    A) [1,2]    B) [1,2,3]    C) [3]    D) Error

Q20. a=[1,2]; b=a[:]; b.append(3); print(a) = ?
    A) [1,2]    B) [1,2,3]    C) [3]    D) Error

Q21. a=[1,2]; b=a; a=a+[3]; print(b) = ?
    A) [1,2]    B) [1,2,3]    C) [3]    D) Error

Q22. a=[1,2]; b=a; a+=[3]; print(b) = ?
    A) [1,2]    B) [1,2,3]    C) [3]    D) Error

Q23. lst=[[0]]*3; lst[0][0]=1; print(lst) = ?
    A) [[1],[0],[0]]    B) [[1],[1],[1]]    C) [[0],[0],[1]]    D) Error

Q24. print([1,2]*3) = ?
    A) [3,6]    B) [1,2,1,2,1,2]    C) [[1,2]*3]    D) Error

Q25. print(not not not True) = ?
    A) True    B) False    C) Error    D) None

Q26. print(3 > 2 > 2) = ?
    A) True    B) False    C) Error    D) 2

Q27. i=0; while i<3: print(i,end=""); i+=1 → output?
    A) 123    B) 012    C) 0123    D) Error

Q28. for i in range(5):
         if i==3: continue
         print(i,end="") → output?
    A) 01234    B) 0124    C) 012    D) 34

Q29. print(list(range(2,8,2))) = ?
    A) [2,4,6,8]    B) [2,4,6]    C) [2,3,4,5,6,7]    D) Error

Q30. print(type(range(5))) = ?
    A) list    B) range    C) tuple    D) iterator

Q31. lst=[1,2,3]; print(2 in lst) = ?
    A) True    B) False    C) 1    D) Error

Q32. x=10; y=20; x,y=y,x; print(x,y) = ?
    A) 10 20    B) 20 10    C) Error    D) 20 20

Q33. print("hello"[::-1]) = ?
    A) hello    B) olleh    C) h    D) Error

Q34. lst=[1,2,3,4,5]; print(lst[-3:-1]) = ?
    A) [3,4]    B) [3,4,5]    C) [2,3,4]    D) Error

Q35. print(0 and "hello") = ?
    A) 0    B) "hello"    C) False    D) True

Q36. print("" or "default") = ?
    A) ""    B) "default"    C) True    D) False

Q37. for i in range(0): print(i)
     else: print("X") → output?
    A) X    B) (niente)    C) 0X    D) Error

Q38. lst=[1,2,3]; lst[1:2]=[4,5,6]; print(lst) = ?
    A) [1,4,5,6,3]    B) [1,4,5,6,2,3]    C) [4,5,6,2,3]    D) Error

Q39. print(bool([0])) = ?
    A) True    B) False    C) 0    D) Error

Q40. print(1 or 2 and 3) = ?
    A) 1    B) 3    C) True    D) Error


══════════════════════════════════════════════════════════════════════════════
                              FINE TEST
══════════════════════════════════════════════════════════════════════════════
"""


MODULE_3_TEST_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                    MODULE 3 - RISPOSTE TEST FINALE
══════════════════════════════════════════════════════════════════════════════

Q1:  A) True (and prima di or)
Q2:  C) 3 (and restituisce ultimo se entrambi truthy)
Q3:  C) [] (or restituisce ultimo se tutti falsy)
Q4:  A) True
Q5:  A) True (" " non è vuoto!)
Q6:  A) A
Q7:  B) 012
Q8:  A) 543
Q9:  A) X (loop completa → else eseguito)
Q10: B) (niente - break → no else)
Q11: B) 3
Q12: B) 4
Q13: B) [1,9,2,3]
Q14: B) [1,3,2] (remove prima occorrenza)
Q15: B) 3 [1,2]
Q16: B) [1,2,3]
Q17: A) [0,2,4]
Q18: B) []
Q19: B) [1,2,3] (riferimento)
Q20: A) [1,2] (copia)
Q21: A) [1,2] (+ crea nuova lista)
Q22: B) [1,2,3] (+= modifica in place)
Q23: B) [[1],[1],[1]] (stesso riferimento!)
Q24: B) [1,2,1,2,1,2]
Q25: B) False
Q26: B) False (3>2 True, 2>2 False)
Q27: B) 012
Q28: B) 0124
Q29: B) [2,4,6]
Q30: B) range
Q31: A) True
Q32: B) 20 10
Q33: B) olleh
Q34: A) [3,4]
Q35: A) 0 (short-circuit)
Q36: B) "default"
Q37: A) X (range(0) è vuoto ma completa → else)
Q38: A) [1,4,5,6,3]
Q39: A) True ([0] contiene qualcosa!)
Q40: A) 1 (or short-circuit)

PUNTEGGIO:
36-40: Eccellente!
32-35: Ottimo!
28-31: Buono, target PCEP raggiunto
<28:   Rivedi teoria
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    ESECUZIONE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 78)
    print("PYTHON ESSENTIALS 1 - MODULE 3")
    print("Boolean, Conditionals, Loops, Lists")
    print("=" * 78)
    print("""
    
    Questo è il MODULO PIÙ IMPORTANTE per PCEP!
    Copre Block 2 (29%) e parte di Block 3 (25%)
    
    CONTENUTO:
    ──────────
    - Section 3.1: Boolean e operatori logici (10 quiz)
    - Section 3.2: if/elif/else (4 quiz)
    - Section 3.3: Loops while/for (10 quiz)
    - Section 3.4: Liste basics (6 quiz)
    - Section 3.6: Slicing e riferimenti (10 quiz)
    - Test finale: 40 domande
    
    ARGOMENTI CRITICI:
    ──────────────────
    ⚠️  Precedenza: not > and > or
    ⚠️  and/or restituiscono VALORI, non bool
    ⚠️  else nei loop (eseguito se NO break)
    ⚠️  Riferimenti vs copie di liste
    ⚠️  += vs + per liste
    
    COMANDI:
    ────────
    print(QUIZ_3_1)      # Boolean e operatori
    print(QUIZ_3_3)      # Loops
    print(QUIZ_3_6)      # Slicing e riferimenti
    print(MODULE_3_TEST) # Test finale
    
    """)
    print("\n" + "=" * 78)
    print("MODULE 3 PRONTO!")
    print("Passa a: pe1_m4_functions.py")
    print("=" * 78)
