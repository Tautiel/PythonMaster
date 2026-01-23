"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PYTHON ESSENTIALS 1 - MODULE 1                            ║
║           Introduction to Python and Computer Programming                    ║
║                                                                              ║
║                    Allineato al Syllabus PCEP-30-02                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

STRUTTURA MODULO:
├── Section 1.1: Introduction to Programming
├── Section 1.2: Introduction to Python  
├── Section 1.3: Your First Program
└── Quiz Modulo 1

TEMPO STIMATO: 2-3 ore
OBIETTIVO: Capire i fondamenti della programmazione e l'ecosistema Python

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1.1: INTRODUCTION TO PROGRAMMING
# ══════════════════════════════════════════════════════════════════════════════
"""
CONCETTI CHIAVE:
- Come funziona un computer program
- Linguaggio naturale vs linguaggio di programmazione
- Compilation vs Interpretation
- Lexis, Syntax, Semantics

───────────────────────────────────────────────────────────────────────────────
1.1.1 - HOW DOES A COMPUTER PROGRAM WORK?
───────────────────────────────────────────────────────────────────────────────

Un computer esegue operazioni molto semplici, ma MOLTO velocemente.
Il processore capisce solo il LINGUAGGIO MACCHINA (sequenze di 0 e 1).

Problema: scrivere in linguaggio macchina è praticamente impossibile per un umano.
Soluzione: LINGUAGGI DI PROGRAMMAZIONE (come Python)

Il programma scritto in linguaggio di programmazione si chiama SOURCE CODE.
Il file che contiene il source code si chiama SOURCE FILE.

───────────────────────────────────────────────────────────────────────────────
1.1.2 - COMPILATION vs INTERPRETATION
───────────────────────────────────────────────────────────────────────────────

Due modi per convertire source code in linguaggio macchina:

COMPILATION (es. C, C++, Rust):
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Source Code  │ ──► │   Compiler   │ ──► │  Executable  │
│   (.c file)  │     │              │     │  (.exe file) │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
                                          [Runs directly]

Pro: Veloce esecuzione, codice distribuibile senza sorgente
Contro: Compilazione lenta, dipendente dalla piattaforma


INTERPRETATION (es. Python, JavaScript):
┌──────────────┐     ┌──────────────┐
│ Source Code  │ ──► │ Interpreter  │ ──► [Executes line by line]
│   (.py file) │     │   (Python)   │
└──────────────┘     └──────────────┘

Pro: Cross-platform, sviluppo rapido, facile debug
Contro: Esecuzione più lenta, richiede interprete installato


PYTHON È UN IBRIDO:
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Source Code  │ ──► │   Compile    │ ──► │   Bytecode   │
│   (.py file) │     │  to bytecode │     │  (.pyc file) │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │     PVM      │
                                          │ Python VM    │
                                          └──────────────┘
                                                  │
                                                  ▼
                                          [Executes bytecode]

Python compila il source code in BYTECODE, poi la PVM (Python Virtual Machine)
interpreta il bytecode. Questo è invisibile all'utente.

───────────────────────────────────────────────────────────────────────────────
1.1.3 - LEXIS, SYNTAX, SEMANTICS
───────────────────────────────────────────────────────────────────────────────

Ogni linguaggio (naturale o di programmazione) ha tre livelli:

LEXIS (Lessico):
- L'insieme di parole/simboli riconosciuti dal linguaggio
- In Python: keywords (if, for, while...), operatori (+, -, *...), literals (42, "hello"...)
- Errore lessicale: usare un simbolo non riconosciuto

SYNTAX (Sintassi):
- Le regole per combinare le parole in frasi valide
- In Python: come strutturare if statement, funzioni, etc.
- Errore sintattico: SyntaxError

SEMANTICS (Semantica):
- Il SIGNIFICATO delle frasi
- Sintassi corretta NON significa semantica corretta!

Esempio:
    "Il gatto mangia il topo"  → Lexis ✓, Syntax ✓, Semantics ✓
    "Topo il mangia gatto il"  → Lexis ✓, Syntax ✗
    "Il sasso mangia il tempo" → Lexis ✓, Syntax ✓, Semantics ?

In Python:
    print("Hello")      # Lexis ✓, Syntax ✓, Semantics ✓
    print "Hello"       # Lexis ✓, Syntax ✗ (Python 3)
    print(Hello)        # Lexis ✓, Syntax ✓, Semantics: NameError (Hello non definito)
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1.2: INTRODUCTION TO PYTHON
# ══════════════════════════════════════════════════════════════════════════════
"""
CONCETTI CHIAVE:
- Storia di Python
- Python 2 vs Python 3
- Implementazioni di Python
- Dove si usa Python

───────────────────────────────────────────────────────────────────────────────
1.2.1 - STORIA DI PYTHON
───────────────────────────────────────────────────────────────────────────────

- Creato da Guido van Rossum (Olanda)
- Prima versione: 1991
- Nome: dai Monty Python (comedy group britannico)
- Filosofia: "Readability counts" - codice leggibile

───────────────────────────────────────────────────────────────────────────────
1.2.2 - PYTHON 2 vs PYTHON 3
───────────────────────────────────────────────────────────────────────────────

Python 2:
- Ultima versione: 2.7 (2010)
- End of Life: 1 Gennaio 2020
- NON usare per nuovi progetti!

Python 3:
- Prima versione: 2008
- Versione corrente: 3.12+ (2024)
- NON retrocompatibile con Python 2

Differenze principali (per l'esame):

┌─────────────────────┬─────────────────────┬─────────────────────┐
│      Feature        │      Python 2       │      Python 3       │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ print               │ print "hello"       │ print("hello")      │
│ Division            │ 5/2 = 2             │ 5/2 = 2.5           │
│ Integer division    │ 5/2 = 2             │ 5//2 = 2            │
│ input()             │ raw_input()         │ input()             │
│ Unicode             │ u"string"           │ "string" (default)  │
└─────────────────────┴─────────────────────┴─────────────────────┘

L'ESAME PCEP USA PYTHON 3.x

───────────────────────────────────────────────────────────────────────────────
1.2.3 - IMPLEMENTAZIONI DI PYTHON
───────────────────────────────────────────────────────────────────────────────

CPython:
- Implementazione STANDARD e di riferimento
- Scritto in C
- Quando dici "Python", intendi CPython
- È quello che usi normalmente

Cython:
- Python che compila in C per performance
- NON è CPython!

Jython:
- Python implementato in Java
- Gira sulla JVM (Java Virtual Machine)
- Può usare librerie Java

PyPy:
- Python implementato in Python (subset chiamato RPython)
- Usa JIT (Just-In-Time) compilation
- Più veloce di CPython per certi task

MicroPython / CircuitPython:
- Python per microcontrollori
- Usato in IoT, embedded systems

L'ESAME PCEP SI RIFERISCE A CPYTHON

───────────────────────────────────────────────────────────────────────────────
1.2.4 - DOVE SI USA PYTHON
───────────────────────────────────────────────────────────────────────────────

- Web Development (Django, Flask, FastAPI)
- Data Science / Machine Learning (NumPy, Pandas, TensorFlow)
- Automation / Scripting
- Scientific Computing
- Game Development
- Desktop Applications
- DevOps / System Administration
- Trading Bots (ccxt, backtrader) ← IL TUO OBIETTIVO!
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1.3: YOUR FIRST PROGRAM
# ══════════════════════════════════════════════════════════════════════════════
"""
CONCETTI CHIAVE:
- La funzione print()
- Stringhe e quotes
- Escape characters
- Esecuzione di un programma

───────────────────────────────────────────────────────────────────────────────
1.3.1 - LA FUNZIONE print()
───────────────────────────────────────────────────────────────────────────────

print() è una FUNZIONE BUILT-IN che stampa output sulla console.

Anatomia di una chiamata a funzione:

    print("Hello, World!")
    │     │             │
    │     └─────────────┴── argomento (quello che vuoi stampare)
    └── nome funzione

Regole:
- Il nome funzione NON può avere spazi
- Le parentesi sono OBBLIGATORIE (anche senza argomenti)
- Gli argomenti vanno DENTRO le parentesi
"""

# Esempi print():
print("Hello, World!")          # Stringa con doppi apici
print('Hello, World!')          # Stringa con singoli apici (equivalente)
print()                         # Stampa una riga vuota
print("Hello", "World")         # Due argomenti, separati da spazio


"""
───────────────────────────────────────────────────────────────────────────────
1.3.2 - STRINGHE: QUOTES E APOSTROFI
───────────────────────────────────────────────────────────────────────────────

Puoi usare singoli (') o doppi (") apici, ma devi essere CONSISTENTE:
"""

print("Hello")      # ✓ OK
print('Hello')      # ✓ OK
# print("Hello')    # ✗ ERRORE: apici misti

# Se la stringa contiene un apostrofo:
print("It's a test")            # Usa doppi apici fuori
print('It\'s a test')           # Oppure escape con \

# Se la stringa contiene doppi apici:
print('He said "Hello"')        # Usa singoli apici fuori
print("He said \"Hello\"")      # Oppure escape con \


"""
───────────────────────────────────────────────────────────────────────────────
1.3.3 - ESCAPE CHARACTERS
───────────────────────────────────────────────────────────────────────────────

Il backslash (\) introduce sequenze speciali:

    \n  → newline (vai a capo)
    \t  → tab
    \\  → backslash letterale
    \'  → apostrofo
    \"  → doppio apice
"""

print("Line1\nLine2")           # Stampa su due righe
print("Col1\tCol2")             # Tab tra le colonne
print("Path: C:\\Users\\Name")  # Backslash letterali
print("She said: \"Hi!\"")      # Doppi apici nella stringa


"""
───────────────────────────────────────────────────────────────────────────────
1.3.4 - PARAMETRI DI print(): sep E end
───────────────────────────────────────────────────────────────────────────────

print() ha due parametri keyword importanti:

    sep  → separatore tra argomenti (default: " " spazio)
    end  → cosa stampare alla fine (default: "\n" newline)
"""

print("a", "b", "c")                    # Output: a b c
print("a", "b", "c", sep="-")           # Output: a-b-c
print("a", "b", "c", sep="")            # Output: abc
print("a", "b", "c", sep="***")         # Output: a***b***c

print("Hello", end=" ")
print("World")                          # Output: Hello World (stessa riga)

print("A", end="")
print("B", end="")
print("C")                              # Output: ABC


"""
───────────────────────────────────────────────────────────────────────────────
1.3.5 - ERRORI COMUNI
───────────────────────────────────────────────────────────────────────────────
"""

# SyntaxError: parentesi mancante
# print "Hello"                 # Python 2 style, ERRORE in Python 3

# SyntaxError: apici non chiusi
# print("Hello)

# SyntaxError: apici misti
# print("Hello')

# NameError: variabile non definita
# print(Hello)                  # Hello senza apici → Python cerca una variabile



# ══════════════════════════════════════════════════════════════════════════════
#                              QUIZ MODULE 1
# ══════════════════════════════════════════════════════════════════════════════
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                         QUIZ MODULE 1                                      ║
║                                                                            ║
║  ISTRUZIONI:                                                               ║
║  1. Leggi ogni domanda                                                     ║
║  2. Scrivi la tua risposta su carta/notes PRIMA di verificare             ║
║  3. Solo DOPO controlla la risposta                                        ║
║  4. Se sbagli: FERMATI e capisci il perché                                ║
║                                                                            ║
║  Target: 8/10 per passare                                                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

# ──────────────────────────────────────────────────────────────────────────────
# QUIZ 1.1 - Compilation vs Interpretation
# ──────────────────────────────────────────────────────────────────────────────
"""
Q1: Quale affermazione su Python è CORRETTA?

A) Python è un linguaggio puramente compilato
B) Python è un linguaggio puramente interpretato
C) Python compila il source code in bytecode, poi la PVM esegue il bytecode
D) Python compila direttamente in codice macchina

Tua risposta: ___
"""
# RISPOSTA: C
# Python prima compila in bytecode (.pyc), poi la Python Virtual Machine
# interpreta il bytecode. È un approccio ibrido.


# ──────────────────────────────────────────────────────────────────────────────
# QUIZ 1.2 - Implementazioni
# ──────────────────────────────────────────────────────────────────────────────
"""
Q2: Qual è l'implementazione STANDARD di Python?

A) PyPy
B) Jython
C) CPython
D) Cython

Tua risposta: ___
"""
# RISPOSTA: C
# CPython è l'implementazione di riferimento, scritta in C.
# È quella che scarichi da python.org


# ──────────────────────────────────────────────────────────────────────────────
# QUIZ 1.3 - Python 2 vs 3
# ──────────────────────────────────────────────────────────────────────────────
"""
Q3: Quale istruzione è valida SOLO in Python 2?

A) print("Hello")
B) print "Hello"
C) print('Hello')
D) print()

Tua risposta: ___
"""
# RISPOSTA: B
# In Python 2, print era uno statement: print "Hello"
# In Python 3, print è una funzione: print("Hello")


# ──────────────────────────────────────────────────────────────────────────────
# QUIZ 1.4 - Escape Characters
# ──────────────────────────────────────────────────────────────────────────────
"""
Q4: Cosa stampa questo codice?

print("Hello\nWorld")

A) Hello\nWorld
B) HelloWorld
C) Hello
   World
D) Hello World

Tua risposta: ___
"""
# RISPOSTA: C
# \n è il carattere newline, quindi va a capo


# ──────────────────────────────────────────────────────────────────────────────
# QUIZ 1.5 - sep parameter
# ──────────────────────────────────────────────────────────────────────────────
"""
Q5: Cosa stampa questo codice?

print("a", "b", "c", sep="*")

A) a b c
B) a*b*c
C) abc
D) a * b * c

Tua risposta: ___
"""
# RISPOSTA: B
# sep="*" sostituisce lo spazio di default con asterisco


# ──────────────────────────────────────────────────────────────────────────────
# QUIZ 1.6 - end parameter
# ──────────────────────────────────────────────────────────────────────────────
"""
Q6: Cosa stampa questo codice?

print("Hello", end=" ")
print("World")

A) Hello
   World
B) HelloWorld
C) Hello World
D) Hello
    World

Tua risposta: ___
"""
# RISPOSTA: C
# end=" " sostituisce il newline con uno spazio, quindi restano sulla stessa riga


# ──────────────────────────────────────────────────────────────────────────────
# QUIZ 1.7 - Combinazione sep e end
# ──────────────────────────────────────────────────────────────────────────────
"""
Q7: Cosa stampa questo codice?

print("a", "b", sep="-", end="!")
print("c")

A) a-b!c
B) a-b!
   c
C) a-b-c!
D) a b!c

Tua risposta: ___
"""
# RISPOSTA: A
# Prima print: "a-b!" (sep="-", end="!")
# Seconda print: "c" (sulla stessa riga perché end non era \n)
# Risultato: a-b!c


# ──────────────────────────────────────────────────────────────────────────────
# QUIZ 1.8 - Escape backslash
# ──────────────────────────────────────────────────────────────────────────────
"""
Q8: Cosa stampa questo codice?

print("C:\\Users\\Name")

A) C:\Users\Name
B) C:\\Users\\Name
C) C:UsersName
D) Error

Tua risposta: ___
"""
# RISPOSTA: A
# \\ è l'escape per un singolo backslash letterale


# ──────────────────────────────────────────────────────────────────────────────
# QUIZ 1.9 - Errori
# ──────────────────────────────────────────────────────────────────────────────
"""
Q9: Quale codice produce un ERRORE?

A) print("It's OK")
B) print('It\'s OK')
C) print("Hello", "World")
D) print("Hello' + 'World")

Tua risposta: ___
"""
# RISPOSTA: D
# print("Hello' + 'World") - gli apici non sono bilanciati correttamente
# La stringa inizia con " ma trova ' all'interno in modo problematico
# Nota: in realtà D è sintatticamente corretto ma stampa: Hello' + 'World
# Nessuno di questi produce errore. Domanda tricky - tutte sono valide!

# CORREZIONE: Tutte le opzioni sono valide!
# A) "It's OK" → stringa con apostrofo dentro doppi apici ✓
# B) 'It\'s OK' → apostrofo escaped ✓
# C) Due argomenti ✓
# D) Stringa letterale "Hello' + 'World" ✓


# ──────────────────────────────────────────────────────────────────────────────
# QUIZ 1.10 - Lexis, Syntax, Semantics
# ──────────────────────────────────────────────────────────────────────────────
"""
Q10: Il codice seguente ha un errore. Di che tipo?

print(Hello)

A) Errore lessicale (lexical error)
B) Errore sintattico (syntax error)
C) Errore semantico (semantic error) / NameError
D) Nessun errore

Tua risposta: ___
"""
# RISPOSTA: C
# La sintassi è corretta (una funzione con un argomento)
# Ma Hello senza apici è interpretato come una variabile
# che non esiste → NameError (errore semantico/runtime)


# ══════════════════════════════════════════════════════════════════════════════
#                              LAB MODULE 1
# ══════════════════════════════════════════════════════════════════════════════
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                          LAB MODULE 1                                       ║
║                                                                            ║
║  Esercizi pratici - scrivi il codice tu stesso                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

# ──────────────────────────────────────────────────────────────────────────────
# LAB 1.1 - Hello World
# ──────────────────────────────────────────────────────────────────────────────
"""
Scrivi un programma che stampi:
Hello, World!

Il tuo codice:
"""
# SOLUZIONE:
# print("Hello, World!")


# ──────────────────────────────────────────────────────────────────────────────
# LAB 1.2 - Multiline
# ──────────────────────────────────────────────────────────────────────────────
"""
Usando UNA SOLA istruzione print(), stampa:

Line 1
Line 2
Line 3

Il tuo codice:
"""
# SOLUZIONE:
# print("Line 1\nLine 2\nLine 3")


# ──────────────────────────────────────────────────────────────────────────────
# LAB 1.3 - Formatting output
# ──────────────────────────────────────────────────────────────────────────────
"""
Usando print() con sep, stampa:

2024-01-15

Partendo da tre variabili separate per anno, mese, giorno.

Il tuo codice:
"""
# SOLUZIONE:
# print("2024", "01", "15", sep="-")
# oppure:
# year, month, day = "2024", "01", "15"
# print(year, month, day, sep="-")


# ──────────────────────────────────────────────────────────────────────────────
# LAB 1.4 - Arrow pattern
# ──────────────────────────────────────────────────────────────────────────────
"""
Stampa questa freccia usando escape characters:

    *
   * *
  *   *
 *     *
***   ***
  *   *
  *   *
  *****

Il tuo codice:
"""
# SOLUZIONE:
# print("    *")
# print("   * *")
# print("  *   *")
# print(" *     *")
# print("***   ***")
# print("  *   *")
# print("  *   *")
# print("  *****")


# ══════════════════════════════════════════════════════════════════════════════
#                         CHECKLIST MODULO 1
# ══════════════════════════════════════════════════════════════════════════════
"""
Prima di procedere al Modulo 2, assicurati di saper rispondere a:

[ ] Cos'è la differenza tra compilation e interpretation?
[ ] Cos'è il bytecode Python e cos'è la PVM?
[ ] Qual è l'implementazione standard di Python?
[ ] Quali sono le differenze principali tra Python 2 e 3?
[ ] Come funziona la funzione print()?
[ ] Cosa sono sep e end in print()?
[ ] Quali sono i principali escape characters?
[ ] Qual è la differenza tra errore sintattico e semantico?

Se hai risposto correttamente a 8/10 quiz, procedi al Modulo 2!
"""

if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("PE1 MODULE 1 COMPLETATO!")
    print("═" * 70)
    print("""
    Hai completato il Module 1: Introduction to Python
    
    Prossimo: Module 2 - Data Types, Variables, Operators, I/O
    
    Ricorda: l'obiettivo NON è solo scrivere codice,
    ma CAPIRE come Python funziona internamente!
    """)
