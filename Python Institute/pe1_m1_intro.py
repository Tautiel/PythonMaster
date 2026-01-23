"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PYTHON ESSENTIALS 1 - MODULE 1                            ║
║           Introduction to Python and Computer Programming                    ║
║                                                                              ║
║                     Allineato al Syllabus PCEP-30-02                         ║
║                         OpenEDG Python Institute                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PCEP-30-02 Exam Block 1 (Partial): Computer Programming and Python Fundamentals
Peso nell'esame: ~18% del totale (questo modulo copre la parte introduttiva)

STRUTTURA MODULO:
├── Section 1.1: Introduction to Programming
├── Section 1.2: Introduction to Python  
├── Section 1.3: Downloading and Installing Python
└── Module 1 Quiz (15 domande)

TEMPO STIMATO: 2-3 ore

METODO DI STUDIO:
1. Leggi ogni sezione di teoria
2. Rispondi ai quiz SENZA eseguire il codice
3. Verifica le risposte
4. Se sbagli, CAPISCI il perché prima di procedere

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.1: INTRODUCTION TO PROGRAMMING
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                         1.1 TEORIA: COME FUNZIONA UN COMPUTER                │
└──────────────────────────────────────────────────────────────────────────────┘

UN COMPUTER È COMPOSTO DA:

1. HARDWARE - La parte fisica
   - CPU (Central Processing Unit): Il "cervello", esegue le istruzioni
   - RAM (Random Access Memory): Memoria veloce ma volatile
   - Storage (HDD/SSD): Memoria permanente
   - I/O Devices: Tastiera, mouse, monitor, etc.

2. SOFTWARE - I programmi
   - Sistema Operativo (OS): Windows, macOS, Linux
   - Applicazioni: Browser, editor, giochi, etc.


IL LINGUAGGIO MACCHINA (Machine Language):
─────────────────────────────────────────
La CPU capisce SOLO sequenze di 0 e 1 (binary).
Esempio: 10110000 01100001 potrebbe significare "carica 97 nel registro"

Questo si chiama "Instruction List" (IL) o Machine Code.

Problema: Scrivere in binario è IMPOSSIBILE per gli umani.
Soluzione: Linguaggi di programmazione ad alto livello.


LINGUAGGI AD ALTO LIVELLO:
──────────────────────────
Permettono di scrivere codice comprensibile agli umani:

    print("Hello")     # Python
    printf("Hello");   # C
    System.out.println("Hello");  # Java

Ma il computer non capisce questo! Serve una TRADUZIONE.


DUE MODI DI TRADURRE: COMPILAZIONE vs INTERPRETAZIONE
─────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│                            COMPILAZIONE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Source Code ──────► COMPILER ──────► Executable                           │
│   (hello.c)           (gcc)            (hello.exe)                          │
│                                                                             │
│   - Traduce TUTTO il codice in una volta                                    │
│   - Crea un file eseguibile                                                 │
│   - L'eseguibile gira SENZA il compilatore                                  │
│   - Errori trovati PRIMA dell'esecuzione                                    │
│   - Generalmente PIÙ VELOCE a runtime                                       │
│   - Esempi: C, C++, Rust, Go                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           INTERPRETAZIONE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Source Code ──────► INTERPRETER ──────► Output                            │
│   (hello.py)          (python)            (Hello)                           │
│                                                                             │
│   - Traduce ed esegue LINEA PER LINEA                                       │
│   - NON crea un file eseguibile                                             │
│   - Serve SEMPRE l'interprete per eseguire                                  │
│   - Errori trovati DURANTE l'esecuzione                                     │
│   - Generalmente PIÙ LENTO a runtime                                        │
│   - Esempi: Python, JavaScript, Ruby                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

NOTA IMPORTANTE SU PYTHON:
──────────────────────────
Python è TECNICAMENTE un linguaggio IBRIDO:
1. Il codice .py viene prima COMPILATO in BYTECODE (.pyc)
2. Il bytecode viene poi INTERPRETATO dalla PVM (Python Virtual Machine)

Ma per semplicità, Python è classificato come linguaggio INTERPRETATO.


VANTAGGI E SVANTAGGI:
─────────────────────

COMPILAZIONE:
✅ Esecuzione veloce
✅ Codice sorgente protetto (distribuisci solo l'eseguibile)
✅ Errori trovati prima dell'esecuzione
❌ Compilazione richiede tempo
❌ Eseguibile specifico per OS/architettura
❌ Ciclo sviluppo più lento (modifica → compila → esegui)

INTERPRETAZIONE:
✅ Sviluppo più veloce (modifica → esegui)
✅ Portabilità (stesso codice su OS diversi)
✅ Debugging più facile
❌ Esecuzione più lenta
❌ Serve sempre l'interprete
❌ Codice sorgente visibile


LEXIS, SYNTAX, SEMANTICS:
─────────────────────────
Ogni linguaggio ha regole a tre livelli:

1. LEXIS (Lessico): L'insieme dei "simboli" validi
   - Keywords: if, for, while, def, class...
   - Operatori: +, -, *, /, ==, !=...
   - Literals: 42, 3.14, "hello", True...

2. SYNTAX (Sintassi): Le regole di STRUTTURA
   - Come combinare i simboli in modo valido
   - Esempio: "if x > 5:" è valido, "if > x 5:" NO

3. SEMANTICS (Semantica): Il SIGNIFICATO
   - Cosa FA effettivamente il codice
   - Esempio: "x = 5" significa "assegna il valore 5 alla variabile x"
"""


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 1.1 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                         QUIZ SECTION 1.1                                     │
│                  Scrivi le risposte PRIMA di verificare                      │
└──────────────────────────────────────────────────────────────────────────────┘
"""

QUIZ_1_1 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 1.1.1
══════════════════════════════════════════════════════════════════════════════
Quale componente del computer esegue effettivamente le istruzioni del programma?

A) RAM
B) Hard Drive
C) CPU
D) Monitor

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 1.1.2
══════════════════════════════════════════════════════════════════════════════
Qual è la differenza PRINCIPALE tra compilazione e interpretazione?

A) I compilatori sono più moderni degli interpreti
B) I compilatori traducono tutto il codice prima dell'esecuzione,
   gli interpreti traducono ed eseguono linea per linea
C) I compilatori funzionano solo su Windows
D) Non c'è differenza, sono sinonimi

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 1.1.3
══════════════════════════════════════════════════════════════════════════════
Quale delle seguenti è una caratteristica dei linguaggi COMPILATI?

A) Richiedono l'interprete per ogni esecuzione
B) Gli errori vengono trovati durante l'esecuzione
C) Creano un file eseguibile indipendente
D) Sono sempre più lenti dei linguaggi interpretati

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 1.1.4
══════════════════════════════════════════════════════════════════════════════
Se hai un file Python "script.py", cosa serve per eseguirlo?

A) Un compilatore C
B) L'interprete Python installato
C) Niente, si esegue da solo
D) Microsoft Word

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 1.1.5
══════════════════════════════════════════════════════════════════════════════
Cosa significa "semantica" in un linguaggio di programmazione?

A) Le regole di formattazione del codice
B) L'insieme dei simboli validi
C) Il significato delle istruzioni
D) La velocità di esecuzione

Tua risposta: ___
"""


RISPOSTE_1_1 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 1.1
══════════════════════════════════════════════════════════════════════════════

1.1.1: C) CPU
       La CPU (Central Processing Unit) è il "processore" che esegue
       le istruzioni. RAM memorizza dati temporanei, HDD dati permanenti.

1.1.2: B) I compilatori traducono tutto prima, gli interpreti linea per linea
       Questa è LA differenza fondamentale tra i due approcci.

1.1.3: C) Creano un file eseguibile indipendente
       Il compilatore produce un .exe (o equivalente) che gira da solo.

1.1.4: B) L'interprete Python installato
       Python è interpretato, serve sempre l'interprete per eseguire.

1.1.5: C) Il significato delle istruzioni
       Lexis = simboli, Syntax = struttura, Semantics = significato.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.2: INTRODUCTION TO PYTHON
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                         1.2 TEORIA: PYTHON                                   │
└──────────────────────────────────────────────────────────────────────────────┘

STORIA DI PYTHON:
─────────────────
- Creato da: Guido van Rossum (Olanda)
- Anno: 1991 (prima release pubblica)
- Nome: Dal gruppo comico "Monty Python's Flying Circus"
- Filosofia: "Readability counts" - Il codice deve essere leggibile


PYTHON 2 vs PYTHON 3:
─────────────────────
- Python 2: 2000-2020 (End of Life: 1 Gennaio 2020)
- Python 3: 2008-oggi (versione attuale)

DIFFERENZE CHIAVE da conoscere per l'esame:

┌─────────────────────┬─────────────────────┬─────────────────────┐
│     Funzionalità    │      Python 2       │      Python 3       │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ print               │ print "hello"       │ print("hello")      │
│ Division            │ 5/2 = 2             │ 5/2 = 2.5           │
│ Integer Division    │ 5/2 = 2             │ 5//2 = 2            │
│ input()             │ raw_input()         │ input()             │
│ Unicode             │ Opzionale           │ Default             │
└─────────────────────┴─────────────────────┴─────────────────────┘

Per l'esame PCEP: Si usa SOLO Python 3.x


IMPLEMENTAZIONI DI PYTHON:
──────────────────────────
"Python" è una SPECIFICA del linguaggio. Esistono diverse IMPLEMENTAZIONI:

1. CPython (STANDARD)
   - Scritto in C
   - L'implementazione di riferimento
   - Quella che usi normalmente
   - Compila in bytecode → eseguito dalla PVM

2. Cython
   - Superset di Python che compila in C
   - Per performance (estensioni C)

3. Jython
   - Python su Java Virtual Machine (JVM)
   - Può usare librerie Java

4. PyPy
   - Implementazione con JIT compiler
   - Spesso più veloce di CPython

5. MicroPython / CircuitPython
   - Per microcontrollori (Arduino, Raspberry Pi Pico)
   - Subset di Python


PYTHON È:
─────────
✅ Interpretato (tecnicamente ibrido: compila in bytecode)
✅ Ad alto livello (astrae dettagli macchina)
✅ General-purpose (non limitato a un dominio)
✅ Multi-paradigma (procedurale, OOP, funzionale)
✅ Dinamicamente tipizzato (tipi determinati a runtime)
✅ Fortemente tipizzato (no conversioni implicite pericolose)
✅ Open source e gratuito


PERCHÉ PYTHON È POPOLARE:
─────────────────────────
1. Sintassi pulita e leggibile
2. Curva di apprendimento dolce
3. Enorme ecosistema di librerie
4. Comunità attiva e supportiva
5. Usato in: Web, Data Science, AI/ML, Automazione, Trading...


LIMITAZIONI DI PYTHON:
──────────────────────
1. Più lento di C/C++/Rust (ma spesso non importa)
2. GIL (Global Interpreter Lock) limita il multithreading
3. Mobile development non ideale
4. Memory-intensive per alcuni use case
"""


# ══════════════════════════════════════════════════════════════════════════════
#                     SECTION 1.2 - QUIZ
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_1_2 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 1.2.1
══════════════════════════════════════════════════════════════════════════════
Chi ha creato Python?

A) Linus Torvalds
B) Guido van Rossum
C) Dennis Ritchie
D) James Gosling

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 1.2.2
══════════════════════════════════════════════════════════════════════════════
Da dove deriva il nome "Python"?

A) Dal serpente pitone
B) Dal gruppo comico Monty Python
C) Da un acronimo tecnico
D) Dal nome del creatore

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 1.2.3
══════════════════════════════════════════════════════════════════════════════
Qual è l'implementazione STANDARD/di riferimento di Python?

A) PyPy
B) Jython
C) CPython
D) IronPython

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 1.2.4
══════════════════════════════════════════════════════════════════════════════
Quale versione di Python è usata per l'esame PCEP?

A) Python 2.7
B) Python 3.x
C) Entrambe
D) Dipende dal paese

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 1.2.5
══════════════════════════════════════════════════════════════════════════════
Cosa produce CPython quando esegui un file .py?

A) Un file .exe direttamente
B) Bytecode che viene eseguito dalla PVM
C) Codice macchina nativo
D) Un file HTML

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 1.2.6
══════════════════════════════════════════════════════════════════════════════
In Python 3, cosa restituisce 5 / 2?

A) 2
B) 2.5
C) 2.0
D) Error

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 1.2.7
══════════════════════════════════════════════════════════════════════════════
Quale implementazione Python è ottimizzata per microcontrollori?

A) CPython
B) Jython
C) MicroPython
D) PyPy

Tua risposta: ___
"""


RISPOSTE_1_2 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 1.2
══════════════════════════════════════════════════════════════════════════════

1.2.1: B) Guido van Rossum
       Programmatore olandese, ha creato Python nel 1991.
       (Linus = Linux, Ritchie = C, Gosling = Java)

1.2.2: B) Dal gruppo comico Monty Python
       "Monty Python's Flying Circus" - gruppo comico britannico.

1.2.3: C) CPython
       Scritto in C, è l'implementazione di riferimento.
       Quando installi Python, installi CPython.

1.2.4: B) Python 3.x
       Python 2 è obsoleto (EOL 2020). L'esame usa Python 3.

1.2.5: B) Bytecode che viene eseguito dalla PVM
       Il codice .py viene compilato in bytecode (.pyc),
       poi la Python Virtual Machine lo interpreta.

1.2.6: B) 2.5
       In Python 3, / è "true division" (risultato float).
       Per integer division, usa // (5 // 2 = 2).

1.2.7: C) MicroPython
       Progettato per microcontrollori con risorse limitate.
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    SECTION 1.3: DOWNLOADING AND INSTALLING PYTHON
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                         1.3 TEORIA: INSTALLAZIONE                            │
└──────────────────────────────────────────────────────────────────────────────┘

DOVE SCARICARE PYTHON:
──────────────────────
Sito ufficiale: https://www.python.org/downloads/

NOTA: Per l'esame PCEP non serve installare nulla!
Puoi usare edube.org che ha un ambiente integrato.

Ma per il VERO sviluppo, installa Python localmente.


VERIFICARE L'INSTALLAZIONE:
───────────────────────────
Apri il terminale/prompt e digita:

    python --version
    # oppure
    python3 --version

Output atteso: Python 3.x.x (es. Python 3.11.4)


L'INTERPRETE INTERATTIVO (REPL):
────────────────────────────────
Digita solo "python" (o "python3") per entrare in modalità interattiva:

    $ python
    Python 3.11.4 (main, Jun 24 2023, 10:18:00)
    >>> print("Hello")
    Hello
    >>> 2 + 2
    4
    >>> exit()

REPL = Read-Eval-Print Loop
- Read: Legge il tuo input
- Eval: Valuta/esegue
- Print: Stampa il risultato
- Loop: Ripete

Utile per test rapidi e sperimentazione.


ESEGUIRE UN FILE .py:
─────────────────────
Crea un file "hello.py" con:

    print("Hello, World!")

Esegui da terminale:

    python hello.py

Output: Hello, World!


SHEBANG (Linux/macOS):
──────────────────────
La prima riga può specificare l'interprete:

    #!/usr/bin/env python3
    print("Hello")

Poi puoi rendere il file eseguibile:

    chmod +x hello.py
    ./hello.py


IDE E EDITOR CONSIGLIATI:
─────────────────────────
1. VS Code (quello che usi) - Ottimo con estensione Python
2. PyCharm - IDE completo per Python
3. IDLE - Editor base incluso con Python
4. Jupyter Notebook - Per data science/sperimentazione
"""


QUIZ_1_3 = """
══════════════════════════════════════════════════════════════════════════════
QUIZ 1.3.1
══════════════════════════════════════════════════════════════════════════════
Quale comando verifica la versione di Python installata?

A) python -v
B) python --version
C) python -check
D) python /version

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 1.3.2
══════════════════════════════════════════════════════════════════════════════
Cosa significa REPL?

A) Real Execution Python Language
B) Read-Eval-Print Loop
C) Run-Execute-Process Loop
D) Remote Execution Protocol Layer

Tua risposta: ___


══════════════════════════════════════════════════════════════════════════════
QUIZ 1.3.3
══════════════════════════════════════════════════════════════════════════════
Come si esce dall'interprete interattivo Python?

A) quit
B) exit()
C) Ctrl+C
D) close()

Tua risposta: ___
"""


RISPOSTE_1_3 = """
══════════════════════════════════════════════════════════════════════════════
RISPOSTE QUIZ 1.3
══════════════════════════════════════════════════════════════════════════════

1.3.1: B) python --version
       Oppure python -V (con V maiuscola). Entrambi funzionano.

1.3.2: B) Read-Eval-Print Loop
       Descrive il ciclo dell'interprete interattivo.

1.3.3: B) exit()
       Oppure quit() o Ctrl+D (Linux/Mac) o Ctrl+Z+Enter (Windows).
       exit senza parentesi mostra "Use exit() or Ctrl-D".
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    MODULE 1 - TEST FINALE
# ══════════════════════════════════════════════════════════════════════════════
"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                         MODULE 1 - TEST FINALE                               │
│                                                                              │
│                    15 domande - Target: 70% (11/15)                          │
│                     Tempo consigliato: 15 minuti                             │
└──────────────────────────────────────────────────────────────────────────────┘
"""

MODULE_1_TEST = """
══════════════════════════════════════════════════════════════════════════════
                         MODULE 1 - TEST FINALE
                    Rispondi SENZA guardare le risposte
══════════════════════════════════════════════════════════════════════════════

Q1. Quale componente hardware esegue le istruzioni del programma?
    A) RAM    B) SSD    C) CPU    D) GPU

Q2. Un compilatore:
    A) Traduce ed esegue il codice linea per linea
    B) Traduce tutto il codice sorgente in un eseguibile
    C) È sempre più lento di un interprete
    D) Funziona solo con Python

Q3. Python è principalmente classificato come:
    A) Compilato
    B) Interpretato
    C) Assembly
    D) Markup

Q4. Quale file viene prodotto quando CPython compila il codice Python?
    A) .exe
    B) .pyc (bytecode)
    C) .bin
    D) .out

Q5. Chi ha creato Python?
    A) Dennis Ritchie
    B) Guido van Rossum
    C) James Gosling
    D) Bjarne Stroustrup

Q6. Python 2 è:
    A) La versione corrente
    B) Obsoleto (End of Life dal 2020)
    C) Più veloce di Python 3
    D) Usato per l'esame PCEP

Q7. Quale implementazione di Python è scritta in C ed è quella standard?
    A) PyPy
    B) Jython
    C) CPython
    D) IronPython

Q8. In Python 3, 7 / 2 restituisce:
    A) 3
    B) 3.5
    C) 3.0
    D) Error

Q9. In Python 3, 7 // 2 restituisce:
    A) 3
    B) 3.5
    C) 3.0
    D) Error

Q10. La PVM è:
     A) Python Virtual Machine - esegue il bytecode
     B) Python Version Manager
     C) Python Variable Memory
     D) Pre-compiled Virtual Module

Q11. Quale di questi è un VANTAGGIO dell'interpretazione?
     A) Esecuzione più veloce
     B) Sviluppo più rapido (modifica → esegui)
     C) Codice sorgente protetto
     D) Non serve installare nulla

Q12. "Syntax" in un linguaggio di programmazione si riferisce a:
     A) La velocità del codice
     B) Le regole di struttura delle istruzioni
     C) Il significato delle istruzioni
     D) L'insieme dei simboli validi

Q13. MicroPython è progettato per:
     A) Applicazioni web
     B) Data science
     C) Microcontrollori
     D) Gaming

Q14. Quale comando avvia l'interprete interattivo Python?
     A) python run
     B) python
     C) python --interactive
     D) python -start

Q15. Python è definito "dinamicamente tipizzato" perché:
     A) Il codice cambia durante l'esecuzione
     B) I tipi delle variabili sono determinati a runtime
     C) Supporta solo tipi numerici
     D) Non usa variabili


══════════════════════════════════════════════════════════════════════════════
                              FINE TEST
              Calcola il tuo punteggio prima di vedere le risposte
══════════════════════════════════════════════════════════════════════════════
"""


MODULE_1_TEST_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                    MODULE 1 - RISPOSTE TEST FINALE
══════════════════════════════════════════════════════════════════════════════

Q1:  C) CPU
Q2:  B) Traduce tutto il codice sorgente in un eseguibile
Q3:  B) Interpretato
Q4:  B) .pyc (bytecode)
Q5:  B) Guido van Rossum
Q6:  B) Obsoleto (End of Life dal 2020)
Q7:  C) CPython
Q8:  B) 3.5 (true division in Python 3)
Q9:  A) 3 (floor/integer division)
Q10: A) Python Virtual Machine - esegue il bytecode
Q11: B) Sviluppo più rapido (modifica → esegui)
Q12: B) Le regole di struttura delle istruzioni
Q13: C) Microcontrollori
Q14: B) python
Q15: B) I tipi delle variabili sono determinati a runtime


PUNTEGGIO:
──────────
15/15: Eccellente! Pronto per il Module 2
13-14: Ottimo! Rivedi gli errori e procedi
11-12: Buono! Target PCEP raggiunto (70%)
<11:   Riguarda la teoria prima di procedere


ARGOMENTI DA RIVEDERE SE HAI SBAGLIATO:

Q1-4:    Concetti base di compilazione/interpretazione
Q5-7:    Storia e implementazioni di Python
Q8-9:    Differenze Python 2/3 (MOLTO IMPORTANTE!)
Q10:     Come funziona Python internamente
Q11-12:  Terminologia (syntax, semantics, lexis)
Q13-15:  Caratteristiche specifiche di Python
"""


# ══════════════════════════════════════════════════════════════════════════════
#                    ESECUZIONE E VERIFICA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 78)
    print("PYTHON ESSENTIALS 1 - MODULE 1")
    print("Introduction to Python and Computer Programming")
    print("=" * 78)
    print("""
    
    CONTENUTO DEL MODULO:
    ─────────────────────
    1. Section 1.1: Introduction to Programming (5 quiz)
    2. Section 1.2: Introduction to Python (7 quiz)
    3. Section 1.3: Downloading and Installing (3 quiz)
    4. Module 1 Test Finale (15 domande)
    
    COME USARE QUESTO FILE:
    ───────────────────────
    1. Leggi le sezioni di teoria (commenti multilinea)
    2. Copia i quiz in un editor/foglio
    3. Rispondi SENZA eseguire codice
    4. Confronta con le risposte
    
    PER VEDERE I QUIZ:
    ──────────────────
    print(QUIZ_1_1)       # Quiz sezione 1.1
    print(RISPOSTE_1_1)   # Risposte sezione 1.1
    print(QUIZ_1_2)       # Quiz sezione 1.2
    print(RISPOSTE_1_2)   # Risposte sezione 1.2
    print(QUIZ_1_3)       # Quiz sezione 1.3
    print(RISPOSTE_1_3)   # Risposte sezione 1.3
    print(MODULE_1_TEST)  # Test finale
    print(MODULE_1_TEST_ANSWERS)  # Risposte test finale
    
    """)
    
    print("\n" + "=" * 78)
    print("MODULE 1 COMPLETATO!")
    print("Passa a: pe1_m2_datatypes.py")
    print("=" * 78)
