"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PCEP-30-02 EXAM SIMULATION #1                             ║
║                                                                              ║
║                    Formato Pearson VUE Realistico                            ║
║                    40 Domande | 45 Minuti | 70% Pass                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

ISTRUZIONI:
1. Imposta un timer per 45 minuti
2. NON eseguire codice - simula l'esame reale
3. Segna le risposte su un foglio
4. Controlla le risposte SOLO alla fine
5. Target: 28/40 (70%)

FORMATO DOMANDE:
- Single choice (una risposta corretta)
- Multiple choice (più risposte corrette - indicato)
- Drag & drop (riordina - simulato come ordinamento)
- Fill in the gap (completa il codice)

═══════════════════════════════════════════════════════════════════════════════
"""

EXAM_SIMULATION_1 = """
══════════════════════════════════════════════════════════════════════════════
                         PCEP-30-02 EXAM SIMULATION #1
                              START YOUR TIMER: 45:00
══════════════════════════════════════════════════════════════════════════════

SECTION 1: Computer Programming & Python Fundamentals (18%)
───────────────────────────────────────────────────────────────────────────────

Q1. What is the output of the following code?

    print("Hello", "World", sep="***")

    A) Hello World
    B) Hello***World
    C) HelloWorld
    D) Hello World***

───────────────────────────────────────────────────────────────────────────────

Q2. Which of the following is NOT a valid Python variable name?

    A) _variable
    B) variable1
    C) 1variable
    D) __variable__

───────────────────────────────────────────────────────────────────────────────

Q3. Python is:

    A) A compiled language only
    B) An interpreted language only
    C) Both compiled and interpreted
    D) Neither compiled nor interpreted

───────────────────────────────────────────────────────────────────────────────

Q4. What is the output?

    x = "Python"
    print(x[0], x[-1])

    A) P n
    B) P o
    C) y n
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q5. Which keyword is used to define a function in Python?

    A) function
    B) def
    C) define
    D) func

───────────────────────────────────────────────────────────────────────────────

Q6. What does the following code print?

    print(type(3.14).__name__)

    A) float
    B) <class 'float'>
    C) 3.14
    D) type

───────────────────────────────────────────────────────────────────────────────

Q7. What is the result of: bool("")

    A) True
    B) False
    C) ""
    D) Error

───────────────────────────────────────────────────────────────────────────────

SECTION 2: Data Types, Evaluations, and Basic I/O (29%)
───────────────────────────────────────────────────────────────────────────────

Q8. What is the output?

    print(17 // 3)

    A) 5.666666666666667
    B) 5
    C) 6
    D) 5.0

───────────────────────────────────────────────────────────────────────────────

Q9. What is the output?

    print(17 % 3)

    A) 5
    B) 2
    C) 5.666666666666667
    D) 0

───────────────────────────────────────────────────────────────────────────────

Q10. What is the result of: 2 ** 3 ** 2

    A) 64
    B) 512
    C) 81
    D) 256

───────────────────────────────────────────────────────────────────────────────

Q11. What is the output?

    x = 5
    y = 2
    print(x / y)

    A) 2
    B) 2.5
    C) 2.0
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q12. What is the output?

    print(-7 // 2)

    A) -3
    B) -4
    C) -3.5
    D) 3

───────────────────────────────────────────────────────────────────────────────

Q13. What is the output?

    print(-7 % 2)

    A) -1
    B) 1
    C) -3
    D) 3

───────────────────────────────────────────────────────────────────────────────

Q14. What is the output?

    x = "3"
    y = 2
    print(x * y)

    A) 6
    B) "6"
    C) 33
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q15. What is the output?

    x = input("Enter: ")  # User enters: 5
    print(type(x).__name__)

    A) int
    B) str
    C) float
    D) input

───────────────────────────────────────────────────────────────────────────────

Q16. What is the result of: 0.1 + 0.2 == 0.3

    A) True
    B) False
    C) Error
    D) 0.3

───────────────────────────────────────────────────────────────────────────────

Q17. What is the output?

    print(int(3.9))

    A) 3
    B) 4
    C) 3.0
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q18. What is the output?

    print("A" > "a")

    A) True
    B) False
    C) Error
    D) None

───────────────────────────────────────────────────────────────────────────────

SECTION 3: Control Flow - Conditional Blocks and Loops (25%)
───────────────────────────────────────────────────────────────────────────────

Q19. What is the output?

    x = 5
    if x > 3:
        print("A")
    elif x > 4:
        print("B")
    else:
        print("C")

    A) A
    B) B
    C) A B
    D) C

───────────────────────────────────────────────────────────────────────────────

Q20. What is the output?

    for i in range(3):
        print(i, end=" ")

    A) 1 2 3
    B) 0 1 2
    C) 0 1 2 3
    D) 1 2

───────────────────────────────────────────────────────────────────────────────

Q21. What is the output?

    i = 0
    while i < 3:
        i += 1
        print(i, end=" ")

    A) 0 1 2
    B) 1 2 3
    C) 0 1 2 3
    D) 1 2

───────────────────────────────────────────────────────────────────────────────

Q22. What is the output?

    for i in range(5):
        if i == 3:
            break
        print(i, end=" ")

    A) 0 1 2
    B) 0 1 2 3
    C) 0 1 2 4
    D) 0 1 2 3 4

───────────────────────────────────────────────────────────────────────────────

Q23. What is the output?

    for i in range(5):
        if i == 3:
            continue
        print(i, end=" ")

    A) 0 1 2
    B) 0 1 2 4
    C) 0 1 2 3 4
    D) 3

───────────────────────────────────────────────────────────────────────────────

Q24. What is the output?

    for i in range(3):
        pass
    else:
        print("Done")

    A) Done
    B) Nothing (no output)
    C) Error
    D) 0 1 2 Done

───────────────────────────────────────────────────────────────────────────────

Q25. What is the output?

    for i in range(3):
        if i == 1:
            break
    else:
        print("Done")

    A) Done
    B) Nothing (no output)
    C) Error
    D) 0 Done

───────────────────────────────────────────────────────────────────────────────

Q26. What is the output?

    lst = [1, 2, 3]
    print(lst[-1])

    A) 1
    B) 3
    C) -1
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q27. What is the output?

    lst = [1, 2, 3, 4, 5]
    print(lst[1:4])

    A) [1, 2, 3, 4]
    B) [2, 3, 4]
    C) [2, 3, 4, 5]
    D) [1, 2, 3]

───────────────────────────────────────────────────────────────────────────────

Q28. What is the output?

    lst = [1, 2, 3]
    lst.append([4, 5])
    print(len(lst))

    A) 3
    B) 4
    C) 5
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q29. What is the output?

    a = [1, 2, 3]
    b = a
    b.append(4)
    print(a)

    A) [1, 2, 3]
    B) [1, 2, 3, 4]
    C) [4]
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q30. What is the output?

    print(not 0 and 1 or 2)

    A) True
    B) False
    C) 1
    D) 2

───────────────────────────────────────────────────────────────────────────────

SECTION 4: Functions, Tuples, Dictionaries, Exceptions (28%)
───────────────────────────────────────────────────────────────────────────────

Q31. What is the output?

    def func(a, b=2):
        return a * b
    
    print(func(3))

    A) 3
    B) 6
    C) Error
    D) None

───────────────────────────────────────────────────────────────────────────────

Q32. What is the output?

    def func(lst=[]):
        lst.append(1)
        return lst
    
    print(func())
    print(func())

    A) [1] [1]
    B) [1] [1, 1]
    C) [] [1]
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q33. What is the output?

    def func(*args):
        return sum(args)
    
    print(func(1, 2, 3, 4))

    A) (1, 2, 3, 4)
    B) 10
    C) [1, 2, 3, 4]
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q34. What is the output?

    x = 10
    def func():
        x = 20
        return x
    
    func()
    print(x)

    A) 10
    B) 20
    C) Error
    D) None

───────────────────────────────────────────────────────────────────────────────

Q35. What is the output?

    t = (1, 2, 3)
    t[0] = 10
    print(t)

    A) (10, 2, 3)
    B) (1, 2, 3)
    C) Error
    D) [10, 2, 3]

───────────────────────────────────────────────────────────────────────────────

Q36. What is the output?

    t = (1)
    print(type(t).__name__)

    A) tuple
    B) int
    C) list
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q37. What is the output?

    d = {"a": 1, "b": 2}
    print(d.get("c", 0))

    A) None
    B) 0
    C) Error
    D) "c"

───────────────────────────────────────────────────────────────────────────────

Q38. What is the output?

    d = {"a": 1, "b": 2, "a": 3}
    print(d["a"])

    A) 1
    B) 3
    C) Error
    D) [1, 3]

───────────────────────────────────────────────────────────────────────────────

Q39. What is the output?

    try:
        x = 1 / 0
    except ZeroDivisionError:
        print("A", end=" ")
    finally:
        print("B", end=" ")

    A) A
    B) B
    C) A B
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q40. What is the output?

    def func():
        try:
            return 1
        finally:
            return 2
    
    print(func())

    A) 1
    B) 2
    C) 1 2
    D) Error

───────────────────────────────────────────────────────────────────────────────

                              END OF EXAM
                         CHECK YOUR ANSWERS BELOW
══════════════════════════════════════════════════════════════════════════════
"""

EXAM_SIMULATION_1_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                    PCEP-30-02 EXAM SIMULATION #1 - ANSWERS
══════════════════════════════════════════════════════════════════════════════

Q1:  B) Hello***World
     → sep="***" sostituisce lo spazio tra argomenti

Q2:  C) 1variable
     → I nomi non possono iniziare con un numero

Q3:  C) Both compiled and interpreted
     → Python compila in bytecode, poi interpreta

Q4:  A) P n
     → x[0]='P' (primo), x[-1]='n' (ultimo)

Q5:  B) def
     → Keyword per definire funzioni

Q6:  A) float
     → __name__ restituisce solo il nome della classe

Q7:  B) False
     → Stringa vuota è falsy

Q8:  B) 5
     → // è floor division, restituisce int

Q9:  B) 2
     → 17 = 3*5 + 2, quindi resto = 2

Q10: B) 512
     → ** è right-associative: 2 ** (3 ** 2) = 2 ** 9 = 512

Q11: B) 2.5
     → / restituisce SEMPRE float

Q12: B) -4
     → Floor va verso -∞, non verso zero

Q13: B) 1
     → Il segno del risultato segue il DIVISORE (2 è positivo)

Q14: C) 33
     → Stringa * intero = ripetizione stringa

Q15: B) str
     → input() restituisce SEMPRE stringa

Q16: B) False
     → Floating point precision: 0.1+0.2 ≠ 0.3 esattamente

Q17: A) 3
     → int() tronca verso zero (non arrotonda)

Q18: B) False
     → 'A' (65) < 'a' (97) in ASCII/Unicode

Q19: A) A
     → Prima condizione vera, elif non viene valutato

Q20: B) 0 1 2
     → range(3) = 0, 1, 2

Q21: B) 1 2 3
     → i viene incrementato PRIMA del print

Q22: A) 0 1 2
     → break esce quando i==3, prima di stampare 3

Q23: B) 0 1 2 4
     → continue salta solo i==3

Q24: A) Done
     → else eseguito perché loop completa senza break

Q25: B) Nothing (no output)
     → else NON eseguito perché c'è stato break

Q26: B) 3
     → Indice negativo conta dalla fine

Q27: B) [2, 3, 4]
     → Slice [1:4] = indici 1, 2, 3 (4 escluso)

Q28: B) 4
     → append aggiunge UN elemento (la lista intera)

Q29: B) [1, 2, 3, 4]
     → b = a crea riferimento, non copia

Q30: C) 1
     → not 0 = True, True and 1 = 1, 1 or 2 = 1 (short-circuit)

Q31: B) 6
     → 3 * 2 (b usa default)

Q32: B) [1] [1, 1]
     → TRAP! Default mutable è condiviso tra chiamate

Q33: B) 10
     → *args raccoglie in tupla, sum() somma

Q34: A) 10
     → x dentro func() è locale, non modifica globale

Q35: C) Error
     → Tuple sono IMMUTABILI

Q36: B) int
     → (1) è solo 1 con parentesi, non tupla. Serve (1,)

Q37: B) 0
     → get() con default restituisce default se chiave mancante

Q38: B) 3
     → Chiave duplicata: l'ultimo valore vince

Q39: C) A B
     → except cattura errore, finally esegue SEMPRE

Q40: B) 2
     → finally SOVRASCRIVE il return del try!

══════════════════════════════════════════════════════════════════════════════
                              SCORE CALCULATION
══════════════════════════════════════════════════════════════════════════════

Count your correct answers:
- 0-19:  Need more study (< 50%)
- 20-27: Almost there (50-67%)  
- 28-32: PASS (70-80%)
- 33-36: Good (82-90%)
- 37-40: Excellent (92-100%)

PASS THRESHOLD: 28/40 (70%)

══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print("PCEP-30-02 EXAM SIMULATION #1")
    print("=" * 50)
    print("1. print(EXAM_SIMULATION_1) - Start exam")
    print("2. print(EXAM_SIMULATION_1_ANSWERS) - Check answers")
