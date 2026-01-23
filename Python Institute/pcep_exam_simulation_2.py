"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PCEP-30-02 EXAM SIMULATION #2                             ║
║                                                                              ║
║                    DIFFICOLTÀ: AVANZATA (Edge Cases)                         ║
║                    40 Domande | 45 Minuti | 70% Pass                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Questa simulazione include le domande più insidiose che Python Institute usa.
Focus su edge cases e trappole comuni.

═══════════════════════════════════════════════════════════════════════════════
"""

EXAM_SIMULATION_2 = """
══════════════════════════════════════════════════════════════════════════════
                         PCEP-30-02 EXAM SIMULATION #2
                              ADVANCED EDGE CASES
                              START YOUR TIMER: 45:00
══════════════════════════════════════════════════════════════════════════════

Q1. What is the output?

    print(2 ** 2 ** 3)

    A) 64
    B) 256
    C) 16
    D) 512

───────────────────────────────────────────────────────────────────────────────

Q2. What is the output?

    x = 5
    x += x -= 2  # Hint: Is this valid?

    A) 6
    B) 8
    C) 3
    D) SyntaxError

───────────────────────────────────────────────────────────────────────────────

Q3. What is the output?

    print([] == False)

    A) True
    B) False
    C) Error
    D) None

───────────────────────────────────────────────────────────────────────────────

Q4. What is the output?

    print(bool([]) == False)

    A) True
    B) False
    C) Error
    D) []

───────────────────────────────────────────────────────────────────────────────

Q5. What is the output?

    x = [1, 2, 3]
    y = x[:]
    y.append(4)
    print(x)

    A) [1, 2, 3]
    B) [1, 2, 3, 4]
    C) [4]
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q6. What is the output?

    x = [[1, 2], [3, 4]]
    y = x[:]
    y[0][0] = 99
    print(x[0][0])

    A) 1
    B) 99
    C) [99, 2]
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q7. What is the output?

    print("abc" * 0)

    A) "abc"
    B) ""
    C) 0
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q8. What is the output?

    print(1 < 2 < 3)

    A) True
    B) False
    C) Error
    D) 1

───────────────────────────────────────────────────────────────────────────────

Q9. What is the output?

    print(1 < 2 > 1)

    A) True
    B) False
    C) Error
    D) None

───────────────────────────────────────────────────────────────────────────────

Q10. What is the output?

    x = None
    print(x == None, x is None)

    A) True True
    B) True False
    C) False True
    D) False False

───────────────────────────────────────────────────────────────────────────────

Q11. What is the output?

    a = 256
    b = 256
    print(a is b)

    A) True
    B) False
    C) Error
    D) Depends on implementation

───────────────────────────────────────────────────────────────────────────────

Q12. What is the output?

    a = 257
    b = 257
    print(a is b)

    A) True
    B) False
    C) Error
    D) Depends on implementation

───────────────────────────────────────────────────────────────────────────────

Q13. What is the output?

    print("10" > "9")

    A) True
    B) False
    C) Error
    D) None

───────────────────────────────────────────────────────────────────────────────

Q14. What is the output?

    print([1, 2] + [3, 4])
    print([1, 2].extend([3, 4]))

    A) [1, 2, 3, 4] and [1, 2, 3, 4]
    B) [1, 2, 3, 4] and None
    C) [1, 2, 3, 4] and Error
    D) Error and None

───────────────────────────────────────────────────────────────────────────────

Q15. What is the output?

    lst = [1, 2, 3]
    lst.insert(10, 4)
    print(lst)

    A) Error (index out of range)
    B) [1, 2, 3, 4]
    C) [1, 2, 3, None, None, None, None, None, None, None, 4]
    D) [1, 2, 3]

───────────────────────────────────────────────────────────────────────────────

Q16. What is the output?

    lst = [1, 2, 3, 2, 4, 2]
    lst.remove(2)
    print(lst)

    A) [1, 3, 4]
    B) [1, 3, 2, 4, 2]
    C) [1, 2, 3, 4]
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q17. What is the output?

    print(round(2.5))
    print(round(3.5))

    A) 2 and 3
    B) 3 and 4
    C) 2 and 4
    D) 3 and 3

───────────────────────────────────────────────────────────────────────────────

Q18. What is the output?

    def f(a, b, /, c, *, d):
        return a + b + c + d
    
    print(f(1, 2, 3, d=4))

    A) 10
    B) Error
    C) (1, 2, 3, 4)
    D) None

───────────────────────────────────────────────────────────────────────────────

Q19. What is the output?

    x = {"a": 1}
    y = {"b": 2}
    z = {**x, **y}
    print(z)

    A) {"a": 1, "b": 2}
    B) Error
    C) {"a": 1}
    D) ({"a": 1}, {"b": 2})

───────────────────────────────────────────────────────────────────────────────

Q20. What is the output?

    print({1, 2, 3} & {2, 3, 4})

    A) {1, 2, 3, 4}
    B) {2, 3}
    C) {1, 4}
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q21. What is the output?

    d = {}
    d[[1, 2]] = "value"
    print(d)

    A) {[1, 2]: "value"}
    B) {(1, 2): "value"}
    C) Error
    D) {"[1, 2]": "value"}

───────────────────────────────────────────────────────────────────────────────

Q22. What is the output?

    def f():
        global x
        x = 10
    
    f()
    print(x)

    A) Error (x not defined)
    B) 10
    C) None
    D) 0

───────────────────────────────────────────────────────────────────────────────

Q23. What is the output?

    def f():
        print(x)
        x = 10
    
    x = 5
    f()

    A) 5
    B) 10
    C) Error (UnboundLocalError)
    D) None

───────────────────────────────────────────────────────────────────────────────

Q24. What is the output?

    x = [i for i in range(3)]
    print(x)
    print(i)

    A) [0, 1, 2] and Error
    B) [0, 1, 2] and 2
    C) Error
    D) [0, 1, 2] and None

───────────────────────────────────────────────────────────────────────────────

Q25. What is the output?

    x = (i for i in range(3))
    print(type(x).__name__)

    A) list
    B) tuple
    C) generator
    D) range

───────────────────────────────────────────────────────────────────────────────

Q26. What is the output?

    lst = [1, 2, 3]
    for i in lst:
        lst.append(i)
        if len(lst) > 6:
            break
    print(lst)

    A) [1, 2, 3, 1, 2, 3]
    B) [1, 2, 3, 1, 2, 3, 1]
    C) Infinite loop
    D) [1, 2, 3]

───────────────────────────────────────────────────────────────────────────────

Q27. What is the output?

    try:
        try:
            raise ValueError
        except TypeError:
            print("A")
        finally:
            print("B")
    except ValueError:
        print("C")

    A) A B
    B) B C
    C) A B C
    D) C B

───────────────────────────────────────────────────────────────────────────────

Q28. What is the output?

    class A:
        x = 1
    
    a = A()
    b = A()
    a.x = 2
    print(A.x, a.x, b.x)

    A) 2 2 2
    B) 1 2 1
    C) 1 2 2
    D) 2 2 1

───────────────────────────────────────────────────────────────────────────────

Q29. What is the output?

    print(0 or "" or [] or "hello" or "world")

    A) 0
    B) ""
    C) "hello"
    D) "world"

───────────────────────────────────────────────────────────────────────────────

Q30. What is the output?

    print(1 and 2 and 3)

    A) True
    B) 1
    C) 3
    D) 6

───────────────────────────────────────────────────────────────────────────────

Q31. What is the output?

    x = "hello"
    print(x[::2])

    A) "hlo"
    B) "el"
    C) "hel"
    D) "llo"

───────────────────────────────────────────────────────────────────────────────

Q32. What is the output?

    x = "hello"
    print(x[::-1])

    A) "hello"
    B) "olleh"
    C) "o"
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q33. What is the output?

    def f(**kwargs):
        return len(kwargs)
    
    print(f(a=1, b=2, c=3))

    A) 3
    B) (a=1, b=2, c=3)
    C) {'a': 1, 'b': 2, 'c': 3}
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q34. What is the output?

    lst = [[]] * 3
    lst[0].append(1)
    print(lst)

    A) [[1], [], []]
    B) [[1], [1], [1]]
    C) [[1, 1, 1]]
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q35. What is the output?

    print(sum([1, 2, 3], 10))

    A) 6
    B) 16
    C) [10, 1, 2, 3]
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q36. What is the output?

    d = {"a": 1, "b": 2}
    for k in d:
        print(k, end=" ")

    A) a b
    B) 1 2
    C) a 1 b 2
    D) ('a', 1) ('b', 2)

───────────────────────────────────────────────────────────────────────────────

Q37. What is the output?

    x = [1, 2, 3]
    y = [1, 2, 3]
    print(x == y, x is y)

    A) True True
    B) True False
    C) False True
    D) False False

───────────────────────────────────────────────────────────────────────────────

Q38. What is the output?

    def f(x=[]):
        x.append(len(x))
        return x
    
    print(f())
    print(f())
    print(f([]))

    A) [0] [0, 1] [0]
    B) [0] [0] []
    C) [0] [1] []
    D) [0] [0, 1] []

───────────────────────────────────────────────────────────────────────────────

Q39. What is the output?

    print(type(lambda: None).__name__)

    A) lambda
    B) function
    C) NoneType
    D) method

───────────────────────────────────────────────────────────────────────────────

Q40. What is the output?

    x = 5
    def f():
        x += 1
        return x
    
    print(f())

    A) 6
    B) 5
    C) Error (UnboundLocalError)
    D) None

───────────────────────────────────────────────────────────────────────────────

                              END OF EXAM
══════════════════════════════════════════════════════════════════════════════
"""

EXAM_SIMULATION_2_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                    PCEP-30-02 EXAM SIMULATION #2 - ANSWERS
══════════════════════════════════════════════════════════════════════════════

Q1:  B) 256
     → ** right-associative: 2 ** (2 ** 3) = 2 ** 8 = 256

Q2:  D) SyntaxError
     → Python non permette assegnazioni concatenate come x += x -= 2

Q3:  B) False
     → [] e False sono oggetti diversi, == confronta valore E tipo

Q4:  A) True
     → bool([]) è False, e False == False è True

Q5:  A) [1, 2, 3]
     → x[:] crea una SHALLOW COPY, y è indipendente

Q6:  B) 99
     → SHALLOW COPY! Le liste interne sono ancora condivise

Q7:  B) ""
     → Stringa * 0 = stringa vuota

Q8:  A) True
     → Chained comparison: (1 < 2) and (2 < 3)

Q9:  A) True
     → Chained: (1 < 2) and (2 > 1) = True and True

Q10: A) True True
     → Per None, sia == che is funzionano (singleton)

Q11: A) True
     → CPython cache interi -5 a 256 (stesso oggetto)

Q12: D) Depends on implementation
     → 257 fuori dalla cache, ma può variare. Nell'interprete interattivo: False

Q13: B) False
     → Confronto lessicografico: '1' (49) < '9' (57)

Q14: B) [1, 2, 3, 4] and None
     → extend() modifica in-place e restituisce None

Q15: B) [1, 2, 3, 4]
     → insert() con indice > len semplicemente appende

Q16: B) [1, 3, 2, 4, 2]
     → remove() rimuove solo la PRIMA occorrenza

Q17: C) 2 and 4
     → Python 3 usa "banker's rounding" (verso il pari più vicino)

Q18: A) 10
     → / = positional-only, * = keyword-only, c può essere entrambi

Q19: A) {"a": 1, "b": 2}
     → ** unpacking per merge dizionari

Q20: B) {2, 3}
     → & è intersezione di set

Q21: C) Error
     → Liste non sono hashable, non possono essere chiavi dict

Q22: B) 10
     → global crea la variabile globale dalla funzione

Q23: C) Error (UnboundLocalError)
     → x = 10 rende x locale, ma print(x) lo usa prima dell'assegnazione

Q24: B) [0, 1, 2] and 2
     → In Python 3, la variabile del loop comprehension LEAK nel namespace

Q25: C) generator
     → () invece di [] crea un generator expression

Q26: B) [1, 2, 3, 1, 2, 3, 1]
     → Il loop itera sulla lista che cresce, break a len > 6

Q27: B) B C
     → finally esegue sempre, poi l'eccezione propaga all'except esterno

Q28: B) 1 2 1
     → a.x = 2 crea attributo d'istanza, non modifica classe

Q29: C) "hello"
     → or restituisce il primo valore truthy

Q30: C) 3
     → and restituisce l'ultimo valore se tutti truthy

Q31: A) "hlo"
     → [::2] = ogni secondo carattere (step 2)

Q32: B) "olleh"
     → [::-1] = stringa al contrario

Q33: A) 3
     → **kwargs è un dict, len() restituisce il numero di chiavi

Q34: B) [[1], [1], [1]]
     → [[]] * 3 crea 3 riferimenti alla STESSA lista!

Q35: B) 16
     → sum(iterable, start) → start + sum(iterable) = 10 + 6

Q36: A) a b
     → Iterare su dict itera sulle CHIAVI

Q37: B) True False
     → == confronta contenuto (uguale), is confronta identità (diversa)

Q38: D) [0] [0, 1] []
     → Default mutable condiviso! f([]) usa una NUOVA lista

Q39: B) function
     → Lambda sono funzioni, il tipo è 'function'

Q40: C) Error (UnboundLocalError)
     → x += 1 implica x = x + 1, che rende x locale, ma x non è definita

══════════════════════════════════════════════════════════════════════════════
                              TRAPS SUMMARY
══════════════════════════════════════════════════════════════════════════════

Le trappole più comuni in questo esame:
1. ** è right-associative
2. Shallow copy vs deep copy
3. Integer caching (-5 to 256)
4. String comparison è lessicografica
5. extend() returns None
6. insert() non solleva errore per indici grandi
7. remove() rimuove solo prima occorrenza
8. Banker's rounding in Python 3
9. UnboundLocalError con assegnazioni
10. Default mutable arguments
11. List multiplication crea riferimenti
12. sum() ha un parametro start

══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print("PCEP-30-02 EXAM SIMULATION #2 - ADVANCED")
    print("=" * 50)
    print("1. print(EXAM_SIMULATION_2) - Start exam")
    print("2. print(EXAM_SIMULATION_2_ANSWERS) - Check answers")
