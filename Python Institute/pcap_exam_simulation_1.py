"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PCAP-31-03 EXAM SIMULATION #1                             ║
║                                                                              ║
║                    Certified Associate in Python                             ║
║                    40 Domande | 65 Minuti | 70% Pass                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PESO SEZIONI:
- Modules & Packages: 12%
- Strings: 18%
- OOP: 34% (PIÙ IMPORTANTE!)
- Miscellaneous: 36%

═══════════════════════════════════════════════════════════════════════════════
"""

PCAP_EXAM_1 = """
══════════════════════════════════════════════════════════════════════════════
                         PCAP-31-03 EXAM SIMULATION #1
                              START YOUR TIMER: 65:00
══════════════════════════════════════════════════════════════════════════════

SECTION 1: MODULES & PACKAGES (12%)
───────────────────────────────────────────────────────────────────────────────

Q1. What is the output?

    # file: mymodule.py
    print(__name__)
    
    # Executed as: python mymodule.py

    A) __main__
    B) mymodule
    C) None
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q2. What is the output?

    import math
    print(math.floor(-2.5))

    A) -2
    B) -3
    C) -2.0
    D) -3.0

───────────────────────────────────────────────────────────────────────────────

Q3. What is the output?

    import random
    random.seed(0)
    lst = [1, 2, 3]
    result = random.shuffle(lst)
    print(result)

    A) [2, 3, 1]
    B) [1, 2, 3]
    C) None
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q4. Which statement correctly imports only the sqrt function from math?

    A) import sqrt from math
    B) from math import sqrt
    C) import math.sqrt
    D) from math import sqrt()

───────────────────────────────────────────────────────────────────────────────

Q5. What does sys.path contain?

    A) Only the current directory
    B) List of directories where Python looks for modules
    C) The Python executable path
    D) Environment variables

───────────────────────────────────────────────────────────────────────────────

SECTION 2: STRINGS (18%)
───────────────────────────────────────────────────────────────────────────────

Q6. What is the output?

    s = "Hello World"
    print(s.find("o"), s.rfind("o"))

    A) 4 4
    B) 4 7
    C) 7 4
    D) 7 7

───────────────────────────────────────────────────────────────────────────────

Q7. What is the output?

    s = "hello"
    print(s.index("x"))

    A) -1
    B) None
    C) ValueError
    D) 0

───────────────────────────────────────────────────────────────────────────────

Q8. What is the output?

    s = "hello world"
    print(s.title())

    A) Hello world
    B) Hello World
    C) HELLO WORLD
    D) hello world

───────────────────────────────────────────────────────────────────────────────

Q9. What is the output?

    s = "  hello  "
    print(f"'{s.strip()}'")

    A) '  hello  '
    B) 'hello'
    C) '  hello'
    D) 'hello  '

───────────────────────────────────────────────────────────────────────────────

Q10. What is the output?

    s = "a,b,c"
    print(s.split(","))

    A) "a,b,c"
    B) ['a', 'b', 'c']
    C) ('a', 'b', 'c')
    D) ["a,b,c"]

───────────────────────────────────────────────────────────────────────────────

Q11. What is the output?

    lst = ['a', 'b', 'c']
    print("-".join(lst))

    A) ['a-b-c']
    B) a-b-c
    C) -a-b-c-
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q12. What is the output?

    print(ord('A'), chr(66))

    A) 65 B
    B) A 66
    C) 65 66
    D) A B

───────────────────────────────────────────────────────────────────────────────

SECTION 3: OOP (34% - MOST IMPORTANT!)
───────────────────────────────────────────────────────────────────────────────

Q13. What is the output?

    class A:
        x = 1
        
    class B(A):
        pass
        
    class C(A):
        pass
    
    B.x = 2
    print(A.x, B.x, C.x)

    A) 1 2 1
    B) 2 2 2
    C) 1 1 1
    D) 2 2 1

───────────────────────────────────────────────────────────────────────────────

Q14. What is the output?

    class Counter:
        count = 0
        
        def __init__(self):
            Counter.count += 1
    
    a = Counter()
    b = Counter()
    c = Counter()
    print(Counter.count)

    A) 0
    B) 1
    C) 3
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q15. What is the output?

    class A:
        def __init__(self):
            self.x = 1
            self.__y = 2
    
    a = A()
    print(a._A__y)

    A) 2
    B) Error (AttributeError)
    C) None
    D) __y

───────────────────────────────────────────────────────────────────────────────

Q16. What is the output?

    class A:
        def method(self):
            return "A"
    
    class B(A):
        def method(self):
            return "B" + super().method()
    
    b = B()
    print(b.method())

    A) A
    B) B
    C) BA
    D) AB

───────────────────────────────────────────────────────────────────────────────

Q17. What is the output?

    class A:
        pass
    
    class B(A):
        pass
    
    b = B()
    print(isinstance(b, A), isinstance(b, B))

    A) True True
    B) True False
    C) False True
    D) False False

───────────────────────────────────────────────────────────────────────────────

Q18. What is the output?

    class A:
        pass
    
    class B(A):
        pass
    
    print(issubclass(B, A), issubclass(A, B))

    A) True True
    B) True False
    C) False True
    D) False False

───────────────────────────────────────────────────────────────────────────────

Q19. What is the output?

    class A:
        def __str__(self):
            return "A str"
        
        def __repr__(self):
            return "A repr"
    
    a = A()
    print(a)

    A) A str
    B) A repr
    C) <A object>
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q20. What is the output?

    class A:
        def __init__(self, x):
            self.x = x
        
        def __add__(self, other):
            return A(self.x + other.x)
    
    a = A(1)
    b = A(2)
    c = a + b
    print(c.x)

    A) 1
    B) 2
    C) 3
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q21. What is the output?

    class A:
        def __len__(self):
            return 5
    
    a = A()
    print(len(a))

    A) 5
    B) 0
    C) None
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q22. What is the output?

    class A:
        @staticmethod
        def method():
            return "static"
        
        @classmethod
        def class_method(cls):
            return cls.__name__
    
    print(A.method(), A.class_method())

    A) static A
    B) Error Error
    C) static <class A>
    D) None A

───────────────────────────────────────────────────────────────────────────────

Q23. What is the MRO (Method Resolution Order) of D?

    class A: pass
    class B(A): pass
    class C(A): pass
    class D(B, C): pass

    A) D, B, C, A, object
    B) D, B, A, C, object
    C) D, C, B, A, object
    D) D, A, B, C, object

───────────────────────────────────────────────────────────────────────────────

Q24. What is the output?

    class A:
        def __init__(self):
            print("A", end=" ")
    
    class B(A):
        def __init__(self):
            print("B", end=" ")
    
    b = B()

    A) A B
    B) B A
    C) B
    D) A

───────────────────────────────────────────────────────────────────────────────

Q25. Which keyword is used to check if a class inherits from another?

    A) isinstance
    B) issubclass
    C) inherits
    D) extends

───────────────────────────────────────────────────────────────────────────────

Q26. What is the output?

    class A:
        def __init__(self, x=[]):
            self.x = x
    
    a = A()
    a.x.append(1)
    b = A()
    print(b.x)

    A) []
    B) [1]
    C) None
    D) Error

───────────────────────────────────────────────────────────────────────────────

SECTION 4: MISCELLANEOUS (36%)
───────────────────────────────────────────────────────────────────────────────

Q27. What is the output?

    def gen():
        yield 1
        yield 2
        yield 3
    
    g = gen()
    print(next(g), next(g))

    A) 1 2
    B) 1 1
    C) 2 3
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q28. What is the output?

    lst = [x**2 for x in range(5) if x % 2 == 0]
    print(lst)

    A) [0, 4, 16]
    B) [1, 9]
    C) [0, 1, 4, 9, 16]
    D) [0, 2, 4]

───────────────────────────────────────────────────────────────────────────────

Q29. What is the output?

    d = {x: x**2 for x in range(3)}
    print(d)

    A) {0: 0, 1: 1, 2: 4}
    B) [(0, 0), (1, 1), (2, 4)]
    C) {0, 1, 4}
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q30. What is the output?

    f = lambda x, y: x + y
    print(f(2, 3))

    A) 5
    B) (2, 3)
    C) x + y
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q31. What is the output?

    lst = [1, 2, 3, 4, 5]
    result = list(filter(lambda x: x > 2, lst))
    print(result)

    A) [1, 2]
    B) [3, 4, 5]
    C) [True, True, True]
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q32. What is the output?

    result = list(map(lambda x: x * 2, [1, 2, 3]))
    print(result)

    A) [1, 2, 3]
    B) [2, 4, 6]
    C) 12
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q33. What is the output?

    with open("test.txt", "w") as f:
        f.write("Hello")
    
    with open("test.txt", "r") as f:
        print(f.read())

    A) Hello
    B) "Hello"
    C) None
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q34. What mode opens a file for writing AND reading?

    A) "w"
    B) "r+"
    C) "rw"
    D) "wr"

───────────────────────────────────────────────────────────────────────────────

Q35. What is the output?

    import os
    print(os.path.splitext("file.tar.gz"))

    A) ('file', '.tar.gz')
    B) ('file.tar', '.gz')
    C) ('file', 'tar', 'gz')
    D) ('file.tar.gz', '')

───────────────────────────────────────────────────────────────────────────────

Q36. What is the output?

    from datetime import date
    d = date(2024, 1, 15)
    print(d.weekday())  # Monday = 0

    A) 0
    B) 1
    C) 7
    D) 15

───────────────────────────────────────────────────────────────────────────────

Q37. What is the output?

    class MyException(Exception):
        pass
    
    try:
        raise MyException("error")
    except Exception as e:
        print(type(e).__name__)

    A) Exception
    B) MyException
    C) error
    D) Error

───────────────────────────────────────────────────────────────────────────────

Q38. What is the output?

    def outer():
        x = 10
        def inner():
            return x
        return inner
    
    f = outer()
    print(f())

    A) 10
    B) None
    C) Error
    D) inner

───────────────────────────────────────────────────────────────────────────────

Q39. What is the output?

    g = (x**2 for x in range(3))
    print(type(g).__name__)

    A) list
    B) tuple
    C) generator
    D) genexpr

───────────────────────────────────────────────────────────────────────────────

Q40. What is the output?

    result = list(filter(None, [0, 1, "", "a", [], [1]]))
    print(result)

    A) [0, 1, "", "a", [], [1]]
    B) [1, "a", [1]]
    C) []
    D) Error

───────────────────────────────────────────────────────────────────────────────

                              END OF EXAM
══════════════════════════════════════════════════════════════════════════════
"""

PCAP_EXAM_1_ANSWERS = """
══════════════════════════════════════════════════════════════════════════════
                    PCAP-31-03 EXAM SIMULATION #1 - ANSWERS
══════════════════════════════════════════════════════════════════════════════

SECTION 1: MODULES & PACKAGES
─────────────────────────────
Q1:  A) __main__
     → Quando eseguito direttamente, __name__ è "__main__"

Q2:  B) -3
     → floor() va verso -∞, non verso zero

Q3:  C) None
     → shuffle() modifica in-place e restituisce None!

Q4:  B) from math import sqrt
     → Sintassi corretta per import singolo

Q5:  B) List of directories where Python looks for modules
     → sys.path è la lista di ricerca moduli

SECTION 2: STRINGS
──────────────────
Q6:  B) 4 7
     → find() trova prima "o" a indice 4, rfind() trova ultima a 7

Q7:  C) ValueError
     → index() solleva eccezione se non trova (find() restituisce -1)

Q8:  B) Hello World
     → title() capitalizza ogni parola

Q9:  B) 'hello'
     → strip() rimuove spazi da entrambi i lati

Q10: B) ['a', 'b', 'c']
     → split() restituisce lista di stringhe

Q11: B) a-b-c
     → join() unisce elementi con il separatore

Q12: A) 65 B
     → ord('A')=65, chr(66)='B'

SECTION 3: OOP
──────────────
Q13: A) 1 2 1
     → B.x = 2 crea attributo di classe separato per B

Q14: C) 3
     → Ogni istanza incrementa Counter.count (class variable)

Q15: A) 2
     → Name mangling: __y diventa _A__y, accessibile

Q16: C) BA
     → B.method() restituisce "B" + "A" = "BA"

Q17: A) True True
     → b è istanza di B e anche di A (ereditarietà)

Q18: B) True False
     → B è sottoclasse di A, ma A non è sottoclasse di B

Q19: A) A str
     → print() usa __str__, __repr__ è per repr()

Q20: C) 3
     → __add__ somma self.x + other.x = 1 + 2

Q21: A) 5
     → len() chiama __len__()

Q22: A) static A
     → @staticmethod non riceve self, @classmethod riceve cls

Q23: A) D, B, C, A, object
     → MRO C3 linearization

Q24: C) B
     → B.__init__ non chiama super().__init__()

Q25: B) issubclass
     → issubclass(Child, Parent) verifica ereditarietà

Q26: B) [1]
     → TRAP! Default mutable argument condiviso tra istanze

SECTION 4: MISCELLANEOUS
────────────────────────
Q27: A) 1 2
     → Generator restituisce valori uno alla volta

Q28: A) [0, 4, 16]
     → Quadrati di numeri pari: 0², 2², 4²

Q29: A) {0: 0, 1: 1, 2: 4}
     → Dict comprehension

Q30: A) 5
     → Lambda function: 2 + 3 = 5

Q31: B) [3, 4, 5]
     → filter() tiene elementi dove lambda è True

Q32: B) [2, 4, 6]
     → map() applica funzione a ogni elemento

Q33: A) Hello
     → File scritto e letto correttamente

Q34: B) "r+"
     → "r+" apre per lettura e scrittura (file deve esistere)

Q35: B) ('file.tar', '.gz')
     → splitext() separa solo l'ULTIMA estensione

Q36: A) 0
     → 15 gennaio 2024 è lunedì (weekday() = 0)

Q37: B) MyException
     → type(e).__name__ restituisce il nome della classe

Q38: A) 10
     → Closure: inner() ricorda x dalla outer()

Q39: C) generator
     → () crea generator expression

Q40: B) [1, "a", [1]]
     → filter(None, ...) rimuove valori falsy

══════════════════════════════════════════════════════════════════════════════
                              SCORE CALCULATION
══════════════════════════════════════════════════════════════════════════════

PASS THRESHOLD: 28/40 (70%)

Per sezione:
- Modules: 5 domande (~12%)
- Strings: 7 domande (~18%)
- OOP: 14 domande (~34%)
- Misc: 14 domande (~36%)

══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print("PCAP-31-03 EXAM SIMULATION #1")
    print("=" * 50)
    print("1. print(PCAP_EXAM_1) - Start exam")
    print("2. print(PCAP_EXAM_1_ANSWERS) - Check answers")
