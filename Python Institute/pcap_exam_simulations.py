#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PCAP-31-03 EXAM SIMULATIONS                               ║
║                    2 Complete Mock Exams (80 Questions)                      ║
║                    Allineato 100% al Syllabus Ufficiale                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

FORMATO ESAME REALE:
- 40 domande | 65 minuti | 70% per passare (28/40)
- Section 1: Modules & Packages (12%) ~5 domande
- Section 2: Exceptions (14%) ~6 domande
- Section 3: Strings (18%) ~7 domande
- Section 4: OOP (34%) ~14 domande
- Section 5: Misc (22%) ~8 domande
"""

# ══════════════════════════════════════════════════════════════════════════════
#                           SIMULATION 1
# ══════════════════════════════════════════════════════════════════════════════

SIM1_Q = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PCAP SIMULATION 1 - 40 Questions - 65 Minutes             ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══ SECTION 1: MODULES (Q1-5) ═══

Q1. math.floor(-4.7) = ?
    A) -4  B) -5  C) 4  D) 5

Q2. random.sample([1,2,3,4,5], 3) restituisce?
    A) Un elemento  B) 3 elementi con ripetizione  C) 3 elementi senza ripetizione  D) Errore

Q3. Quale file rende una directory un package?
    A) __main__.py  B) __init__(self).py  C) __init__.py  D) package.py

Q4. platform.python_version_tuple() restituisce?
    A) String  B) Int  C) Tuple  D) List

Q5. from math import * importa nomi che iniziano con _?
    A) Sì  B) No  C) Solo pubblici  D) Errore

═══ SECTION 2: EXCEPTIONS (Q6-11) ═══

Q6. Quale eccezione per chiave dizionario non trovata?
    A) IndexError  B) KeyError  C) ValueError  D) LookupError

Q7. try-except-else: quando viene eseguito else?
    A) Sempre  B) Se eccezione  C) Se NO eccezione  D) Mai

Q8. finally viene eseguito?
    A) Solo se eccezione  B) Solo se no eccezione  C) Sempre  D) Mai

Q9. raise ValueError from e imposta?
    A) __context__  B) __cause__  C) __traceback__  D) __error__

Q10. BaseException è?
    A) Eccezione base per errori  B) Root di TUTTE le eccezioni  C) Solo per system exit  D) Deprecato

Q11. assert x > 0 solleva quale eccezione se falso?
    A) ValueError  B) TypeError  C) AssertionError  D) RuntimeError

═══ SECTION 3: STRINGS (Q12-18) ═══

Q12. ord('A') = ?
    A) 65  B) 97  C) 48  D) 32

Q13. chr(97) = ?
    A) 'A'  B) 'a'  C) '9'  D) ' '

Q14. 'Hello'[1:4] = ?
    A) 'Hell'  B) 'ell'  C) 'ello'  D) 'Hel'

Q15. 'Python'[::-1] = ?
    A) 'Python'  B) 'nohtyP'  C) 'Pytho'  D) Errore

Q16. '123'.isdigit() = ?
    A) True  B) False  C) '123'  D) Errore

Q17. '-'.join(['a','b','c']) = ?
    A) 'abc'  B) 'a-b-c'  C) ['a-b-c']  D) Errore

Q18. 'Hello'.find('xyz') = ?
    A) 0  B) -1  C) None  D) ValueError

═══ SECTION 4: OOP (Q19-32) ═══

Q19. Quale variabile è condivisa tra tutte le istanze?
    A) Instance variable  B) Class variable  C) Local variable  D) Global

Q20. obj.__dict__ contiene?
    A) Class variables  B) Instance variables  C) Methods  D) All

Q21. __name diventa (in class MyClass)?
    A) __name  B) _name  C) _MyClass__name  D) MyClass__name

Q22. isinstance(dog, Animal) con Dog(Animal)?
    A) True  B) False  C) Errore  D) None

Q23. Dog.__bases__ mostra?
    A) Tutte le classi parent  B) Solo il parent diretto  C) MRO  D) Metodi

Q24. super().__init__() chiama?
    A) object.__init__  B) Parent.__init__  C) Current __init__  D) Errore

Q25. @classmethod riceve come primo parametro?
    A) self  B) cls  C) None  D) *args

Q26. @staticmethod riceve?
    A) self  B) cls  C) Nessuno  D) Both

Q27. @property crea?
    A) Attribute  B) Getter  C) Setter  D) Deleter

Q28. Quale metodo è chiamato da print(obj)?
    A) __repr__  B) __str__  C) __print__  D) __display__

Q29. MRO di D(B, C) con B(A), C(A)?
    A) D,A,B,C  B) D,B,C,A,object  C) D,C,B,A  D) Errore

Q30. hasattr(obj, 'name') restituisce?
    A) Il valore  B) True/False  C) L'attributo  D) Errore

Q31. ABC sta per?
    A) Any Base Class  B) Abstract Base Class  C) Another Base Class  D) All Base Classes

Q32. @abstractmethod richiede?
    A) Implementazione nella classe  B) Implementazione nelle subclass  C) Nessuna implementazione  D) Solo definizione

═══ SECTION 5: MISC (Q33-40) ═══

Q33. list(filter(lambda x: x>2, [1,2,3,4])) = ?
    A) [1,2]  B) [3,4]  C) [2,3,4]  D) Errore

Q34. list(map(lambda x: x*2, [1,2,3])) = ?
    A) [1,2,3]  B) [2,4,6]  C) [1,4,9]  D) Errore

Q35. Cos'è una closure?
    A) Funzione che chiude file  B) Funzione che ricorda scope esterno  C) Funzione ricorsiva  D) Classe

Q36. Quale mode file crea e solleva errore se esiste?
    A) 'w'  B) 'a'  C) 'x'  D) 'r+'

Q37. readline() restituisce?
    A) Tutto il file  B) Una riga  C) Lista di righe  D) Un carattere

Q38. bytearray è?
    A) Immutabile  B) Mutabile  C) Solo lettura  D) Solo scrittura

Q39. [x**2 for x in range(3)] = ?
    A) [0,1,4]  B) [1,4,9]  C) [0,1,2]  D) Errore

Q40. Generator expression usa?
    A) []  B) ()  C) {}  D) <>
"""

SIM1_A = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PCAP SIMULATION 1 - ANSWERS                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Q1:  B) -5           | floor arrotonda verso il basso
Q2:  C) 3 senza rip  | sample = senza ripetizione
Q3:  C) __init__.py  | File obbligatorio per package
Q4:  C) Tuple        | ('3', '11', '0')
Q5:  B) No           | * non importa nomi con _

Q6:  B) KeyError     | Dict key missing
Q7:  C) Se NO eccezione | else esegue solo se try completa
Q8:  C) Sempre       | finally esegue sempre
Q9:  B) __cause__    | raise from = causa esplicita
Q10: B) Root tutte   | BaseException è la base di tutto
Q11: C) AssertionError | assert solleva AssertionError

Q12: A) 65           | 'A' ha code point 65
Q13: B) 'a'          | code point 97 = 'a'
Q14: B) 'ell'        | Indices 1,2,3
Q15: B) 'nohtyP'     | Reverse
Q16: A) True         | Solo cifre
Q17: B) 'a-b-c'      | Join con '-'
Q18: B) -1           | find ritorna -1 se non trovato

Q19: B) Class variable | Condivisa tra istanze
Q20: B) Instance vars  | __dict__ = solo instance
Q21: C) _MyClass__name | Name mangling
Q22: A) True         | Dog è istanza di Animal
Q23: A) Parent diretti | __bases__ = tuple parent
Q24: B) Parent.__init__ | super chiama parent
Q25: B) cls          | classmethod riceve la classe
Q26: C) Nessuno      | staticmethod non riceve nulla
Q27: B) Getter       | @property = getter
Q28: B) __str__      | print usa __str__
Q29: B) D,B,C,A,object | MRO linearization
Q30: B) True/False   | hasattr = check esistenza
Q31: B) Abstract Base Class
Q32: B) Subclass     | abstractmethod deve essere implementato in subclass

Q33: B) [3,4]        | filter mantiene x>2
Q34: B) [2,4,6]      | map applica *2
Q35: B) Scope esterno | Closure ricorda variabili
Q36: C) 'x'          | Exclusive create
Q37: B) Una riga     | readline = singola riga
Q38: B) Mutabile     | bytearray è mutabile
Q39: A) [0,1,4]      | 0²,1²,2²
Q40: B) ()           | Generator usa parentesi tonde

SCORE: ___/40 | PASS: 28+ | TARGET: 32+
"""

# ══════════════════════════════════════════════════════════════════════════════
#                           SIMULATION 2
# ══════════════════════════════════════════════════════════════════════════════

SIM2_Q = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PCAP SIMULATION 2 - 40 Questions - 65 Minutes             ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══ SECTION 1: MODULES (Q1-5) ═══

Q1. math.trunc(-4.7) = ?
    A) -4  B) -5  C) 4  D) 5

Q2. math.hypot(3, 4) = ?
    A) 7  B) 12  C) 5.0  D) 25

Q3. random.choice(['a','b','c']) restituisce?
    A) Lista  B) Un elemento  C) Tre elementi  D) Tupla

Q4. platform.system() su Windows restituisce?
    A) 'Win'  B) 'Windows'  C) 'WIN32'  D) 'NT'

Q5. Quando __name__ == "__main__"?
    A) Sempre  B) Se importato  C) Se eseguito direttamente  D) Mai

═══ SECTION 2: EXCEPTIONS (Q6-11) ═══

Q6. Quale eccezione per indice lista non valido?
    A) KeyError  B) IndexError  C) ValueError  D) RangeError

Q7. except Exception cattura SystemExit?
    A) Sì  B) No  C) Solo se esplicito  D) Errore

Q8. try-finally senza except: se eccezione?
    A) finally non esegue  B) finally esegue, poi propaga  C) finally esegue, eccezione soppressa  D) Errore

Q9. __context__ è impostato quando?
    A) raise from  B) Durante handling di altra eccezione  C) Sempre  D) Mai

Q10. Quale di queste NON è sottoclasse di Exception?
    A) ValueError  B) TypeError  C) KeyboardInterrupt  D) RuntimeError

Q11. with statement chiama quali metodi?
    A) __init__, __del__  B) __enter__, __exit__  C) __open__, __close__  D) __start__, __stop__

═══ SECTION 3: STRINGS (Q12-18) ═══

Q12. ord('a') - ord('A') = ?
    A) 0  B) 26  C) 32  D) -32

Q13. 'hello'.upper() = ?
    A) 'Hello'  B) 'HELLO'  C) 'hello'  D) Errore

Q14. 'abcdef'[1:5:2] = ?
    A) 'bd'  B) 'bcd'  C) 'ace'  D) 'bcde'

Q15. '  hello  '.strip() = ?
    A) 'hello  '  B) '  hello'  C) 'hello'  D) 'hello '

Q16. 'Hello123'.isalpha() = ?
    A) True  B) False  C) Errore  D) None

Q17. 'hello'.split() = ?
    A) ['hello']  B) ['h','e','l','l','o']  C) 'hello'  D) Errore

Q18. 'Hello World'.find('o') = ?
    A) 4  B) 7  C) -1  D) [4,7]

═══ SECTION 4: OOP (Q19-32) ═══

Q19. class A: pass - A.__bases__ = ?
    A) ()  B) (object,)  C) None  D) Errore

Q20. getattr(obj, 'x', 'default') se x non esiste?
    A) Errore  B) None  C) 'default'  D) False

Q21. delattr(obj, 'x') fa cosa?
    A) Imposta x a None  B) Elimina attributo x  C) Restituisce x  D) Errore sempre

Q22. issubclass(bool, int) = ?
    A) True  B) False  C) Errore  D) None

Q23. type(type) = ?
    A) object  B) type  C) class  D) Errore

Q24. Quale decorator per metodo che non usa self né cls?
    A) @classmethod  B) @staticmethod  C) @property  D) Nessuno

Q25. @property.setter si usa per?
    A) Getter  B) Setter  C) Deleter  D) Validator

Q26. copy.copy() crea?
    A) Deep copy  B) Shallow copy  C) Reference  D) Clone

Q27. copy.deepcopy() su lista nested: modifica originale?
    A) Sì  B) No  C) Solo primo livello  D) Errore

Q28. pickle.dumps() restituisce?
    A) String  B) Dict  C) Bytes  D) File

Q29. shelve si usa come?
    A) Lista  B) Dizionario  C) Set  D) Tupla

Q30. Multiple inheritance D(B, C): quale metodo vince?
    A) C  B) B  C) Errore  D) Random

Q31. __slots__ serve per?
    A) Limitare attributi  B) Velocizzare accesso  C) Entrambi  D) Nessuno

Q32. Metaclass è?
    A) Classe di una classe  B) Classe astratta  C) Interfaccia  D) Mixin

═══ SECTION 5: MISC (Q33-40) ═══

Q33. lambda x, y: x + y è equivalente a?
    A) def f(x,y): return x+y  B) def f(x,y): x+y  C) f = x+y  D) Nessuno

Q34. def f(): yield 1 - f() restituisce?
    A) 1  B) None  C) Generator  D) Errore

Q35. next(gen) su generator esaurito solleva?
    A) GeneratorError  B) StopIteration  C) EndOfGenerator  D) None

Q36. open('f.txt', 'a') fa cosa se file esiste?
    A) Sovrascrive  B) Aggiunge alla fine  C) Errore  D) Crea backup

Q37. readlines() restituisce?
    A) String  B) Una riga  C) Lista di righe  D) Bytes

Q38. 'rb' mode significa?
    A) Read binary  B) Raw binary  C) Return binary  D) Rotate binary

Q39. {x: x**2 for x in range(3)} = ?
    A) [0,1,4]  B) {0:0, 1:1, 2:4}  C) {0,1,4}  D) Errore

Q40. (x**2 for x in range(3)) è?
    A) List  B) Tuple  C) Generator  D) Set
"""

SIM2_A = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PCAP SIMULATION 2 - ANSWERS                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Q1:  A) -4           | trunc verso zero
Q2:  C) 5.0          | sqrt(3²+4²) = 5
Q3:  B) Un elemento  | choice = singolo elemento
Q4:  B) 'Windows'    | Nome sistema
Q5:  C) Eseguito     | __main__ quando run direttamente

Q6:  B) IndexError   | Lista index out of range
Q7:  B) No           | Exception non cattura SystemExit
Q8:  B) Esegue,propaga | finally sempre, poi propaga
Q9:  B) Durante handling | __context__ = eccezione implicita
Q10: C) KeyboardInterrupt | Deriva da BaseException, non Exception
Q11: B) __enter__,__exit__ | Context manager protocol

Q12: C) 32           | 97-65 = 32
Q13: B) 'HELLO'      | upper = tutto maiuscolo
Q14: A) 'bd'         | Indices 1,3
Q15: C) 'hello'      | strip rimuove spazi
Q16: B) False        | Contiene numeri
Q17: A) ['hello']    | split senza sep = una parola
Q18: A) 4            | Prima occorrenza

Q19: B) (object,)    | Tutte le classi ereditano da object
Q20: C) 'default'    | Fallback value
Q21: B) Elimina      | delattr rimuove attributo
Q22: A) True         | bool è sottoclasse di int!
Q23: B) type         | type è istanza di se stesso
Q24: B) @staticmethod | Non usa self né cls
Q25: B) Setter       | @x.setter definisce setter
Q26: B) Shallow      | copy = shallow
Q27: B) No           | deepcopy = completamente indipendente
Q28: C) Bytes        | dumps = serialize to bytes
Q29: B) Dizionario   | shelve = persistent dict
Q30: B) B            | Primo nella MRO vince
Q31: C) Entrambi     | Limita attributi E velocizza
Q32: A) Classe di classe | Metaclass crea classi

Q33: A) def f(x,y)   | Lambda equivale a funzione con return
Q34: C) Generator    | yield crea generator
Q35: B) StopIteration | Generator esaurito
Q36: B) Aggiunge     | 'a' = append
Q37: C) Lista righe  | readlines = lista
Q38: A) Read binary  | r=read, b=binary
Q39: B) {0:0,1:1,2:4} | Dict comprehension
Q40: C) Generator    | Parentesi = generator expression

SCORE: ___/40 | PASS: 28+ | TARGET: 32+
"""

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PCAP-31-03 EXAM SIMULATIONS                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  1. Simulation 1                                                             ║
║  2. Simulation 2                                                             ║
║  3. Show All Answers                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    choice = input("Select (1-3): ").strip()
    
    if choice == "1":
        print(SIM1_Q)
        input("\nPress ENTER to see answers...")
        print(SIM1_A)
    elif choice == "2":
        print(SIM2_Q)
        input("\nPress ENTER to see answers...")
        print(SIM2_A)
    elif choice == "3":
        print(SIM1_A)
        print(SIM2_A)

if __name__ == "__main__":
    main()
