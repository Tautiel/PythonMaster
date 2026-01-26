#!/usr/bin/env python3
"""
PCPP1-32-101 EXAM SIMULATION - 45 Questions - 65 Minutes
70% per passare (32/45)
"""

EXAM_Q = """
═══ SECTION 1: ADVANCED OOP (Q1-16) ═══

Q1. copy.copy() su nested list: nested objects sono?
    A) Copiati  B) Riferimenti  C) None  D) Errore

Q2. pickle.dumps() restituisce?
    A) String  B) Dict  C) Bytes  D) File

Q3. shelve si usa come?
    A) List  B) Dict  C) Set  D) Tuple

Q4. @classmethod primo parametro?
    A) self  B) cls  C) None  D) args

Q5. @staticmethod primo parametro?
    A) self  B) cls  C) Nulla  D) args

Q6. @property crea?
    A) Attribute  B) Getter  C) Variable  D) Constant

Q7. raise X from e imposta?
    A) __context__  B) __cause__  C) __traceback__  D) __error__

Q8. ABC significa?
    A) Any Base  B) Abstract Base Class  C) Another Base  D) All Base

Q9. @abstractmethod richiede implementazione in?
    A) Stessa classe  B) Subclass  C) Metaclass  D) Nessuna

Q10. type(type) = ?
    A) object  B) type  C) class  D) meta

Q11. Metaclass default è?
    A) object  B) type  C) ABC  D) None

Q12. __slots__ limita?
    A) Metodi  B) Attributi  C) Ereditarietà  D) Istanze

Q13. Subclassing list: override __getitem__ per?
    A) Aggiungere  B) Tracciare accessi  C) Eliminare  D) Nulla

Q14. functools.wraps preserva?
    A) Argomenti  B) Metadata funzione  C) Return value  D) Scope

Q15. Decorator con argomenti ha quanti livelli di funzioni?
    A) 1  B) 2  C) 3  D) 4

Q16. Factory method usa spesso?
    A) @staticmethod  B) @classmethod  C) @property  D) @abstractmethod

═══ SECTION 2: PEP & CONVENTIONS (Q17-21) ═══

Q17. PEP 8 raccomanda indentazione di?
    A) 2 spaces  B) 4 spaces  C) Tab  D) 8 spaces

Q18. Lunghezza massima linea PEP 8?
    A) 72  B) 79  C) 80  D) 120

Q19. Naming per costanti?
    A) lowercase  B) UPPERCASE  C) CamelCase  D) mixedCase

Q20. PEP 20 si accede con?
    A) import pep20  B) import this  C) import zen  D) help(pep)

Q21. Type hint Optional[str] equivale a?
    A) str  B) None  C) Union[str,None]  D) List[str]

═══ SECTION 3: GUI TKINTER (Q22-30) ═══

Q22. Main loop Tkinter?
    A) start()  B) mainloop()  C) run()  D) loop()

Q23. Geometry manager con row/column?
    A) pack()  B) place()  C) grid()  D) layout()

Q24. Entry.get() restituisce?
    A) Widget  B) String  C) Int  D) None

Q25. Button command riceve?
    A) Event  B) Callback function  C) String  D) Widget

Q26. Evento click sinistro?
    A) <Click>  B) <Button-1>  C) <LeftClick>  D) <Mouse-1>

Q27. Canvas.create_rectangle restituisce?
    A) Widget  B) None  C) Item ID  D) Coordinates

Q28. StringVar.get() restituisce?
    A) StringVar  B) String  C) Entry  D) None

Q29. messagebox.askyesno restituisce?
    A) String  B) True/False  C) 'yes'/'no'  D) 1/0

Q30. Frame è usato per?
    A) Testo  B) Input  C) Container  D) Disegno

═══ SECTION 4: NETWORK (Q31-38) ═══

Q31. TCP socket type?
    A) SOCK_DGRAM  B) SOCK_STREAM  C) SOCK_RAW  D) SOCK_TCP

Q32. UDP socket type?
    A) SOCK_DGRAM  B) SOCK_STREAM  C) SOCK_UDP  D) SOCK_RAW

Q33. server.accept() restituisce?
    A) Data  B) (socket, address)  C) Port  D) Connection

Q34. json.dumps() converte?
    A) JSON→Python  B) Python→JSON  C) File→Python  D) Python→File

Q35. json.loads() accetta?
    A) File  B) String  C) Bytes  D) Dict

Q36. HTTP POST serve per?
    A) Read  B) Create  C) Update  D) Delete

Q37. HTTP status 404 significa?
    A) OK  B) Created  C) Not Found  D) Error

Q38. socket.recv(1024) - 1024 è?
    A) Port  B) Timeout  C) Max bytes  D) Min bytes

═══ SECTION 5: FILE PROCESSING (Q39-45) ═══

Q39. csv.DictReader legge righe come?
    A) Lists  B) Tuples  C) Dicts  D) Strings

Q40. logging.DEBUG valore numerico?
    A) 0  B) 10  C) 20  D) 30

Q41. Quale logging level più grave?
    A) DEBUG  B) INFO  C) ERROR  D) CRITICAL

Q42. ConfigParser legge formato?
    A) JSON  B) XML  C) INI  D) YAML

Q43. config.getint() restituisce?
    A) String  B) Int  C) Float  D) Bool

Q44. ET.SubElement crea?
    A) Root  B) Child  C) Attribute  D) Text

Q45. element.find('.//tag') cerca?
    A) Solo figli  B) Ricorsivamente  C) Attributi  D) Testo
"""

EXAM_A = """
═══ ANSWERS ═══

SECTION 1: ADVANCED OOP
Q1:  B) Riferimenti  | Shallow copy = nested sono riferimenti
Q2:  C) Bytes        | pickle.dumps → bytes
Q3:  B) Dict         | shelve = persistent dictionary
Q4:  B) cls          | classmethod riceve la classe
Q5:  C) Nulla        | staticmethod non riceve nulla
Q6:  B) Getter       | @property crea getter
Q7:  B) __cause__    | from = causa esplicita
Q8:  B) Abstract Base Class
Q9:  B) Subclass     | abstractmethod richiede implementazione
Q10: B) type         | type è istanza di se stesso
Q11: B) type         | Default metaclass
Q12: B) Attributi    | __slots__ limita attributi
Q13: B) Tracciare    | Override per monitorare
Q14: B) Metadata     | wraps preserva __name__, __doc__
Q15: C) 3            | decorator_factory → decorator → wrapper
Q16: B) @classmethod | Factory method pattern

SECTION 2: PEP
Q17: B) 4 spaces
Q18: B) 79
Q19: B) UPPERCASE
Q20: B) import this
Q21: C) Union[str,None]

SECTION 3: GUI
Q22: B) mainloop()
Q23: C) grid()
Q24: B) String
Q25: B) Callback function
Q26: B) <Button-1>
Q27: C) Item ID
Q28: B) String
Q29: B) True/False
Q30: C) Container

SECTION 4: NETWORK
Q31: B) SOCK_STREAM
Q32: A) SOCK_DGRAM
Q33: B) (socket, address)
Q34: B) Python→JSON
Q35: B) String
Q36: B) Create
Q37: C) Not Found
Q38: C) Max bytes

SECTION 5: FILE PROCESSING
Q39: C) Dicts
Q40: B) 10
Q41: D) CRITICAL
Q42: C) INI
Q43: B) Int
Q44: B) Child
Q45: B) Ricorsivamente

SCORE: ___/45 | PASS: 32+ | TARGET: 36+
"""

def main():
    print("PCPP1-32-101 EXAM SIMULATION")
    print("=" * 50)
    print(EXAM_Q)
    input("\nPress ENTER for answers...")
    print(EXAM_A)

if __name__ == "__main__":
    main()
