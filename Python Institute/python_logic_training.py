"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PYTHON LOGIC TRAINING - Stile Certificazione                    ║
║                    "Leggi il Codice con la Mente"                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Le certificazioni Python Institute NON testano solo "sai scrivere codice?"
Testano: "CAPISCI cosa fa Python internamente?"

REGOLA D'ORO: Devi saper predire l'output SENZA eseguire il codice.

Questo file contiene 80 esercizi di PURA LOGICA:
- Niente coding
- Solo ragionamento
- Predici l'output
- Poi verifica

METODO DI STUDIO:
1. Leggi il codice
2. Scrivi la tua risposta su carta
3. SOLO DOPO esegui per verificare
4. Se sbagli, CAPISCI il perché prima di andare avanti

═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 1: OPERATORI E PRECEDENZA
# ══════════════════════════════════════════════════════════════════════════════
"""
La precedenza è CRITICA negli esami.
Memorizza: () > ** > +x,-x,~x > *,/,//,% > +,- > <<,>> > & > ^ > | > comparisons > not > and > or
"""

# ------------------------------------------------------------------------------
# ESERCIZIO L1.1
# ------------------------------------------------------------------------------
"""
Cosa stampa? Scrivi la risposta PRIMA di eseguire.

print(2 + 3 * 4 ** 2)

La tua risposta: ___

Ragionamento:
- Prima: 4 ** 2 = ?
- Poi: 3 * ? = ?
- Infine: 2 + ? = ?
"""
# Esegui per verificare:
# print(2 + 3 * 4 ** 2)  # Risposta: 50 (4**2=16, 3*16=48, 2+48=50)


# ------------------------------------------------------------------------------
# ESERCIZIO L1.2
# ------------------------------------------------------------------------------
"""
Cosa stampa?

print(2 ** 3 ** 2)

La tua risposta: ___

ATTENZIONE: ** è associativo a DESTRA!
- NON è (2**3)**2 = 8**2 = 64
- È 2**(3**2) = 2**9 = ?
"""
# print(2 ** 3 ** 2)  # Risposta: 512


# ------------------------------------------------------------------------------
# ESERCIZIO L1.3
# ------------------------------------------------------------------------------
"""
Cosa stampa?

print(10 - 5 - 2)

La tua risposta: ___

- è associativo a SINISTRA
- (10-5)-2 = ?
"""
# print(10 - 5 - 2)  # Risposta: 3


# ------------------------------------------------------------------------------
# ESERCIZIO L1.4
# ------------------------------------------------------------------------------
"""
Cosa stampa?

print(-3 ** 2)
print((-3) ** 2)

Le tue risposte: ___, ___

TRAPPOLA: -3**2 viene interpretato come -(3**2), non (-3)**2
"""
# print(-3 ** 2)    # -9 (prima 3**2=9, poi negativo)
# print((-3) ** 2)  # 9 (prima -3, poi al quadrato)


# ------------------------------------------------------------------------------
# ESERCIZIO L1.5
# ------------------------------------------------------------------------------
"""
Cosa stampano?

print(17 // 5)
print(17 % 5)
print(-17 // 5)
print(-17 % 5)

Le tue risposte: ___, ___, ___, ___

REGOLA PYTHON per numeri negativi:
- // arrotonda verso -infinito (floor)
- % ha il segno del DIVISORE
"""
# print(17 // 5)   # 3
# print(17 % 5)    # 2
# print(-17 // 5)  # -4 (NON -3! Floor verso -infinito)
# print(-17 % 5)   # 3 (NON -2! Segno del divisore)

# Verifica: -17 = 5 * (-4) + 3 ✓


# ------------------------------------------------------------------------------
# ESERCIZIO L1.6
# ------------------------------------------------------------------------------
"""
Cosa stampa?

print(5 / 2)
print(5 // 2)
print(5.0 // 2)

Le tue risposte: ___, ___, ___

REGOLA: / sempre float, // dipende dagli operandi
"""
# print(5 / 2)    # 2.5 (true division, sempre float)
# print(5 // 2)   # 2 (floor division, int se entrambi int)
# print(5.0 // 2) # 2.0 (floor division, ma float perché 5.0)


# ------------------------------------------------------------------------------
# ESERCIZIO L1.7
# ------------------------------------------------------------------------------
"""
Cosa stampa?

x = 5
print(x == 5 == 5.0 == True + 4)

La tua risposta: ___

NOTA: Python supporta chained comparisons
E True ha valore numerico 1
"""
# True + 4 = 1 + 4 = 5
# 5 == 5 == 5.0 == 5 → True
# print(x == 5 == 5.0 == True + 4)  # True


# ------------------------------------------------------------------------------
# ESERCIZIO L1.8
# ------------------------------------------------------------------------------
"""
Cosa stampa?

print(1 < 2 < 3)
print(1 < 2 > 0)
print(3 > 2 > 2)

Le tue risposte: ___, ___, ___
"""
# print(1 < 2 < 3)  # True (1<2 AND 2<3)
# print(1 < 2 > 0)  # True (1<2 AND 2>0)
# print(3 > 2 > 2)  # False (3>2 AND 2>2 → False)


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 2: BOOLEAN E SHORT-CIRCUIT
# ══════════════════════════════════════════════════════════════════════════════
"""
PRECEDENZA LOGICA: not > and > or
SHORT-CIRCUIT: Python smette di valutare appena sa il risultato
"""

# ------------------------------------------------------------------------------
# ESERCIZIO L2.1
# ------------------------------------------------------------------------------
"""
Cosa stampa?

print(True or False and False)

La tua risposta: ___

ORDINE: and si valuta PRIMA di or
- False and False = False
- True or False = True
"""
# print(True or False and False)  # True


# ------------------------------------------------------------------------------
# ESERCIZIO L2.2
# ------------------------------------------------------------------------------
"""
Cosa stampa?

print(not True or True and not False)

La tua risposta: ___

ORDINE: not > and > or
1. not True = False
2. not False = True
3. True and True = True
4. False or True = True
"""
# print(not True or True and not False)  # True


# ------------------------------------------------------------------------------
# ESERCIZIO L2.3
# ------------------------------------------------------------------------------
"""
Cosa stampa?

x = 0
y = 5

print(x and y)
print(x or y)
print(y and x)
print(y or x)

Le tue risposte: ___, ___, ___, ___

REGOLA: and/or restituiscono un VALORE, non necessariamente bool!
- and: restituisce il primo falsy o l'ultimo valore
- or: restituisce il primo truthy o l'ultimo valore
"""
# print(x and y)  # 0 (x è falsy, restituisce x)
# print(x or y)   # 5 (x è falsy, valuta y, restituisce y)
# print(y and x)  # 0 (y è truthy, valuta x, restituisce x)
# print(y or x)   # 5 (y è truthy, restituisce y senza valutare x)


# ------------------------------------------------------------------------------
# ESERCIZIO L2.4
# ------------------------------------------------------------------------------
"""
Cosa stampa?

print("hello" and "world")
print("" and "world")
print("hello" or "world")
print("" or "world")

Le tue risposte: ___, ___, ___, ___
"""
# print("hello" and "world")  # "world" (primo truthy, valuta secondo)
# print("" and "world")       # "" (primo falsy, restituisce subito)
# print("hello" or "world")   # "hello" (primo truthy, restituisce subito)
# print("" or "world")        # "world" (primo falsy, valuta secondo)


# ------------------------------------------------------------------------------
# ESERCIZIO L2.5
# ------------------------------------------------------------------------------
"""
Cosa stampa questo codice?

def f():
    print("f called")
    return False

def g():
    print("g called")
    return True

result = f() or g()
print(result)

Output completo: ___
"""
# f called
# g called
# True
# (f() restituisce False, quindi or deve valutare g())


# ------------------------------------------------------------------------------
# ESERCIZIO L2.6
# ------------------------------------------------------------------------------
"""
Cosa stampa?

def f():
    print("f called")
    return True

def g():
    print("g called")
    return False

result = f() or g()
print(result)

Output completo: ___
"""
# f called
# True
# (f() restituisce True, or fa short-circuit, g() MAI chiamata)


# ------------------------------------------------------------------------------
# ESERCIZIO L2.7
# ------------------------------------------------------------------------------
"""
Cosa stampa?

result = f() and g()  # Usando le stesse f() e g() di sopra

Output completo: ___
"""
# f called
# g called
# False
# (f() è True, and deve valutare g(), g() restituisce False)


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 3: MUTABILITÀ E RIFERIMENTI
# ══════════════════════════════════════════════════════════════════════════════
"""
CRITICO per gli esami! Devi capire quando Python copia e quando riferisce.
"""

# ------------------------------------------------------------------------------
# ESERCIZIO L3.1
# ------------------------------------------------------------------------------
"""
Cosa stampa?

a = [1, 2, 3]
b = a
b.append(4)
print(a)
print(b)

Le tue risposte: ___, ___

a e b sono lo STESSO oggetto in memoria!
"""
# print(a)  # [1, 2, 3, 4]
# print(b)  # [1, 2, 3, 4]


# ------------------------------------------------------------------------------
# ESERCIZIO L3.2
# ------------------------------------------------------------------------------
"""
Cosa stampa?

a = [1, 2, 3]
b = a[:]  # oppure a.copy() o list(a)
b.append(4)
print(a)
print(b)

Le tue risposte: ___, ___
"""
# print(a)  # [1, 2, 3] (INTATTA, b è una copia)
# print(b)  # [1, 2, 3, 4]


# ------------------------------------------------------------------------------
# ESERCIZIO L3.3
# ------------------------------------------------------------------------------
"""
TRAPPOLA! Cosa stampa?

a = [[1, 2], [3, 4]]
b = a[:]
b[0][0] = 99
print(a)
print(b)

Le tue risposte: ___, ___

Shallow copy copia solo il PRIMO livello!
"""
# print(a)  # [[99, 2], [3, 4]] - MODIFICATA!
# print(b)  # [[99, 2], [3, 4]]
# Le liste interne sono ancora condivise!


# ------------------------------------------------------------------------------
# ESERCIZIO L3.4
# ------------------------------------------------------------------------------
"""
Cosa stampa?

a = [1, 2, 3]
b = a
a = a + [4]
print(a)
print(b)

Le tue risposte: ___, ___

ATTENZIONE: a + [4] crea una NUOVA lista!
"""
# print(a)  # [1, 2, 3, 4]
# print(b)  # [1, 2, 3] - b ancora punta alla vecchia lista!


# ------------------------------------------------------------------------------
# ESERCIZIO L3.5
# ------------------------------------------------------------------------------
"""
Cosa stampa?

a = [1, 2, 3]
b = a
a += [4]
print(a)
print(b)

Le tue risposte: ___, ___

ATTENZIONE: += per liste è DIVERSO da + !
+= modifica IN PLACE (estende la lista esistente)
"""
# print(a)  # [1, 2, 3, 4]
# print(b)  # [1, 2, 3, 4] - b vede la modifica!


# ------------------------------------------------------------------------------
# ESERCIZIO L3.6
# ------------------------------------------------------------------------------
"""
Cosa stampa?

x = 10
y = x
x += 5
print(x)
print(y)

Le tue risposte: ___, ___

Gli interi sono IMMUTABILI, quindi += crea un nuovo oggetto
"""
# print(x)  # 15
# print(y)  # 10 (y punta ancora al vecchio valore)


# ------------------------------------------------------------------------------
# ESERCIZIO L3.7
# ------------------------------------------------------------------------------
"""
Cosa stampa?

def modify(lst):
    lst.append(4)
    
my_list = [1, 2, 3]
modify(my_list)
print(my_list)

La tua risposta: ___
"""
# print(my_list)  # [1, 2, 3, 4] - la funzione modifica l'originale!


# ------------------------------------------------------------------------------
# ESERCIZIO L3.8
# ------------------------------------------------------------------------------
"""
Cosa stampa?

def modify(lst):
    lst = [4, 5, 6]
    
my_list = [1, 2, 3]
modify(my_list)
print(my_list)

La tua risposta: ___

lst = [4,5,6] crea una NUOVA lista locale, non modifica l'originale
"""
# print(my_list)  # [1, 2, 3] - INVARIATA!


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 4: SCOPE E VARIABILI
# ══════════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------------------
# ESERCIZIO L4.1
# ------------------------------------------------------------------------------
"""
Cosa stampa?

x = "global"

def f():
    x = "local"
    print(x)

f()
print(x)

Output completo: ___
"""
# local
# global


# ------------------------------------------------------------------------------
# ESERCIZIO L4.2
# ------------------------------------------------------------------------------
"""
Cosa stampa?

x = "global"

def f():
    print(x)

f()

Output: ___
"""
# global (legge la variabile globale)


# ------------------------------------------------------------------------------
# ESERCIZIO L4.3 - TRAPPOLA CLASSICA
# ------------------------------------------------------------------------------
"""
Cosa stampa?

x = "global"

def f():
    print(x)
    x = "local"

f()

Output: ___

ERRORE! UnboundLocalError
Perché? Python vede x = "local" e decide che x è LOCALE.
Ma il print cerca di usarla PRIMA dell'assegnazione.
"""
# UnboundLocalError: local variable 'x' referenced before assignment


# ------------------------------------------------------------------------------
# ESERCIZIO L4.4
# ------------------------------------------------------------------------------
"""
Cosa stampa?

x = "global"

def f():
    global x
    print(x)
    x = "modified"

f()
print(x)

Output completo: ___
"""
# global
# modified


# ------------------------------------------------------------------------------
# ESERCIZIO L4.5
# ------------------------------------------------------------------------------
"""
Cosa stampa?

def outer():
    x = "outer"
    
    def inner():
        print(x)
    
    inner()

outer()

Output: ___
"""
# outer (inner può vedere variabili di outer)


# ------------------------------------------------------------------------------
# ESERCIZIO L4.6
# ------------------------------------------------------------------------------
"""
Cosa stampa?

def outer():
    x = "outer"
    
    def inner():
        x = "inner"
        print(x)
    
    inner()
    print(x)

outer()

Output completo: ___
"""
# inner
# outer (x in inner è una variabile diversa)


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 5: SLICING E INDICIZZAZIONE
# ══════════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------------------
# ESERCIZIO L5.1
# ------------------------------------------------------------------------------
"""
Data la stringa s = "Python":

s[0] = ?
s[-1] = ?
s[2:4] = ?
s[:3] = ?
s[3:] = ?
s[::2] = ?
s[::-1] = ?
s[1:-1] = ?

Risposte: ___
"""
s = "Python"
# s[0] = 'P'
# s[-1] = 'n'
# s[2:4] = 'th'
# s[:3] = 'Pyt'
# s[3:] = 'hon'
# s[::2] = 'Pto'
# s[::-1] = 'nohtyP'
# s[1:-1] = 'ytho'


# ------------------------------------------------------------------------------
# ESERCIZIO L5.2
# ------------------------------------------------------------------------------
"""
Cosa stampa?

lst = [0, 1, 2, 3, 4, 5]
print(lst[10:])
print(lst[2:10])
print(lst[-10:2])

Le tue risposte: ___, ___, ___

Lo slicing NON dà errore se fuori range!
"""
# print(lst[10:])   # [] (niente dopo indice 10)
# print(lst[2:10])  # [2, 3, 4, 5] (fino alla fine)
# print(lst[-10:2]) # [0, 1] (da inizio a 2)


# ------------------------------------------------------------------------------
# ESERCIZIO L5.3
# ------------------------------------------------------------------------------
"""
Cosa stampa?

lst = [0, 1, 2, 3, 4]
lst[1:3] = [10, 20, 30]
print(lst)

La tua risposta: ___
"""
# print(lst)  # [0, 10, 20, 30, 3, 4]
# Sostituisce 2 elementi con 3!


# ------------------------------------------------------------------------------
# ESERCIZIO L5.4
# ------------------------------------------------------------------------------
"""
Cosa stampa?

lst = [0, 1, 2, 3, 4]
lst[1:4] = []
print(lst)

La tua risposta: ___
"""
# print(lst)  # [0, 4] (rimuove elementi 1,2,3)


# ------------------------------------------------------------------------------
# ESERCIZIO L5.5
# ------------------------------------------------------------------------------
"""
Cosa stampa?

s = "hello"
print(s[1:4:2])
print(s[-2::-1])

Le tue risposte: ___, ___
"""
# print(s[1:4:2])  # 'el' (indici 1, 3)
# print(s[-2::-1]) # 'lleh' (da -2 all'inizio, al contrario)


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 6: LOOP E CONTROL FLOW
# ══════════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------------------
# ESERCIZIO L6.1
# ------------------------------------------------------------------------------
"""
Cosa stampa?

for i in range(3):
    print(i, end=" ")
else:
    print("done")

Output: ___
"""
# 0 1 2 done
# else viene eseguito perché il loop completa normalmente


# ------------------------------------------------------------------------------
# ESERCIZIO L6.2
# ------------------------------------------------------------------------------
"""
Cosa stampa?

for i in range(5):
    if i == 3:
        break
    print(i, end=" ")
else:
    print("done")

Output: ___
"""
# 0 1 2 
# else NON viene eseguito perché c'è break


# ------------------------------------------------------------------------------
# ESERCIZIO L6.3
# ------------------------------------------------------------------------------
"""
Cosa stampa?

for i in range(5):
    if i == 3:
        continue
    print(i, end=" ")
else:
    print("done")

Output: ___
"""
# 0 1 2 4 done
# continue salta solo un'iterazione, else viene eseguito


# ------------------------------------------------------------------------------
# ESERCIZIO L6.4
# ------------------------------------------------------------------------------
"""
Cosa stampa?

x = 0
while x < 3:
    x += 1
    print(x, end=" ")
else:
    print("done")

Output: ___
"""
# 1 2 3 done


# ------------------------------------------------------------------------------
# ESERCIZIO L6.5
# ------------------------------------------------------------------------------
"""
Cosa stampa?

for i in range(3):
    for j in range(3):
        if j == 1:
            break
        print(f"({i},{j})", end=" ")

Output: ___
"""
# (0,0) (1,0) (2,0)
# break esce solo dal loop interno


# ------------------------------------------------------------------------------
# ESERCIZIO L6.6
# ------------------------------------------------------------------------------
"""
Cosa stampa?

lst = [1, 2, 3]
for i in lst:
    lst.append(i + 3)
    if len(lst) > 6:
        break
print(lst)

La tua risposta: ___

ATTENZIONE: modificare una lista durante l'iterazione!
"""
# [1, 2, 3, 4, 5, 6, 7]


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 7: FUNZIONI E PARAMETRI
# ══════════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------------------
# ESERCIZIO L7.1 - DEFAULT MUTABILE (TRAPPOLA CLASSICA!)
# ------------------------------------------------------------------------------
"""
Cosa stampa?

def append_to(element, lst=[]):
    lst.append(element)
    return lst

print(append_to(1))
print(append_to(2))
print(append_to(3))

Output completo: ___

TRAPPOLA: Il default [] viene creato UNA SOLA VOLTA!
"""
# [1]
# [1, 2]
# [1, 2, 3]
# La stessa lista viene riutilizzata!


# ------------------------------------------------------------------------------
# ESERCIZIO L7.2
# ------------------------------------------------------------------------------
"""
Cosa stampa?

def f(a, b=2, c=3):
    return a + b + c

print(f(1))
print(f(1, 4))
print(f(1, c=10))
print(f(c=10, a=1))

Output: ___
"""
# 6 (1+2+3)
# 8 (1+4+3)
# 13 (1+2+10)
# 13 (1+2+10)


# ------------------------------------------------------------------------------
# ESERCIZIO L7.3
# ------------------------------------------------------------------------------
"""
Cosa stampa?

def f(*args, **kwargs):
    print(len(args), len(kwargs))

f(1, 2, 3, a=4, b=5)

Output: ___
"""
# 3 2


# ------------------------------------------------------------------------------
# ESERCIZIO L7.4
# ------------------------------------------------------------------------------
"""
Cosa stampa?

def f(a, b, *, c):
    return a + b + c

print(f(1, 2, c=3))
# print(f(1, 2, 3))  # Cosa succede?

Output: ___
"""
# 6
# f(1, 2, 3) dà TypeError: f() takes 2 positional arguments but 3 were given


# ------------------------------------------------------------------------------
# ESERCIZIO L7.5
# ------------------------------------------------------------------------------
"""
Cosa stampa?

x = 10

def f():
    x = 20
    def g():
        nonlocal x
        x = 30
    g()
    print(x)

f()
print(x)

Output: ___
"""
# 30 (g modifica x di f)
# 10 (x globale non toccata)


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 8: STRUTTURE DATI - COMPORTAMENTI SPECIALI
# ══════════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------------------
# ESERCIZIO L8.1
# ------------------------------------------------------------------------------
"""
Cosa stampa?

d = {'a': 1, 'b': 2, 'a': 3}
print(d)
print(len(d))

Output: ___
"""
# {'a': 3, 'b': 2}
# 2
# Chiave duplicata: l'ultima sovrascrive


# ------------------------------------------------------------------------------
# ESERCIZIO L8.2
# ------------------------------------------------------------------------------
"""
Cosa stampa?

t = (1, 2, [3, 4])
t[2].append(5)
print(t)
# t[2] = [6, 7]  # Cosa succede?

Output: ___
"""
# (1, 2, [3, 4, 5])
# t[2] = [6, 7] dà TypeError: tuple non supporta assegnazione
# MA possiamo modificare l'oggetto mutabile contenuto!


# ------------------------------------------------------------------------------
# ESERCIZIO L8.3
# ------------------------------------------------------------------------------
"""
Cosa stampa?

s = {1, 2, 3}
s.add(2)
s.add(4)
print(s)
print(len(s))

Output: ___
"""
# {1, 2, 3, 4} (ordine può variare)
# 4 (add(2) non aggiunge duplicato)


# ------------------------------------------------------------------------------
# ESERCIZIO L8.4
# ------------------------------------------------------------------------------
"""
Cosa stampa?

print({1, 2, 3} | {3, 4, 5})  # unione
print({1, 2, 3} & {3, 4, 5})  # intersezione
print({1, 2, 3} - {3, 4, 5})  # differenza
print({1, 2, 3} ^ {3, 4, 5})  # differenza simmetrica

Output: ___
"""
# {1, 2, 3, 4, 5}
# {3}
# {1, 2}
# {1, 2, 4, 5}


# ------------------------------------------------------------------------------
# ESERCIZIO L8.5
# ------------------------------------------------------------------------------
"""
Cosa stampa?

lst = [1, 2, 3]
print(lst * 2)
print([lst] * 2)

# Ora:
nested = [lst] * 2
nested[0].append(4)
print(nested)

Output completo: ___
"""
# [1, 2, 3, 1, 2, 3]
# [[1, 2, 3], [1, 2, 3]]
# [[1, 2, 3, 4], [1, 2, 3, 4]]  # ATTENZIONE: stesso oggetto!


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 9: STRINGHE E FORMATTAZIONE
# ══════════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------------------
# ESERCIZIO L9.1
# ------------------------------------------------------------------------------
"""
Cosa stampa?

s = "hello"
print(s.replace("l", "L", 1))
print(s)

Output: ___
"""
# heLlo
# hello (stringhe sono immutabili, replace restituisce nuova stringa)


# ------------------------------------------------------------------------------
# ESERCIZIO L9.2
# ------------------------------------------------------------------------------
"""
Cosa stampa?

print("a,b,,c".split(","))
print("a b  c".split())
print("a b  c".split(" ")

Output: ___
"""
# ['a', 'b', '', 'c'] (split con separatore mantiene vuoti)
# ['a', 'b', 'c'] (split senza arg raggruppa whitespace)
# ['a', 'b', '', 'c'] (split(" ") non raggruppa)


# ------------------------------------------------------------------------------
# ESERCIZIO L9.3
# ------------------------------------------------------------------------------
"""
Cosa stampa?

print("-".join(['a', 'b', 'c']))
print("".join(['a', 'b', 'c']))
print(list("abc"))

Output: ___
"""
# a-b-c
# abc
# ['a', 'b', 'c']


# ------------------------------------------------------------------------------
# ESERCIZIO L9.4
# ------------------------------------------------------------------------------
"""
Cosa stampa?

x = 42
print(f"{x:05d}")
print(f"{x:>10}")
print(f"{x:<10}|")
print(f"{x:^10}|")

Output: ___
"""
# 00042 (padding con zeri, 5 caratteri)
#         42 (allineato a destra, 10 caratteri)
# 42        | (allineato a sinistra)
#     42    | (centrato)


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 10: ECCEZIONI
# ══════════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------------------
# ESERCIZIO L10.1
# ------------------------------------------------------------------------------
"""
Cosa stampa?

try:
    print("A")
    x = 1 / 0
    print("B")
except ZeroDivisionError:
    print("C")
else:
    print("D")
finally:
    print("E")

Output: ___
"""
# A
# C
# E
# (B non eseguito, D non eseguito perché c'è eccezione)


# ------------------------------------------------------------------------------
# ESERCIZIO L10.2
# ------------------------------------------------------------------------------
"""
Cosa stampa?

try:
    print("A")
    x = 1 / 1
    print("B")
except ZeroDivisionError:
    print("C")
else:
    print("D")
finally:
    print("E")

Output: ___
"""
# A
# B
# D
# E
# (nessuna eccezione, else viene eseguito)


# ------------------------------------------------------------------------------
# ESERCIZIO L10.3
# ------------------------------------------------------------------------------
"""
Cosa stampa?

def f():
    try:
        return 1
    finally:
        return 2

print(f())

Output: ___
"""
# 2
# finally viene SEMPRE eseguito, anche con return!
# Il return in finally sovrascrive quello in try


# ------------------------------------------------------------------------------
# ESERCIZIO L10.4
# ------------------------------------------------------------------------------
"""
Cosa stampa?

try:
    try:
        raise ValueError("inner")
    except TypeError:
        print("A")
    finally:
        print("B")
except ValueError:
    print("C")

Output: ___
"""
# B
# C
# (ValueError non catturata nel try interno, finally eseguito, poi catturata fuori)


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 11: COMPREHENSIONS E GENERATORI
# ══════════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------------------
# ESERCIZIO L11.1
# ------------------------------------------------------------------------------
"""
Cosa stampa?

print([x**2 for x in range(5) if x % 2 == 0])

Output: ___
"""
# [0, 4, 16] (quadrati di 0, 2, 4)


# ------------------------------------------------------------------------------
# ESERCIZIO L11.2
# ------------------------------------------------------------------------------
"""
Cosa stampa?

print([x*y for x in range(3) for y in range(3)])

Output: ___

Equivale a:
for x in range(3):
    for y in range(3):
        result.append(x*y)
"""
# [0, 0, 0, 0, 1, 2, 0, 2, 4]


# ------------------------------------------------------------------------------
# ESERCIZIO L11.3
# ------------------------------------------------------------------------------
"""
Cosa stampa?

g = (x**2 for x in range(3))
print(type(g))
print(list(g))
print(list(g))

Output: ___
"""
# <class 'generator'>
# [0, 1, 4]
# [] (generatore esaurito!)


# ------------------------------------------------------------------------------
# ESERCIZIO L11.4
# ------------------------------------------------------------------------------
"""
Cosa stampa?

print({x: x**2 for x in range(3)})
print({x**2 for x in range(5)})

Output: ___
"""
# {0: 0, 1: 1, 2: 4} (dict comprehension)
# {0, 1, 4, 9, 16} (set comprehension)


# ══════════════════════════════════════════════════════════════════════════════
# SEZIONE 12: QUIZ FINALE - 20 DOMANDE STILE ESAME
# ══════════════════════════════════════════════════════════════════════════════

"""
Completa TUTTO senza eseguire il codice.
Tempo: 20 minuti.
Target: 14/20 (70%)
"""

QUIZ_FINALE = """
Q1. print(10 // 3 * 3 + 10 % 3) = ?

Q2. print(2 ** 2 ** 3) = ?

Q3. x = [1, 2]; y = x; y += [3]; print(x) = ?

Q4. x = [1, 2]; y = x; y = y + [3]; print(x) = ?

Q5. print(bool([]), bool([0]), bool(""), bool(" ")) = ?

Q6. print("abc" < "abd" < "abe") = ?

Q7. print([1,2,3][5:]) = ?

Q8. def f(x=[]): x.append(1); return x
    print(f(), f()) = ?

Q9. x = 5
    def f(): x = 10
    f()
    print(x) = ?

Q10. for i in range(3):
         pass
     else:
         print("done")
     Output = ?

Q11. print(1 or 2 and 3) = ?

Q12. print({1:1, 2:2, 1:3}[1]) = ?

Q13. print("a,b,c".split(",")[1]) = ?

Q14. x = (1, 2, 3)
     print(x[1:]) = ?

Q15. print(type(range(5))) = ?

Q16. try:
         x = 1
     except:
         x = 2
     else:
         x = 3
     finally:
         x = 4
     print(x) = ?

Q17. print([i*j for i in range(2) for j in range(2)]) = ?

Q18. print("hello"[-2::-2]) = ?

Q19. print(not not not True) = ?

Q20. x = "global"
     def f():
         global x
         x = "local"
     f()
     print(x) = ?
"""

RISPOSTE_QUIZ = """
RISPOSTE:

Q1: 10 (10//3=3, 3*3=9, 10%3=1, 9+1=10)
Q2: 256 (2**(2**3) = 2**8 = 256)
Q3: [1, 2, 3] (y+=[] modifica in place)
Q4: [1, 2] (y=y+[] crea nuova lista)
Q5: False, True, False, True
Q6: True (chained comparison)
Q7: [] (slicing oltre fine = vuoto)
Q8: [1] [1, 1] (default mutabile!)
Q9: 5 (x locale non tocca globale)
Q10: done (nessun break)
Q11: 1 (1 è truthy, or short-circuit)
Q12: 3 (chiave duplicata sovrascritta)
Q13: b
Q14: (2, 3)
Q15: <class 'range'>
Q16: 4 (finally sempre eseguito)
Q17: [0, 0, 0, 1]
Q18: lh (da -2 verso inizio, step -2: 'l', poi 'h')
Q19: False (not not not True = not not False = not True = False)
Q20: local (global modifica)
"""

print(QUIZ_FINALE)
print("\n" + "="*70 + "\n")
print(RISPOSTE_QUIZ)


# ══════════════════════════════════════════════════════════════════════════════
# TRACKING PROGRESS
# ══════════════════════════════════════════════════════════════════════════════

"""
CHECKLIST PROGRESSO:

Sezione 1 - Operatori:      [ ] 8/8
Sezione 2 - Boolean:        [ ] 7/7
Sezione 3 - Mutabilità:     [ ] 8/8
Sezione 4 - Scope:          [ ] 6/6
Sezione 5 - Slicing:        [ ] 5/5
Sezione 6 - Loop:           [ ] 6/6
Sezione 7 - Funzioni:       [ ] 5/5
Sezione 8 - Strutture:      [ ] 5/5
Sezione 9 - Stringhe:       [ ] 4/4
Sezione 10 - Eccezioni:     [ ] 4/4
Sezione 11 - Comprehension: [ ] 4/4
Sezione 12 - Quiz Finale:   [ ] /20

TOTALE: ___/82

Se hai sbagliato un esercizio:
1. Segna il numero
2. Capisci PERCHÉ hai sbagliato
3. Ripeti domani
4. Ripeti tra 3 giorni
5. Ripeti tra 1 settimana
"""


if __name__ == "__main__":
    print("\n" + "═" * 70)
    print("PYTHON LOGIC TRAINING COMPLETATO!")
    print("═" * 70)
    print("""
    Questo approccio è ESATTAMENTE quello che serve per l'esame!
    
    Le certificazioni Python Institute testano:
    ✓ Sai PREVEDERE cosa fa il codice?
    ✓ Conosci i COMPORTAMENTI SPECIALI di Python?
    ✓ Capisci PRECEDENZA e ASSOCIATIVITÀ?
    ✓ Conosci le TRAPPOLE comuni?
    
    METODO STUDIO CONSIGLIATO:
    1. Ogni giorno: 10-15 esercizi di logica (30 min)
    2. Scrivi SEMPRE la risposta PRIMA di eseguire
    3. Se sbagli: NON andare avanti finché non capisci
    4. Ripeti gli errori nei giorni successivi
    
    Questo tipo di pratica è più efficace di scrivere 100 programmi!
    """)
