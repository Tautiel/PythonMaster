"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              PROGETTI GUIDATI PYTHON INSTITUTE                               ║
║              Con Soluzioni Complete e Step-by-Step                           ║
║                                                                              ║
║              🎯 Allineati a: PCEP → PCAP → PCPP1 → PCPP2                    ║
║              📊 Research-Based: 70-80% Pratica                               ║
║              ✅ Verificato: Gennaio 2026                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

STRUTTURA DEL FILE:
══════════════════

LIVELLO 1: PROGETTI PCEP (Entry-Level)
  - Progetto 1.1: Calcolatrice Interattiva
  - Progetto 1.2: Quiz Game
  - Progetto 1.3: Gestore Password
  - Progetto 1.4: Convertitore Universale

LIVELLO 2: PROGETTI PCAP (Associate)
  - Progetto 2.1: Rubrica Avanzata (OOP)
  - Progetto 2.2: Sistema Gestione Inventario
  - Progetto 2.3: File Manager con Eccezioni
  - Progetto 2.4: Analizzatore di Testo

LIVELLO 3: PROGETTI PCPP1 (Professional 1)
  - Progetto 3.1: REST API Client
  - Progetto 3.2: GUI Application (Tkinter)
  - Progetto 3.3: Multi-threaded Downloader
  - Progetto 3.4: Design Patterns in Pratica

LIVELLO 4: PROGETTI PCPP2 (Professional 2)
  - Progetto 4.1: Database Manager (SQLite + PostgreSQL)
  - Progetto 4.2: Network Scanner
  - Progetto 4.3: Trading Bot Base (Paper Trading)
  - Progetto 4.4: Sistema Completo con Testing

COME USARE QUESTO FILE:
═══════════════════════

1. LEGGI la descrizione del progetto
2. PROVA a costruirlo da solo (usa gli step come guida)
3. Se bloccato > 30 min → guarda l'hint dello step
4. SOLO DOPO aver provato → confronta con la soluzione
5. RISCRIVI la soluzione da zero senza guardare
6. AGGIUNGI una tua funzionalità extra

"""


# ══════════════════════════════════════════════════════════════════════════════
#                         LIVELLO 1: PROGETTI PCEP
#                         (Entry-Level Python Programmer)
# ══════════════════════════════════════════════════════════════════════════════

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PROGETTO 1.1: CALCOLATRICE INTERATTIVA                                      ║
║                                                                              ║
║  Competenze PCEP testate:                                                    ║
║  ✓ Input/Output                                                              ║
║  ✓ Variabili e tipi dati                                                     ║
║  ✓ Operatori aritmetici                                                      ║
║  ✓ Condizioni (if/elif/else)                                                 ║
║  ✓ Loop (while)                                                              ║
║  ✓ Funzioni base                                                             ║
║                                                                              ║
║  Tempo stimato: 1-2 ore                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Crea la struttura base
# ─────────────────────────────────────────────────────────────────────────────
"""
OBIETTIVO: Creare un menu che mostra le operazioni disponibili

REQUISITI:
- Mostra un menu con: +, -, *, /, //, %, **
- Chiedi all'utente di scegliere un'operazione
- Permetti di uscire con 'q'

HINT: Usa print() per il menu e input() per la scelta
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Aggiungi l'input dei numeri
# ─────────────────────────────────────────────────────────────────────────────
"""
OBIETTIVO: Chiedere due numeri all'utente

REQUISITI:
- Chiedi il primo numero
- Chiedi il secondo numero
- Converti in float per supportare decimali

HINT: float(input("..."))
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Implementa le operazioni
# ─────────────────────────────────────────────────────────────────────────────
"""
OBIETTIVO: Eseguire l'operazione scelta

REQUISITI:
- Usa if/elif/else per ogni operazione
- Gestisci la divisione per zero
- Mostra il risultato formattato

HINT: if operazione == '+': risultato = num1 + num2
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Aggiungi il loop principale
# ─────────────────────────────────────────────────────────────────────────────
"""
OBIETTIVO: Permettere calcoli multipli

REQUISITI:
- Il programma continua finché l'utente non sceglie 'q'
- Dopo ogni calcolo, torna al menu
- Mostra messaggio di arrivederci quando esce

HINT: while True: ... if scelta == 'q': break
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Refactoring in funzioni
# ─────────────────────────────────────────────────────────────────────────────
"""
OBIETTIVO: Organizzare il codice in funzioni

REQUISITI:
- mostra_menu(): stampa il menu
- ottieni_numeri(): chiede e ritorna i due numeri
- calcola(num1, num2, operazione): esegue il calcolo
- main(): funzione principale

HINT: def nome_funzione(parametri): ...
"""

# ═══════════════════════════════════════════════════════════════════════════
# SOLUZIONE COMPLETA PROGETTO 1.1
# ═══════════════════════════════════════════════════════════════════════════

def progetto_1_1_calcolatrice():
    """
    CALCOLATRICE INTERATTIVA - Soluzione Completa
    Allineata a PCEP: input/output, operatori, condizioni, loop, funzioni
    """
    
    def mostra_menu():
        """Mostra il menu delle operazioni disponibili"""
        print("\n" + "="*40)
        print("      CALCOLATRICE PYTHON")
        print("="*40)
        print("  +  : Addizione")
        print("  -  : Sottrazione")
        print("  *  : Moltiplicazione")
        print("  /  : Divisione")
        print("  // : Divisione intera")
        print("  %  : Modulo (resto)")
        print("  ** : Potenza")
        print("  q  : Esci")
        print("="*40)
    
    def ottieni_numeri():
        """Chiede e ritorna due numeri all'utente"""
        while True:
            try:
                num1 = float(input("Inserisci il primo numero: "))
                num2 = float(input("Inserisci il secondo numero: "))
                return num1, num2
            except ValueError:
                print("❌ Errore: inserisci numeri validi!")
    
    def calcola(num1, num2, operazione):
        """Esegue il calcolo e ritorna il risultato"""
        if operazione == '+':
            return num1 + num2
        elif operazione == '-':
            return num1 - num2
        elif operazione == '*':
            return num1 * num2
        elif operazione == '/':
            if num2 == 0:
                return "Errore: divisione per zero!"
            return num1 / num2
        elif operazione == '//':
            if num2 == 0:
                return "Errore: divisione per zero!"
            return num1 // num2
        elif operazione == '%':
            if num2 == 0:
                return "Errore: divisione per zero!"
            return num1 % num2
        elif operazione == '**':
            return num1 ** num2
        else:
            return "Operazione non valida"
    
    def main():
        """Funzione principale"""
        print("\n🧮 Benvenuto nella Calcolatrice Python!")
        
        operazioni_valide = ['+', '-', '*', '/', '//', '%', '**']
        
        while True:
            mostra_menu()
            scelta = input("\nScegli operazione: ").strip()
            
            if scelta.lower() == 'q':
                print("\n👋 Grazie per aver usato la calcolatrice!")
                break
            
            if scelta not in operazioni_valide:
                print("❌ Operazione non valida. Riprova.")
                continue
            
            num1, num2 = ottieni_numeri()
            risultato = calcola(num1, num2, scelta)
            
            print(f"\n✅ Risultato: {num1} {scelta} {num2} = {risultato}")
    
    # Avvia il programma
    main()


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PROGETTO 1.2: QUIZ GAME                                                     ║
║                                                                              ║
║  Competenze PCEP testate:                                                    ║
║  ✓ Liste e dizionari                                                         ║
║  ✓ Loop (for, while)                                                         ║
║  ✓ Condizioni                                                                ║
║  ✓ Funzioni                                                                  ║
║  ✓ Formattazione stringhe                                                    ║
║                                                                              ║
║  Tempo stimato: 2-3 ore                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Crea la struttura dati delle domande
# ─────────────────────────────────────────────────────────────────────────────
"""
OBIETTIVO: Creare una lista di domande con risposte

REQUISITI:
- Ogni domanda è un dizionario con: domanda, opzioni, risposta_corretta
- Almeno 5 domande su Python
- Le opzioni sono una lista

HINT: domande = [{"domanda": "...", "opzioni": ["A", "B", "C"], "corretta": 0}]
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Mostra una domanda
# ─────────────────────────────────────────────────────────────────────────────
"""
OBIETTIVO: Funzione che mostra domanda e opzioni

REQUISITI:
- Mostra il numero della domanda
- Mostra il testo della domanda
- Mostra le opzioni numerate (1, 2, 3, ...)

HINT: for i, opzione in enumerate(opzioni, 1): print(f"{i}. {opzione}")
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Verifica la risposta
# ─────────────────────────────────────────────────────────────────────────────
"""
OBIETTIVO: Controllare se la risposta è corretta

REQUISITI:
- Chiedi all'utente di inserire il numero della risposta
- Confronta con la risposta corretta
- Mostra feedback (corretto/sbagliato)

HINT: if risposta_utente == risposta_corretta: ...
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Calcola e mostra il punteggio
# ─────────────────────────────────────────────────────────────────────────────
"""
OBIETTIVO: Tenere traccia del punteggio

REQUISITI:
- Conta le risposte corrette
- Mostra punteggio finale (es: 4/5)
- Mostra percentuale
- Mostra messaggio basato sul risultato

HINT: punteggio += 1 se corretto
"""

# ═══════════════════════════════════════════════════════════════════════════
# SOLUZIONE COMPLETA PROGETTO 1.2
# ═══════════════════════════════════════════════════════════════════════════

def progetto_1_2_quiz_game():
    """
    QUIZ GAME - Soluzione Completa
    Allineata a PCEP: liste, dizionari, loop, condizioni, funzioni
    """
    
    # Database domande (lista di dizionari)
    domande = [
        {
            "domanda": "Quale funzione stampa output in Python?",
            "opzioni": ["echo()", "print()", "write()", "output()"],
            "corretta": 1  # indice 1 = print()
        },
        {
            "domanda": "Come si dichiara una lista in Python?",
            "opzioni": ["lista = {}", "lista = ()", "lista = []", "lista = <>"],
            "corretta": 2  # indice 2 = []
        },
        {
            "domanda": "Quale operatore calcola il resto della divisione?",
            "opzioni": ["/", "//", "%", "**"],
            "corretta": 2  # indice 2 = %
        },
        {
            "domanda": "Come si inizia un commento in Python?",
            "opzioni": ["//", "/*", "#", "--"],
            "corretta": 2  # indice 2 = #
        },
        {
            "domanda": "Quale tipo di dato è True/False?",
            "opzioni": ["string", "integer", "boolean", "float"],
            "corretta": 2  # indice 2 = boolean
        },
        {
            "domanda": "Come si definisce una funzione in Python?",
            "opzioni": ["function nome():", "def nome():", "fun nome():", "define nome():"],
            "corretta": 1  # indice 1 = def
        },
        {
            "domanda": "Quale metodo aggiunge un elemento alla fine di una lista?",
            "opzioni": ["add()", "insert()", "append()", "push()"],
            "corretta": 2  # indice 2 = append()
        },
        {
            "domanda": "Come si converte una stringa in intero?",
            "opzioni": ["str()", "int()", "float()", "convert()"],
            "corretta": 1  # indice 1 = int()
        },
        {
            "domanda": "Quale keyword esce da un loop?",
            "opzioni": ["exit", "stop", "break", "end"],
            "corretta": 2  # indice 2 = break
        },
        {
            "domanda": "Come si accede all'ultimo elemento di una lista?",
            "opzioni": ["lista[0]", "lista[-1]", "lista[last]", "lista.last()"],
            "corretta": 1  # indice 1 = lista[-1]
        }
    ]
    
    def mostra_domanda(numero, domanda_dict):
        """Mostra una domanda con le sue opzioni"""
        print(f"\n{'='*50}")
        print(f"  DOMANDA {numero}")
        print('='*50)
        print(f"\n{domanda_dict['domanda']}\n")
        
        for i, opzione in enumerate(domanda_dict['opzioni'], 1):
            print(f"  {i}. {opzione}")
        
        print()
    
    def ottieni_risposta(num_opzioni):
        """Chiede e valida la risposta dell'utente"""
        while True:
            try:
                risposta = int(input("La tua risposta (numero): "))
                if 1 <= risposta <= num_opzioni:
                    return risposta - 1  # Converti in indice (0-based)
                else:
                    print(f"❌ Inserisci un numero tra 1 e {num_opzioni}")
            except ValueError:
                print("❌ Inserisci un numero valido!")
    
    def verifica_risposta(risposta_utente, risposta_corretta, opzioni):
        """Verifica se la risposta è corretta e mostra feedback"""
        if risposta_utente == risposta_corretta:
            print("✅ CORRETTO! Ottimo lavoro!")
            return True
        else:
            print(f"❌ Sbagliato! La risposta corretta era: {opzioni[risposta_corretta]}")
            return False
    
    def mostra_risultato_finale(punteggio, totale):
        """Mostra il risultato finale con feedback"""
        percentuale = (punteggio / totale) * 100
        
        print("\n" + "="*50)
        print("         RISULTATO FINALE")
        print("="*50)
        print(f"\n  Punteggio: {punteggio}/{totale}")
        print(f"  Percentuale: {percentuale:.1f}%")
        print()
        
        # Feedback basato sul risultato
        if percentuale == 100:
            print("  🏆 PERFETTO! Sei un esperto di Python!")
        elif percentuale >= 80:
            print("  🌟 ECCELLENTE! Ottima conoscenza!")
        elif percentuale >= 60:
            print("  👍 BUONO! Continua a studiare!")
        elif percentuale >= 40:
            print("  📚 SUFFICIENTE! Ripassa i concetti!")
        else:
            print("  💪 Da migliorare! Non arrenderti!")
        
        print("="*50)
    
    def main():
        """Funzione principale del quiz"""
        print("\n" + "="*50)
        print("      🎮 QUIZ PYTHON - LIVELLO PCEP")
        print("="*50)
        print("\nBenvenuto! Rispondi alle domande su Python.")
        print(f"Totale domande: {len(domande)}")
        
        input("\nPremi INVIO per iniziare...")
        
        punteggio = 0
        
        for i, domanda in enumerate(domande, 1):
            mostra_domanda(i, domanda)
            risposta = ottieni_risposta(len(domanda['opzioni']))
            
            if verifica_risposta(risposta, domanda['corretta'], domanda['opzioni']):
                punteggio += 1
        
        mostra_risultato_finale(punteggio, len(domande))
    
    # Avvia il quiz
    main()


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PROGETTO 1.3: GESTORE PASSWORD                                              ║
║                                                                              ║
║  Competenze PCEP testate:                                                    ║
║  ✓ Stringhe e metodi stringa                                                 ║
║  ✓ Liste                                                                     ║
║  ✓ Modulo random                                                             ║
║  ✓ Loop e condizioni                                                         ║
║  ✓ Funzioni con parametri default                                            ║
║                                                                              ║
║  Tempo stimato: 2-3 ore                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════
# SOLUZIONE COMPLETA PROGETTO 1.3
# ═══════════════════════════════════════════════════════════════════════════

def progetto_1_3_gestore_password():
    """
    GESTORE PASSWORD - Soluzione Completa
    Allineata a PCEP: stringhe, liste, random, funzioni
    """
    import random
    import string
    
    def genera_password(lunghezza=12, usa_maiuscole=True, usa_numeri=True, usa_speciali=True):
        """
        Genera una password casuale con criteri specificati
        
        Args:
            lunghezza: lunghezza della password (default 12)
            usa_maiuscole: include lettere maiuscole (default True)
            usa_numeri: include numeri (default True)
            usa_speciali: include caratteri speciali (default True)
        
        Returns:
            Una stringa con la password generata
        """
        # Caratteri base (sempre inclusi)
        caratteri = string.ascii_lowercase  # a-z
        
        # Aggiungi caratteri opzionali
        if usa_maiuscole:
            caratteri += string.ascii_uppercase  # A-Z
        if usa_numeri:
            caratteri += string.digits  # 0-9
        if usa_speciali:
            caratteri += "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        # Genera la password
        password = ''.join(random.choice(caratteri) for _ in range(lunghezza))
        
        return password
    
    def valuta_forza_password(password):
        """
        Valuta la forza di una password
        
        Returns:
            Tupla (punteggio, valutazione, suggerimenti)
        """
        punteggio = 0
        suggerimenti = []
        
        # Criterio 1: Lunghezza
        if len(password) >= 8:
            punteggio += 1
        else:
            suggerimenti.append("Usa almeno 8 caratteri")
        
        if len(password) >= 12:
            punteggio += 1
        
        if len(password) >= 16:
            punteggio += 1
        
        # Criterio 2: Lettere minuscole
        if any(c.islower() for c in password):
            punteggio += 1
        else:
            suggerimenti.append("Aggiungi lettere minuscole")
        
        # Criterio 3: Lettere maiuscole
        if any(c.isupper() for c in password):
            punteggio += 1
        else:
            suggerimenti.append("Aggiungi lettere maiuscole")
        
        # Criterio 4: Numeri
        if any(c.isdigit() for c in password):
            punteggio += 1
        else:
            suggerimenti.append("Aggiungi numeri")
        
        # Criterio 5: Caratteri speciali
        speciali = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if any(c in speciali for c in password):
            punteggio += 1
        else:
            suggerimenti.append("Aggiungi caratteri speciali")
        
        # Criterio 6: No sequenze ovvie
        sequenze_comuni = ['123', 'abc', 'qwerty', 'password', '111', 'aaa']
        password_lower = password.lower()
        if not any(seq in password_lower for seq in sequenze_comuni):
            punteggio += 1
        else:
            suggerimenti.append("Evita sequenze comuni (123, abc, ecc.)")
        
        # Determina la valutazione
        if punteggio >= 8:
            valutazione = "🔒 MOLTO FORTE"
        elif punteggio >= 6:
            valutazione = "✅ FORTE"
        elif punteggio >= 4:
            valutazione = "⚠️ MEDIA"
        elif punteggio >= 2:
            valutazione = "❌ DEBOLE"
        else:
            valutazione = "💀 MOLTO DEBOLE"
        
        return punteggio, valutazione, suggerimenti
    
    def mostra_menu():
        """Mostra il menu principale"""
        print("\n" + "="*50)
        print("      🔐 GESTORE PASSWORD")
        print("="*50)
        print("  1. Genera nuova password")
        print("  2. Valuta forza password")
        print("  3. Genera password personalizzata")
        print("  q. Esci")
        print("="*50)
    
    def genera_personalizzata():
        """Chiede parametri e genera password personalizzata"""
        print("\n📝 CONFIGURAZIONE PASSWORD:")
        
        # Lunghezza
        while True:
            try:
                lunghezza = int(input("Lunghezza (8-50): "))
                if 8 <= lunghezza <= 50:
                    break
                print("❌ Inserisci un numero tra 8 e 50")
            except ValueError:
                print("❌ Inserisci un numero valido")
        
        # Opzioni
        usa_maiuscole = input("Includere maiuscole? (s/n): ").lower() == 's'
        usa_numeri = input("Includere numeri? (s/n): ").lower() == 's'
        usa_speciali = input("Includere caratteri speciali? (s/n): ").lower() == 's'
        
        password = genera_password(lunghezza, usa_maiuscole, usa_numeri, usa_speciali)
        
        print(f"\n🔑 Password generata: {password}")
        
        # Valuta la password generata
        punteggio, valutazione, _ = valuta_forza_password(password)
        print(f"   Forza: {valutazione} ({punteggio}/9)")
    
    def main():
        """Funzione principale"""
        print("\n🔐 Benvenuto nel Gestore Password!")
        
        while True:
            mostra_menu()
            scelta = input("\nScegli opzione: ").strip()
            
            if scelta == '1':
                # Genera password standard
                password = genera_password()
                print(f"\n🔑 Password generata: {password}")
                punteggio, valutazione, _ = valuta_forza_password(password)
                print(f"   Forza: {valutazione}")
                
            elif scelta == '2':
                # Valuta password esistente
                password = input("\nInserisci la password da valutare: ")
                punteggio, valutazione, suggerimenti = valuta_forza_password(password)
                
                print(f"\n📊 ANALISI PASSWORD:")
                print(f"   Forza: {valutazione}")
                print(f"   Punteggio: {punteggio}/9")
                
                if suggerimenti:
                    print("\n💡 Suggerimenti per migliorare:")
                    for s in suggerimenti:
                        print(f"   • {s}")
                        
            elif scelta == '3':
                # Password personalizzata
                genera_personalizzata()
                
            elif scelta.lower() == 'q':
                print("\n👋 Arrivederci!")
                break
            
            else:
                print("❌ Opzione non valida")
    
    # Avvia il programma
    main()


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PROGETTO 1.4: CONVERTITORE UNIVERSALE                                       ║
║                                                                              ║
║  Competenze PCEP testate:                                                    ║
║  ✓ Dizionari (fattori di conversione)                                        ║
║  ✓ Funzioni con return                                                       ║
║  ✓ Gestione input/output                                                     ║
║  ✓ Formattazione numeri                                                      ║
║                                                                              ║
║  Tempo stimato: 2 ore                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════
# SOLUZIONE COMPLETA PROGETTO 1.4
# ═══════════════════════════════════════════════════════════════════════════

def progetto_1_4_convertitore():
    """
    CONVERTITORE UNIVERSALE - Soluzione Completa
    Allineata a PCEP: dizionari, funzioni, formattazione
    """
    
    # Fattori di conversione (tutto verso unità base)
    CONVERSIONI = {
        'lunghezza': {
            'unita_base': 'metri',
            'fattori': {
                'millimetri': 0.001,
                'centimetri': 0.01,
                'metri': 1,
                'chilometri': 1000,
                'pollici': 0.0254,
                'piedi': 0.3048,
                'miglia': 1609.344
            }
        },
        'peso': {
            'unita_base': 'grammi',
            'fattori': {
                'milligrammi': 0.001,
                'grammi': 1,
                'chilogrammi': 1000,
                'once': 28.3495,
                'libbre': 453.592
            }
        },
        'temperatura': {
            'unita_base': 'celsius',
            'speciale': True  # Le temperature richiedono formule speciali
        },
        'tempo': {
            'unita_base': 'secondi',
            'fattori': {
                'secondi': 1,
                'minuti': 60,
                'ore': 3600,
                'giorni': 86400,
                'settimane': 604800
            }
        },
        'dati': {
            'unita_base': 'bytes',
            'fattori': {
                'bytes': 1,
                'kilobytes': 1024,
                'megabytes': 1048576,
                'gigabytes': 1073741824,
                'terabytes': 1099511627776
            }
        }
    }
    
    def converti_temperatura(valore, da_unita, a_unita):
        """Converte temperature (richiede formule speciali)"""
        # Prima converti tutto in Celsius
        if da_unita == 'celsius':
            celsius = valore
        elif da_unita == 'fahrenheit':
            celsius = (valore - 32) * 5/9
        elif da_unita == 'kelvin':
            celsius = valore - 273.15
        else:
            return None
        
        # Poi converti da Celsius all'unità di destinazione
        if a_unita == 'celsius':
            return celsius
        elif a_unita == 'fahrenheit':
            return celsius * 9/5 + 32
        elif a_unita == 'kelvin':
            return celsius + 273.15
        else:
            return None
    
    def converti(valore, da_unita, a_unita, categoria):
        """
        Converte un valore da un'unità all'altra
        
        Args:
            valore: il numero da convertire
            da_unita: unità di partenza
            a_unita: unità di destinazione
            categoria: tipo di conversione (lunghezza, peso, ecc.)
        
        Returns:
            Il valore convertito o None se errore
        """
        if categoria not in CONVERSIONI:
            return None
        
        config = CONVERSIONI[categoria]
        
        # Gestione speciale per le temperature
        if config.get('speciale'):
            return converti_temperatura(valore, da_unita, a_unita)
        
        fattori = config['fattori']
        
        if da_unita not in fattori or a_unita not in fattori:
            return None
        
        # Converti prima in unità base, poi nell'unità di destinazione
        valore_base = valore * fattori[da_unita]
        risultato = valore_base / fattori[a_unita]
        
        return risultato
    
    def mostra_menu_principale():
        """Mostra il menu principale"""
        print("\n" + "="*50)
        print("      🔄 CONVERTITORE UNIVERSALE")
        print("="*50)
        print("  1. Lunghezza")
        print("  2. Peso")
        print("  3. Temperatura")
        print("  4. Tempo")
        print("  5. Dati (bytes)")
        print("  q. Esci")
        print("="*50)
    
    def mostra_unita(categoria):
        """Mostra le unità disponibili per una categoria"""
        if categoria == 'temperatura':
            unita = ['celsius', 'fahrenheit', 'kelvin']
        else:
            unita = list(CONVERSIONI[categoria]['fattori'].keys())
        
        print(f"\nUnità disponibili:")
        for i, u in enumerate(unita, 1):
            print(f"  {i}. {u}")
        
        return unita
    
    def esegui_conversione(categoria):
        """Esegue una conversione per la categoria specificata"""
        unita = mostra_unita(categoria)
        
        print("\n" + "-"*30)
        
        # Chiedi unità di partenza
        while True:
            try:
                idx = int(input("Da quale unità? (numero): ")) - 1
                if 0 <= idx < len(unita):
                    da_unita = unita[idx]
                    break
                print("❌ Numero non valido")
            except ValueError:
                print("❌ Inserisci un numero")
        
        # Chiedi unità di destinazione
        while True:
            try:
                idx = int(input("A quale unità? (numero): ")) - 1
                if 0 <= idx < len(unita):
                    a_unita = unita[idx]
                    break
                print("❌ Numero non valido")
            except ValueError:
                print("❌ Inserisci un numero")
        
        # Chiedi il valore
        while True:
            try:
                valore = float(input(f"Valore in {da_unita}: "))
                break
            except ValueError:
                print("❌ Inserisci un numero valido")
        
        # Esegui conversione
        risultato = converti(valore, da_unita, a_unita, categoria)
        
        if risultato is not None:
            # Formatta il risultato
            if abs(risultato) < 0.001 or abs(risultato) > 1000000:
                formato = f"{risultato:.6e}"
            else:
                formato = f"{risultato:.4f}"
            
            print(f"\n✅ {valore} {da_unita} = {formato} {a_unita}")
        else:
            print("❌ Errore nella conversione")
    
    def main():
        """Funzione principale"""
        categorie = {
            '1': 'lunghezza',
            '2': 'peso',
            '3': 'temperatura',
            '4': 'tempo',
            '5': 'dati'
        }
        
        print("\n🔄 Benvenuto nel Convertitore Universale!")
        
        while True:
            mostra_menu_principale()
            scelta = input("\nScegli categoria: ").strip()
            
            if scelta.lower() == 'q':
                print("\n👋 Arrivederci!")
                break
            
            if scelta in categorie:
                esegui_conversione(categorie[scelta])
            else:
                print("❌ Opzione non valida")
    
    # Avvia il programma
    main()


# ══════════════════════════════════════════════════════════════════════════════
#                         LIVELLO 2: PROGETTI PCAP
#                         (Associate Python Programmer)
# ══════════════════════════════════════════════════════════════════════════════

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PROGETTO 2.1: RUBRICA AVANZATA (OOP)                                        ║
║                                                                              ║
║  Competenze PCAP testate:                                                    ║
║  ✓ Classi e oggetti                                                          ║
║  ✓ Metodi speciali (__init__, __str__, __repr__)                             ║
║  ✓ Incapsulamento (attributi privati)                                        ║
║  ✓ Gestione eccezioni                                                        ║
║  ✓ File I/O (salvataggio/caricamento)                                        ║
║                                                                              ║
║  Tempo stimato: 3-4 ore                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════
# SOLUZIONE COMPLETA PROGETTO 2.1
# ═══════════════════════════════════════════════════════════════════════════

def progetto_2_1_rubrica_oop():
    """
    RUBRICA AVANZATA CON OOP - Soluzione Completa
    Allineata a PCAP: classi, metodi speciali, eccezioni, file I/O
    """
    import json
    from datetime import datetime
    
    class Contatto:
        """
        Classe che rappresenta un singolo contatto
        Dimostra: __init__, __str__, __repr__, @property, validazione
        """
        
        def __init__(self, nome, telefono, email=None, note=None):
            """
            Inizializza un nuovo contatto
            
            Args:
                nome: Nome del contatto (obbligatorio)
                telefono: Numero di telefono (obbligatorio)
                email: Indirizzo email (opzionale)
                note: Note aggiuntive (opzionale)
            """
            self._nome = None
            self._telefono = None
            self._email = None
            self._note = note
            self._data_creazione = datetime.now()
            
            # Usa i setter per validazione
            self.nome = nome
            self.telefono = telefono
            if email:
                self.email = email
        
        @property
        def nome(self):
            """Getter per il nome"""
            return self._nome
        
        @nome.setter
        def nome(self, valore):
            """Setter per il nome con validazione"""
            if not valore or not valore.strip():
                raise ValueError("Il nome non può essere vuoto")
            self._nome = valore.strip().title()
        
        @property
        def telefono(self):
            """Getter per il telefono"""
            return self._telefono
        
        @telefono.setter
        def telefono(self, valore):
            """Setter per il telefono con validazione"""
            # Rimuovi spazi e caratteri non numerici (eccetto +)
            pulito = ''.join(c for c in valore if c.isdigit() or c == '+')
            if len(pulito) < 6:
                raise ValueError("Numero di telefono non valido")
            self._telefono = pulito
        
        @property
        def email(self):
            """Getter per l'email"""
            return self._email
        
        @email.setter
        def email(self, valore):
            """Setter per l'email con validazione base"""
            if valore and '@' not in valore:
                raise ValueError("Email non valida")
            self._email = valore
        
        def to_dict(self):
            """Converte il contatto in dizionario (per JSON)"""
            return {
                'nome': self._nome,
                'telefono': self._telefono,
                'email': self._email,
                'note': self._note,
                'data_creazione': self._data_creazione.isoformat()
            }
        
        @classmethod
        def from_dict(cls, dati):
            """Crea un contatto da un dizionario"""
            contatto = cls(dati['nome'], dati['telefono'], dati.get('email'), dati.get('note'))
            if 'data_creazione' in dati:
                contatto._data_creazione = datetime.fromisoformat(dati['data_creazione'])
            return contatto
        
        def __str__(self):
            """Rappresentazione stringa user-friendly"""
            risultato = f"📇 {self._nome}\n"
            risultato += f"   📞 {self._telefono}\n"
            if self._email:
                risultato += f"   📧 {self._email}\n"
            if self._note:
                risultato += f"   📝 {self._note}\n"
            return risultato
        
        def __repr__(self):
            """Rappresentazione stringa per debug"""
            return f"Contatto(nome='{self._nome}', telefono='{self._telefono}')"
    
    
    class Rubrica:
        """
        Classe che gestisce una collezione di contatti
        Dimostra: composizione, file I/O, ricerca, ordinamento
        """
        
        def __init__(self, nome_file="rubrica.json"):
            """Inizializza la rubrica"""
            self._contatti = []
            self._nome_file = nome_file
            self._carica()
        
        def _carica(self):
            """Carica i contatti dal file JSON"""
            try:
                with open(self._nome_file, 'r', encoding='utf-8') as f:
                    dati = json.load(f)
                    self._contatti = [Contatto.from_dict(c) for c in dati]
                print(f"✅ Caricati {len(self._contatti)} contatti")
            except FileNotFoundError:
                print("📝 Rubrica nuova creata")
                self._contatti = []
            except json.JSONDecodeError:
                print("⚠️ File corrotto, rubrica vuota")
                self._contatti = []
        
        def _salva(self):
            """Salva i contatti nel file JSON"""
            try:
                with open(self._nome_file, 'w', encoding='utf-8') as f:
                    dati = [c.to_dict() for c in self._contatti]
                    json.dump(dati, f, indent=2, ensure_ascii=False)
                return True
            except IOError as e:
                print(f"❌ Errore salvataggio: {e}")
                return False
        
        def aggiungi(self, contatto):
            """Aggiunge un contatto alla rubrica"""
            self._contatti.append(contatto)
            self._salva()
            print(f"✅ Contatto '{contatto.nome}' aggiunto!")
        
        def rimuovi(self, nome):
            """Rimuove un contatto per nome"""
            nome_lower = nome.lower()
            for i, c in enumerate(self._contatti):
                if c.nome.lower() == nome_lower:
                    rimosso = self._contatti.pop(i)
                    self._salva()
                    print(f"✅ Contatto '{rimosso.nome}' rimosso!")
                    return True
            print(f"❌ Contatto '{nome}' non trovato")
            return False
        
        def cerca(self, query):
            """Cerca contatti per nome, telefono o email"""
            query_lower = query.lower()
            risultati = []
            
            for c in self._contatti:
                if (query_lower in c.nome.lower() or 
                    query in c.telefono or 
                    (c.email and query_lower in c.email.lower())):
                    risultati.append(c)
            
            return risultati
        
        def lista_tutti(self):
            """Ritorna tutti i contatti ordinati per nome"""
            return sorted(self._contatti, key=lambda c: c.nome)
        
        def __len__(self):
            """Ritorna il numero di contatti"""
            return len(self._contatti)
        
        def __iter__(self):
            """Permette di iterare sui contatti"""
            return iter(self._contatti)
    
    
    def main():
        """Funzione principale - CLI della rubrica"""
        rubrica = Rubrica()
        
        def mostra_menu():
            print("\n" + "="*50)
            print("      📒 RUBRICA TELEFONICA")
            print("="*50)
            print(f"  Contatti salvati: {len(rubrica)}")
            print("-"*50)
            print("  1. Aggiungi contatto")
            print("  2. Cerca contatto")
            print("  3. Mostra tutti")
            print("  4. Rimuovi contatto")
            print("  5. Modifica contatto")
            print("  q. Esci")
            print("="*50)
        
        while True:
            mostra_menu()
            scelta = input("\nScegli opzione: ").strip()
            
            if scelta == '1':
                # Aggiungi contatto
                print("\n📝 NUOVO CONTATTO")
                try:
                    nome = input("Nome: ")
                    telefono = input("Telefono: ")
                    email = input("Email (opzionale): ") or None
                    note = input("Note (opzionale): ") or None
                    
                    contatto = Contatto(nome, telefono, email, note)
                    rubrica.aggiungi(contatto)
                except ValueError as e:
                    print(f"❌ Errore: {e}")
            
            elif scelta == '2':
                # Cerca contatto
                query = input("\n🔍 Cerca: ")
                risultati = rubrica.cerca(query)
                
                if risultati:
                    print(f"\n📋 Trovati {len(risultati)} contatti:")
                    for c in risultati:
                        print(c)
                else:
                    print("❌ Nessun contatto trovato")
            
            elif scelta == '3':
                # Mostra tutti
                contatti = rubrica.lista_tutti()
                if contatti:
                    print(f"\n📋 TUTTI I CONTATTI ({len(contatti)}):")
                    print("-"*40)
                    for c in contatti:
                        print(c)
                else:
                    print("📭 La rubrica è vuota")
            
            elif scelta == '4':
                # Rimuovi contatto
                nome = input("\n🗑️ Nome da rimuovere: ")
                rubrica.rimuovi(nome)
            
            elif scelta == '5':
                # Modifica contatto
                nome = input("\n✏️ Nome da modificare: ")
                risultati = rubrica.cerca(nome)
                
                if risultati:
                    print("Contatto trovato:")
                    print(risultati[0])
                    
                    nuovo_tel = input("Nuovo telefono (invio per mantenere): ")
                    nuova_email = input("Nuova email (invio per mantenere): ")
                    
                    if nuovo_tel:
                        try:
                            risultati[0].telefono = nuovo_tel
                        except ValueError as e:
                            print(f"❌ {e}")
                    
                    if nuova_email:
                        try:
                            risultati[0].email = nuova_email
                        except ValueError as e:
                            print(f"❌ {e}")
                    
                    rubrica._salva()
                    print("✅ Contatto aggiornato!")
                else:
                    print("❌ Contatto non trovato")
            
            elif scelta.lower() == 'q':
                print("\n👋 Arrivederci!")
                break
            
            else:
                print("❌ Opzione non valida")
    
    # Avvia il programma
    main()


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PROGETTO 2.2: SISTEMA GESTIONE INVENTARIO                                   ║
║                                                                              ║
║  Competenze PCAP testate:                                                    ║
║  ✓ Ereditarietà                                                              ║
║  ✓ Polimorfismo                                                              ║
║  ✓ Classi astratte (ABC)                                                     ║
║  ✓ Decoratori @property, @classmethod                                        ║
║  ✓ Gestione eccezioni personalizzate                                         ║
║                                                                              ║
║  Tempo stimato: 4-5 ore                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════
# SOLUZIONE COMPLETA PROGETTO 2.2
# ═══════════════════════════════════════════════════════════════════════════

def progetto_2_2_inventario():
    """
    SISTEMA GESTIONE INVENTARIO - Soluzione Completa
    Allineata a PCAP: ereditarietà, polimorfismo, ABC, decoratori
    """
    from abc import ABC, abstractmethod
    from datetime import datetime
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECCEZIONI PERSONALIZZATE
    # ─────────────────────────────────────────────────────────────────────────
    
    class InventarioError(Exception):
        """Eccezione base per errori di inventario"""
        pass
    
    class QuantitaInsufficienteError(InventarioError):
        """Eccezione quando la quantità richiesta supera quella disponibile"""
        def __init__(self, prodotto, richiesta, disponibile):
            self.prodotto = prodotto
            self.richiesta = richiesta
            self.disponibile = disponibile
            super().__init__(
                f"Quantità insufficiente per '{prodotto}': "
                f"richiesti {richiesta}, disponibili {disponibile}"
            )
    
    class ProdottoNonTrovatoError(InventarioError):
        """Eccezione quando un prodotto non esiste"""
        def __init__(self, codice):
            self.codice = codice
            super().__init__(f"Prodotto con codice '{codice}' non trovato")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CLASSE ASTRATTA BASE
    # ─────────────────────────────────────────────────────────────────────────
    
    class Prodotto(ABC):
        """
        Classe astratta base per tutti i prodotti
        Dimostra: ABC, @abstractmethod, @property
        """
        
        _contatore_codice = 0  # Variabile di classe
        
        def __init__(self, nome, prezzo, quantita=0):
            Prodotto._contatore_codice += 1
            self._codice = f"PROD{Prodotto._contatore_codice:04d}"
            self._nome = nome
            self._prezzo = prezzo
            self._quantita = quantita
            self._data_aggiunta = datetime.now()
        
        @property
        def codice(self):
            return self._codice
        
        @property
        def nome(self):
            return self._nome
        
        @property
        def prezzo(self):
            return self._prezzo
        
        @prezzo.setter
        def prezzo(self, valore):
            if valore < 0:
                raise ValueError("Il prezzo non può essere negativo")
            self._prezzo = valore
        
        @property
        def quantita(self):
            return self._quantita
        
        @quantita.setter
        def quantita(self, valore):
            if valore < 0:
                raise ValueError("La quantità non può essere negativa")
            self._quantita = valore
        
        @property
        def valore_totale(self):
            """Calcola il valore totale dello stock"""
            return self._prezzo * self._quantita
        
        @abstractmethod
        def descrizione(self):
            """Metodo astratto - ogni sottoclasse deve implementarlo"""
            pass
        
        @abstractmethod
        def calcola_sconto(self, percentuale):
            """Calcola prezzo scontato - può variare per tipo"""
            pass
        
        def __str__(self):
            return f"[{self._codice}] {self._nome} - €{self._prezzo:.2f} (Qtà: {self._quantita})"
    
    # ─────────────────────────────────────────────────────────────────────────
    # SOTTOCLASSI (EREDITARIETÀ)
    # ─────────────────────────────────────────────────────────────────────────
    
    class ProdottoElettronico(Prodotto):
        """Prodotti elettronici con garanzia"""
        
        def __init__(self, nome, prezzo, quantita=0, garanzia_mesi=12):
            super().__init__(nome, prezzo, quantita)
            self._garanzia_mesi = garanzia_mesi
        
        @property
        def garanzia_mesi(self):
            return self._garanzia_mesi
        
        def descrizione(self):
            return f"Elettronico: {self._nome} con {self._garanzia_mesi} mesi di garanzia"
        
        def calcola_sconto(self, percentuale):
            # Elettronici: sconto max 20%
            percentuale_effettiva = min(percentuale, 20)
            return self._prezzo * (1 - percentuale_effettiva / 100)
    
    class ProdottoAlimentare(Prodotto):
        """Prodotti alimentari con scadenza"""
        
        def __init__(self, nome, prezzo, quantita=0, data_scadenza=None):
            super().__init__(nome, prezzo, quantita)
            self._data_scadenza = data_scadenza
        
        @property
        def data_scadenza(self):
            return self._data_scadenza
        
        @property
        def giorni_alla_scadenza(self):
            if self._data_scadenza:
                delta = self._data_scadenza - datetime.now()
                return delta.days
            return None
        
        def descrizione(self):
            if self._data_scadenza:
                giorni = self.giorni_alla_scadenza
                return f"Alimentare: {self._nome} - Scade tra {giorni} giorni"
            return f"Alimentare: {self._nome}"
        
        def calcola_sconto(self, percentuale):
            # Alimentari vicini a scadenza: sconto extra
            giorni = self.giorni_alla_scadenza
            if giorni and giorni < 7:
                percentuale += 20  # +20% extra se scade entro 7 giorni
            return self._prezzo * (1 - percentuale / 100)
    
    class ProdottoAbbigliamento(Prodotto):
        """Prodotti di abbigliamento con taglia"""
        
        TAGLIE_VALIDE = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
        
        def __init__(self, nome, prezzo, quantita=0, taglia='M', colore='Nero'):
            super().__init__(nome, prezzo, quantita)
            if taglia.upper() not in self.TAGLIE_VALIDE:
                raise ValueError(f"Taglia non valida. Usa: {self.TAGLIE_VALIDE}")
            self._taglia = taglia.upper()
            self._colore = colore
        
        def descrizione(self):
            return f"Abbigliamento: {self._nome} - Taglia {self._taglia}, {self._colore}"
        
        def calcola_sconto(self, percentuale):
            # Abbigliamento: nessun limite sconto
            return self._prezzo * (1 - percentuale / 100)
    
    # ─────────────────────────────────────────────────────────────────────────
    # CLASSE INVENTARIO
    # ─────────────────────────────────────────────────────────────────────────
    
    class Inventario:
        """Gestisce la collezione di prodotti"""
        
        def __init__(self):
            self._prodotti = {}  # codice -> prodotto
        
        def aggiungi(self, prodotto):
            """Aggiunge un prodotto all'inventario"""
            self._prodotti[prodotto.codice] = prodotto
            print(f"✅ Aggiunto: {prodotto}")
        
        def rimuovi(self, codice):
            """Rimuove un prodotto dall'inventario"""
            if codice not in self._prodotti:
                raise ProdottoNonTrovatoError(codice)
            rimosso = self._prodotti.pop(codice)
            print(f"✅ Rimosso: {rimosso.nome}")
            return rimosso
        
        def cerca(self, codice):
            """Cerca un prodotto per codice"""
            if codice not in self._prodotti:
                raise ProdottoNonTrovatoError(codice)
            return self._prodotti[codice]
        
        def cerca_per_nome(self, nome):
            """Cerca prodotti per nome (parziale)"""
            nome_lower = nome.lower()
            return [p for p in self._prodotti.values() 
                    if nome_lower in p.nome.lower()]
        
        def modifica_quantita(self, codice, delta):
            """Modifica la quantità di un prodotto"""
            prodotto = self.cerca(codice)
            nuova_quantita = prodotto.quantita + delta
            
            if nuova_quantita < 0:
                raise QuantitaInsufficienteError(
                    prodotto.nome, abs(delta), prodotto.quantita
                )
            
            prodotto.quantita = nuova_quantita
            print(f"✅ {prodotto.nome}: quantità aggiornata a {nuova_quantita}")
        
        def valore_totale_inventario(self):
            """Calcola il valore totale dell'inventario"""
            return sum(p.valore_totale for p in self._prodotti.values())
        
        def prodotti_sotto_scorta(self, soglia=5):
            """Ritorna prodotti con quantità sotto la soglia"""
            return [p for p in self._prodotti.values() if p.quantita < soglia]
        
        def lista_tutti(self):
            """Ritorna tutti i prodotti"""
            return list(self._prodotti.values())
        
        def __len__(self):
            return len(self._prodotti)
    
    # ─────────────────────────────────────────────────────────────────────────
    # MAIN
    # ─────────────────────────────────────────────────────────────────────────
    
    def main():
        """Funzione principale"""
        inventario = Inventario()
        
        # Aggiungi prodotti di esempio
        inventario.aggiungi(ProdottoElettronico("Laptop Dell", 899.99, 10, 24))
        inventario.aggiungi(ProdottoElettronico("Mouse Wireless", 29.99, 50, 12))
        inventario.aggiungi(ProdottoAlimentare("Pasta Barilla", 1.50, 100))
        inventario.aggiungi(ProdottoAbbigliamento("T-Shirt Basic", 19.99, 30, 'L', 'Blu'))
        
        def mostra_menu():
            print("\n" + "="*50)
            print("      📦 GESTIONE INVENTARIO")
            print("="*50)
            print(f"  Prodotti: {len(inventario)}")
            print(f"  Valore totale: €{inventario.valore_totale_inventario():,.2f}")
            print("-"*50)
            print("  1. Lista prodotti")
            print("  2. Cerca prodotto")
            print("  3. Aggiungi prodotto")
            print("  4. Modifica quantità")
            print("  5. Prodotti sotto scorta")
            print("  6. Calcola sconto")
            print("  q. Esci")
            print("="*50)
        
        while True:
            mostra_menu()
            scelta = input("\nScegli opzione: ").strip()
            
            try:
                if scelta == '1':
                    # Lista prodotti
                    print("\n📋 LISTA PRODOTTI:")
                    for p in inventario.lista_tutti():
                        print(f"  {p}")
                        print(f"     → {p.descrizione()}")
                
                elif scelta == '2':
                    # Cerca
                    query = input("\n🔍 Cerca (nome o codice): ")
                    if query.startswith("PROD"):
                        risultati = [inventario.cerca(query)]
                    else:
                        risultati = inventario.cerca_per_nome(query)
                    
                    if risultati:
                        print(f"\n📋 Trovati {len(risultati)} prodotti:")
                        for p in risultati:
                            print(f"  {p}")
                    else:
                        print("❌ Nessun prodotto trovato")
                
                elif scelta == '3':
                    # Aggiungi
                    print("\n📝 NUOVO PRODOTTO")
                    print("Tipo: 1=Elettronico, 2=Alimentare, 3=Abbigliamento")
                    tipo = input("Tipo: ")
                    nome = input("Nome: ")
                    prezzo = float(input("Prezzo: "))
                    quantita = int(input("Quantità: "))
                    
                    if tipo == '1':
                        garanzia = int(input("Mesi garanzia: "))
                        prodotto = ProdottoElettronico(nome, prezzo, quantita, garanzia)
                    elif tipo == '2':
                        prodotto = ProdottoAlimentare(nome, prezzo, quantita)
                    elif tipo == '3':
                        taglia = input("Taglia (XS-XXL): ")
                        colore = input("Colore: ")
                        prodotto = ProdottoAbbigliamento(nome, prezzo, quantita, taglia, colore)
                    else:
                        print("❌ Tipo non valido")
                        continue
                    
                    inventario.aggiungi(prodotto)
                
                elif scelta == '4':
                    # Modifica quantità
                    codice = input("\nCodice prodotto: ")
                    delta = int(input("Variazione (+/-): "))
                    inventario.modifica_quantita(codice, delta)
                
                elif scelta == '5':
                    # Sotto scorta
                    soglia = int(input("\nSoglia scorta minima: ") or "5")
                    sotto_scorta = inventario.prodotti_sotto_scorta(soglia)
                    
                    if sotto_scorta:
                        print(f"\n⚠️ PRODOTTI SOTTO SCORTA (< {soglia}):")
                        for p in sotto_scorta:
                            print(f"  {p}")
                    else:
                        print("✅ Tutti i prodotti hanno scorte sufficienti")
                
                elif scelta == '6':
                    # Calcola sconto
                    codice = input("\nCodice prodotto: ")
                    percentuale = float(input("Percentuale sconto: "))
                    prodotto = inventario.cerca(codice)
                    prezzo_scontato = prodotto.calcola_sconto(percentuale)
                    print(f"\n💰 Prezzo originale: €{prodotto.prezzo:.2f}")
                    print(f"💰 Prezzo scontato: €{prezzo_scontato:.2f}")
                
                elif scelta.lower() == 'q':
                    print("\n👋 Arrivederci!")
                    break
                
                else:
                    print("❌ Opzione non valida")
                    
            except InventarioError as e:
                print(f"❌ {e}")
            except ValueError as e:
                print(f"❌ Errore input: {e}")
    
    # Avvia il programma
    main()


# ══════════════════════════════════════════════════════════════════════════════
#                         LIVELLO 3: PROGETTI PCPP1
#                         (Professional Python Programming 1)
# ══════════════════════════════════════════════════════════════════════════════

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PROGETTO 3.1: REST API CLIENT                                               ║
║                                                                              ║
║  Competenze PCPP1 testate:                                                   ║
║  ✓ Modulo requests                                                           ║
║  ✓ JSON parsing                                                              ║
║  ✓ Gestione errori HTTP                                                      ║
║  ✓ Design pattern (Factory, Singleton)                                       ║
║  ✓ Logging                                                                   ║
║                                                                              ║
║  Tempo stimato: 4-5 ore                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════
# SOLUZIONE COMPLETA PROGETTO 3.1
# ═══════════════════════════════════════════════════════════════════════════

def progetto_3_1_api_client():
    """
    REST API CLIENT - Soluzione Completa
    Allineata a PCPP1: requests, JSON, error handling, design patterns
    
    Usa CoinGecko API per dati crypto (gratuita, no auth)
    """
    import requests
    import json
    import logging
    from datetime import datetime
    from typing import Optional, Dict, List, Any
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONFIGURAZIONE LOGGING
    # ─────────────────────────────────────────────────────────────────────────
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECCEZIONI PERSONALIZZATE
    # ─────────────────────────────────────────────────────────────────────────
    
    class APIError(Exception):
        """Eccezione base per errori API"""
        pass
    
    class RateLimitError(APIError):
        """Eccezione per rate limiting"""
        pass
    
    class NetworkError(APIError):
        """Eccezione per errori di rete"""
        pass
    
    # ─────────────────────────────────────────────────────────────────────────
    # SINGLETON PATTERN - API CLIENT
    # ─────────────────────────────────────────────────────────────────────────
    
    class CryptoAPIClient:
        """
        Client per CoinGecko API
        Implementa Singleton Pattern
        """
        
        _instance = None
        BASE_URL = "https://api.coingecko.com/api/v3"
        
        def __new__(cls):
            """Singleton: una sola istanza"""
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
        
        def __init__(self):
            if self._initialized:
                return
            
            self._session = requests.Session()
            self._session.headers.update({
                'Accept': 'application/json',
                'User-Agent': 'PythonCryptoTracker/1.0'
            })
            self._cache = {}
            self._cache_duration = 60  # secondi
            self._initialized = True
            logger.info("API Client inizializzato")
        
        def _get(self, endpoint: str, params: Dict = None) -> Dict:
            """
            Esegue una richiesta GET con gestione errori
            
            Args:
                endpoint: endpoint API (senza base URL)
                params: parametri query string
            
            Returns:
                Response JSON come dizionario
            
            Raises:
                APIError: per errori API generici
                RateLimitError: per rate limiting
                NetworkError: per errori di rete
            """
            url = f"{self.BASE_URL}/{endpoint}"
            
            # Check cache
            cache_key = f"{url}:{json.dumps(params or {})}"
            cached = self._cache.get(cache_key)
            if cached:
                timestamp, data = cached
                if (datetime.now() - timestamp).seconds < self._cache_duration:
                    logger.debug(f"Cache hit per {endpoint}")
                    return data
            
            try:
                logger.info(f"GET {endpoint}")
                response = self._session.get(url, params=params, timeout=10)
                
                # Gestione codici HTTP
                if response.status_code == 200:
                    data = response.json()
                    # Salva in cache
                    self._cache[cache_key] = (datetime.now(), data)
                    return data
                
                elif response.status_code == 429:
                    raise RateLimitError("Rate limit raggiunto. Riprova più tardi.")
                
                elif response.status_code == 404:
                    raise APIError(f"Risorsa non trovata: {endpoint}")
                
                else:
                    raise APIError(f"Errore HTTP {response.status_code}")
                    
            except requests.exceptions.Timeout:
                raise NetworkError("Timeout della richiesta")
            except requests.exceptions.ConnectionError:
                raise NetworkError("Errore di connessione")
            except requests.exceptions.RequestException as e:
                raise NetworkError(f"Errore di rete: {e}")
        
        def get_price(self, coin_ids: List[str], vs_currencies: List[str] = None) -> Dict:
            """
            Ottiene i prezzi correnti
            
            Args:
                coin_ids: lista di ID coin (es: ['bitcoin', 'ethereum'])
                vs_currencies: valute target (default: ['usd', 'eur'])
            
            Returns:
                Dizionario con prezzi
            """
            if vs_currencies is None:
                vs_currencies = ['usd', 'eur']
            
            params = {
                'ids': ','.join(coin_ids),
                'vs_currencies': ','.join(vs_currencies)
            }
            
            return self._get('simple/price', params)
        
        def get_coin_details(self, coin_id: str) -> Dict:
            """Ottiene dettagli completi di una coin"""
            params = {
                'localization': 'false',
                'tickers': 'false',
                'community_data': 'false',
                'developer_data': 'false'
            }
            return self._get(f'coins/{coin_id}', params)
        
        def get_market_data(self, vs_currency: str = 'usd', 
                           per_page: int = 10, page: int = 1) -> List[Dict]:
            """Ottiene dati di mercato per le top coins"""
            params = {
                'vs_currency': vs_currency,
                'order': 'market_cap_desc',
                'per_page': per_page,
                'page': page,
                'sparkline': 'false'
            }
            return self._get('coins/markets', params)
        
        def search_coins(self, query: str) -> List[Dict]:
            """Cerca coins per nome o simbolo"""
            data = self._get('search', {'query': query})
            return data.get('coins', [])
    
    # ─────────────────────────────────────────────────────────────────────────
    # FACTORY PATTERN - COIN OBJECTS
    # ─────────────────────────────────────────────────────────────────────────
    
    class Coin:
        """Rappresenta una cryptocurrency"""
        
        def __init__(self, data: Dict):
            self.id = data.get('id')
            self.symbol = data.get('symbol', '').upper()
            self.name = data.get('name')
            self.current_price = data.get('current_price')
            self.market_cap = data.get('market_cap')
            self.price_change_24h = data.get('price_change_percentage_24h')
            self.volume_24h = data.get('total_volume')
        
        def __str__(self):
            change = self.price_change_24h or 0
            emoji = "📈" if change >= 0 else "📉"
            return (f"{self.name} ({self.symbol}): "
                   f"${self.current_price:,.2f} {emoji} {change:+.2f}%")
    
    class CoinFactory:
        """Factory per creare oggetti Coin"""
        
        @staticmethod
        def create_from_market_data(data: Dict) -> Coin:
            return Coin(data)
        
        @staticmethod
        def create_multiple(data_list: List[Dict]) -> List[Coin]:
            return [CoinFactory.create_from_market_data(d) for d in data_list]
    
    # ─────────────────────────────────────────────────────────────────────────
    # MAIN APPLICATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def main():
        """Applicazione principale"""
        client = CryptoAPIClient()
        
        def mostra_menu():
            print("\n" + "="*50)
            print("      🪙 CRYPTO TRACKER")
            print("="*50)
            print("  1. Top 10 Cryptocurrencies")
            print("  2. Prezzo Bitcoin/Ethereum")
            print("  3. Cerca Cryptocurrency")
            print("  4. Dettagli Coin")
            print("  q. Esci")
            print("="*50)
        
        while True:
            mostra_menu()
            scelta = input("\nScegli opzione: ").strip()
            
            try:
                if scelta == '1':
                    # Top 10
                    print("\n📊 TOP 10 CRYPTOCURRENCIES:")
                    print("-"*50)
                    data = client.get_market_data(per_page=10)
                    coins = CoinFactory.create_multiple(data)
                    
                    for i, coin in enumerate(coins, 1):
                        print(f"{i:2}. {coin}")
                
                elif scelta == '2':
                    # BTC/ETH
                    print("\n💰 PREZZI BTC/ETH:")
                    prices = client.get_price(['bitcoin', 'ethereum'], ['usd', 'eur'])
                    
                    for coin_id, coin_prices in prices.items():
                        print(f"\n{coin_id.upper()}:")
                        for currency, price in coin_prices.items():
                            print(f"  {currency.upper()}: {price:,.2f}")
                
                elif scelta == '3':
                    # Cerca
                    query = input("\n🔍 Cerca: ")
                    results = client.search_coins(query)
                    
                    if results:
                        print(f"\n📋 Trovate {len(results)} coin:")
                        for c in results[:10]:
                            print(f"  • {c['name']} ({c['symbol'].upper()}) - ID: {c['id']}")
                    else:
                        print("❌ Nessun risultato")
                
                elif scelta == '4':
                    # Dettagli
                    coin_id = input("\nInserisci ID coin (es: bitcoin): ").lower()
                    details = client.get_coin_details(coin_id)
                    
                    print(f"\n📊 DETTAGLI {details['name'].upper()}:")
                    print(f"  Simbolo: {details['symbol'].upper()}")
                    print(f"  Rank: #{details.get('market_cap_rank', 'N/A')}")
                    
                    market_data = details.get('market_data', {})
                    if market_data:
                        print(f"  Prezzo USD: ${market_data.get('current_price', {}).get('usd', 'N/A'):,.2f}")
                        print(f"  Market Cap: ${market_data.get('market_cap', {}).get('usd', 'N/A'):,.0f}")
                        print(f"  Volume 24h: ${market_data.get('total_volume', {}).get('usd', 'N/A'):,.0f}")
                
                elif scelta.lower() == 'q':
                    print("\n👋 Arrivederci!")
                    break
                
                else:
                    print("❌ Opzione non valida")
                    
            except RateLimitError as e:
                print(f"⚠️ {e}")
            except NetworkError as e:
                print(f"🔌 Errore di rete: {e}")
            except APIError as e:
                print(f"❌ Errore API: {e}")
    
    # Avvia
    main()


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PROGETTO 3.2: GUI APPLICATION (TKINTER)                                     ║
║                                                                              ║
║  Competenze PCPP1 testate:                                                   ║
║  ✓ Tkinter widgets                                                           ║
║  ✓ Event handling                                                            ║
║  ✓ Layout management                                                         ║
║  ✓ MVC Pattern                                                               ║
║                                                                              ║
║  Tempo stimato: 5-6 ore                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# NOTA: Tkinter richiede display. Codice di esempio:

GUI_TKINTER_CODE = '''
import tkinter as tk
from tkinter import ttk, messagebox
import json

class TodoApp:
    """
    TODO List Application con Tkinter
    Implementa pattern MVC-like
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("📝 Todo List")
        self.root.geometry("500x600")
        
        # Model
        self.todos = []
        
        # Configura UI
        self._setup_ui()
        self._load_todos()
    
    def _setup_ui(self):
        """Configura l'interfaccia utente"""
        # Frame principale
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titolo
        title_label = ttk.Label(main_frame, text="📝 TODO LIST", 
                               font=('Helvetica', 20, 'bold'))
        title_label.pack(pady=10)
        
        # Frame input
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=10)
        
        self.entry = ttk.Entry(input_frame, font=('Helvetica', 12))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.entry.bind('<Return>', lambda e: self._add_todo())
        
        add_btn = ttk.Button(input_frame, text="Aggiungi", command=self._add_todo)
        add_btn.pack(side=tk.RIGHT)
        
        # Lista todo
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = tk.Listbox(list_frame, font=('Helvetica', 12),
                                  selectmode=tk.SINGLE,
                                  yscrollcommand=scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        
        # Pulsanti azione
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        
        complete_btn = ttk.Button(btn_frame, text="✓ Completa", 
                                 command=self._toggle_complete)
        complete_btn.pack(side=tk.LEFT, padx=5)
        
        delete_btn = ttk.Button(btn_frame, text="🗑 Elimina", 
                               command=self._delete_todo)
        delete_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(btn_frame, text="🧹 Pulisci completati", 
                              command=self._clear_completed)
        clear_btn.pack(side=tk.RIGHT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="0 task")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var)
        status_bar.pack(pady=10)
    
    def _add_todo(self):
        """Aggiunge un nuovo todo"""
        text = self.entry.get().strip()
        if text:
            self.todos.append({'text': text, 'completed': False})
            self._update_list()
            self.entry.delete(0, tk.END)
            self._save_todos()
    
    def _toggle_complete(self):
        """Segna come completato/non completato"""
        selection = self.listbox.curselection()
        if selection:
            idx = selection[0]
            self.todos[idx]['completed'] = not self.todos[idx]['completed']
            self._update_list()
            self._save_todos()
    
    def _delete_todo(self):
        """Elimina il todo selezionato"""
        selection = self.listbox.curselection()
        if selection:
            idx = selection[0]
            del self.todos[idx]
            self._update_list()
            self._save_todos()
    
    def _clear_completed(self):
        """Rimuove tutti i todo completati"""
        self.todos = [t for t in self.todos if not t['completed']]
        self._update_list()
        self._save_todos()
    
    def _update_list(self):
        """Aggiorna la visualizzazione della lista"""
        self.listbox.delete(0, tk.END)
        
        for todo in self.todos:
            text = todo['text']
            if todo['completed']:
                text = f"✓ {text}"
            self.listbox.insert(tk.END, text)
            
            # Colora i completati
            if todo['completed']:
                self.listbox.itemconfig(tk.END, fg='gray')
        
        # Aggiorna status
        total = len(self.todos)
        completed = sum(1 for t in self.todos if t['completed'])
        self.status_var.set(f"{total} task, {completed} completati")
    
    def _save_todos(self):
        """Salva i todo su file"""
        with open('todos.json', 'w') as f:
            json.dump(self.todos, f)
    
    def _load_todos(self):
        """Carica i todo da file"""
        try:
            with open('todos.json', 'r') as f:
                self.todos = json.load(f)
            self._update_list()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = TodoApp(root)
    root.mainloop()
'''


# ══════════════════════════════════════════════════════════════════════════════
#                         LIVELLO 4: PROGETTI PCPP2
#                         (Professional Python Programming 2)
# ══════════════════════════════════════════════════════════════════════════════

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PROGETTO 4.1: DATABASE MANAGER                                              ║
║                                                                              ║
║  Competenze PCPP2 testate:                                                   ║
║  ✓ SQLite3                                                                   ║
║  ✓ Context managers per DB                                                   ║
║  ✓ ORM-like patterns                                                         ║
║  ✓ Migrazioni database                                                       ║
║                                                                              ║
║  Tempo stimato: 5-6 ore                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════
# SOLUZIONE COMPLETA PROGETTO 4.1
# ═══════════════════════════════════════════════════════════════════════════

def progetto_4_1_database_manager():
    """
    DATABASE MANAGER - Soluzione Completa
    Allineata a PCPP2: SQLite, context managers, ORM patterns
    """
    import sqlite3
    from contextlib import contextmanager
    from datetime import datetime
    from typing import Optional, List, Dict, Any
    
    # ─────────────────────────────────────────────────────────────────────────
    # DATABASE CONNECTION MANAGER
    # ─────────────────────────────────────────────────────────────────────────
    
    class DatabaseManager:
        """
        Gestisce connessioni SQLite con context manager
        """
        
        def __init__(self, db_path: str = "trading_data.db"):
            self.db_path = db_path
            self._init_database()
        
        @contextmanager
        def get_connection(self):
            """Context manager per connessioni database"""
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Accesso per nome colonna
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()
        
        def _init_database(self):
            """Inizializza le tabelle del database"""
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Tabella trades
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
                        quantity REAL NOT NULL,
                        price REAL NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        strategy TEXT,
                        profit_loss REAL,
                        status TEXT DEFAULT 'OPEN'
                    )
                ''')
                
                # Tabella portfolio
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT UNIQUE NOT NULL,
                        quantity REAL NOT NULL DEFAULT 0,
                        avg_price REAL,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Tabella price_history
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS price_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL,
                        timestamp DATETIME NOT NULL,
                        UNIQUE(symbol, timestamp)
                    )
                ''')
                
                # Indici per performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_symbol_time ON price_history(symbol, timestamp)')
    
    # ─────────────────────────────────────────────────────────────────────────
    # MODEL CLASSES (ORM-like)
    # ─────────────────────────────────────────────────────────────────────────
    
    class Trade:
        """Rappresenta un singolo trade"""
        
        def __init__(self, symbol: str, side: str, quantity: float, 
                     price: float, strategy: str = None, id: int = None):
            self.id = id
            self.symbol = symbol.upper()
            self.side = side.upper()
            self.quantity = quantity
            self.price = price
            self.strategy = strategy
            self.timestamp = datetime.now()
            self.profit_loss = None
            self.status = 'OPEN'
        
        @property
        def total_value(self) -> float:
            return self.quantity * self.price
        
        def __repr__(self):
            return f"Trade({self.side} {self.quantity} {self.symbol} @ {self.price})"
    
    
    class TradeRepository:
        """Repository pattern per gestione trades"""
        
        def __init__(self, db_manager: DatabaseManager):
            self.db = db_manager
        
        def save(self, trade: Trade) -> int:
            """Salva un trade e ritorna l'ID"""
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trades (symbol, side, quantity, price, strategy, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (trade.symbol, trade.side, trade.quantity, 
                      trade.price, trade.strategy, trade.status))
                trade.id = cursor.lastrowid
                return trade.id
        
        def find_by_id(self, trade_id: int) -> Optional[Trade]:
            """Trova un trade per ID"""
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM trades WHERE id = ?', (trade_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_trade(row)
                return None
        
        def find_by_symbol(self, symbol: str) -> List[Trade]:
            """Trova tutti i trade per un simbolo"""
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC',
                    (symbol.upper(),)
                )
                return [self._row_to_trade(row) for row in cursor.fetchall()]
        
        def find_open_trades(self) -> List[Trade]:
            """Trova tutti i trade aperti"""
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY timestamp DESC"
                )
                return [self._row_to_trade(row) for row in cursor.fetchall()]
        
        def close_trade(self, trade_id: int, close_price: float) -> float:
            """Chiude un trade e calcola P/L"""
            trade = self.find_by_id(trade_id)
            if not trade:
                raise ValueError(f"Trade {trade_id} non trovato")
            
            # Calcola P/L
            if trade.side == 'BUY':
                profit_loss = (close_price - trade.price) * trade.quantity
            else:
                profit_loss = (trade.price - close_price) * trade.quantity
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE trades 
                    SET status = 'CLOSED', profit_loss = ?
                    WHERE id = ?
                ''', (profit_loss, trade_id))
            
            return profit_loss
        
        def get_statistics(self) -> Dict[str, Any]:
            """Calcola statistiche dei trade"""
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Trade totali e P/L
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) as closed_trades,
                        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
                        SUM(profit_loss) as total_profit_loss,
                        AVG(profit_loss) as avg_profit_loss
                    FROM trades
                ''')
                row = cursor.fetchone()
                
                return {
                    'total_trades': row['total_trades'],
                    'closed_trades': row['closed_trades'],
                    'winning_trades': row['winning_trades'] or 0,
                    'losing_trades': row['losing_trades'] or 0,
                    'total_profit_loss': row['total_profit_loss'] or 0,
                    'avg_profit_loss': row['avg_profit_loss'] or 0,
                    'win_rate': (row['winning_trades'] / row['closed_trades'] * 100) 
                               if row['closed_trades'] else 0
                }
        
        def _row_to_trade(self, row) -> Trade:
            """Converte una riga DB in oggetto Trade"""
            trade = Trade(
                symbol=row['symbol'],
                side=row['side'],
                quantity=row['quantity'],
                price=row['price'],
                strategy=row['strategy'],
                id=row['id']
            )
            trade.status = row['status']
            trade.profit_loss = row['profit_loss']
            return trade
    
    # ─────────────────────────────────────────────────────────────────────────
    # PRICE HISTORY REPOSITORY
    # ─────────────────────────────────────────────────────────────────────────
    
    class PriceHistoryRepository:
        """Repository per storico prezzi"""
        
        def __init__(self, db_manager: DatabaseManager):
            self.db = db_manager
        
        def save_candle(self, symbol: str, open_: float, high: float, 
                       low: float, close: float, volume: float, 
                       timestamp: datetime):
            """Salva una candela OHLCV"""
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO price_history 
                    (symbol, open, high, low, close, volume, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (symbol.upper(), open_, high, low, close, volume, timestamp))
        
        def get_history(self, symbol: str, limit: int = 100) -> List[Dict]:
            """Ottiene lo storico prezzi"""
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM price_history 
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (symbol.upper(), limit))
                
                return [dict(row) for row in cursor.fetchall()]
    
    # ─────────────────────────────────────────────────────────────────────────
    # MAIN APPLICATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def main():
        """Applicazione principale"""
        db = DatabaseManager()
        trade_repo = TradeRepository(db)
        price_repo = PriceHistoryRepository(db)
        
        def mostra_menu():
            print("\n" + "="*50)
            print("      💾 DATABASE MANAGER - TRADING")
            print("="*50)
            print("  1. Registra nuovo trade")
            print("  2. Visualizza trade aperti")
            print("  3. Chiudi trade")
            print("  4. Cerca trade per simbolo")
            print("  5. Statistiche")
            print("  6. Aggiungi dati prezzo")
            print("  q. Esci")
            print("="*50)
        
        while True:
            mostra_menu()
            scelta = input("\nScegli opzione: ").strip()
            
            try:
                if scelta == '1':
                    # Nuovo trade
                    print("\n📝 NUOVO TRADE")
                    symbol = input("Simbolo (es: BTC): ").upper()
                    side = input("Lato (BUY/SELL): ").upper()
                    quantity = float(input("Quantità: "))
                    price = float(input("Prezzo: "))
                    strategy = input("Strategia (opzionale): ") or None
                    
                    trade = Trade(symbol, side, quantity, price, strategy)
                    trade_id = trade_repo.save(trade)
                    print(f"✅ Trade salvato con ID: {trade_id}")
                
                elif scelta == '2':
                    # Trade aperti
                    trades = trade_repo.find_open_trades()
                    if trades:
                        print("\n📊 TRADE APERTI:")
                        for t in trades:
                            print(f"  ID:{t.id} | {t.side} {t.quantity} {t.symbol} @ ${t.price:.2f}")
                    else:
                        print("📭 Nessun trade aperto")
                
                elif scelta == '3':
                    # Chiudi trade
                    trade_id = int(input("\nID trade da chiudere: "))
                    close_price = float(input("Prezzo di chiusura: "))
                    
                    profit_loss = trade_repo.close_trade(trade_id, close_price)
                    emoji = "💰" if profit_loss >= 0 else "💸"
                    print(f"\n{emoji} Trade chiuso. P/L: ${profit_loss:+.2f}")
                
                elif scelta == '4':
                    # Cerca per simbolo
                    symbol = input("\nSimbolo: ").upper()
                    trades = trade_repo.find_by_symbol(symbol)
                    
                    if trades:
                        print(f"\n📊 TRADE {symbol}:")
                        for t in trades:
                            status = "🟢" if t.status == 'OPEN' else "⚪"
                            pl = f"P/L: ${t.profit_loss:+.2f}" if t.profit_loss else ""
                            print(f"  {status} {t.side} {t.quantity} @ ${t.price:.2f} {pl}")
                    else:
                        print(f"❌ Nessun trade per {symbol}")
                
                elif scelta == '5':
                    # Statistiche
                    stats = trade_repo.get_statistics()
                    print("\n📈 STATISTICHE:")
                    print(f"  Trade totali: {stats['total_trades']}")
                    print(f"  Trade chiusi: {stats['closed_trades']}")
                    print(f"  Vincenti: {stats['winning_trades']}")
                    print(f"  Perdenti: {stats['losing_trades']}")
                    print(f"  Win Rate: {stats['win_rate']:.1f}%")
                    print(f"  P/L Totale: ${stats['total_profit_loss']:+.2f}")
                    print(f"  P/L Medio: ${stats['avg_profit_loss']:+.2f}")
                
                elif scelta == '6':
                    # Aggiungi prezzo
                    print("\n📊 AGGIUNGI CANDELA")
                    symbol = input("Simbolo: ").upper()
                    open_ = float(input("Open: "))
                    high = float(input("High: "))
                    low = float(input("Low: "))
                    close = float(input("Close: "))
                    volume = float(input("Volume: "))
                    
                    price_repo.save_candle(symbol, open_, high, low, close, 
                                          volume, datetime.now())
                    print("✅ Candela salvata!")
                
                elif scelta.lower() == 'q':
                    print("\n👋 Arrivederci!")
                    break
                
                else:
                    print("❌ Opzione non valida")
                    
            except ValueError as e:
                print(f"❌ Errore input: {e}")
            except Exception as e:
                print(f"❌ Errore: {e}")
    
    # Avvia
    main()


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PROGETTO 4.3: TRADING BOT BASE (PAPER TRADING)                              ║
║                                                                              ║
║  Competenze PCPP2 testate:                                                   ║
║  ✓ Integrazione completa (API + DB + OOP)                                    ║
║  ✓ Strategy pattern                                                          ║
║  ✓ Event-driven architecture                                                 ║
║  ✓ Logging avanzato                                                          ║
║  ✓ Configuration management                                                  ║
║                                                                              ║
║  Tempo stimato: 8-10 ore                                                     ║
║                                                                              ║
║  ⚠️ QUESTO È UN BOT DI PAPER TRADING (SIMULAZIONE)                          ║
║     NON USARE CON SOLDI REALI SENZA ULTERIORI TEST                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

TRADING_BOT_STRUCTURE = '''
# STRUTTURA CONSIGLIATA PER IL TRADING BOT

trading_bot/
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configurazioni
│   └── logging_config.py    # Setup logging
│
├── core/
│   ├── __init__.py
│   ├── bot.py               # Classe principale Bot
│   ├── portfolio.py         # Gestione portfolio
│   └── order.py             # Ordini
│
├── strategies/
│   ├── __init__.py
│   ├── base.py              # Strategy base (ABC)
│   ├── sma_crossover.py     # SMA Crossover
│   └── rsi_strategy.py      # RSI Strategy
│
├── indicators/
│   ├── __init__.py
│   ├── sma.py               # Simple Moving Average
│   ├── ema.py               # Exponential MA
│   └── rsi.py               # Relative Strength Index
│
├── data/
│   ├── __init__.py
│   ├── fetcher.py           # API data fetcher
│   └── database.py          # Database handler
│
├── utils/
│   ├── __init__.py
│   └── helpers.py           # Utility functions
│
├── tests/
│   ├── __init__.py
│   ├── test_strategies.py
│   └── test_indicators.py
│
├── main.py                  # Entry point
├── requirements.txt
└── README.md
'''


# ══════════════════════════════════════════════════════════════════════════════
#                              INDICE ESECUZIONE
# ══════════════════════════════════════════════════════════════════════════════

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         COME ESEGUIRE I PROGETTI                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Per testare un progetto, chiama la sua funzione:

    # LIVELLO PCEP
    progetto_1_1_calcolatrice()
    progetto_1_2_quiz_game()
    progetto_1_3_gestore_password()
    progetto_1_4_convertitore()
    
    # LIVELLO PCAP
    progetto_2_1_rubrica_oop()
    progetto_2_2_inventario()
    
    # LIVELLO PCPP1
    progetto_3_1_api_client()  # Richiede: pip install requests
    
    # LIVELLO PCPP2
    progetto_4_1_database_manager()

PROGRESSIONE CONSIGLIATA:
═════════════════════════

SETTIMANA 1-2 (dopo PCEP Module 4):
  → Progetto 1.1: Calcolatrice
  → Progetto 1.2: Quiz Game

SETTIMANA 3-4 (consolidamento PCEP):
  → Progetto 1.3: Gestore Password
  → Progetto 1.4: Convertitore

SETTIMANA 5-8 (durante PCAP):
  → Progetto 2.1: Rubrica OOP
  → Progetto 2.2: Inventario

SETTIMANA 9-12 (durante PCPP1):
  → Progetto 3.1: API Client
  → Progetto 3.2: GUI Tkinter

SETTIMANA 13-16 (durante PCPP2):
  → Progetto 4.1: Database Manager
  → Progetto 4.3: Trading Bot

══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║          PROGETTI GUIDATI PYTHON INSTITUTE                    ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║  Questo file contiene 10+ progetti completi con soluzioni    ║
    ║  allineati alle certificazioni PCEP, PCAP, PCPP1, PCPP2      ║
    ║                                                               ║
    ║  Per eseguire un progetto:                                    ║
    ║  >>> progetto_1_1_calcolatrice()                             ║
    ║  >>> progetto_1_2_quiz_game()                                ║
    ║  >>> ... etc                                                  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
