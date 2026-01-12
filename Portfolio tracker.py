# ==========================================
# PORFOLIO TRACKER
# ==========================================
"""
STATO
 ├── portfolio
 └── transactions

OPERAZIONI
 ├── buy()
 ├── sell()
 ├── show_portfolio()
 ├── show_transactions()
 └── calculate_pnl()
 
 OPPURE PIù CHIARAMENTE
 
 STATO
 ├── portfolio (cosa possiedi ora)
 └── transactions (storia degli eventi)

OPERAZIONI
 ├── buy()  → modifica portfolio + registra evento
 ├── sell() → modifica portfolio + registra evento
 ├── show_portfolio() → legge portfolio
 ├── show_transactions() → legge transactions
 └── calculate_pnl() → legge entrambi
"""

#---STATO---

def portfolio_tracker():
    
    portfolio = {}
    transactions = []
    
    #---OPERAZIONI---
    
    def buy(symbol, qty, price):
        '''Registra un acquisto nel portfolio e nello storico '''
        
        '''aggiorna il portfolio'''
        portfolio[symbol] = portfolio.get(symbol, 0) + qty
        
        '''Crea la transazione'''
        transaction = {
            'type': 'BUY',
            'symbol': symbol,
            'qty': qty,
            'price': price,
            'value': qty * price
        }
        
        '''Aggiungi allo storico'''
        transactions.append(transaction)
        
        '''feedback'''
        print(f"BUY {qty} {symbol} @ ${price}")
        pass
    
    def sell(symbol, qty, price):
        '''Registra una vendita nel portfolio e nello storico'''
        if symbol not in portfolio:
            print(f"Errore non possiedi: {symbol}")
            return
        
        '''Aggiorna il portfolio e controlla'''
        if portfolio[symbol] < qty:
            print(f"Errore: non possiedi sufficienti quantità di : {symbol}")
            return
        
        '''Aggiorna'''
        portfolio[symbol] -= qty
        
        '''Rimuovi se quantità pari a 0'''
        if portfolio[symbol] == 0:
            del portfolio[symbol]
            return
                
        '''Crea la transazione'''
        transaction = {
            'type': 'SELL',
            'symbol': symbol,
            'qty': qty,
            'price': price,
            'value': qty * price
        }
        
        '''Aggiungi allo storico'''
        transactions.append(transaction)
        
        '''Feedback'''
        print(f"SELL {qty} {symbol} @ ${price}")
        pass
    
    def show_portfolio():
        print("\n PORTFOLIO ATTUALE")
        print("-" * 30)
        
        if not portfolio:
            print("Portfolio vuoto")
            return
        for symbol, qty in portfolio.items():
            print(f"{symbol}: {qty}")
        pass
    
    def show_trasactions():
        print("\n---TRANSACTIONS---")
        print("-" * 30)
        
        if not transactions:
            print("Nessuna transactions registrata")
            return
        for t in transactions:
            print(f"{t['type']}, {t['symbol']}, {t['qty']}, {t['price']}, {t['value']}")
            
        pass
    
    def calculate_pnl(current_prices):
        print("\n---CALCOLO P&L---")
        print("-" * 30)
        
        invested = 0
        realized = 0
        
        '''Scorri tutte le transazioni'''
        for t in transactions:
            if t['type'] == 'BUY':
                invested += t['value']
            elif t['type'] == 'SELL':
                realized += t['value']
                
        '''Calcola valore attuale del portfolio'''
        current_value = 0
        for symbol, qty in portfolio.items():
            if symbol in current_prices:
                current_value += qty * current_prices[symbol]
            else:
                print(f"Nessun prezzo corrente per : {symbol}")
                
        pnl = current_value + realized - invested
        
        '''Stampa report'''
        print(f"Investito totale: ${invested}") 
        print(f"Incassato totale: ${realized}") 
        print(f"Valore attuale: ${current_value}") 
        print(f"📈 P&L totale: ${pnl}")
        
        return pnl
            
        pass
    
    #---RESTITUISCI LE OPERAZIONI---
    
    return {
        'buy': buy,
        'sell': sell,
        'show_portfolio': show_portfolio,
        'show_transactions': show_trasactions,
        'calculate_pnl': calculate_pnl
    }
    

tracker = portfolio_tracker()

tracker['buy']('BTC', 0.1, 96555)
tracker['buy']('ETH', 2.0, 3125)

tracker['sell']('BTC', 0.05, 98000)
tracker['sell']('ETH', 1.5, 4570)

prices = {'BTC': 55000,
          'ETH': 1999}

tracker['show_portfolio']()
tracker['show_transactions']()
tracker['calculate_pnl'](prices)
