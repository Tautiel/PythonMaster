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
        pass
    
    def show_portfolio():
        pass
    
    def show_trasactions():
        pass
    
    def calculate_pnl(current_prices):
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