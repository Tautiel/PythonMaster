"""
🚀 SESSIONE 1 - PARTE 2: TRE PROGETTI COMPLETI
==============================================
Super Intensive Python Master Course
Durata: 120 minuti di pratica con progetti reali
"""

import json
import sqlite3
import random
import time
import os
import re
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import lru_cache, wraps
from enum import Enum, auto
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("="*80)
print("💻 PARTE 2: TRE PROGETTI DIVERSI")
print("="*80)
print("\n1. Trading System v4.0 (Enhanced)")
print("2. Text Adventure Game")  
print("3. Task Automation Tool")
print("="*80)

# ==============================================================================
# PROGETTO 1: TRADING SYSTEM v4.0
# ==============================================================================

print("\n" + "="*60)
print("📈 PROGETTO 1: TRADING SYSTEM v4.0")
print("="*60)

# Protocol per Strategy Pattern
class TradingStrategy(Protocol):
    """Protocol per definire interfaccia strategy"""
    def analyze(self, data: Dict) -> float:
        """Return signal strength -1 to 1"""
        ...

@dataclass
class Trade:
    """Trade con validation automatica"""
    symbol: str
    quantity: float
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.price <= 0:
            raise ValueError("Price must be positive")
    
    @property
    def value(self) -> float:
        return self.quantity * self.price

class SignalType(Enum):
    """Signals using auto()"""
    STRONG_BUY = auto()
    BUY = auto()
    NEUTRAL = auto()
    SELL = auto()
    STRONG_SELL = auto()

class TradingBot:
    """Bot con context manager e caching"""
    
    def __init__(self, db_path: str = "trading.db"):
        self.db_path = db_path
        self._init_db()
        self.trades_executed = 0
        self.total_profit = 0.0
    
    def _init_db(self):
        """Initialize database"""
        with self._get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    profit REAL
                )
            """)
            logger.info("Database initialized")
    
    @contextmanager
    def _get_db(self):
        """Context manager per database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    @lru_cache(maxsize=128)
    def calculate_indicators(self, symbol: str) -> Dict:
        """Cached indicator calculation"""
        # Simulato - in realtà prenderebbe dati reali
        return {
            'rsi': 45 + hash(symbol) % 30,
            'macd': hash(symbol) % 100 / 100,
            'volume': abs(hash(symbol)) % 1000000
        }
    
    def execute_trade(self, trade: Trade) -> bool:
        """Execute and store trade"""
        try:
            with self._get_db() as conn:
                conn.execute("""
                    INSERT INTO trades (symbol, quantity, price, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (trade.symbol, trade.quantity, trade.price, 
                      trade.timestamp.isoformat()))
            
            self.trades_executed += 1
            logger.info(f"Trade #{self.trades_executed}: {trade.symbol} "
                       f"{trade.quantity} @ ${trade.price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Trade failed: {e}")
            return False
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        with self._get_db() as conn:
            result = conn.execute("""
                SELECT SUM(quantity * price) as total
                FROM trades
            """).fetchone()
            
            return result['total'] or 0.0
    
    def get_trade_statistics(self) -> Dict:
        """Get trading statistics"""
        with self._get_db() as conn:
            trades = conn.execute("""
                SELECT * FROM trades
                ORDER BY timestamp DESC
                LIMIT 100
            """).fetchall()
            
            if not trades:
                return {"total_trades": 0, "total_value": 0}
            
            total_value = sum(t['quantity'] * t['price'] for t in trades)
            
            return {
                "total_trades": len(trades),
                "total_value": total_value,
                "avg_trade_size": total_value / len(trades) if trades else 0,
                "last_trade": trades[0]['timestamp'] if trades else None
            }
    
    def simulate_market_data(self, symbol: str) -> Dict:
        """Simula dati di mercato"""
        base_prices = {
            "BTC/USDT": 43000,
            "ETH/USDT": 2800,
            "SOL/USDT": 98
        }
        
        base = base_prices.get(symbol, 100)
        variation = random.uniform(0.98, 1.02)
        
        return {
            "symbol": symbol,
            "price": base * variation,
            "volume": random.uniform(1000000, 10000000),
            "change_24h": random.uniform(-5, 5)
        }
    
    def analyze_and_trade(self, symbol: str):
        """Analizza e esegue trade se opportuno"""
        # Get market data
        market_data = self.simulate_market_data(symbol)
        indicators = self.calculate_indicators(symbol)
        
        # Simple strategy
        signal_strength = 0
        
        if indicators['rsi'] < 30:
            signal_strength = 1  # Buy
        elif indicators['rsi'] > 70:
            signal_strength = -1  # Sell
        
        # Execute trade based on signal
        if abs(signal_strength) > 0.5:
            quantity = 0.1 if signal_strength > 0 else -0.1
            trade = Trade(
                symbol=symbol,
                quantity=abs(quantity),
                price=market_data['price']
            )
            
            success = self.execute_trade(trade)
            
            if success:
                action = "BUY" if signal_strength > 0 else "SELL"
                print(f"  📊 {action} Signal: {symbol} @ ${market_data['price']:.2f}")
    
    def run_simulation(self, cycles: int = 10):
        """Esegue simulazione di trading"""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        
        print(f"\n▶️ Starting simulation ({cycles} cycles)...")
        
        for i in range(cycles):
            print(f"\n📍 Cycle {i+1}/{cycles}")
            
            for symbol in symbols:
                self.analyze_and_trade(symbol)
            
            # Show stats every 5 cycles
            if (i + 1) % 5 == 0:
                stats = self.get_trade_statistics()
                print(f"\n📊 Stats: {stats['total_trades']} trades, "
                      f"Value: ${stats['total_value']:.2f}")
            
            time.sleep(0.5)  # Simulate time passing
        
        # Final report
        final_stats = self.get_trade_statistics()
        portfolio_value = self.get_portfolio_value()
        
        print("\n" + "="*40)
        print("📈 FINAL REPORT")
        print("="*40)
        print(f"Total Trades: {final_stats['total_trades']}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Average Trade Size: ${final_stats['avg_trade_size']:.2f}")
    
    def __enter__(self):
        """Context manager entry"""
        logger.info("Trading bot started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type:
            logger.error(f"Bot stopped with error: {exc_val}")
        else:
            logger.info(f"Bot stopped. Total trades: {self.trades_executed}")
        
        # Cleanup
        self.calculate_indicators.cache_clear()
        
        # Cleanup database file for demo
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        return False

# ==============================================================================
# PROGETTO 2: TEXT ADVENTURE GAME
# ==============================================================================

print("\n" + "="*60)
print("🎮 PROGETTO 2: TEXT ADVENTURE GAME")
print("="*60)

class Direction(Enum):
    """Direzioni di movimento"""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    
    @classmethod
    def from_string(cls, s: str) -> Optional['Direction']:
        """Parse direction from user input"""
        s = s.lower().strip()
        for direction in cls:
            if direction.value.startswith(s):
                return direction
        return None

@dataclass
class Item:
    """Item del gioco"""
    name: str
    description: str
    weight: float = 1.0
    value: int = 0
    usable: bool = False

@dataclass
class Room:
    """Stanza del gioco"""
    name: str
    description: str
    exits: Dict[Direction, str] = field(default_factory=dict)
    items: List[Item] = field(default_factory=list)
    visited: bool = False
    
    def get_description(self) -> str:
        """Get room description"""
        desc = f"\n{'='*40}\n"
        desc += f"📍 {self.name}\n"
        desc += f"{'='*40}\n"
        desc += self.description
        
        if self.items:
            desc += f"\n\n🎁 Items here: {', '.join(i.name for i in self.items)}"
        
        if self.exits:
            desc += f"\n🚪 Exits: {', '.join(d.value for d in self.exits.keys())}"
        
        return desc

class Player:
    """Giocatore"""
    def __init__(self, name: str):
        self.name = name
        self.inventory: List[Item] = []
        self.current_room = "entrance"
        self.score = 0
    
    def take_item(self, item: Item) -> bool:
        """Prende un oggetto"""
        if sum(i.weight for i in self.inventory) + item.weight > 10:
            return False  # Too heavy
        
        self.inventory.append(item)
        self.score += item.value
        return True

class AdventureGame:
    """Motore del gioco"""
    
    def __init__(self):
        self.rooms: Dict[str, Room] = {}
        self.player: Optional[Player] = None
        self.game_over = False
        self.moves = 0
        self._create_world()
    
    def _create_world(self):
        """Crea il mondo di gioco"""
        # Entrance
        self.rooms['entrance'] = Room(
            "🏰 Castle Entrance",
            "You stand before a massive castle gate. The ancient stones whisper secrets.",
            exits={Direction.NORTH: 'hall'},
            items=[
                Item("torch", "A burning torch", 1, 10, True),
                Item("map", "An old map of the castle", 0.1, 50)
            ]
        )
        
        # Hall
        self.rooms['hall'] = Room(
            "🏛️ Great Hall",
            "A magnificent hall with tapestries on the walls. Dust motes dance in the light.",
            exits={
                Direction.SOUTH: 'entrance',
                Direction.EAST: 'library',
                Direction.WEST: 'armory',
                Direction.NORTH: 'throne'
            }
        )
        
        # Library
        self.rooms['library'] = Room(
            "📚 Ancient Library",
            "Shelves tower to the ceiling, filled with forgotten knowledge.",
            exits={Direction.WEST: 'hall'},
            items=[
                Item("spellbook", "A book of ancient spells", 2, 100, True),
                Item("scroll", "A mysterious scroll", 0.5, 75)
            ]
        )
        
        # Armory
        self.rooms['armory'] = Room(
            "⚔️ Armory",
            "Weapons and armor line the walls, remnants of ancient battles.",
            exits={Direction.EAST: 'hall'},
            items=[
                Item("sword", "A gleaming silver sword", 3, 150),
                Item("shield", "A sturdy iron shield", 5, 100)
            ]
        )
        
        # Throne Room
        self.rooms['throne'] = Room(
            "👑 Throne Room",
            "The golden throne sits empty, waiting for a worthy ruler.",
            exits={Direction.SOUTH: 'hall'},
            items=[
                Item("crown", "The royal crown", 2, 500, True),
                Item("scepter", "The royal scepter", 3, 300)
            ]
        )
    
    def start(self):
        """Inizia il gioco"""
        print("\n" + "🎮"*20)
        print("     CASTLE ADVENTURE     ")
        print("🎮"*20)
        
        name = input("\nEnter your name, brave adventurer: ")
        self.player = Player(name)
        
        print(f"\nWelcome, {name}!")
        print("Your quest: Find the royal crown and claim the throne!")
        print("\nCommands: go [direction], take [item], inventory, score, help, quit")
        
        self.game_loop()
    
    def game_loop(self):
        """Loop principale del gioco"""
        while not self.game_over:
            # Show current room
            room = self.rooms[self.player.current_room]
            
            if not room.visited:
                print(room.get_description())
                room.visited = True
            else:
                print(f"\n📍 {room.name}")
            
            # Get command
            command = input("\n> ").lower().strip()
            
            if command:
                self.process_command(command)
                self.moves += 1
                
                # Check win condition
                if self.check_win():
                    self.game_won()
    
    def process_command(self, command: str):
        """Processa i comandi"""
        parts = command.split()
        
        if not parts:
            return
        
        action = parts[0]
        
        if action in ['go', 'move', 'walk'] and len(parts) > 1:
            self.move_player(parts[1])
        elif action in ['n', 'north', 's', 'south', 'e', 'east', 'w', 'west']:
            self.move_player(action)
        elif action in ['take', 'get'] and len(parts) > 1:
            self.take_item(' '.join(parts[1:]))
        elif action == 'inventory':
            self.show_inventory()
        elif action == 'score':
            print(f"Score: {self.player.score} points")
        elif action == 'help':
            self.show_help()
        elif action in ['quit', 'exit']:
            print("Thanks for playing!")
            self.game_over = True
        else:
            print("I don't understand that command.")
    
    def move_player(self, direction_str: str):
        """Muove il giocatore"""
        direction = Direction.from_string(direction_str)
        
        if not direction:
            print("That's not a valid direction.")
            return
        
        room = self.rooms[self.player.current_room]
        
        if direction in room.exits:
            self.player.current_room = room.exits[direction]
            print(f"\nYou go {direction.value}...")
        else:
            print("You can't go that way.")
    
    def take_item(self, item_name: str):
        """Prende un oggetto"""
        room = self.rooms[self.player.current_room]
        
        for item in room.items:
            if item.name.lower() == item_name.lower():
                if self.player.take_item(item):
                    room.items.remove(item)
                    print(f"✅ You take the {item.name}. (+{item.value} points)")
                else:
                    print("❌ Your inventory is too full!")
                return
        
        print("There's no such item here.")
    
    def show_inventory(self):
        """Mostra inventario"""
        if not self.player.inventory:
            print("Your inventory is empty.")
        else:
            print("\n🎒 Inventory:")
            for item in self.player.inventory:
                print(f"  • {item.name}: {item.description}")
            
            total_weight = sum(i.weight for i in self.player.inventory)
            print(f"\nTotal weight: {total_weight}/10")
    
    def show_help(self):
        """Mostra aiuto"""
        print("""
Commands:
  go/move [direction] - Move in a direction
  n/s/e/w - Quick movement
  take/get [item] - Take an item
  inventory - Show your items
  score - Show your score
  help - Show this help
  quit - Exit game
        """)
    
    def check_win(self):
        """Controlla condizione di vittoria"""
        has_crown = any(i.name == "crown" for i in self.player.inventory)
        in_throne_room = self.player.current_room == "throne"
        return has_crown and in_throne_room
    
    def game_won(self):
        """Vittoria!"""
        print("\n" + "🎉"*20)
        print("     CONGRATULATIONS!     ")
        print("🎉"*20)
        print(f"\n{self.player.name}, you have claimed the crown and the throne!")
        print(f"Final Score: {self.player.score} points")
        print(f"Moves: {self.moves}")
        print("\nYou are the new ruler of the castle!")
        self.game_over = True

# ==============================================================================
# PROGETTO 3: TASK AUTOMATION TOOL
# ==============================================================================

print("\n" + "="*60)
print("🤖 PROGETTO 3: TASK AUTOMATION TOOL")
print("="*60)

class FileOrganizer:
    """Organizza file automaticamente"""
    
    FILE_CATEGORIES = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
        'videos': ['.mp4', '.avi', '.mkv', '.mov'],
        'documents': ['.pdf', '.doc', '.docx', '.txt'],
        'code': ['.py', '.js', '.html', '.css', '.java'],
        'archives': ['.zip', '.rar', '.7z', '.tar']
    }
    
    @classmethod
    def organize_directory(cls, directory: Path, dry_run: bool = True):
        """Organizza file in categorie"""
        directory = Path(directory)
        
        if not directory.exists():
            print(f"❌ Directory {directory} doesn't exist")
            return []
        
        organized = []
        
        for file_path in directory.iterdir():
            if file_path.is_file():
                category = cls._get_category(file_path)
                
                if category:
                    target_dir = directory / category
                    
                    if not dry_run:
                        target_dir.mkdir(exist_ok=True)
                        target = target_dir / file_path.name
                        file_path.rename(target)
                    
                    organized.append({
                        'file': file_path.name,
                        'category': category
                    })
        
        return organized
    
    @classmethod
    def _get_category(cls, file_path: Path) -> Optional[str]:
        """Determina categoria del file"""
        ext = file_path.suffix.lower()
        
        for category, extensions in cls.FILE_CATEGORIES.items():
            if ext in extensions:
                return category
        return None

class TextProcessor:
    """Processa e analizza testo"""
    
    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """Estrae email dal testo"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Estrae URL dal testo"""
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_phone_numbers(text: str) -> List[str]:
        """Estrae numeri di telefono"""
        patterns = [
            r'\+\d{1,3}\s?\d{1,14}',
            r'\(\d{3}\)\s?\d{3}-\d{4}',
            r'\d{3}-\d{3}-\d{4}'
        ]
        
        results = []
        for pattern in patterns:
            results.extend(re.findall(pattern, text))
        
        return results
    
    @staticmethod
    def word_frequency(text: str, top_n: int = 5) -> Dict[str, int]:
        """Calcola frequenza parole"""
        # Clean text
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        
        # Count
        frequency = {}
        for word in words:
            frequency[word] = frequency.get(word, 0) + 1
        
        # Top N
        sorted_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_freq[:top_n])

class BackupManager:
    """Gestisce backup automatici"""
    
    def __init__(self, backup_dir: Path = Path("backups")):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def backup_file(self, file_path: Path) -> Path:
        """Crea backup di un file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        # Copy file
        backup_path.write_bytes(file_path.read_bytes())
        
        return backup_path
    
    def list_backups(self) -> List[Dict]:
        """Lista tutti i backup"""
        backups = []
        
        for backup_path in self.backup_dir.iterdir():
            stat = backup_path.stat()
            backups.append({
                'name': backup_path.name,
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_mtime)
            })
        
        return sorted(backups, key=lambda x: x['created'], reverse=True)

def automation_demo():
    """Demo delle funzionalità di automazione"""
    
    print("\n🤖 AUTOMATION TOOL DEMO")
    print("="*40)
    
    # 1. File Organization
    print("\n1️⃣ FILE ORGANIZATION")
    print("-"*30)
    
    # Crea file di esempio
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    test_files = [
        "document.pdf",
        "photo.jpg",
        "script.py",
        "video.mp4",
        "archive.zip"
    ]
    
    for filename in test_files:
        (test_dir / filename).touch()
    
    organized = FileOrganizer.organize_directory(test_dir, dry_run=True)
    
    print(f"📁 Found {len(organized)} files to organize:")
    for item in organized:
        print(f"  • {item['file']} → {item['category']}/")
    
    # 2. Text Processing
    print("\n2️⃣ TEXT PROCESSING")
    print("-"*30)
    
    sample_text = """
    Contact us at info@example.com or support@test.org
    Call: (123) 456-7890 or +1-555-123-4567
    Visit https://www.example.com for more info
    Python Python Java Python programming
    """
    
    emails = TextProcessor.extract_emails(sample_text)
    phones = TextProcessor.extract_phone_numbers(sample_text)
    urls = TextProcessor.extract_urls(sample_text)
    freq = TextProcessor.word_frequency(sample_text)
    
    print(f"📧 Emails found: {emails}")
    print(f"📞 Phones found: {phones}")
    print(f"🔗 URLs found: {urls}")
    print(f"📊 Top words: {freq}")
    
    # 3. Backup System
    print("\n3️⃣ BACKUP SYSTEM")
    print("-"*30)
    
    # Crea file da backuppare
    test_file = Path("important_data.txt")
    test_file.write_text("Important data that needs backup")
    
    backup_mgr = BackupManager()
    backup_path = backup_mgr.backup_file(test_file)
    
    print(f"✅ Backup created: {backup_path.name}")
    
    backups = backup_mgr.list_backups()
    print(f"📦 Total backups: {len(backups)}")
    
    # 4. File Hashing
    print("\n4️⃣ FILE INTEGRITY")
    print("-"*30)
    
    def calculate_hash(file_path: Path) -> str:
        """Calcola hash SHA256"""
        sha256 = hashlib.sha256()
        sha256.update(file_path.read_bytes())
        return sha256.hexdigest()
    
    hash1 = calculate_hash(test_file)
    hash2 = calculate_hash(backup_path)
    
    print(f"🔐 Original hash: {hash1[:16]}...")
    print(f"🔐 Backup hash:   {hash2[:16]}...")
    print(f"✅ Integrity check: {'PASSED' if hash1 == hash2 else 'FAILED'}")
    
    # Cleanup
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
    if test_file.exists():
        test_file.unlink()
    if backup_mgr.backup_dir.exists():
        shutil.rmtree(backup_mgr.backup_dir)
    
    print("\n✨ Demo completed! All test files cleaned up.")

# ==============================================================================
# MAIN - Menu per eseguire i progetti
# ==============================================================================

def main():
    """Menu principale per i progetti"""
    
    print("\n" + "="*60)
    print("🚀 SCEGLI UN PROGETTO DA ESEGUIRE")
    print("="*60)
    
    print("\n1. Trading Bot Simulation")
    print("2. Adventure Game")
    print("3. Automation Tool Demo")
    print("4. Run All Projects")
    print("0. Exit")
    
    choice = input("\nScegli (0-4): ")
    
    if choice == "1":
        print("\n" + "📈"*20)
        with TradingBot() as bot:
            bot.run_simulation(cycles=10)
    
    elif choice == "2":
        print("\n" + "🎮"*20)
        game = AdventureGame()
        game.start()
    
    elif choice == "3":
        automation_demo()
    
    elif choice == "4":
        print("\n🚀 Running all projects...\n")
        
        # Trading Bot
        print("1. TRADING BOT")
        with TradingBot() as bot:
            bot.run_simulation(cycles=5)
        
        input("\nPress ENTER for next project...")
        
        # Automation Demo
        print("\n3. AUTOMATION DEMO")
        automation_demo()
        
        print("\n✅ All projects completed!")
        print("Note: Adventure Game skipped (requires interaction)")
    
    elif choice == "0":
        print("👋 Goodbye!")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
