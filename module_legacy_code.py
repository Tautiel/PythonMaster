#!/usr/bin/env python3
"""
🏚️ LEGACY CODE NAVIGATION MODULE
Working with Existing Codebases

Duration: 1 Week + Ongoing
Level: From Greenfield to Brownfield Expert
"""

import os
import ast
import re
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import textwrap

# ============================================================================
# PART 1: UNDERSTANDING LEGACY CODE
# ============================================================================

class LegacyCodeAnalysis:
    """Analizzare e capire codice legacy"""
    
    def __init__(self, codebase_path: str = "."):
        self.codebase_path = Path(codebase_path)
        self.analysis_results = {}
        
    def first_steps_in_legacy_code(self):
        """I primi passi quando entri in un codebase legacy"""
        
        print("\n🏚️ APPROACHING LEGACY CODE")
        print("=" * 60)
        
        steps = """
        📋 THE LEGACY CODE CHECKLIST
        
        1️⃣ RECONNAISSANCE (Day 1)
           ├─ Find the entry point (main.py, app.py, etc.)
           ├─ Identify core business logic
           ├─ Locate configuration files
           ├─ Find database schemas
           └─ Check for documentation
        
        2️⃣ MAPPING (Day 2-3)
           ├─ Create dependency graph
           ├─ Identify critical paths
           ├─ Find common patterns
           ├─ Locate test files
           └─ Map data flow
        
        3️⃣ UNDERSTANDING (Week 1)
           ├─ Read existing tests
           ├─ Run the application
           ├─ Trace key workflows
           ├─ Identify pain points
           └─ Talk to original authors
        
        4️⃣ DOCUMENTING (Ongoing)
           ├─ Add comments as you learn
           ├─ Create architecture diagrams
           ├─ Document assumptions
           ├─ Note technical debt
           └─ Track mysteries
        """
        
        print(steps)
        
        # Practical tools
        print("\n🔧 ESSENTIAL TOOLS:")
        tools = {
            "grep/ripgrep": "Find patterns across files",
            "find/fd": "Locate files by name/type",
            "ctags": "Jump to definitions",
            "git log/blame": "Understand history",
            "dependency tools": "pipdeptree, pydeps",
            "IDE features": "Go to definition, Find usages",
            "debugger": "Step through execution",
            "profiler": "Find hot paths"
        }
        
        for tool, purpose in tools.items():
            print(f"  {tool:20} → {purpose}")
    
    def code_archaeology(self):
        """Tecniche di archeologia del codice"""
        
        print("\n🔍 CODE ARCHAEOLOGY TECHNIQUES")
        print("=" * 60)
        
        class CodeArchaeologist:
            def __init__(self, repo_path: str):
                self.repo_path = repo_path
                
            def find_hotspots(self) -> Dict[str, int]:
                """Find files that change frequently (hotspots)"""
                # git log --pretty=format: --name-only | sort | uniq -c | sort -rg
                print("Finding hotspots (frequently changed files):")
                print("""
                git log --pretty=format: --name-only | \\
                    sort | uniq -c | sort -rg | head -20
                """)
                
                # Simulated results
                hotspots = {
                    "trading/executor.py": 234,
                    "trading/strategy.py": 189,
                    "utils/helpers.py": 156,  # ⚠️ Utils often = dumping ground
                    "models/order.py": 134,
                    "config.py": 122  # ⚠️ Config changes = instability
                }
                
                return hotspots
            
            def find_authors(self, filepath: str) -> Dict[str, int]:
                """Find who knows this code"""
                # git shortlog -sn -- filepath
                print(f"\nFinding experts for {filepath}:")
                print(f"git shortlog -sn -- {filepath}")
                
                # Simulated results
                return {
                    "John Doe": 45,
                    "Jane Smith": 32,
                    "Bob Wilson": 12
                }
            
            def find_coupling(self) -> List[Tuple[str, str]]:
                """Find files that change together"""
                print("\nFinding coupled files:")
                print("""
                # Files that change together often are coupled
                git log --name-only --pretty="format:" | \\
                    grep -v '^$' | \\
                    python find_coupling.py
                """)
                
                # Example coupled files
                return [
                    ("model.py", "schema.sql"),
                    ("api.py", "serializers.py"),
                    ("trade.py", "risk.py")
                ]
            
            def analyze_commit_messages(self) -> Dict[str, int]:
                """Analyze commit patterns"""
                print("\nAnalyzing commit messages for insights:")
                
                patterns = {
                    "fix": 145,  # Lots of bugs
                    "hotfix": 23,  # Emergency fixes
                    "revert": 18,  # Unstable changes
                    "TODO": 34,   # Incomplete work
                    "hack": 12,   # Technical debt
                    "temporary": 8  # More debt
                }
                
                print("Warning signs in commit history:")
                for pattern, count in patterns.items():
                    if count > 10:
                        print(f"  ⚠️ {pattern}: {count} occurrences")
                
                return patterns
            
            def find_dead_code(self) -> List[str]:
                """Find potentially dead code"""
                print("\nFinding potentially dead code:")
                
                # Techniques
                techniques = [
                    "Look for unused imports",
                    "Find unreferenced functions",
                    "Check for commented code blocks",
                    "Find unreachable code",
                    "Look for obsolete features"
                ]
                
                for technique in techniques:
                    print(f"  • {technique}")
                
                # Tools
                print("\nTools for dead code detection:")
                print("  - vulture (Python)")
                print("  - coverage.py (via tests)")
                print("  - ast analysis")
                
                return ["old_feature.py", "utils/deprecated.py"]
        
        return CodeArchaeologist(".")

# ============================================================================
# PART 2: LEGACY CODE PATTERNS
# ============================================================================

class LegacyCodePatterns:
    """Pattern comuni nel codice legacy"""
    
    def common_code_smells(self):
        """Code smell comuni nel legacy"""
        
        print("\n👃 COMMON LEGACY CODE SMELLS")
        print("=" * 60)
        
        code_smells = {
            "🍝 Spaghetti Code": {
                "symptoms": "No clear structure, goto-like flow",
                "example": """
                # BAD: Spaghetti
                def process_trade():
                    if check_balance():
                        if validate_price():
                            if check_risk():
                                execute()
                                if confirm():
                                    update_db()
                                    if notify():
                                        log()
                """,
                "fix": "Extract methods, use guard clauses"
            },
            
            "🗿 God Class": {
                "symptoms": "One class does everything",
                "example": """
                # BAD: God Class
                class TradingSystem:
                    def connect_to_exchange(self): pass
                    def authenticate(self): pass
                    def get_prices(self): pass
                    def calculate_indicators(self): pass
                    def execute_trade(self): pass
                    def send_email(self): pass
                    def generate_report(self): pass
                    # ... 2000 more lines
                """,
                "fix": "Split responsibilities, use composition"
            },
            
            "🔮 Magic Numbers": {
                "symptoms": "Hardcoded values everywhere",
                "example": """
                # BAD: Magic numbers
                if price > 100:  # What's 100?
                    fee = amount * 0.002  # What's 0.002?
                    if volume > 1000000:  # What unit?
                        apply_discount(0.1)  # What's 0.1?
                """,
                "fix": "Extract constants with meaningful names"
            },
            
            "🧟 Dead Code": {
                "symptoms": "Commented code, unused functions",
                "example": """
                # BAD: Dead code
                def process_order(order):
                    # Old implementation
                    # if order.type == 'old':
                    #     old_process(order)
                    # else:
                    #     other_old_process(order)
                    
                    # New implementation (2019)
                    # TODO: Remove old code after testing
                    new_process(order)
                """,
                "fix": "Delete it! (Git has history)"
            },
            
            "📚 Copy-Paste Programming": {
                "symptoms": "Same code in multiple places",
                "example": """
                # BAD: Duplication
                def process_buy_order(order):
                    if order.amount < 0:
                        raise ValueError("Invalid amount")
                    if order.price < 0:
                        raise ValueError("Invalid price")
                    # ... 50 lines of processing
                
                def process_sell_order(order):
                    if order.amount < 0:
                        raise ValueError("Invalid amount")
                    if order.price < 0:
                        raise ValueError("Invalid price")
                    # ... same 50 lines of processing
                """,
                "fix": "Extract common functionality"
            }
        }
        
        for smell, details in code_smells.items():
            print(f"\n{smell}")
            print(f"Symptoms: {details['symptoms']}")
            print(f"Example:{details['example']}")
            print(f"Fix: {details['fix']}")
    
    def legacy_architecture_patterns(self):
        """Pattern architetturali nel legacy"""
        
        print("\n🏛️ LEGACY ARCHITECTURE PATTERNS")
        print("=" * 60)
        
        patterns = {
            "Big Ball of Mud": {
                "description": "No architecture, just growth",
                "signs": [
                    "No clear layers",
                    "Circular dependencies",
                    "Everything knows everything",
                    "Can't test in isolation"
                ],
                "approach": "Identify seams, extract modules gradually"
            },
            
            "Layered but Leaky": {
                "description": "Layers exist but violate boundaries",
                "signs": [
                    "UI directly queries database",
                    "Business logic in views",
                    "Data access in controllers"
                ],
                "approach": "Enforce boundaries, use dependency injection"
            },
            
            "Database as Integration": {
                "description": "All systems communicate via shared DB",
                "signs": [
                    "Multiple apps write same tables",
                    "No APIs, just database access",
                    "Stored procedures as business logic"
                ],
                "approach": "Add API layer, eventual consistency"
            },
            
            "The Singleton Disease": {
                "description": "Singletons everywhere",
                "signs": [
                    "Global state via singletons",
                    "Hard to test",
                    "Hidden dependencies"
                ],
                "approach": "Dependency injection, explicit dependencies"
            }
        }
        
        for pattern, details in patterns.items():
            print(f"\n{pattern}:")
            print(f"  {details['description']}")
            print(f"\n  Signs:")
            for sign in details['signs']:
                print(f"    • {sign}")
            print(f"\n  Approach: {details['approach']}")

# ============================================================================
# PART 3: REFACTORING STRATEGIES
# ============================================================================

class RefactoringStrategies:
    """Strategie per refactoring sicuro"""
    
    def characterization_testing(self):
        """Testing di caratterizzazione per legacy code"""
        
        print("\n🧪 CHARACTERIZATION TESTING")
        print("=" * 60)
        
        print("""
        When code has no tests, write tests that document 
        current behavior (even if wrong!)
        """)
        
        # Example legacy code
        legacy_code = """
        # LEGACY CODE: No tests, unclear behavior
        def calculate_fee(amount, user_type, day_of_week):
            fee = amount * 0.02
            
            if user_type == 'premium':
                fee = fee * 0.5
            elif user_type == 'vip':
                fee = fee * 0.3
            
            if day_of_week in ['Saturday', 'Sunday']:
                fee = fee * 1.5
            
            if amount > 10000:
                fee = fee - 10
            
            return max(fee, 1)  # Minimum fee
        """
        
        print(f"\nLegacy code:{legacy_code}")
        
        # Characterization tests
        characterization_tests = """
        # CHARACTERIZATION TESTS: Document current behavior
        def test_calculate_fee_current_behavior():
            '''
            These tests document CURRENT behavior.
            They might be wrong, but they prevent regression.
            '''
            
            # Document basic behavior
            assert calculate_fee(100, 'regular', 'Monday') == 2.0
            assert calculate_fee(100, 'premium', 'Monday') == 1.0
            assert calculate_fee(100, 'vip', 'Monday') == 0.6
            
            # Document weekend behavior
            assert calculate_fee(100, 'regular', 'Saturday') == 3.0
            assert calculate_fee(100, 'premium', 'Saturday') == 1.5
            
            # Document large amount behavior
            assert calculate_fee(10001, 'regular', 'Monday') == 190.02
            
            # Document minimum fee
            assert calculate_fee(1, 'vip', 'Monday') == 1.0
            
            # Document edge cases (even if they're bugs!)
            assert calculate_fee(-100, 'regular', 'Monday') == 1.0  # Bug?
            assert calculate_fee(0, 'regular', 'Monday') == 1.0
            assert calculate_fee(100, 'unknown', 'Monday') == 2.0  # Bug?
        """
        
        print(f"\nCharacterization tests:{characterization_tests}")
        
        print("\n📋 CHARACTERIZATION TESTING STEPS:")
        steps = [
            "1. Call the code with various inputs",
            "2. Record the outputs (even if wrong)",
            "3. Write tests that expect these outputs",
            "4. Now you can refactor safely",
            "5. Fix bugs in separate commits"
        ]
        
        for step in steps:
            print(f"  {step}")
    
    def strangler_fig_pattern(self):
        """Strangler Fig pattern per sostituire legacy"""
        
        print("\n🌱 STRANGLER FIG PATTERN")
        print("=" * 60)
        
        print("""
        Gradually replace legacy system like a strangler fig
        tree that grows around and eventually replaces host tree.
        """)
        
        # Evolution stages
        stages = {
            "Stage 1: Identify Boundary": {
                "old_system": """
                class LegacyTradingSystem:
                    def process_order(self, order):
                        # 500 lines of tangled logic
                        validate()
                        calculate_fees()
                        check_risk()
                        execute()
                        notify()
                """,
                "new_system": """
                # Start by identifying a clear boundary
                class OrderProcessor:
                    def __init__(self, legacy_system):
                        self.legacy = legacy_system
                    
                    def process_order(self, order):
                        # Delegate to legacy for now
                        return self.legacy.process_order(order)
                """
            },
            
            "Stage 2: Intercept & Delegate": {
                "description": "Route through new system",
                "code": """
                class OrderProcessor:
                    def process_order(self, order):
                        # Add new functionality
                        self.log_order(order)
                        
                        # Still delegate to legacy
                        result = self.legacy.process_order(order)
                        
                        # Add monitoring
                        self.monitor_result(result)
                        return result
                """
            },
            
            "Stage 3: Gradual Migration": {
                "description": "Move functionality piece by piece",
                "code": """
                class OrderProcessor:
                    def process_order(self, order):
                        # New validation
                        self.validate_order(order)  # NEW
                        
                        # New fee calculation
                        fees = self.calculate_fees(order)  # NEW
                        
                        # Still use legacy for complex parts
                        self.legacy.check_risk(order)  # LEGACY
                        self.legacy.execute(order)  # LEGACY
                        
                        # New notification
                        self.notify(order)  # NEW
                """
            },
            
            "Stage 4: Complete Replacement": {
                "description": "Legacy is gone",
                "code": """
                class OrderProcessor:
                    def process_order(self, order):
                        # Completely new implementation
                        self.validate_order(order)
                        fees = self.calculate_fees(order)
                        risk = self.check_risk(order)
                        result = self.execute_trade(order)
                        self.notify(order, result)
                        return result
                    
                # Legacy can be deleted!
                """
            }
        }
        
        for stage, details in stages.items():
            print(f"\n{stage}:")
            if isinstance(details, dict):
                if "description" in details:
                    print(f"  {details['description']}")
                if "code" in details:
                    print(f"{details['code']}")
                if "old_system" in details:
                    print(f"  Old:{details['old_system']}")
                if "new_system" in details:
                    print(f"  New:{details['new_system']}")
    
    def refactoring_patterns(self):
        """Pattern di refactoring comuni"""
        
        print("\n🔧 COMMON REFACTORING PATTERNS")
        print("=" * 60)
        
        patterns = {
            "Extract Method": {
                "before": """
                def process_trade(trade):
                    # Validation
                    if trade.amount <= 0:
                        raise ValueError("Invalid amount")
                    if trade.price <= 0:
                        raise ValueError("Invalid price")
                    if trade.symbol not in VALID_SYMBOLS:
                        raise ValueError("Invalid symbol")
                    
                    # Risk check
                    exposure = portfolio.get_exposure(trade.symbol)
                    if exposure + trade.amount > MAX_EXPOSURE:
                        raise ValueError("Exceeds exposure limit")
                    
                    # Execute
                    # ... more code
                """,
                "after": """
                def process_trade(trade):
                    validate_trade(trade)
                    check_risk_limits(trade)
                    execute_trade(trade)
                
                def validate_trade(trade):
                    if trade.amount <= 0:
                        raise ValueError("Invalid amount")
                    if trade.price <= 0:
                        raise ValueError("Invalid price")
                    if trade.symbol not in VALID_SYMBOLS:
                        raise ValueError("Invalid symbol")
                
                def check_risk_limits(trade):
                    exposure = portfolio.get_exposure(trade.symbol)
                    if exposure + trade.amount > MAX_EXPOSURE:
                        raise ValueError("Exceeds exposure limit")
                """
            },
            
            "Replace Magic Numbers": {
                "before": """
                if user.age >= 18:
                    if balance > 10000:
                        rate = 0.05
                    else:
                        rate = 0.02
                """,
                "after": """
                LEGAL_AGE = 18
                HIGH_BALANCE_THRESHOLD = 10000
                HIGH_BALANCE_RATE = 0.05
                STANDARD_RATE = 0.02
                
                if user.age >= LEGAL_AGE:
                    if balance > HIGH_BALANCE_THRESHOLD:
                        rate = HIGH_BALANCE_RATE
                    else:
                        rate = STANDARD_RATE
                """
            },
            
            "Introduce Parameter Object": {
                "before": """
                def create_order(symbol, quantity, price, 
                                order_type, time_in_force,
                                stop_price, limit_price):
                    # Too many parameters!
                    pass
                """,
                "after": """
                @dataclass
                class OrderParams:
                    symbol: str
                    quantity: float
                    price: float
                    order_type: str = 'market'
                    time_in_force: str = 'day'
                    stop_price: Optional[float] = None
                    limit_price: Optional[float] = None
                
                def create_order(params: OrderParams):
                    # Clean interface
                    pass
                """
            }
        }
        
        for pattern, example in patterns.items():
            print(f"\n{pattern}:")
            print(f"  Before:{example['before']}")
            print(f"  After:{example['after']}")

# ============================================================================
# PART 4: WORKING WITH TECHNICAL DEBT
# ============================================================================

class TechnicalDebtManagement:
    """Gestire il debito tecnico"""
    
    def identify_technical_debt(self):
        """Identificare e catalogare debito tecnico"""
        
        print("\n💳 IDENTIFYING TECHNICAL DEBT")
        print("=" * 60)
        
        @dataclass
        class TechnicalDebt:
            id: str
            description: str
            type: str  # 'design', 'code', 'test', 'documentation'
            impact: str  # 'high', 'medium', 'low'
            effort: str  # 'high', 'medium', 'low'
            location: str
            added_date: datetime
            notes: str
        
        # Debt categories
        debt_categories = {
            "🏗️ Design Debt": [
                "Improper abstractions",
                "Violated SOLID principles",
                "Missing design patterns",
                "Tight coupling",
                "No clear architecture"
            ],
            
            "💻 Code Debt": [
                "Code duplication",
                "Long methods",
                "Complex conditionals",
                "Dead code",
                "Poor naming"
            ],
            
            "🧪 Test Debt": [
                "Missing tests",
                "Flaky tests",
                "Slow tests",
                "No integration tests",
                "Poor test coverage"
            ],
            
            "📚 Documentation Debt": [
                "Missing documentation",
                "Outdated docs",
                "No architecture diagram",
                "Missing API docs",
                "No onboarding guide"
            ],
            
            "🔧 Infrastructure Debt": [
                "Manual deployments",
                "No monitoring",
                "Missing backups",
                "No disaster recovery",
                "Outdated dependencies"
            ]
        }
        
        for category, items in debt_categories.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  • {item}")
        
        # Debt tracking template
        print("\n📊 DEBT TRACKING TEMPLATE:")
        debt_template = """
        ## Technical Debt Register
        
        ### TD-001: God Class in TradingSystem
        **Type**: Design Debt
        **Impact**: High - Blocks all refactoring
        **Effort**: High - 2 weeks to split
        **Location**: trading/system.py
        **Added**: 2024-01-15
        
        **Description**:
        TradingSystem class has 3000+ lines and 50+ methods.
        Violates SRP, hard to test, frequent merge conflicts.
        
        **Proposed Solution**:
        1. Extract OrderManager class
        2. Extract RiskManager class
        3. Extract PriceEngine class
        4. Use dependency injection
        
        **Business Impact**:
        - Slows feature development by 30%
        - Causes 2-3 bugs per release
        - Blocks team scaling
        """
        
        print(debt_template)
    
    def debt_payment_strategies(self):
        """Strategie per ripagare debito tecnico"""
        
        print("\n💰 DEBT PAYMENT STRATEGIES")
        print("=" * 60)
        
        strategies = {
            "🚚 Boy Scout Rule": {
                "description": "Leave code better than you found it",
                "implementation": "Fix small issues when you touch code",
                "example": "Rename variables, extract small methods",
                "effort": "Continuous, small",
                "impact": "Gradual improvement"
            },
            
            "🎯 Focused Sprints": {
                "description": "Dedicate sprints to debt reduction",
                "implementation": "20% of sprints for tech debt",
                "example": "Refactoring sprint every 5th sprint",
                "effort": "Scheduled, medium",
                "impact": "Significant improvements"
            },
            
            "🔄 Refactor on Feature": {
                "description": "Refactor when adding features",
                "implementation": "Budget refactoring time in estimates",
                "example": "Adding feature? Refactor that module first",
                "effort": "Feature-driven",
                "impact": "Targeted improvements"
            },
            
            "💣 Debt Ceiling": {
                "description": "Stop everything when debt is too high",
                "implementation": "Set metrics thresholds",
                "example": "If coverage < 60%, stop features",
                "effort": "Crisis-driven",
                "impact": "Forced improvement"
            },
            
            "🎰 Tech Debt Friday": {
                "description": "Regular time for debt work",
                "implementation": "Every Friday afternoon",
                "example": "Team chooses what debt to tackle",
                "effort": "Regular, small",
                "impact": "Consistent progress"
            }
        }
        
        for strategy, details in strategies.items():
            print(f"\n{strategy}")
            for key, value in details.items():
                print(f"  {key}: {value}")

# ============================================================================
# PART 5: LEGACY CODE PROJECTS
# ============================================================================

class LegacyCodeProjects:
    """Progetti pratici con legacy code"""
    
    def project_dependency_mapper(self):
        """Map dependencies in legacy code"""
        
        print("\n🗺️ PROJECT: Dependency Mapper")
        print("=" * 60)
        
        class DependencyMapper:
            def __init__(self, root_path: str):
                self.root_path = Path(root_path)
                self.dependencies = defaultdict(set)
                
            def analyze_imports(self, file_path: Path) -> Set[str]:
                """Extract imports from Python file"""
                imports = set()
                
                with open(file_path, 'r') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module)
                
                return imports
            
            def build_dependency_graph(self):
                """Build complete dependency graph"""
                for py_file in self.root_path.rglob("*.py"):
                    if 'venv' not in str(py_file):
                        try:
                            imports = self.analyze_imports(py_file)
                            relative_path = py_file.relative_to(self.root_path)
                            self.dependencies[str(relative_path)] = imports
                        except:
                            pass
                
                return self.dependencies
            
            def find_circular_dependencies(self) -> List[Tuple[str, str]]:
                """Find circular import dependencies"""
                circular = []
                
                for file1, deps1 in self.dependencies.items():
                    for file2, deps2 in self.dependencies.items():
                        if file1 != file2:
                            # Check if they import each other
                            if file2 in deps1 and file1 in deps2:
                                if (file2, file1) not in circular:
                                    circular.append((file1, file2))
                
                return circular
            
            def find_unused_files(self) -> List[str]:
                """Find potentially unused files"""
                all_files = set(self.dependencies.keys())
                imported_files = set()
                
                for deps in self.dependencies.values():
                    imported_files.update(deps)
                
                # Files never imported
                unused = all_files - imported_files
                
                # Exclude common entry points
                entry_points = {'main.py', 'app.py', '__main__.py'}
                unused = [f for f in unused if Path(f).name not in entry_points]
                
                return list(unused)
            
            def generate_report(self) -> str:
                """Generate dependency analysis report"""
                report = "# Dependency Analysis Report\n\n"
                
                # Stats
                report += "## Statistics\n"
                report += f"- Total files: {len(self.dependencies)}\n"
                report += f"- Total unique imports: {len(set().union(*self.dependencies.values()))}\n"
                
                # Circular dependencies
                circular = self.find_circular_dependencies()
                if circular:
                    report += "\n## ⚠️ Circular Dependencies\n"
                    for file1, file2 in circular:
                        report += f"- {file1} ↔ {file2}\n"
                
                # Unused files
                unused = self.find_unused_files()
                if unused:
                    report += "\n## 🗑️ Potentially Unused Files\n"
                    for file in unused:
                        report += f"- {file}\n"
                
                # High coupling
                report += "\n## 🔗 Highly Coupled Files\n"
                for file, deps in self.dependencies.items():
                    if len(deps) > 10:
                        report += f"- {file}: {len(deps)} dependencies\n"
                
                return report
        
        print("Mapper features:")
        print("✅ Import analysis")
        print("✅ Circular dependency detection")
        print("✅ Unused file detection")
        print("✅ Coupling metrics")
        print("✅ Visual graph generation")
        
        return DependencyMapper(".")

# ============================================================================
# EXERCISES
# ============================================================================

def legacy_code_exercises():
    """50 legacy code exercises"""
    
    print("\n🏚️ LEGACY CODE EXERCISES")
    print("=" * 60)
    
    exercises = {
        "Analysis (1-15)": [
            "Map dependencies in legacy project",
            "Find circular dependencies",
            "Identify code hotspots",
            "Find dead code",
            "Analyze code complexity",
            "Create architecture diagram",
            "Find duplicate code",
            "Identify design patterns used",
            "Find hardcoded values",
            "Analyze test coverage",
            "Find security vulnerabilities",
            "Identify performance bottlenecks",
            "Document business rules",
            "Find hidden dependencies",
            "Create data flow diagram"
        ],
        
        "Refactoring (16-35)": [
            "Extract method from 100-line function",
            "Replace magic numbers",
            "Remove dead code safely",
            "Split god class",
            "Introduce parameter object",
            "Replace conditionals with polymorphism",
            "Extract interface from class",
            "Remove circular dependency",
            "Introduce dependency injection",
            "Replace global with parameter",
            "Consolidate duplicate code",
            "Rename unclear variables",
            "Simplify complex boolean",
            "Replace nested conditionals",
            "Extract class from method",
            "Move method to proper class",
            "Replace inheritance with composition",
            "Introduce null object pattern",
            "Replace constructor with factory",
            "Encapsulate collection"
        ],
        
        "Testing Legacy (36-50)": [
            "Write characterization tests",
            "Add tests to untested code",
            "Break dependencies for testing",
            "Mock external dependencies",
            "Test database interactions",
            "Create integration tests",
            "Add contract tests",
            "Test error conditions",
            "Add performance tests",
            "Create regression tests",
            "Test configuration loading",
            "Add smoke tests",
            "Create end-to-end tests",
            "Test backward compatibility",
            "Add property-based tests"
        ]
    }
    
    for category, items in exercises.items():
        print(f"\n{category}:")
        for i, exercise in enumerate(items, 1):
            print(f"  {i}. {exercise}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run legacy code module"""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║              🏚️ LEGACY CODE NAVIGATION MODULE               ║
    ║            From Greenfield to Brownfield Expert             ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    modules = {
        "1": ("Understanding Legacy", LegacyCodeAnalysis),
        "2": ("Legacy Patterns", LegacyCodePatterns),
        "3": ("Refactoring Strategies", RefactoringStrategies),
        "4": ("Technical Debt", TechnicalDebtManagement),
        "5": ("Projects", LegacyCodeProjects),
        "6": ("Exercises", legacy_code_exercises)
    }
    
    while True:
        print("\n📚 SELECT MODULE:")
        for key, (name, _) in modules.items():
            print(f"  {key}. {name}")
        print("  Q. Quit")
        
        choice = input("\nChoice: ").strip().upper()
        
        if choice == 'Q':
            break
        elif choice == '6':
            legacy_code_exercises()
        else:
            # Run selected module
            pass

if __name__ == "__main__":
    main()
