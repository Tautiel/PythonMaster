#!/usr/bin/env python3
"""
👀 CODE REVIEW & TEAM COLLABORATION MODULE
Professional Development Practices

Duration: 1 Week + Ongoing
Level: From Solo Coder to Team Player
"""

import ast
import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import difflib
from pathlib import Path

# ============================================================================
# PART 1: CODE REVIEW FUNDAMENTALS
# ============================================================================

class CodeReviewFundamentals:
    """Fondamenti del code review professionale"""
    
    def __init__(self):
        self.review_checklist = self.create_checklist()
        
    def create_checklist(self) -> Dict[str, List[str]]:
        """Checklist completa per code review"""
        
        return {
            "🎯 Functionality": [
                "Does the code do what it's supposed to do?",
                "Are all requirements met?",
                "Are edge cases handled?",
                "Is error handling appropriate?",
                "Are there any obvious bugs?"
            ],
            
            "🏗️ Design & Architecture": [
                "Does it follow SOLID principles?",
                "Is the code maintainable?",
                "Is it over-engineered or under-engineered?",
                "Are design patterns used appropriately?",
                "Is there unnecessary complexity?"
            ],
            
            "⚡ Performance": [
                "Are there obvious performance issues?",
                "Is the algorithm optimal (Big-O)?",
                "Are database queries efficient?",
                "Is caching used appropriately?",
                "Memory usage reasonable?"
            ],
            
            "🔒 Security": [
                "Input validation present?",
                "SQL injection prevented?",
                "XSS protection in place?",
                "Authentication/authorization correct?",
                "Sensitive data protected?"
            ],
            
            "✨ Code Quality": [
                "Is the code readable?",
                "Are names meaningful?",
                "Is it DRY (Don't Repeat Yourself)?",
                "Appropriate comments?",
                "Consistent style?"
            ],
            
            "🧪 Testing": [
                "Are there adequate tests?",
                "Do tests cover edge cases?",
                "Are tests maintainable?",
                "Is test coverage sufficient?",
                "Are mocks used appropriately?"
            ],
            
            "📚 Documentation": [
                "Is the code self-documenting?",
                "Are complex parts explained?",
                "Is API documentation complete?",
                "README updated if needed?",
                "Changelog updated?"
            ]
        }
    
    def review_workflow(self):
        """Il workflow completo del code review"""
        
        print("\n📋 CODE REVIEW WORKFLOW")
        print("=" * 60)
        
        workflow = """
        1️⃣ PREPARATION (Author)
           ├─ Self-review your code first
           ├─ Run all tests locally
           ├─ Update documentation
           └─ Write clear PR description
        
        2️⃣ SUBMISSION
           ├─ Create Pull Request
           ├─ Link related issues
           ├─ Add reviewers
           └─ Set labels/milestones
        
        3️⃣ REVIEW (Reviewer)
           ├─ Understand context
           ├─ Check out branch locally
           ├─ Run tests
           ├─ Review systematically
           └─ Leave constructive feedback
        
        4️⃣ DISCUSSION
           ├─ Respond to comments
           ├─ Explain decisions
           ├─ Ask questions
           └─ Reach consensus
        
        5️⃣ ITERATION
           ├─ Address feedback
           ├─ Push fixes
           ├─ Re-request review
           └─ Repeat as needed
        
        6️⃣ APPROVAL & MERGE
           ├─ Get required approvals
           ├─ Ensure CI passes
           ├─ Squash if needed
           └─ Merge to main
        """
        
        print(workflow)
        
        # Timing guidelines
        print("\n⏰ TIMING GUIDELINES:")
        guidelines = {
            "Response Time": "< 24 hours",
            "Small PR (<100 lines)": "15-30 minutes review",
            "Medium PR (100-500 lines)": "30-60 minutes review",
            "Large PR (>500 lines)": "Consider splitting",
            "Iteration cycle": "< 1 business day"
        }
        
        for key, value in guidelines.items():
            print(f"  {key}: {value}")

# ============================================================================
# PART 2: GIVING CONSTRUCTIVE FEEDBACK
# ============================================================================

class GivingFeedback:
    """Come dare feedback costruttivo"""
    
    def feedback_framework(self):
        """Framework per feedback efficace"""
        
        print("\n💬 EFFECTIVE FEEDBACK FRAMEWORK")
        print("=" * 60)
        
        # The SBI Model
        print("THE SBI MODEL:")
        print("""
        S - Situation: Context del codice
        B - Behavior: Cosa osservi nel codice
        I - Impact: Conseguenze o suggerimenti
        """)
        
        # Examples of good vs bad feedback
        feedback_examples = {
            "❌ BAD": {
                "Vague": "This code is messy",
                "Harsh": "This is terrible, rewrite everything",
                "Prescriptive": "Change this to my way",
                "Nitpicky": "Missing period in comment",
                "Personal": "You always write bad code"
            },
            
            "✅ GOOD": {
                "Specific": "This function has 6 responsibilities, consider splitting",
                "Constructive": "This works, but using a dict would be more efficient",
                "Explanatory": "Consider X because Y (with link to docs)",
                "Questioning": "What do you think about using pattern X here?",
                "Positive": "Great error handling! One suggestion..."
            }
        }
        
        for category, examples in feedback_examples.items():
            print(f"\n{category} Feedback:")
            for style, example in examples.items():
                print(f"  {style}: '{example}'")
        
        return self.comment_templates()
    
    def comment_templates(self) -> Dict[str, str]:
        """Template per commenti comuni"""
        
        templates = {
            "suggest_refactoring": """
                Consider extracting this logic into a separate method 
                for better testability and reusability.
                
                Example:
                ```python
                def calculate_trading_fee(amount, fee_rate):
                    return amount * fee_rate
                ```
            """,
            
            "performance_concern": """
                This operation is O(n²). For large datasets, 
                consider using a dict for O(1) lookup:
                
                ```python
                # Instead of:
                for item in list1:
                    if item in list2:  # O(n)
                
                # Use:
                set2 = set(list2)
                for item in list1:
                    if item in set2:  # O(1)
                ```
            """,
            
            "security_issue": """
                ⚠️ Security: This could be vulnerable to SQL injection.
                Use parameterized queries:
                
                ```python
                # Vulnerable:
                query = f"SELECT * FROM users WHERE id = {user_id}"
                
                # Secure:
                query = "SELECT * FROM users WHERE id = ?"
                cursor.execute(query, (user_id,))
                ```
            """,
            
            "praise_good_code": """
                Excellent error handling here! This pattern of 
                catching specific exceptions and providing 
                meaningful error messages is exactly what we need. 🎉
            """,
            
            "suggest_test": """
                This edge case could use a test. Consider adding:
                
                ```python
                def test_handle_empty_input():
                    assert function([]) == expected_result
                ```
            """,
            
            "question_approach": """
                I see this implements a custom sort. Have you considered 
                using `sorted()` with a key function? It might be more 
                maintainable. What are your thoughts?
            """,
            
            "documentation_needed": """
                This algorithm is complex. Could you add a docstring 
                explaining the approach? Future maintainers 
                (including future you!) will appreciate it.
            """
        }
        
        return templates
    
    def code_review_etiquette(self):
        """Etichetta professionale nel code review"""
        
        print("\n🤝 CODE REVIEW ETIQUETTE")
        print("=" * 60)
        
        dos_and_donts = {
            "✅ DO": [
                "Review the code, not the person",
                "Provide specific examples",
                "Explain the 'why' behind suggestions",
                "Acknowledge good code",
                "Be humble - you might be wrong",
                "Respond promptly",
                "Test the code locally if complex",
                "Consider the author's experience level",
                "Focus on important issues first",
                "Use 'we' instead of 'you' language"
            ],
            
            "❌ DON'T": [
                "Make personal attacks",
                "Bikeshed on minor style issues",
                "Demand changes without explanation",
                "Review when tired or frustrated",
                "Approve without actually reviewing",
                "Block PR for perfect code",
                "Ignore the PR context/requirements",
                "Rewrite their code in comments",
                "Use sarcasm or hostile tone",
                "Review more than 400 lines at once"
            ]
        }
        
        for category, items in dos_and_donts.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  • {item}")

# ============================================================================
# PART 3: RECEIVING FEEDBACK
# ============================================================================

class ReceivingFeedback:
    """Come ricevere e gestire feedback"""
    
    def feedback_response_strategies(self):
        """Strategie per rispondere al feedback"""
        
        print("\n📥 RECEIVING FEEDBACK GRACEFULLY")
        print("=" * 60)
        
        strategies = {
            "🧘 Mindset": [
                "Feedback is a gift, not an attack",
                "Reviewers want to help improve the code",
                "Everyone's code can be improved",
                "It's about the code, not about you",
                "Learning opportunity, not criticism"
            ],
            
            "💭 Processing": [
                "Read all comments before responding",
                "Take a break if feeling defensive",
                "Ask for clarification if unclear",
                "Consider the reviewer's perspective",
                "Look for patterns in feedback"
            ],
            
            "💬 Responding": [
                "Thank reviewers for their time",
                "Acknowledge valid points",
                "Explain your reasoning calmly",
                "Ask questions to understand better",
                "Propose alternatives if disagreeing"
            ]
        }
        
        for category, items in strategies.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  • {item}")
        
        # Response templates
        print("\n📝 RESPONSE TEMPLATES:")
        
        response_templates = {
            "Agreeing": "Good catch! I'll fix this.",
            
            "Clarifying": "Thanks for the feedback! Could you elaborate on what you mean by X?",
            
            "Explaining": "I chose this approach because [reason]. Do you think [alternative] would be better?",
            
            "Disagreeing": "I see your point. My thinking was [explanation]. What do you think about [compromise]?",
            
            "Learning": "I wasn't aware of that pattern! Could you point me to docs/examples?",
            
            "Deferring": "Good suggestion! I'll create a follow-up ticket for this to keep this PR focused."
        }
        
        for situation, template in response_templates.items():
            print(f"\n{situation}:")
            print(f"  '{template}'")

# ============================================================================
# PART 4: AUTOMATED CODE REVIEW
# ============================================================================

class AutomatedCodeReview:
    """Tool automatici per code review"""
    
    def __init__(self):
        self.issues = []
        
    def python_linter(self, code: str) -> List[Dict[str, Any]]:
        """Linter personalizzato per Python"""
        
        print("\n🤖 AUTOMATED CODE REVIEW TOOLS")
        print("=" * 60)
        
        issues = []
        lines = code.split('\n')
        
        # Check various code quality issues
        for i, line in enumerate(lines, 1):
            # Line too long
            if len(line) > 79:
                issues.append({
                    'line': i,
                    'type': 'style',
                    'message': f'Line too long ({len(line)} > 79 characters)',
                    'severity': 'minor'
                })
            
            # Trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                issues.append({
                    'line': i,
                    'type': 'style',
                    'message': 'Trailing whitespace',
                    'severity': 'minor'
                })
            
            # TODO comments
            if 'TODO' in line or 'FIXME' in line:
                issues.append({
                    'line': i,
                    'type': 'maintenance',
                    'message': 'TODO/FIXME comment found',
                    'severity': 'info'
                })
            
            # Print statements (debugging left in)
            if 'print(' in line and not '#' in line[:line.find('print(')]:
                issues.append({
                    'line': i,
                    'type': 'quality',
                    'message': 'Print statement found (debugging code?)',
                    'severity': 'warning'
                })
            
            # Hardcoded passwords/secrets
            if any(secret in line.lower() for secret in ['password=', 'api_key=', 'secret=']):
                if '"' in line or "'" in line:
                    issues.append({
                        'line': i,
                        'type': 'security',
                        'message': 'Possible hardcoded secret',
                        'severity': 'critical'
                    })
            
            # Broad exception handling
            if 'except:' in line or 'except Exception:' in line:
                issues.append({
                    'line': i,
                    'type': 'quality',
                    'message': 'Broad exception clause',
                    'severity': 'warning'
                })
        
        return issues
    
    def complexity_analyzer(self, code: str) -> Dict[str, Any]:
        """Analizza complessità del codice"""
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {'error': 'Syntax error in code'}
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1  # Base complexity
                self.functions = {}
                self.current_function = None
                
            def visit_FunctionDef(self, node):
                self.current_function = node.name
                self.functions[node.name] = {
                    'complexity': 1,
                    'lines': node.end_lineno - node.lineno + 1,
                    'params': len(node.args.args)
                }
                self.generic_visit(node)
                self.current_function = None
                
            def visit_If(self, node):
                if self.current_function:
                    self.functions[self.current_function]['complexity'] += 1
                else:
                    self.complexity += 1
                self.generic_visit(node)
                
            def visit_For(self, node):
                if self.current_function:
                    self.functions[self.current_function]['complexity'] += 1
                else:
                    self.complexity += 1
                self.generic_visit(node)
                
            def visit_While(self, node):
                if self.current_function:
                    self.functions[self.current_function]['complexity'] += 1
                else:
                    self.complexity += 1
                self.generic_visit(node)
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        # Analyze results
        analysis = {
            'total_complexity': visitor.complexity,
            'functions': visitor.functions,
            'recommendations': []
        }
        
        # Add recommendations
        for func_name, metrics in visitor.functions.items():
            if metrics['complexity'] > 10:
                analysis['recommendations'].append(
                    f"Function '{func_name}' has high complexity ({metrics['complexity']}). Consider refactoring."
                )
            if metrics['lines'] > 50:
                analysis['recommendations'].append(
                    f"Function '{func_name}' is too long ({metrics['lines']} lines). Consider splitting."
                )
            if metrics['params'] > 5:
                analysis['recommendations'].append(
                    f"Function '{func_name}' has too many parameters ({metrics['params']}). Consider using objects."
                )
        
        return analysis
    
    def setup_ci_tools(self):
        """Setup CI/CD tools per review automatico"""
        
        print("\n🔧 CI/CD REVIEW TOOLS")
        print("=" * 60)
        
        tools = {
            "Linting": {
                "Python": ["flake8", "pylint", "black", "mypy"],
                "JavaScript": ["eslint", "prettier"],
                "General": ["pre-commit", "super-linter"]
            },
            
            "Testing": {
                "Coverage": ["pytest-cov", "coverage.py"],
                "Quality": ["pytest", "tox", "nox"],
                "Security": ["bandit", "safety", "snyk"]
            },
            
            "Code Quality": {
                "Complexity": ["radon", "mccabe", "cognitive-complexity"],
                "Duplication": ["pmd", "jscpd"],
                "Tech Debt": ["sonarqube", "codeclimate"]
            },
            
            "Documentation": {
                "Docstrings": ["pydocstyle", "darglint"],
                "API Docs": ["sphinx", "mkdocs"],
                "README": ["markdown-lint"]
            }
        }
        
        for category, subcategories in tools.items():
            print(f"\n{category}:")
            for subcat, items in subcategories.items():
                print(f"  {subcat}: {', '.join(items)}")
        
        # Example pre-commit config
        print("\n📝 EXAMPLE .pre-commit-config.yaml:")
        pre_commit_config = """
        repos:
        - repo: https://github.com/pre-commit/pre-commit-hooks
          rev: v4.4.0
          hooks:
            - id: trailing-whitespace
            - id: end-of-file-fixer
            - id: check-yaml
            - id: check-added-large-files
            
        - repo: https://github.com/psf/black
          rev: 22.10.0
          hooks:
            - id: black
              language_version: python3.9
              
        - repo: https://github.com/pycqa/flake8
          rev: 5.0.4
          hooks:
            - id: flake8
              args: ['--max-line-length=88']
              
        - repo: https://github.com/pycqa/isort
          rev: 5.10.1
          hooks:
            - id: isort
              args: ["--profile", "black"]
        """
        
        print(pre_commit_config)

# ============================================================================
# PART 5: TEAM COLLABORATION
# ============================================================================

class TeamCollaboration:
    """Collaborazione efficace nel team"""
    
    def communication_patterns(self):
        """Pattern di comunicazione nel team"""
        
        print("\n💬 TEAM COMMUNICATION PATTERNS")
        print("=" * 60)
        
        patterns = {
            "Daily Standup": {
                "frequency": "Daily",
                "duration": "15 minutes",
                "format": "Yesterday/Today/Blockers",
                "purpose": "Sync and unblock"
            },
            
            "Code Pairing": {
                "frequency": "2-3 times/week",
                "duration": "1-2 hours",
                "format": "Driver/Navigator",
                "purpose": "Knowledge sharing"
            },
            
            "Tech Talks": {
                "frequency": "Weekly",
                "duration": "30-60 minutes",
                "format": "Presentation + Q&A",
                "purpose": "Learning and sharing"
            },
            
            "Retrospectives": {
                "frequency": "Bi-weekly",
                "duration": "1 hour",
                "format": "Start/Stop/Continue",
                "purpose": "Continuous improvement"
            },
            
            "Architecture Reviews": {
                "frequency": "Per feature",
                "duration": "1-2 hours",
                "format": "Design doc review",
                "purpose": "Alignment on approach"
            }
        }
        
        for meeting, details in patterns.items():
            print(f"\n{meeting}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
    
    def documentation_standards(self):
        """Standard di documentazione del team"""
        
        print("\n📚 DOCUMENTATION STANDARDS")
        print("=" * 60)
        
        # Python docstring example
        print("PYTHON DOCSTRING FORMAT:")
        docstring_example = '''
        def calculate_portfolio_return(
            positions: List[Position],
            start_date: datetime,
            end_date: datetime,
            benchmark: Optional[str] = None
        ) -> Dict[str, float]:
            """
            Calculate portfolio return over a given period.
            
            This function computes the time-weighted return of a portfolio,
            optionally comparing it against a benchmark index.
            
            Args:
                positions: List of Position objects representing holdings
                start_date: Start of the calculation period
                end_date: End of the calculation period
                benchmark: Optional benchmark symbol for comparison
            
            Returns:
                Dictionary containing:
                    - 'portfolio_return': Portfolio return as decimal
                    - 'benchmark_return': Benchmark return if provided
                    - 'alpha': Excess return over benchmark
                    - 'sharpe_ratio': Risk-adjusted return
            
            Raises:
                ValueError: If date range is invalid
                PortfolioError: If positions data is corrupted
            
            Example:
                >>> positions = [Position('AAPL', 100), Position('GOOGL', 50)]
                >>> returns = calculate_portfolio_return(
                ...     positions, 
                ...     datetime(2024, 1, 1),
                ...     datetime(2024, 12, 31)
                ... )
                >>> print(f"Return: {returns['portfolio_return']:.2%}")
                Return: 15.34%
            
            Note:
                This calculation assumes daily rebalancing and
                includes dividends in the return calculation.
            """
        '''
        
        print(docstring_example)
        
        # README template
        print("\nREADME.md TEMPLATE:")
        readme_template = """
        # Project Name
        
        Brief description of what this project does.
        
        ## 🚀 Quick Start
        
        ```bash
        pip install -r requirements.txt
        python main.py
        ```
        
        ## 📋 Prerequisites
        
        - Python 3.8+
        - PostgreSQL 12+
        - Redis 6+
        
        ## 🔧 Installation
        
        1. Clone the repo
        2. Install dependencies
        3. Configure environment
        4. Run migrations
        
        ## 🧪 Testing
        
        ```bash
        pytest tests/
        ```
        
        ## 📖 API Documentation
        
        See [API.md](docs/API.md)
        
        ## 🤝 Contributing
        
        See [CONTRIBUTING.md](CONTRIBUTING.md)
        
        ## 📝 License
        
        MIT License - see [LICENSE](LICENSE)
        """
        
        print(readme_template)

# ============================================================================
# PART 6: CODE REVIEW PROJECTS
# ============================================================================

class CodeReviewProjects:
    """Progetti pratici per code review"""
    
    def project_review_bot(self):
        """Build an automated review bot"""
        
        print("\n🤖 PROJECT: Automated Review Bot")
        print("=" * 60)
        
        class ReviewBot:
            def __init__(self):
                self.rules = []
                self.stats = {'files_reviewed': 0, 'issues_found': 0}
            
            def add_rule(self, pattern: str, message: str, severity: str = 'info'):
                """Add a review rule"""
                self.rules.append({
                    'pattern': re.compile(pattern),
                    'message': message,
                    'severity': severity
                })
            
            def review_file(self, filepath: str) -> List[Dict]:
                """Review a single file"""
                issues = []
                
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines, 1):
                    for rule in self.rules:
                        if rule['pattern'].search(line):
                            issues.append({
                                'file': filepath,
                                'line': i,
                                'message': rule['message'],
                                'severity': rule['severity'],
                                'code': line.strip()
                            })
                
                self.stats['files_reviewed'] += 1
                self.stats['issues_found'] += len(issues)
                
                return issues
            
            def generate_report(self, issues: List[Dict]) -> str:
                """Generate review report"""
                report = "# 🤖 Automated Code Review Report\n\n"
                report += f"Files reviewed: {self.stats['files_reviewed']}\n"
                report += f"Issues found: {self.stats['issues_found']}\n\n"
                
                if not issues:
                    report += "✅ No issues found!\n"
                    return report
                
                # Group by severity
                by_severity = {'critical': [], 'warning': [], 'info': []}
                for issue in issues:
                    by_severity[issue['severity']].append(issue)
                
                for severity in ['critical', 'warning', 'info']:
                    if by_severity[severity]:
                        report += f"\n## {severity.upper()} ({len(by_severity[severity])})\n\n"
                        for issue in by_severity[severity]:
                            report += f"- **{issue['file']}:{issue['line']}**\n"
                            report += f"  {issue['message']}\n"
                            report += f"  ```python\n  {issue['code']}\n  ```\n\n"
                
                return report
        
        # Setup bot with rules
        bot = ReviewBot()
        bot.add_rule(r'print\(', 'Remove debug print statements', 'warning')
        bot.add_rule(r'TODO|FIXME', 'Unresolved TODO/FIXME', 'info')
        bot.add_rule(r'password\s*=\s*["\']', 'Hardcoded password', 'critical')
        bot.add_rule(r'except\s*:', 'Broad exception handling', 'warning')
        
        print("Bot features:")
        print("✅ Configurable rules")
        print("✅ Multiple severity levels")
        print("✅ Detailed reporting")
        print("✅ GitHub integration ready")
        
        return bot
    
    def project_pr_analyzer(self):
        """Analyze PR patterns"""
        
        print("\n📊 PROJECT: PR Pattern Analyzer")
        print("=" * 60)
        
        @dataclass
        class PullRequest:
            id: int
            author: str
            title: str
            files_changed: int
            lines_added: int
            lines_deleted: int
            comments: int
            review_time_hours: float
            approved: bool
            merged: bool
        
        class PRAnalyzer:
            def __init__(self):
                self.prs = []
            
            def analyze_patterns(self) -> Dict[str, Any]:
                """Analyze PR patterns"""
                if not self.prs:
                    return {}
                
                analysis = {
                    'total_prs': len(self.prs),
                    'merge_rate': sum(1 for pr in self.prs if pr.merged) / len(self.prs),
                    'avg_files_changed': sum(pr.files_changed for pr in self.prs) / len(self.prs),
                    'avg_lines_changed': sum(pr.lines_added + pr.lines_deleted for pr in self.prs) / len(self.prs),
                    'avg_review_time': sum(pr.review_time_hours for pr in self.prs) / len(self.prs),
                    'avg_comments': sum(pr.comments for pr in self.prs) / len(self.prs)
                }
                
                # Identify patterns
                analysis['patterns'] = []
                
                # Large PRs
                large_prs = [pr for pr in self.prs if pr.lines_added + pr.lines_deleted > 500]
                if large_prs:
                    analysis['patterns'].append(
                        f"⚠️ {len(large_prs)} PRs are too large (>500 lines)"
                    )
                
                # Slow reviews
                slow_reviews = [pr for pr in self.prs if pr.review_time_hours > 48]
                if slow_reviews:
                    analysis['patterns'].append(
                        f"⏰ {len(slow_reviews)} PRs took >48 hours to review"
                    )
                
                # Low discussion
                no_discussion = [pr for pr in self.prs if pr.comments == 0]
                if no_discussion:
                    analysis['patterns'].append(
                        f"💬 {len(no_discussion)} PRs had no discussion"
                    )
                
                return analysis
            
            def recommend_improvements(self) -> List[str]:
                """Recommend process improvements"""
                recommendations = []
                analysis = self.analyze_patterns()
                
                if analysis.get('avg_lines_changed', 0) > 400:
                    recommendations.append("Consider smaller, more focused PRs")
                
                if analysis.get('avg_review_time', 0) > 24:
                    recommendations.append("Set up review reminders or rotation")
                
                if analysis.get('avg_comments', 0) < 2:
                    recommendations.append("Encourage more discussion in reviews")
                
                if analysis.get('merge_rate', 1) < 0.8:
                    recommendations.append("Investigate why PRs aren't being merged")
                
                return recommendations
        
        print("Analyzer features:")
        print("✅ PR size analysis")
        print("✅ Review time tracking")
        print("✅ Pattern identification")
        print("✅ Process recommendations")
        
        return PRAnalyzer()

# ============================================================================
# EXERCISES
# ============================================================================

def code_review_exercises():
    """50 code review exercises"""
    
    print("\n👀 CODE REVIEW EXERCISES")
    print("=" * 60)
    
    exercises = {
        "Review Skills (1-20)": [
            "Review a 100-line Python function",
            "Find 5 bugs in provided code",
            "Identify security vulnerabilities",
            "Suggest performance improvements",
            "Review code for SOLID violations",
            "Write constructive feedback",
            "Review a database schema",
            "Check test coverage",
            "Review API design",
            "Identify code smells",
            "Review error handling",
            "Check for race conditions",
            "Review documentation",
            "Validate input sanitization",
            "Review logging strategy",
            "Check for memory leaks",
            "Review configuration management",
            "Validate business logic",
            "Review code style consistency",
            "Check accessibility compliance"
        ],
        
        "Feedback Practice (21-35)": [
            "Rewrite harsh feedback constructively",
            "Give feedback to senior developer",
            "Give feedback to junior developer",
            "Handle defensive responses",
            "Explain complex issue simply",
            "Suggest alternatives diplomatically",
            "Praise good code effectively",
            "Ask clarifying questions",
            "Disagree respectfully",
            "Build consensus on approach",
            "Escalate blocking issues",
            "Mentor through review",
            "Document review decisions",
            "Handle urgent reviews",
            "Balance speed vs thoroughness"
        ],
        
        "Process Improvement (36-50)": [
            "Setup pre-commit hooks",
            "Configure CI/CD pipeline",
            "Create review checklist",
            "Implement review metrics",
            "Setup automated linting",
            "Create PR template",
            "Setup code coverage",
            "Implement security scanning",
            "Create review guidelines",
            "Setup review rotation",
            "Track review velocity",
            "Reduce review cycle time",
            "Implement pair programming",
            "Create coding standards",
            "Setup review automation"
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
    """Run code review module"""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║           👀 CODE REVIEW & TEAM COLLABORATION               ║
    ║              From Solo Coder to Team Player                 ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    modules = {
        "1": ("Review Fundamentals", CodeReviewFundamentals),
        "2": ("Giving Feedback", GivingFeedback),
        "3": ("Receiving Feedback", ReceivingFeedback),
        "4": ("Automated Review", AutomatedCodeReview),
        "5": ("Team Collaboration", TeamCollaboration),
        "6": ("Projects", CodeReviewProjects),
        "7": ("Exercises", code_review_exercises)
    }
    
    while True:
        print("\n📚 SELECT MODULE:")
        for key, (name, _) in modules.items():
            print(f"  {key}. {name}")
        print("  Q. Quit")
        
        choice = input("\nChoice: ").strip().upper()
        
        if choice == 'Q':
            break
        elif choice == '7':
            code_review_exercises()
        else:
            # Run selected module
            pass

if __name__ == "__main__":
    main()
