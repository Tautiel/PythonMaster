"""
🚀 SESSIONE 3 - PARTE 4: 4 PROGETTI DATA SCIENCE
================================================
Progetti Production-Ready: Analytics, ML API, Dashboard, Trading Bot
Durata: 120 minuti di progetti pratici
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import json
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("💼 SESSIONE 3 PARTE 4: 4 PROGETTI DATA SCIENCE")
print("="*80)

# ==============================================================================
# PROGETTO 1: COMPLETE DATA ANALYSIS PIPELINE
# ==============================================================================

class DataAnalysisPipeline:
    """Complete ETL and Analysis Pipeline for E-commerce Data"""
    
    def __init__(self, name="E-commerce Analytics"):
        self.name = name
        self.data = None
        self.processed_data = None
        self.insights = {}
        
    def generate_sample_data(self, n_records=10000):
        """Generate realistic e-commerce data"""
        np.random.seed(42)
        
        # Date range
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=x) for x in range(365)]
        
        # Generate transactions
        data = {
            'transaction_id': range(1, n_records + 1),
            'date': np.random.choice(dates, n_records),
            'customer_id': np.random.randint(1, 2000, n_records),
            'product_id': np.random.randint(1, 500, n_records),
            'category': np.random.choice(['Electronics', 'Clothing', 'Books', 
                                        'Home', 'Sports', 'Beauty'], n_records),
            'price': np.random.exponential(50, n_records) + 10,
            'quantity': np.random.randint(1, 5, n_records),
            'payment_method': np.random.choice(['Credit Card', 'PayPal', 
                                              'Debit Card', 'Cash'], n_records),
            'customer_age': np.random.randint(18, 70, n_records),
            'customer_segment': np.random.choice(['Premium', 'Regular', 'New'], 
                                               n_records, p=[0.2, 0.5, 0.3]),
            'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], 
                                     n_records, p=[0.5, 0.35, 0.15]),
            'marketing_channel': np.random.choice(['Organic', 'Paid Search', 
                                                 'Social', 'Email', 'Direct'], 
                                                n_records)
        }
        
        self.data = pd.DataFrame(data)
        self.data['revenue'] = self.data['price'] * self.data['quantity']
        
        # Add some patterns
        # Higher sales on weekends
        self.data.loc[self.data['date'].dt.dayofweek.isin([5, 6]), 'revenue'] *= 1.3
        
        # Premium customers spend more
        self.data.loc[self.data['customer_segment'] == 'Premium', 'revenue'] *= 1.5
        
        print(f"✅ Generated {n_records} transaction records")
        return self.data
    
    def clean_and_preprocess(self):
        """Clean and preprocess data"""
        print("\n📧 Cleaning and preprocessing data...")
        
        # Remove duplicates
        before = len(self.data)
        self.data = self.data.drop_duplicates()
        after = len(self.data)
        print(f"  Removed {before - after} duplicate records")
        
        # Handle outliers
        Q1 = self.data['revenue'].quantile(0.01)
        Q3 = self.data['revenue'].quantile(0.99)
        
        self.data = self.data[(self.data['revenue'] >= Q1) & 
                              (self.data['revenue'] <= Q3)]
        print(f"  Removed outliers (kept 1st-99th percentile)")
        
        # Add derived features
        self.data['month'] = self.data['date'].dt.month
        self.data['quarter'] = self.data['date'].dt.quarter
        self.data['day_of_week'] = self.data['date'].dt.dayofweek
        self.data['is_weekend'] = self.data['day_of_week'].isin([5, 6])
        
        # Customer lifetime value approximation
        customer_stats = self.data.groupby('customer_id').agg({
            'revenue': ['sum', 'mean', 'count'],
            'date': ['min', 'max']
        })
        customer_stats.columns = ['total_revenue', 'avg_order_value', 
                                 'order_count', 'first_purchase', 'last_purchase']
        
        self.processed_data = self.data.copy()
        print("  ✅ Data cleaned and preprocessed")
        
        return self.processed_data
    
    def perform_eda(self):
        """Exploratory Data Analysis"""
        print("\n📊 Performing Exploratory Data Analysis...")
        
        # Basic statistics
        self.insights['total_revenue'] = self.data['revenue'].sum()
        self.insights['avg_order_value'] = self.data['revenue'].mean()
        self.insights['total_customers'] = self.data['customer_id'].nunique()
        self.insights['total_products'] = self.data['product_id'].nunique()
        
        print(f"\n  Key Metrics:")
        print(f"    Total Revenue: ${self.insights['total_revenue']:,.2f}")
        print(f"    Average Order Value: ${self.insights['avg_order_value']:.2f}")
        print(f"    Unique Customers: {self.insights['total_customers']:,}")
        print(f"    Unique Products: {self.insights['total_products']:,}")
        
        # Category analysis
        category_revenue = self.data.groupby('category')['revenue'].agg(['sum', 'mean', 'count'])
        self.insights['top_category'] = category_revenue['sum'].idxmax()
        
        print(f"\n  Category Performance:")
        print(category_revenue.sort_values('sum', ascending=False))
        
        # Customer segmentation analysis
        segment_analysis = self.data.groupby('customer_segment').agg({
            'revenue': ['mean', 'sum'],
            'customer_id': 'nunique'
        })
        
        print(f"\n  Customer Segment Analysis:")
        print(segment_analysis)
        
        # Time series analysis
        daily_revenue = self.data.groupby(self.data['date'].dt.date)['revenue'].sum()
        self.insights['best_day'] = daily_revenue.idxmax()
        self.insights['worst_day'] = daily_revenue.idxmin()
        
        return self.insights
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📈 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('E-commerce Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Revenue by Category
        category_revenue = self.data.groupby('category')['revenue'].sum().sort_values()
        category_revenue.plot(kind='barh', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Revenue by Category')
        axes[0, 0].set_xlabel('Revenue ($)')
        
        # 2. Customer Segment Distribution
        segment_counts = self.data['customer_segment'].value_counts()
        axes[0, 1].pie(segment_counts.values, labels=segment_counts.index, 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Customer Segment Distribution')
        
        # 3. Payment Method Usage
        payment_revenue = self.data.groupby('payment_method')['revenue'].sum()
        payment_revenue.plot(kind='bar', ax=axes[0, 2], color='lightgreen')
        axes[0, 2].set_title('Revenue by Payment Method')
        axes[0, 2].set_ylabel('Revenue ($)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Monthly Revenue Trend
        monthly_revenue = self.data.groupby(self.data['date'].dt.to_period('M'))['revenue'].sum()
        monthly_revenue.plot(ax=axes[1, 0], marker='o', color='darkblue')
        axes[1, 0].set_title('Monthly Revenue Trend')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Revenue ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Device Usage
        device_data = self.data.groupby('device').agg({
            'revenue': 'mean',
            'transaction_id': 'count'
        })
        device_data.plot(kind='bar', ax=axes[1, 1], secondary_y='transaction_id')
        axes[1, 1].set_title('Device Usage Analysis')
        axes[1, 1].set_ylabel('Avg Revenue ($)')
        axes[1, 1].right_ax.set_ylabel('Transaction Count')
        
        # 6. Age vs Revenue Scatter
        axes[1, 2].scatter(self.data['customer_age'], self.data['revenue'], 
                          alpha=0.3, s=10)
        axes[1, 2].set_title('Customer Age vs Revenue')
        axes[1, 2].set_xlabel('Customer Age')
        axes[1, 2].set_ylabel('Revenue ($)')
        
        plt.tight_layout()
        plt.show()
        
        print("  ✅ Visualizations created")
    
    def generate_report(self):
        """Generate analysis report"""
        report = f"""
        ========================================
        E-COMMERCE ANALYTICS REPORT
        ========================================
        
        EXECUTIVE SUMMARY
        -----------------
        Total Revenue: ${self.insights['total_revenue']:,.2f}
        Average Order Value: ${self.insights['avg_order_value']:.2f}
        Total Customers: {self.insights['total_customers']:,}
        Top Category: {self.insights['top_category']}
        
        KEY INSIGHTS
        ------------
        1. Best performing day: {self.insights['best_day']}
        2. Worst performing day: {self.insights['worst_day']}
        3. Weekend sales are 30% higher than weekdays
        4. Premium customers generate 50% more revenue
        
        RECOMMENDATIONS
        ---------------
        1. Focus marketing on {self.insights['top_category']} category
        2. Implement weekend-specific promotions
        3. Develop premium customer retention programs
        4. Optimize mobile experience (50% of transactions)
        
        ========================================
        """
        
        print(report)
        return report
    
    def run_pipeline(self):
        """Run complete analysis pipeline"""
        print(f"\n🚀 Running {self.name} Pipeline...")
        print("="*50)
        
        # Generate data
        self.generate_sample_data(5000)
        
        # Clean and preprocess
        self.clean_and_preprocess()
        
        # Perform EDA
        self.perform_eda()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report()
        
        print("\n✅ Pipeline completed successfully!")
        return self.processed_data

# ==============================================================================
# PROGETTO 2: ML PREDICTION API
# ==============================================================================

class MLPredictionAPI:
    """Machine Learning API for Customer Churn Prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        
    def generate_churn_data(self, n_samples=5000):
        """Generate synthetic customer churn data"""
        np.random.seed(42)
        
        # Generate features
        data = {
            'customer_id': range(1, n_samples + 1),
            'tenure_months': np.random.randint(1, 72, n_samples),
            'monthly_charges': np.random.uniform(20, 120, n_samples),
            'total_charges': np.random.uniform(100, 8000, n_samples),
            'num_services': np.random.randint(1, 8, n_samples),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 
                                             'Two year'], n_samples),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check',
                                              'Bank transfer', 'Credit card'], n_samples),
            'support_tickets': np.random.poisson(2, n_samples),
            'late_payments': np.random.poisson(1, n_samples),
            'satisfaction_score': np.random.uniform(1, 5, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create churn based on patterns
        churn_probability = (
            0.3 * (df['tenure_months'] < 12).astype(int) +
            0.2 * (df['monthly_charges'] > 70).astype(int) +
            0.2 * (df['contract_type'] == 'Month-to-month').astype(int) +
            0.1 * (df['support_tickets'] > 3).astype(int) +
            0.1 * (df['late_payments'] > 2).astype(int) +
            0.1 * (df['satisfaction_score'] < 3).astype(int) +
            np.random.uniform(-0.2, 0.2, n_samples)
        )
        
        df['churn'] = (churn_probability > 0.5).astype(int)
        
        print(f"✅ Generated {n_samples} customer records")
        print(f"Churn rate: {df['churn'].mean():.2%}")
        
        return df
    
    def preprocess_features(self, df):
        """Preprocess features for ML"""
        # Encode categorical variables
        df_encoded = pd.get_dummies(df, columns=['contract_type', 'payment_method'])
        
        # Select features
        feature_cols = [col for col in df_encoded.columns 
                       if col not in ['customer_id', 'churn']]
        
        self.feature_names = feature_cols
        
        return df_encoded[feature_cols], df_encoded.get('churn')
    
    def train_model(self, X_train, y_train):
        """Train churn prediction model"""
        print("\n🤖 Training ML model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        print("  ✅ Model trained successfully")
        
        # Feature importance
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n  Top 5 Important Features:")
        print(feature_importance.head())
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n📊 Evaluating model...")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        self.metrics['accuracy'] = accuracy_score(y_test, y_pred)
        self.metrics['precision'] = precision_score(y_test, y_pred)
        self.metrics['recall'] = recall_score(y_test, y_pred)
        self.metrics['f1'] = f1_score(y_test, y_pred)
        
        print(f"\n  Model Performance:")
        print(f"    Accuracy:  {self.metrics['accuracy']:.3f}")
        print(f"    Precision: {self.metrics['precision']:.3f}")
        print(f"    Recall:    {self.metrics['recall']:.3f}")
        print(f"    F1-Score:  {self.metrics['f1']:.3f}")
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Churn Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return self.metrics
    
    def predict_single(self, customer_data):
        """Predict churn for single customer"""
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Preprocess
        X, _ = self.preprocess_features(df)
        
        # Ensure all features are present
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        
        X = X[self.feature_names]
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        return {
            'customer_id': customer_data.get('customer_id'),
            'churn_prediction': bool(prediction),
            'churn_probability': float(probability[1]),
            'retention_probability': float(probability[0]),
            'risk_level': 'High' if probability[1] > 0.7 else 
                         'Medium' if probability[1] > 0.3 else 'Low'
        }
    
    def create_api_endpoint(self):
        """Simulate API endpoint"""
        print("\n🌐 API Endpoints:")
        print("="*50)
        
        endpoints = {
            'POST /predict': 'Single customer churn prediction',
            'POST /batch_predict': 'Batch prediction for multiple customers',
            'GET /model/metrics': 'Get model performance metrics',
            'GET /model/features': 'Get required feature list',
            'POST /model/retrain': 'Retrain model with new data'
        }
        
        for endpoint, description in endpoints.items():
            print(f"  {endpoint}: {description}")
        
        # Example API usage
        print("\n  Example API Request:")
        print("  POST /predict")
        print("  Content-Type: application/json")
        
        example_request = {
            'customer_id': 12345,
            'tenure_months': 6,
            'monthly_charges': 85.50,
            'total_charges': 513.00,
            'num_services': 3,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'support_tickets': 5,
            'late_payments': 2,
            'satisfaction_score': 2.5
        }
        
        print(f"  Body: {json.dumps(example_request, indent=4)}")
        
        # Make prediction
        result = self.predict_single(example_request)
        
        print("\n  API Response:")
        print(f"  {json.dumps(result, indent=4)}")
        
        return result
    
    def run_ml_api(self):
        """Run complete ML API pipeline"""
        print("\n🚀 Running ML Prediction API...")
        print("="*50)
        
        # Generate data
        df = self.generate_churn_data()
        
        # Preprocess
        X, y = self.preprocess_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.train_model(X_train, y_train)
        
        # Evaluate
        self.evaluate_model(X_test, y_test)
        
        # Create API
        self.create_api_endpoint()
        
        print("\n✅ ML API ready for deployment!")

# ==============================================================================
# PROGETTO 3: REAL-TIME ANALYTICS DASHBOARD
# ==============================================================================

class RealTimeDashboard:
    """Real-time Analytics Dashboard for Monitoring"""
    
    def __init__(self):
        self.metrics_history = []
        self.current_metrics = {}
        
    def generate_streaming_data(self):
        """Generate simulated streaming data"""
        np.random.seed(None)  # Random seed for variability
        
        timestamp = datetime.now()
        
        # Generate metrics
        metrics = {
            'timestamp': timestamp,
            'active_users': np.random.randint(800, 1200),
            'requests_per_second': np.random.randint(50, 200),
            'avg_response_time': np.random.uniform(100, 500),  # ms
            'error_rate': np.random.uniform(0, 0.05),  # 0-5%
            'cpu_usage': np.random.uniform(20, 80),  # percentage
            'memory_usage': np.random.uniform(40, 70),  # percentage
            'conversion_rate': np.random.uniform(0.02, 0.05),
            'revenue_per_minute': np.random.uniform(100, 500)
        }
        
        self.metrics_history.append(metrics)
        self.current_metrics = metrics
        
        return metrics
    
    def create_dashboard(self):
        """Create comprehensive dashboard visualization"""
        print("\n📊 Creating Real-time Dashboard...")
        
        # Generate 60 data points (last hour)
        for _ in range(60):
            self.generate_streaming_data()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.metrics_history)
        
        # Create dashboard
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Real-time Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Active Users (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df.index, df['active_users'], 'b-', linewidth=2)
        ax1.fill_between(df.index, df['active_users'], alpha=0.3)
        ax1.set_title('Active Users')
        ax1.set_ylabel('Users')
        ax1.grid(True, alpha=0.3)
        
        # Add current value
        current_users = df['active_users'].iloc[-1]
        ax1.text(0.95, 0.95, f'Current: {current_users}',
                transform=ax1.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Requests per Second (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(df.index[-20:], df['requests_per_second'].iloc[-20:],
               color='green', alpha=0.7)
        ax2.set_title('Requests/Second (Last 20 min)')
        ax2.set_ylabel('Requests')
        ax2.grid(True, alpha=0.3)
        
        # 3. Error Rate (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        error_colors = ['red' if x > 0.03 else 'orange' if x > 0.01 else 'green' 
                       for x in df['error_rate']]
        ax3.scatter(df.index, df['error_rate'] * 100, c=error_colors, alpha=0.6)
        ax3.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='Critical')
        ax3.set_title('Error Rate (%)')
        ax3.set_ylabel('Error %')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Response Time (middle-left)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(df.index, df['avg_response_time'], 'purple', linewidth=1.5)
        ax4.set_title('Average Response Time')
        ax4.set_ylabel('Time (ms)')
        ax4.set_xlabel('Time (minutes)')
        ax4.grid(True, alpha=0.3)
        
        # Add threshold line
        ax4.axhline(y=400, color='red', linestyle='--', alpha=0.5, 
                   label='SLA Threshold')
        ax4.legend()
        
        # 5. System Resources (middle-center)
        ax5 = fig.add_subplot(gs[1, 1])
        x = np.arange(2)
        width = 0.35
        
        current_cpu = df['cpu_usage'].iloc[-1]
        current_memory = df['memory_usage'].iloc[-1]
        
        bars1 = ax5.bar(x - width/2, [current_cpu, current_memory], width,
                       label='Current', color=['#FF6B6B', '#4ECDC4'])
        
        avg_cpu = df['cpu_usage'].mean()
        avg_memory = df['memory_usage'].mean()
        
        bars2 = ax5.bar(x + width/2, [avg_cpu, avg_memory], width,
                       label='Average', color=['#FF9999', '#7DD4D0'])
        
        ax5.set_ylabel('Usage (%)')
        ax5.set_title('System Resources')
        ax5.set_xticks(x)
        ax5.set_xticklabels(['CPU', 'Memory'])
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 6. Conversion Funnel (middle-right)
        ax6 = fig.add_subplot(gs[1, 2])
        
        funnel_data = {
            'Visitors': 1000,
            'Sign-ups': 150,
            'Active': 100,
            'Converted': 35
        }
        
        y_pos = np.arange(len(funnel_data))
        values = list(funnel_data.values())
        
        bars = ax6.barh(y_pos, values, color='skyblue')
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(funnel_data.keys())
        ax6.set_title('Conversion Funnel')
        ax6.set_xlabel('Count')
        
        # Add conversion rates
        for i, (bar, val) in enumerate(zip(bars, values)):
            if i > 0:
                rate = (val / values[i-1]) * 100
                ax6.text(val + 10, bar.get_y() + bar.get_height()/2,
                        f'{rate:.1f}%', va='center')
        
        # 7. Revenue Trend (bottom)
        ax7 = fig.add_subplot(gs[2, :])
        
        # Calculate cumulative revenue
        df['cumulative_revenue'] = df['revenue_per_minute'].cumsum()
        
        ax7_twin = ax7.twinx()
        
        # Plot revenue per minute
        ax7.bar(df.index, df['revenue_per_minute'], alpha=0.3, 
               color='green', label='Revenue/min')
        ax7.set_ylabel('Revenue per Minute ($)', color='green')
        ax7.tick_params(axis='y', labelcolor='green')
        
        # Plot cumulative revenue
        ax7_twin.plot(df.index, df['cumulative_revenue'], 'b-', 
                     linewidth=2, label='Cumulative')
        ax7_twin.set_ylabel('Cumulative Revenue ($)', color='blue')
        ax7_twin.tick_params(axis='y', labelcolor='blue')
        
        ax7.set_xlabel('Time (minutes)')
        ax7.set_title('Revenue Metrics')
        ax7.grid(True, alpha=0.3)
        
        # Add total revenue text
        total_revenue = df['cumulative_revenue'].iloc[-1]
        ax7.text(0.02, 0.95, f'Total: ${total_revenue:,.0f}',
                transform=ax7.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        plt.show()
        
        print("  ✅ Dashboard created")
        
        # Print summary statistics
        print("\n  📈 Current Metrics Summary:")
        print("  " + "="*40)
        for key, value in self.current_metrics.items():
            if key != 'timestamp':
                if 'rate' in key:
                    print(f"    {key}: {value:.2%}")
                elif 'usage' in key:
                    print(f"    {key}: {value:.1f}%")
                elif key == 'revenue_per_minute':
                    print(f"    {key}: ${value:.2f}")
                else:
                    print(f"    {key}: {value:.1f}")
        
        return df

# ==============================================================================
# PROGETTO 4: AI-POWERED TRADING BOT
# ==============================================================================

class AITradingBot:
    """AI-powered Trading Bot with ML predictions"""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.portfolio = {}
        self.trades_history = []
        self.model = None
        self.predictions = []
        
    def generate_market_data(self, days=365):
        """Generate synthetic market data"""
        np.random.seed(42)
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate price data for multiple stocks
        stocks = ['TECH', 'BANK', 'RETAIL', 'ENERGY']
        data = {}
        
        for stock in stocks:
            # Generate realistic price movement
            initial_price = np.random.uniform(50, 200)
            returns = np.random.randn(days) * 0.02  # 2% daily volatility
            
            # Add trend
            trend = np.linspace(0, np.random.uniform(-0.1, 0.2), days)
            returns = returns + trend/days
            
            prices = initial_price * (1 + returns).cumprod()
            
            # Add volume
            volume = np.random.randint(1000000, 10000000, days)
            
            # Technical indicators
            ma_20 = pd.Series(prices).rolling(20).mean()
            ma_50 = pd.Series(prices).rolling(50).mean()
            
            # RSI
            delta = pd.Series(prices).diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            data[stock] = pd.DataFrame({
                'date': dates,
                'open': prices * np.random.uniform(0.98, 1.02, days),
                'high': prices * np.random.uniform(1.01, 1.05, days),
                'low': prices * np.random.uniform(0.95, 0.99, days),
                'close': prices,
                'volume': volume,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'rsi': rsi
            })
        
        print(f"✅ Generated {days} days of market data for {len(stocks)} stocks")
        return data
    
    def create_features(self, df):
        """Create technical features for ML"""
        features = pd.DataFrame()
        
        # Price features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        features['ma_ratio_20'] = df['close'] / df['ma_20']
        features['ma_ratio_50'] = df['close'] / df['ma_50']
        
        # Volatility
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Volume features
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # RSI
        features['rsi'] = df['rsi']
        
        # Price position
        features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Target: Next day return (1 if positive, 0 if negative)
        features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def train_prediction_model(self, market_data):
        """Train ML model for price prediction"""
        print("\n🤖 Training prediction model...")
        
        all_features = []
        
        for stock, df in market_data.items():
            features = self.create_features(df)
            features['stock'] = stock
            all_features.append(features)
        
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Prepare data
        feature_cols = ['returns', 'log_returns', 'ma_ratio_20', 'ma_ratio_50',
                       'volatility', 'volume_ratio', 'rsi', 'price_position']
        
        X = combined_features[feature_cols]
        y = combined_features['target']
        
        # Split data
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"  Model Performance:")
        print(f"    Train accuracy: {train_score:.3f}")
        print(f"    Test accuracy: {test_score:.3f}")
        
        return self.model
    
    def generate_signals(self, market_data):
        """Generate trading signals"""
        print("\n📊 Generating trading signals...")
        
        signals = {}
        
        for stock, df in market_data.items():
            features = self.create_features(df)
            
            if len(features) > 0:
                # Get latest features
                latest_features = features[['returns', 'log_returns', 'ma_ratio_20', 
                                           'ma_ratio_50', 'volatility', 'volume_ratio',
                                           'rsi', 'price_position']].iloc[-1:]
                
                # Predict
                prediction = self.model.predict_proba(latest_features)[0]
                
                # Generate signal
                signal = {
                    'stock': stock,
                    'price': df['close'].iloc[-1],
                    'prediction': prediction[1],  # Probability of price increase
                    'rsi': df['rsi'].iloc[-1],
                    'volume_ratio': features['volume_ratio'].iloc[-1]
                }
                
                # Determine action
                if prediction[1] > 0.6 and signal['rsi'] < 70:
                    signal['action'] = 'BUY'
                    signal['confidence'] = prediction[1]
                elif prediction[1] < 0.4 and signal['rsi'] > 30:
                    signal['action'] = 'SELL'
                    signal['confidence'] = 1 - prediction[1]
                else:
                    signal['action'] = 'HOLD'
                    signal['confidence'] = 0.5
                
                signals[stock] = signal
        
        return signals
    
    def execute_trades(self, signals):
        """Execute trades based on signals"""
        print("\n💰 Executing trades...")
        
        for stock, signal in signals.items():
            if signal['action'] == 'BUY':
                # Calculate position size (risk management)
                position_size = self.current_capital * 0.1 * signal['confidence']
                shares = int(position_size / signal['price'])
                
                if shares > 0 and self.current_capital >= position_size:
                    # Execute buy
                    self.current_capital -= shares * signal['price']
                    
                    if stock in self.portfolio:
                        self.portfolio[stock]['shares'] += shares
                        self.portfolio[stock]['avg_price'] = (
                            (self.portfolio[stock]['avg_price'] * 
                             self.portfolio[stock]['shares'] + 
                             shares * signal['price']) / 
                            (self.portfolio[stock]['shares'] + shares)
                        )
                    else:
                        self.portfolio[stock] = {
                            'shares': shares,
                            'avg_price': signal['price']
                        }
                    
                    trade = {
                        'timestamp': datetime.now(),
                        'stock': stock,
                        'action': 'BUY',
                        'shares': shares,
                        'price': signal['price'],
                        'value': shares * signal['price']
                    }
                    
                    self.trades_history.append(trade)
                    print(f"  ✅ BUY {shares} shares of {stock} at ${signal['price']:.2f}")
            
            elif signal['action'] == 'SELL' and stock in self.portfolio:
                # Sell position
                shares = self.portfolio[stock]['shares']
                
                if shares > 0:
                    sell_value = shares * signal['price']
                    self.current_capital += sell_value
                    
                    profit = sell_value - (shares * self.portfolio[stock]['avg_price'])
                    
                    trade = {
                        'timestamp': datetime.now(),
                        'stock': stock,
                        'action': 'SELL',
                        'shares': shares,
                        'price': signal['price'],
                        'value': sell_value,
                        'profit': profit
                    }
                    
                    self.trades_history.append(trade)
                    del self.portfolio[stock]
                    
                    print(f"  ✅ SELL {shares} shares of {stock} at ${signal['price']:.2f} "
                          f"(Profit: ${profit:.2f})")
    
    def calculate_performance(self, market_data):
        """Calculate trading performance"""
        print("\n📈 Performance Report:")
        print("="*50)
        
        # Calculate portfolio value
        portfolio_value = self.current_capital
        
        for stock, position in self.portfolio.items():
            current_price = market_data[stock]['close'].iloc[-1]
            portfolio_value += position['shares'] * current_price
        
        # Performance metrics
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        # Win rate
        profitable_trades = [t for t in self.trades_history 
                           if t.get('profit', 0) > 0]
        losing_trades = [t for t in self.trades_history 
                        if t.get('profit', 0) < 0]
        
        win_rate = len(profitable_trades) / max(len(self.trades_history), 1)
        
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Current Portfolio Value: ${portfolio_value:,.2f}")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Cash Available: ${self.current_capital:,.2f}")
        print(f"  Total Trades: {len(self.trades_history)}")
        print(f"  Win Rate: {win_rate:.2%}")
        
        print(f"\n  Current Holdings:")
        for stock, position in self.portfolio.items():
            current_price = market_data[stock]['close'].iloc[-1]
            value = position['shares'] * current_price
            unrealized_pnl = value - (position['shares'] * position['avg_price'])
            
            print(f"    {stock}: {position['shares']} shares @ "
                  f"${position['avg_price']:.2f} (Current: ${current_price:.2f}, "
                  f"P&L: ${unrealized_pnl:.2f})")
        
        return {
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': len(self.trades_history)
        }
    
    def run_trading_bot(self):
        """Run complete trading bot simulation"""
        print("\n🤖 Running AI Trading Bot...")
        print("="*50)
        
        # Generate market data
        market_data = self.generate_market_data(365)
        
        # Train prediction model
        self.train_prediction_model(market_data)
        
        # Simulate trading for last 30 days
        print("\n📊 Simulating trading for last 30 days...")
        
        for day in range(30):
            # Update market data (in real scenario, this would be live data)
            signals = self.generate_signals(market_data)
            
            # Execute trades
            self.execute_trades(signals)
            
            # Simulate price movement for next day
            for stock in market_data:
                # Simple random walk for simulation
                last_price = market_data[stock]['close'].iloc[-1]
                new_price = last_price * (1 + np.random.randn() * 0.02)
                # Update last price (simplified)
                market_data[stock].loc[market_data[stock].index[-1], 'close'] = new_price
        
        # Calculate final performance
        performance = self.calculate_performance(market_data)
        
        print("\n✅ Trading simulation completed!")
        
        return performance

# ==============================================================================
# MAIN - Run all projects
# ==============================================================================

def main():
    """Run all 4 Data Science projects"""
    
    print("\n" + "="*60)
    print("🚀 4 PROGETTI DATA SCIENCE - MENU")
    print("="*60)
    
    projects = [
        ("Data Analysis Pipeline", lambda: DataAnalysisPipeline().run_pipeline()),
        ("ML Prediction API", lambda: MLPredictionAPI().run_ml_api()),
        ("Real-time Dashboard", lambda: RealTimeDashboard().create_dashboard()),
        ("AI Trading Bot", lambda: AITradingBot().run_trading_bot())
    ]
    
    print("\n0. Esegui TUTTI i progetti")
    for i, (name, _) in enumerate(projects, 1):
        print(f"{i}. {name}")
    
    choice = input("\nScegli progetto (0-4): ")
    
    try:
        choice = int(choice)
        if choice == 0:
            for name, func in projects:
                input(f"\n➡️ Press ENTER to run: {name}")
                func()
                print("\n" + "="*60)
        elif 1 <= choice <= len(projects):
            projects[choice-1][1]()
        else:
            print("Scelta non valida")
    except (ValueError, IndexError) as e:
        print(f"Errore: {e}")
    
    print("\n" + "="*60)
    print("✅ PROGETTI COMPLETATI!")
    print("="*60)

if __name__ == "__main__":
    # Import additional required libraries
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    main()
