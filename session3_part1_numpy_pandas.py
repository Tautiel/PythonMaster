"""
🚀 SESSIONE 3 - PARTE 1: NUMPY & PANDAS MASTERY
===============================================
Data Manipulation & Analysis Foundation
Durata: 90 minuti di data science fundamentals
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import json
from typing import List, Dict, Any, Optional, Tuple
import random

print("="*80)
print("📊 SESSIONE 3 PARTE 1: NUMPY & PANDAS MASTERY")
print("="*80)

# ==============================================================================
# SEZIONE 1: NUMPY FUNDAMENTALS
# ==============================================================================

def section1_numpy_fundamentals():
    """NumPy: la base del scientific computing"""
    
    print("\n" + "="*60)
    print("🔢 SEZIONE 1: NUMPY FUNDAMENTALS")
    print("="*60)
    
    # 1.1 ARRAY CREATION & BASICS
    print("\n📌 1.1 ARRAY CREATION & BASICS")
    print("-"*40)
    
    # Creazione arrays
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    
    print(f"1D Array: {arr1}")
    print(f"Shape: {arr1.shape}, Dtype: {arr1.dtype}")
    
    print(f"\n2D Array:\n{arr2}")
    print(f"Shape: {arr2.shape}, Dimensions: {arr2.ndim}")
    
    # Special arrays
    zeros = np.zeros((3, 3))
    ones = np.ones((2, 4))
    identity = np.eye(3)
    random_arr = np.random.random((3, 3))
    
    print(f"\nIdentity matrix:\n{identity}")
    
    # Range arrays
    range_arr = np.arange(0, 10, 2)  # Start, stop, step
    linspace_arr = np.linspace(0, 1, 5)  # 5 punti tra 0 e 1
    
    print(f"\nArange: {range_arr}")
    print(f"Linspace: {linspace_arr}")
    
    # 1.2 ARRAY OPERATIONS
    print("\n⚡ 1.2 ARRAY OPERATIONS")
    print("-"*40)
    
    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])
    
    # Operazioni element-wise
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a ** 2 = {a ** 2}")
    print(f"sqrt(a) = {np.sqrt(a)}")
    
    # Operazioni matriciali
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print(f"\nMatrix multiplication A @ B:\n{A @ B}")
    print(f"Element-wise A * B:\n{A * B}")
    
    # Broadcasting
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"\nOriginal:\n{arr}")
    print(f"Add 10 (broadcasting):\n{arr + 10}")
    print(f"Multiply by column:\n{arr * np.array([1, 2, 3])}")
    
    # 1.3 INDEXING & SLICING
    print("\n🔍 1.3 INDEXING & SLICING")
    print("-"*40)
    
    arr = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])
    
    print(f"Array:\n{arr}")
    print(f"arr[1, 2] = {arr[1, 2]}")
    print(f"arr[:, 1] = {arr[:, 1]}")  # Colonna 1
    print(f"arr[1, :] = {arr[1, :]}")  # Riga 1
    print(f"arr[1:, 2:] =\n{arr[1:, 2:]}")  # Submatrix
    
    # Boolean indexing
    print("\nBoolean indexing:")
    mask = arr > 5
    print(f"Elements > 5: {arr[mask]}")
    
    # Fancy indexing
    indices = [0, 2]
    print(f"Select rows 0 and 2:\n{arr[indices]}")
    
    # 1.4 ARRAY MANIPULATION
    print("\n🔧 1.4 ARRAY MANIPULATION")
    print("-"*40)
    
    arr = np.arange(12)
    print(f"Original: {arr}")
    
    # Reshape
    reshaped = arr.reshape(3, 4)
    print(f"Reshape to 3x4:\n{reshaped}")
    
    # Flatten vs Ravel
    flattened = reshaped.flatten()  # Copia
    raveled = reshaped.ravel()      # Vista
    print(f"Flattened: {flattened}")
    
    # Transpose
    transposed = reshaped.T
    print(f"Transposed:\n{transposed}")
    
    # Stack & Split
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    vstacked = np.vstack([a, b])
    hstacked = np.hstack([a, b])
    
    print(f"\nVstack:\n{vstacked}")
    print(f"Hstack: {hstacked}")
    
    # 1.5 STATISTICAL OPERATIONS
    print("\n📊 1.5 STATISTICAL OPERATIONS")
    print("-"*40)
    
    data = np.random.randn(1000)  # Normal distribution
    
    print(f"Mean: {data.mean():.4f}")
    print(f"Std: {data.std():.4f}")
    print(f"Min: {data.min():.4f}")
    print(f"Max: {data.max():.4f}")
    print(f"Median: {np.median(data):.4f}")
    print(f"25th percentile: {np.percentile(data, 25):.4f}")
    print(f"75th percentile: {np.percentile(data, 75):.4f}")
    
    # Aggregation su assi
    matrix = np.random.randint(1, 10, (3, 4))
    print(f"\nMatrix:\n{matrix}")
    print(f"Sum per row: {matrix.sum(axis=1)}")
    print(f"Sum per column: {matrix.sum(axis=0)}")
    print(f"Mean per column: {matrix.mean(axis=0)}")

# ==============================================================================
# SEZIONE 2: PANDAS FUNDAMENTALS
# ==============================================================================

def section2_pandas_fundamentals():
    """Pandas: data manipulation powerhouse"""
    
    print("\n" + "="*60)
    print("🐼 SEZIONE 2: PANDAS FUNDAMENTALS")
    print("="*60)
    
    # 2.1 SERIES & DATAFRAMES
    print("\n📋 2.1 SERIES & DATAFRAMES")
    print("-"*40)
    
    # Series
    s = pd.Series([1, 3, 5, 7, 9], index=['a', 'b', 'c', 'd', 'e'])
    print(f"Series:\n{s}")
    print(f"\nAccess by index: s['c'] = {s['c']}")
    print(f"Access by position: s.iloc[2] = {s.iloc[2]}")
    
    # DataFrame creation
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 28, 32],
        'City': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'],
        'Salary': [70000, 85000, 95000, 65000, 78000]
    }
    
    df = pd.DataFrame(data)
    print(f"\nDataFrame:\n{df}")
    print(f"\nInfo:")
    print(df.info())
    print(f"\nDescribe:\n{df.describe()}")
    
    # 2.2 DATA SELECTION & FILTERING
    print("\n🔍 2.2 DATA SELECTION & FILTERING")
    print("-"*40)
    
    # Column selection
    print(f"Single column:\n{df['Name']}")
    print(f"\nMultiple columns:\n{df[['Name', 'Salary']]}")
    
    # Row selection
    print(f"\nRow by index:\n{df.loc[2]}")
    print(f"\nRows by slice:\n{df.loc[1:3]}")
    
    # Boolean filtering
    high_salary = df[df['Salary'] > 75000]
    print(f"\nHigh salary (>75000):\n{high_salary}")
    
    # Multiple conditions
    filtered = df[(df['Age'] > 28) & (df['Salary'] < 90000)]
    print(f"\nAge > 28 AND Salary < 90000:\n{filtered}")
    
    # Query method
    result = df.query('Age > 30 and City == "LA"')
    print(f"\nQuery result:\n{result}")
    
    # 2.3 DATA MANIPULATION
    print("\n🔧 2.3 DATA MANIPULATION")
    print("-"*40)
    
    # Adding columns
    df['Bonus'] = df['Salary'] * 0.1
    df['Total'] = df['Salary'] + df['Bonus']
    
    print(f"With new columns:\n{df}")
    
    # Applying functions
    df['Age_Category'] = df['Age'].apply(
        lambda x: 'Young' if x < 30 else 'Adult'
    )
    
    # Map values
    city_map = {'NYC': 'New York', 'LA': 'Los Angeles'}
    df['City_Full'] = df['City'].map(city_map).fillna(df['City'])
    
    print(f"\nAfter transformations:\n{df[['Name', 'Age_Category', 'City_Full']]}")
    
    # 2.4 GROUPBY OPERATIONS
    print("\n📊 2.4 GROUPBY OPERATIONS")
    print("-"*40)
    
    # Create sample data
    sales_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=100),
        'Product': np.random.choice(['A', 'B', 'C'], 100),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'Sales': np.random.randint(100, 1000, 100),
        'Quantity': np.random.randint(1, 50, 100)
    })
    
    # Groupby single column
    product_sales = sales_data.groupby('Product')['Sales'].sum()
    print(f"Sales by Product:\n{product_sales}")
    
    # Groupby multiple columns
    region_product = sales_data.groupby(['Region', 'Product'])['Sales'].mean()
    print(f"\nAverage Sales by Region and Product:\n{region_product}")
    
    # Multiple aggregations
    agg_result = sales_data.groupby('Product').agg({
        'Sales': ['sum', 'mean', 'max'],
        'Quantity': ['sum', 'mean']
    })
    print(f"\nMultiple aggregations:\n{agg_result}")
    
    # 2.5 MERGE & JOIN
    print("\n🔗 2.5 MERGE & JOIN")
    print("-"*40)
    
    # Create sample dataframes
    employees = pd.DataFrame({
        'emp_id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'dept_id': [10, 20, 10, 30]
    })
    
    departments = pd.DataFrame({
        'dept_id': [10, 20, 30],
        'dept_name': ['Sales', 'Marketing', 'IT']
    })
    
    # Inner join
    merged = pd.merge(employees, departments, on='dept_id')
    print(f"Inner Join:\n{merged}")
    
    # Left join
    left_merged = pd.merge(employees, departments, on='dept_id', how='left')
    print(f"\nLeft Join:\n{left_merged}")
    
    # Concat dataframes
    df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    
    concatenated = pd.concat([df1, df2], ignore_index=True)
    print(f"\nConcatenated:\n{concatenated}")

# ==============================================================================
# SEZIONE 3: ADVANCED PANDAS
# ==============================================================================

def section3_advanced_pandas():
    """Advanced Pandas techniques"""
    
    print("\n" + "="*60)
    print("🚀 SEZIONE 3: ADVANCED PANDAS")
    print("="*60)
    
    # 3.1 TIME SERIES
    print("\n⏰ 3.1 TIME SERIES")
    print("-"*40)
    
    # Create time series data
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    ts_data = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(365).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 365)
    })
    ts_data.set_index('date', inplace=True)
    
    print(f"Time Series Data:\n{ts_data.head()}")
    
    # Resampling
    monthly = ts_data.resample('M').agg({
        'value': 'mean',
        'volume': 'sum'
    })
    print(f"\nMonthly aggregation:\n{monthly.head()}")
    
    # Rolling window
    ts_data['MA7'] = ts_data['value'].rolling(window=7).mean()
    ts_data['MA30'] = ts_data['value'].rolling(window=30).mean()
    
    print(f"\nWith moving averages:\n{ts_data[['value', 'MA7', 'MA30']].head(35)}")
    
    # Date filtering
    jan_data = ts_data['2024-01']
    print(f"\nJanuary 2024 data:\n{jan_data.head()}")
    
    # 3.2 PIVOT TABLES
    print("\n📊 3.2 PIVOT TABLES")
    print("-"*40)
    
    # Create sample data
    sales = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=1000),
        'Product': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'Category': np.random.choice(['Electronics', 'Clothing'], 1000),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'Sales': np.random.randint(100, 1000, 1000),
        'Units': np.random.randint(1, 50, 1000)
    })
    
    # Pivot table
    pivot = pd.pivot_table(
        sales,
        values='Sales',
        index='Product',
        columns='Region',
        aggfunc='sum',
        fill_value=0
    )
    
    print(f"Pivot Table:\n{pivot}")
    
    # Multi-index pivot
    multi_pivot = pd.pivot_table(
        sales,
        values=['Sales', 'Units'],
        index=['Category', 'Product'],
        columns='Region',
        aggfunc={'Sales': 'sum', 'Units': 'mean'},
        fill_value=0
    )
    
    print(f"\nMulti-index Pivot:\n{multi_pivot}")
    
    # 3.3 CATEGORICAL DATA
    print("\n🏷️ 3.3 CATEGORICAL DATA")
    print("-"*40)
    
    df = pd.DataFrame({
        'id': range(1000),
        'grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], 1000),
        'size': np.random.choice(['S', 'M', 'L', 'XL'], 1000)
    })
    
    # Convert to categorical
    df['grade'] = pd.Categorical(
        df['grade'], 
        categories=['F', 'D', 'C', 'B', 'A'],
        ordered=True
    )
    
    df['size'] = pd.Categorical(
        df['size'],
        categories=['S', 'M', 'L', 'XL'],
        ordered=True
    )
    
    print(f"Categorical info:")
    print(df.info())
    
    # Categorical operations
    print(f"\nGrade distribution:\n{df['grade'].value_counts().sort_index()}")
    
    # Filter by category
    high_grades = df[df['grade'] > 'C']
    print(f"\nHigh grades (> C): {len(high_grades)} students")
    
    # 3.4 MISSING DATA HANDLING
    print("\n❓ 3.4 MISSING DATA HANDLING")
    print("-"*40)
    
    # Create data with missing values
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5],
        'D': [np.nan, np.nan, np.nan, 4, 5]
    })
    
    print(f"Data with NaN:\n{df}")
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    
    # Fill missing values
    df_filled = df.fillna({
        'A': df['A'].mean(),
        'B': df['B'].median(),
        'D': 0
    })
    
    print(f"\nAfter filling:\n{df_filled}")
    
    # Drop missing values
    df_dropped = df.dropna(thresh=3)  # Keep rows with at least 3 non-NaN
    print(f"\nAfter dropping (thresh=3):\n{df_dropped}")
    
    # Interpolation
    df['B_interpolated'] = df['B'].interpolate()
    print(f"\nWith interpolation:\n{df[['B', 'B_interpolated']]}")
    
    # 3.5 STRING OPERATIONS
    print("\n📝 3.5 STRING OPERATIONS")
    print("-"*40)
    
    df = pd.DataFrame({
        'name': ['John Doe', 'jane smith', 'Bob JONES', '  Alice Brown  '],
        'email': ['john@gmail.com', 'JANE@YAHOO.COM', 'bob@hotmail.com', 'alice@gmail.com'],
        'phone': ['123-456-7890', '(555) 123-4567', '9876543210', '555.123.4567']
    })
    
    print(f"Original:\n{df}")
    
    # String cleaning
    df['name_clean'] = df['name'].str.strip().str.title()
    df['email_lower'] = df['email'].str.lower()
    df['domain'] = df['email'].str.split('@').str[1].str.lower()
    
    # Extract patterns
    df['phone_digits'] = df['phone'].str.replace(r'\D', '', regex=True)
    
    print(f"\nCleaned:\n{df[['name_clean', 'email_lower', 'domain', 'phone_digits']]}")

# ==============================================================================
# SEZIONE 4: NUMPY & PANDAS FOR FINANCE
# ==============================================================================

def section4_finance_applications():
    """Financial applications with NumPy & Pandas"""
    
    print("\n" + "="*60)
    print("💰 SEZIONE 4: FINANCE APPLICATIONS")
    print("="*60)
    
    # 4.1 STOCK DATA SIMULATION
    print("\n📈 4.1 STOCK DATA SIMULATION")
    print("-"*40)
    
    # Simulate stock prices
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='B')  # Business days
    
    # Multiple stocks
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    prices = pd.DataFrame(index=dates)
    
    for stock in stocks:
        initial_price = np.random.uniform(100, 200)
        returns = np.random.randn(252) * 0.02  # 2% daily volatility
        price_series = initial_price * (1 + returns).cumprod()
        prices[stock] = price_series
    
    print(f"Stock Prices:\n{prices.head()}")
    print(f"\nSummary Statistics:\n{prices.describe()}")
    
    # 4.2 RETURNS CALCULATION
    print("\n💹 4.2 RETURNS CALCULATION")
    print("-"*40)
    
    # Daily returns
    returns = prices.pct_change()
    print(f"Daily Returns:\n{returns.head()}")
    
    # Cumulative returns
    cumulative_returns = (1 + returns).cumprod() - 1
    print(f"\nCumulative Returns (last 5 days):\n{cumulative_returns.tail()}")
    
    # Log returns
    log_returns = np.log(prices / prices.shift(1))
    print(f"\nLog Returns:\n{log_returns.head()}")
    
    # 4.3 RISK METRICS
    print("\n⚠️ 4.3 RISK METRICS")
    print("-"*40)
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252)
    print(f"Annualized Volatility:\n{volatility}")
    
    # Sharpe Ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    excess_returns = returns.mean() * 252 - risk_free_rate
    sharpe_ratio = excess_returns / volatility
    print(f"\nSharpe Ratio:\n{sharpe_ratio}")
    
    # Maximum Drawdown
    def calculate_max_drawdown(prices):
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    max_dd = prices.apply(calculate_max_drawdown)
    print(f"\nMaximum Drawdown:\n{max_dd}")
    
    # 4.4 PORTFOLIO OPTIMIZATION
    print("\n🎯 4.4 PORTFOLIO OPTIMIZATION")
    print("-"*40)
    
    # Equal weight portfolio
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    
    # Portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Portfolio metrics
    portfolio_mean = portfolio_returns.mean() * 252
    portfolio_std = portfolio_returns.std() * np.sqrt(252)
    portfolio_sharpe = (portfolio_mean - risk_free_rate) / portfolio_std
    
    print(f"Portfolio Annual Return: {portfolio_mean:.2%}")
    print(f"Portfolio Annual Volatility: {portfolio_std:.2%}")
    print(f"Portfolio Sharpe Ratio: {portfolio_sharpe:.2f}")
    
    # Correlation matrix
    correlation = returns.corr()
    print(f"\nCorrelation Matrix:\n{correlation}")
    
    # 4.5 TECHNICAL INDICATORS
    print("\n📊 4.5 TECHNICAL INDICATORS")
    print("-"*40)
    
    # Focus on one stock
    stock_price = prices['AAPL'].copy()
    
    # Moving Averages
    ma20 = stock_price.rolling(window=20).mean()
    ma50 = stock_price.rolling(window=50).mean()
    
    # RSI (Relative Strength Index)
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(stock_price)
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    bb_middle = stock_price.rolling(window=bb_period).mean()
    bb_upper = bb_middle + bb_std * stock_price.rolling(window=bb_period).std()
    bb_lower = bb_middle - bb_std * stock_price.rolling(window=bb_period).std()
    
    # Create indicators dataframe
    indicators = pd.DataFrame({
        'Price': stock_price,
        'MA20': ma20,
        'MA50': ma50,
        'RSI': rsi,
        'BB_Upper': bb_upper,
        'BB_Lower': bb_lower
    })
    
    print(f"Technical Indicators (last 5 days):\n{indicators.tail()}")

# ==============================================================================
# SEZIONE 5: DATA CLEANING & PREPROCESSING
# ==============================================================================

def section5_data_preprocessing():
    """Real-world data cleaning and preprocessing"""
    
    print("\n" + "="*60)
    print("🧹 SEZIONE 5: DATA CLEANING & PREPROCESSING")
    print("="*60)
    
    # 5.1 CREATING MESSY DATA
    print("\n🗑️ 5.1 CREATING REALISTIC MESSY DATA")
    print("-"*40)
    
    # Simulate messy real-world data
    np.random.seed(42)
    n_records = 1000
    
    messy_data = pd.DataFrame({
        'customer_id': range(1, n_records + 1),
        'name': ['John Doe', 'jane smith', 'BOB JONES', '  alice brown  ', None] * (n_records // 5),
        'age': np.random.randint(18, 80, n_records),
        'email': [f'user{i}@{"gmail" if i%3==0 else "yahoo"}.com' if i%10!=0 else None 
                  for i in range(n_records)],
        'phone': [f'{np.random.randint(100,999)}-{np.random.randint(100,999)}-{np.random.randint(1000,9999)}' 
                  if i%7!=0 else None for i in range(n_records)],
        'purchase_date': pd.date_range('2023-01-01', periods=n_records, freq='H'),
        'amount': np.random.exponential(100, n_records),
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Food', None], n_records),
        'rating': np.random.choice([1, 2, 3, 4, 5, None], n_records)
    })
    
    # Add some outliers
    messy_data.loc[messy_data.index % 100 == 0, 'amount'] = np.random.uniform(5000, 10000)
    messy_data.loc[messy_data.index % 150 == 0, 'age'] = np.random.choice([150, -5, 999])
    
    print(f"Messy Data Sample:\n{messy_data.head(10)}")
    print(f"\nData Info:")
    print(messy_data.info())
    print(f"\nMissing Values:\n{messy_data.isnull().sum()}")
    
    # 5.2 CLEANING PIPELINE
    print("\n🔧 5.2 CLEANING PIPELINE")
    print("-"*40)
    
    def clean_data_pipeline(df):
        """Complete data cleaning pipeline"""
        df_clean = df.copy()
        
        # 1. Handle missing values
        df_clean['name'].fillna('Unknown', inplace=True)
        df_clean['email'].fillna('no_email@unknown.com', inplace=True)
        df_clean['category'].fillna('Other', inplace=True)
        df_clean['rating'].fillna(df_clean['rating'].median(), inplace=True)
        
        # 2. Clean string columns
        df_clean['name'] = df_clean['name'].str.strip().str.title()
        df_clean['email'] = df_clean['email'].str.lower().str.strip()
        df_clean['category'] = df_clean['category'].str.capitalize()
        
        # 3. Fix outliers
        df_clean.loc[df_clean['age'] > 120, 'age'] = df_clean['age'].median()
        df_clean.loc[df_clean['age'] < 0, 'age'] = df_clean['age'].median()
        
        # 4. Cap extreme amounts (winsorizing)
        amount_99 = df_clean['amount'].quantile(0.99)
        df_clean.loc[df_clean['amount'] > amount_99, 'amount'] = amount_99
        
        # 5. Create derived features
        df_clean['purchase_month'] = df_clean['purchase_date'].dt.month
        df_clean['purchase_dayofweek'] = df_clean['purchase_date'].dt.dayofweek
        df_clean['is_weekend'] = df_clean['purchase_dayofweek'].isin([5, 6])
        
        return df_clean
    
    cleaned_data = clean_data_pipeline(messy_data)
    
    print(f"Cleaned Data Sample:\n{cleaned_data.head()}")
    print(f"\nCleaned Data Info:")
    print(cleaned_data.info())
    
    # 5.3 FEATURE ENGINEERING
    print("\n🏗️ 5.3 FEATURE ENGINEERING")
    print("-"*40)
    
    # Binning continuous variables
    cleaned_data['age_group'] = pd.cut(
        cleaned_data['age'], 
        bins=[0, 25, 35, 50, 65, 100],
        labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder']
    )
    
    cleaned_data['amount_category'] = pd.qcut(
        cleaned_data['amount'],
        q=4,
        labels=['Low', 'Medium', 'High', 'VeryHigh']
    )
    
    # One-hot encoding
    category_dummies = pd.get_dummies(cleaned_data['category'], prefix='cat')
    cleaned_data = pd.concat([cleaned_data, category_dummies], axis=1)
    
    print(f"Feature Engineered Sample:\n{cleaned_data[['age_group', 'amount_category']].head()}")
    print(f"\nNew Features: {[col for col in cleaned_data.columns if col.startswith('cat_')]}")
    
    # 5.4 DATA VALIDATION
    print("\n✅ 5.4 DATA VALIDATION")
    print("-"*40)
    
    def validate_data(df):
        """Validate cleaned data"""
        validations = {
            'No nulls in critical fields': df[['customer_id', 'name', 'amount']].isnull().sum().sum() == 0,
            'Age in valid range': ((df['age'] >= 18) & (df['age'] <= 120)).all(),
            'Amount is positive': (df['amount'] >= 0).all(),
            'Ratings in 1-5 range': df['rating'].between(1, 5).all(),
            'Valid email format': df['email'].str.contains('@').all(),
            'No duplicate customer_ids': df['customer_id'].nunique() == len(df)
        }
        
        for check, result in validations.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {check}")
        
        return all(validations.values())
    
    is_valid = validate_data(cleaned_data)
    print(f"\n{'✅ Data is valid!' if is_valid else '❌ Data validation failed!'}")
    
    # 5.5 EXPORT CLEANED DATA
    print("\n💾 5.5 DATA EXPORT OPTIONS")
    print("-"*40)
    
    # Different export formats
    print("Export formats available:")
    print("  • CSV: df.to_csv('data.csv', index=False)")
    print("  • Excel: df.to_excel('data.xlsx', sheet_name='Clean')")
    print("  • JSON: df.to_json('data.json', orient='records')")
    print("  • Parquet: df.to_parquet('data.parquet', compression='snappy')")
    print("  • SQL: df.to_sql('table_name', connection)")
    
    # Sample export (to dict for display)
    sample_export = cleaned_data.head(3).to_dict('records')
    print(f"\nSample export (JSON format):")
    print(json.dumps(sample_export[0], indent=2, default=str)[:500] + "...")

# ==============================================================================
# MAIN - Menu per le sezioni
# ==============================================================================

def main():
    """Menu principale per NumPy & Pandas"""
    
    print("\n" + "="*60)
    print("📊 NUMPY & PANDAS - SCEGLI SEZIONE")
    print("="*60)
    
    sections = [
        ("NumPy Fundamentals", section1_numpy_fundamentals),
        ("Pandas Fundamentals", section2_pandas_fundamentals),
        ("Advanced Pandas", section3_advanced_pandas),
        ("Finance Applications", section4_finance_applications),
        ("Data Preprocessing", section5_data_preprocessing)
    ]
    
    print("\n0. Esegui TUTTO")
    for i, (name, _) in enumerate(sections, 1):
        print(f"{i}. {name}")
    
    choice = input("\nScegli (0-5): ")
    
    try:
        choice = int(choice)
        if choice == 0:
            for name, func in sections:
                input(f"\n➡️ Press ENTER for: {name}")
                func()
        elif 1 <= choice <= len(sections):
            sections[choice-1][1]()
        else:
            print("Scelta non valida")
    except (ValueError, IndexError):
        print("Scelta non valida")
    
    print("\n" + "="*60)
    print("✅ PARTE 1 COMPLETATA!")
    print("Prossimo: session3_part2_visualization.py")
    print("="*60)

if __name__ == "__main__":
    main()
