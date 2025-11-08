"""
🚀 SESSIONE 3 - PARTE 2: DATA VISUALIZATION MASTERY
===================================================
Matplotlib, Seaborn, Plotly - Visualizzazioni professionali
Durata: 60 minuti di data visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configurazione stile
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("📊 SESSIONE 3 PARTE 2: DATA VISUALIZATION MASTERY")
print("="*80)

# ==============================================================================
# SEZIONE 1: MATPLOTLIB FUNDAMENTALS
# ==============================================================================

def section1_matplotlib_basics():
    """Matplotlib: il foundation della visualization"""
    
    print("\n" + "="*60)
    print("📈 SEZIONE 1: MATPLOTLIB FUNDAMENTALS")
    print("="*60)
    
    # 1.1 BASIC PLOTS
    print("\n📊 1.1 BASIC PLOTS")
    print("-"*40)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Matplotlib Basic Plots', fontsize=16, fontweight='bold')
    
    # Line plot
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    axes[0, 0].plot(x, y1, 'b-', label='sin(x)', linewidth=2)
    axes[0, 0].plot(x, y2, 'r--', label='cos(x)', linewidth=2)
    axes[0, 0].set_title('Line Plot')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Scatter plot
    np.random.seed(42)
    x_scatter = np.random.randn(100)
    y_scatter = 2 * x_scatter + np.random.randn(100) * 0.5
    
    axes[0, 1].scatter(x_scatter, y_scatter, c=x_scatter, cmap='viridis', alpha=0.6)
    axes[0, 1].set_title('Scatter Plot')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    
    # Bar plot
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']
    
    axes[0, 2].bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 2].set_title('Bar Plot')
    axes[0, 2].set_ylabel('Values')
    
    # Add value labels on bars
    for i, (cat, val) in enumerate(zip(categories, values)):
        axes[0, 2].text(i, val + 1, str(val), ha='center', fontweight='bold')
    
    # Histogram
    data = np.random.normal(100, 15, 1000)
    
    n, bins, patches = axes[1, 0].hist(data, bins=30, color='skyblue', 
                                       edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(data.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {data.mean():.1f}')
    axes[1, 0].set_title('Histogram')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Pie chart
    sizes = [30, 25, 20, 15, 10]
    explode = (0.1, 0, 0, 0, 0)  # Explode first slice
    
    axes[1, 1].pie(sizes, explode=explode, labels=categories, autopct='%1.1f%%',
                   shadow=True, startangle=90, colors=colors)
    axes[1, 1].set_title('Pie Chart')
    
    # Box plot
    data_box = [np.random.normal(100, 10, 100),
                np.random.normal(90, 20, 100),
                np.random.normal(110, 15, 100),
                np.random.normal(85, 25, 100)]
    
    bp = axes[1, 2].boxplot(data_box, labels=['A', 'B', 'C', 'D'],
                            patch_artist=True, notch=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors[:4]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1, 2].set_title('Box Plot')
    axes[1, 2].set_ylabel('Values')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Basic plots created successfully!")
    
    # 1.2 ADVANCED STYLING
    print("\n🎨 1.2 ADVANCED STYLING")
    print("-"*40)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Generate data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) * np.exp(-x/10)
    y2 = np.cos(x) * np.exp(-x/10)
    
    # Plot with custom styling
    ax.plot(x, y1, color='#FF6B6B', linewidth=2.5, linestyle='-',
            marker='o', markersize=4, markevery=10, label='Damped Sine',
            alpha=0.8)
    
    ax.plot(x, y2, color='#4ECDC4', linewidth=2.5, linestyle='--',
            marker='s', markersize=4, markevery=10, label='Damped Cosine',
            alpha=0.8)
    
    # Fill between
    ax.fill_between(x, y1, y2, where=(y1 > y2), interpolate=True,
                    alpha=0.3, color='gold', label='y1 > y2')
    
    # Customize axes
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax.set_title('Damped Oscillations', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Customize legend
    ax.legend(loc='upper right', frameon=True, fancybox=True,
             shadow=True, borderpad=1, ncol=3)
    
    # Add annotations
    max_idx = np.argmax(y1)
    ax.annotate('Maximum', xy=(x[max_idx], y1[max_idx]),
                xytext=(x[max_idx]+1, y1[max_idx]+0.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red')
    
    # Set limits
    ax.set_xlim([0, 10])
    ax.set_ylim([-1, 1])
    
    # Add text box
    textstr = 'Equation: y = sin(x) * e^(-x/10)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Advanced styling applied!")

# ==============================================================================
# SEZIONE 2: SEABORN STATISTICAL PLOTS
# ==============================================================================

def section2_seaborn_advanced():
    """Seaborn per statistical visualization"""
    
    print("\n" + "="*60)
    print("📊 SEZIONE 2: SEABORN STATISTICAL PLOTS")
    print("="*60)
    
    # 2.1 DISTRIBUTION PLOTS
    print("\n📈 2.1 DISTRIBUTION PLOTS")
    print("-"*40)
    
    # Generate sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'normal': np.random.normal(0, 1, 1000),
        'uniform': np.random.uniform(-3, 3, 1000),
        'exponential': np.random.exponential(1, 1000),
        'bimodal': np.concatenate([np.random.normal(-2, 0.5, 500),
                                   np.random.normal(2, 0.5, 500)])
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Seaborn Distribution Plots', fontsize=14, fontweight='bold')
    
    # Histogram with KDE
    sns.histplot(data['normal'], kde=True, ax=axes[0, 0], 
                color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Normal Distribution')
    
    # KDE plot comparison
    sns.kdeplot(data[['normal', 'uniform', 'exponential']], ax=axes[0, 1])
    axes[0, 1].set_title('KDE Comparison')
    axes[0, 1].legend(['Normal', 'Uniform', 'Exponential'])
    
    # Violin plot
    df_melt = pd.melt(data[['normal', 'uniform', 'exponential']], 
                     var_name='Distribution', value_name='Value')
    sns.violinplot(data=df_melt, x='Distribution', y='Value', ax=axes[1, 0])
    axes[1, 0].set_title('Violin Plot Comparison')
    
    # ECDF plot
    sns.ecdfplot(data=data[['normal', 'bimodal']], ax=axes[1, 1])
    axes[1, 1].set_title('Empirical CDF')
    axes[1, 1].legend(['Normal', 'Bimodal'])
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Distribution plots created!")
    
    # 2.2 REGRESSION PLOTS
    print("\n📉 2.2 REGRESSION PLOTS")
    print("-"*40)
    
    # Generate regression data
    n_points = 200
    x = np.random.randn(n_points)
    y_linear = 2 * x + np.random.randn(n_points) * 0.5
    y_quadratic = x**2 + np.random.randn(n_points) * 0.5
    y_log = np.log(np.abs(x) + 1) * np.sign(x) + np.random.randn(n_points) * 0.3
    
    reg_data = pd.DataFrame({
        'x': x,
        'y_linear': y_linear,
        'y_quadratic': y_quadratic,
        'y_log': y_log
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Seaborn Regression Plots', fontsize=14, fontweight='bold')
    
    # Linear regression
    sns.regplot(data=reg_data, x='x', y='y_linear', ax=axes[0, 0],
               scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    axes[0, 0].set_title('Linear Regression')
    
    # Polynomial regression
    sns.regplot(data=reg_data, x='x', y='y_quadratic', ax=axes[0, 1],
               order=2, scatter_kws={'alpha': 0.5}, line_kws={'color': 'green'})
    axes[0, 1].set_title('Quadratic Regression')
    
    # Residual plot
    sns.residplot(data=reg_data, x='x', y='y_linear', ax=axes[1, 0],
                 scatter_kws={'alpha': 0.5})
    axes[1, 0].set_title('Residual Plot')
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    
    # Joint plot (shown separately due to layout)
    axes[1, 1].text(0.5, 0.5, 'Joint Plot\n(Shown separately)', 
                   ha='center', va='center', fontsize=12)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    # Create joint plot separately
    joint_plot = sns.jointplot(data=reg_data, x='x', y='y_linear',
                               kind='reg', height=8)
    joint_plot.fig.suptitle('Joint Plot with Marginal Distributions', y=1.02)
    plt.show()
    
    print("✅ Regression plots created!")
    
    # 2.3 CATEGORICAL PLOTS
    print("\n📊 2.3 CATEGORICAL PLOTS")
    print("-"*40)
    
    # Generate categorical data
    np.random.seed(42)
    cat_data = pd.DataFrame({
        'category': np.repeat(['A', 'B', 'C', 'D'], 100),
        'group': np.tile(np.repeat(['Group1', 'Group2'], 50), 4),
        'value': np.concatenate([
            np.random.normal(100, 10, 100),  # A
            np.random.normal(110, 15, 100),  # B
            np.random.normal(90, 12, 100),   # C
            np.random.normal(105, 8, 100)    # D
        ]),
        'count': np.random.poisson(5, 400)
    })
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Seaborn Categorical Plots', fontsize=14, fontweight='bold')
    
    # Box plot
    sns.boxplot(data=cat_data, x='category', y='value', hue='group',
               ax=axes[0, 0])
    axes[0, 0].set_title('Box Plot')
    
    # Violin plot
    sns.violinplot(data=cat_data, x='category', y='value', hue='group',
                  split=True, ax=axes[0, 1])
    axes[0, 1].set_title('Split Violin Plot')
    
    # Swarm plot
    sns.swarmplot(data=cat_data.sample(200), x='category', y='value',
                 hue='group', ax=axes[0, 2], size=3)
    axes[0, 2].set_title('Swarm Plot')
    
    # Bar plot with confidence intervals
    sns.barplot(data=cat_data, x='category', y='value', hue='group',
               ax=axes[1, 0], ci=95)
    axes[1, 0].set_title('Bar Plot with 95% CI')
    
    # Point plot
    sns.pointplot(data=cat_data, x='category', y='value', hue='group',
                 ax=axes[1, 1], dodge=True, markers=['o', 's'])
    axes[1, 1].set_title('Point Plot')
    
    # Count plot
    sns.countplot(data=cat_data, x='category', hue='group', ax=axes[1, 2])
    axes[1, 2].set_title('Count Plot')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Categorical plots created!")

# ==============================================================================
# SEZIONE 3: HEATMAPS & CORRELATION
# ==============================================================================

def section3_heatmaps_correlation():
    """Heatmaps e correlation visualization"""
    
    print("\n" + "="*60)
    print("🔥 SEZIONE 3: HEATMAPS & CORRELATION")
    print("="*60)
    
    # 3.1 CORRELATION HEATMAP
    print("\n📊 3.1 CORRELATION HEATMAP")
    print("-"*40)
    
    # Generate correlated data
    np.random.seed(42)
    n_samples = 500
    
    # Create correlated features
    base = np.random.randn(n_samples)
    features = pd.DataFrame({
        'feature1': base + np.random.randn(n_samples) * 0.5,
        'feature2': -base + np.random.randn(n_samples) * 0.5,
        'feature3': base * 2 + np.random.randn(n_samples),
        'feature4': np.random.randn(n_samples),
        'feature5': base * 0.5 + np.random.randn(n_samples) * 2,
        'feature6': -base * 1.5 + np.random.randn(n_samples) * 0.3,
        'feature7': np.sin(base) + np.random.randn(n_samples) * 0.2,
        'feature8': np.random.randn(n_samples) * 3
    })
    
    # Calculate correlation
    corr_matrix = features.corr()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Standard heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, vmin=-1, vmax=1, ax=axes[0],
               cbar_kws={'label': 'Correlation'})
    axes[0].set_title('Correlation Heatmap', fontweight='bold')
    
    # Masked heatmap (show only upper triangle)
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
               cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=axes[1],
               square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
    axes[1].set_title('Masked Correlation Heatmap', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Correlation heatmaps created!")
    
    # 3.2 CLUSTERMAP
    print("\n🗺️ 3.2 CLUSTERMAP")
    print("-"*40)
    
    # Create clustermap
    clustermap = sns.clustermap(corr_matrix, annot=True, fmt='.2f',
                                cmap='coolwarm', center=0,
                                figsize=(10, 10),
                                dendrogram_ratio=0.15,
                                cbar_pos=(0.02, 0.83, 0.03, 0.15))
    
    clustermap.fig.suptitle('Hierarchical Clustering Heatmap', 
                            fontsize=14, fontweight='bold', y=1.02)
    plt.show()
    
    print("✅ Clustermap created!")
    
    # 3.3 PIVOT TABLE HEATMAP
    print("\n📋 3.3 PIVOT TABLE HEATMAP")
    print("-"*40)
    
    # Generate sales data
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    sales_data = pd.DataFrame({
        'date': np.random.choice(dates, 1000),
        'product': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'sales': np.random.randint(100, 1000, 1000)
    })
    
    # Add month and day of week
    sales_data['month'] = sales_data['date'].dt.month
    sales_data['dayofweek'] = sales_data['date'].dt.dayofweek
    
    # Create pivot tables
    pivot_product_region = sales_data.pivot_table(
        values='sales', index='product', columns='region', aggfunc='mean'
    )
    
    pivot_month_day = sales_data.pivot_table(
        values='sales', index='month', columns='dayofweek', aggfunc='mean'
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Product vs Region heatmap
    sns.heatmap(pivot_product_region, annot=True, fmt='.0f',
               cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Avg Sales'})
    axes[0].set_title('Average Sales: Product vs Region', fontweight='bold')
    axes[0].set_xlabel('Region')
    axes[0].set_ylabel('Product')
    
    # Month vs Day of Week heatmap
    sns.heatmap(pivot_month_day, annot=True, fmt='.0f',
               cmap='viridis', ax=axes[1], cbar_kws={'label': 'Avg Sales'})
    axes[1].set_title('Average Sales: Month vs Day of Week', fontweight='bold')
    axes[1].set_xlabel('Day of Week (0=Mon, 6=Sun)')
    axes[1].set_ylabel('Month')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Pivot table heatmaps created!")

# ==============================================================================
# SEZIONE 4: TIME SERIES VISUALIZATION
# ==============================================================================

def section4_time_series():
    """Time series visualization techniques"""
    
    print("\n" + "="*60)
    print("⏰ SEZIONE 4: TIME SERIES VISUALIZATION")
    print("="*60)
    
    # 4.1 STOCK PRICE VISUALIZATION
    print("\n📈 4.1 STOCK PRICE VISUALIZATION")
    print("-"*40)
    
    # Generate stock data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    
    # Simulate multiple stocks
    stock_data = pd.DataFrame(index=dates)
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    for stock in stocks:
        initial_price = np.random.uniform(100, 200)
        returns = np.random.randn(365) * 0.02
        prices = initial_price * (1 + returns).cumprod()
        stock_data[stock] = prices
    
    # Add volume data
    for stock in stocks:
        stock_data[f'{stock}_volume'] = np.random.randint(1000000, 10000000, 365)
    
    # Create comprehensive stock chart
    fig = plt.figure(figsize=(15, 12))
    
    # Main price plot
    ax1 = plt.subplot(3, 1, 1)
    for stock in stocks:
        ax1.plot(stock_data.index, stock_data[stock], label=stock, linewidth=2)
    
    ax1.set_title('Stock Prices - 2023', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add moving averages for AAPL
    ma20 = stock_data['AAPL'].rolling(window=20).mean()
    ma50 = stock_data['AAPL'].rolling(window=50).mean()
    
    ax1.plot(stock_data.index, ma20, 'g--', alpha=0.5, label='AAPL MA20')
    ax1.plot(stock_data.index, ma50, 'r--', alpha=0.5, label='AAPL MA50')
    
    # Volume subplot
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.bar(stock_data.index, stock_data['AAPL_volume'], 
           color='lightblue', alpha=0.5, label='AAPL Volume')
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_title('AAPL Trading Volume', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Returns subplot
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    returns = stock_data[stocks].pct_change()
    
    for stock in stocks:
        ax3.plot(returns.index, returns[stock].rolling(window=7).mean() * 100,
                label=f'{stock} 7-day avg', alpha=0.7)
    
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Returns (%)', fontsize=12)
    ax3.set_title('7-Day Average Returns', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Stock visualization created!")
    
    # 4.2 SEASONAL DECOMPOSITION
    print("\n🌊 4.2 SEASONAL PATTERNS")
    print("-"*40)
    
    # Generate seasonal data
    t = np.arange(0, 365*2)
    trend = t * 0.5
    seasonal = 10 * np.sin(2 * np.pi * t / 365)  # Yearly seasonality
    weekly = 5 * np.sin(2 * np.pi * t / 7)       # Weekly seasonality
    noise = np.random.randn(len(t)) * 2
    
    ts = trend + seasonal + weekly + noise + 100
    
    ts_df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=len(t), freq='D'),
        'value': ts,
        'trend': trend + 100,
        'seasonal': seasonal,
        'weekly': weekly
    })
    ts_df.set_index('date', inplace=True)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('Time Series Decomposition', fontsize=14, fontweight='bold')
    
    # Original series
    axes[0].plot(ts_df.index, ts_df['value'], 'b-', linewidth=1, alpha=0.8)
    axes[0].set_ylabel('Original', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    axes[1].plot(ts_df.index, ts_df['trend'], 'r-', linewidth=2)
    axes[1].set_ylabel('Trend', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal component
    axes[2].plot(ts_df.index, ts_df['seasonal'], 'g-', linewidth=1)
    axes[2].set_ylabel('Seasonal', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    # Weekly component
    axes[3].plot(ts_df.index[:30], ts_df['weekly'][:30], 'purple', linewidth=1)
    axes[3].set_ylabel('Weekly Pattern\n(first 30 days)', fontsize=11)
    axes[3].set_xlabel('Date', fontsize=11)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Seasonal decomposition visualized!")

# ==============================================================================
# SEZIONE 5: INTERACTIVE DASHBOARDS (Simulated)
# ==============================================================================

def section5_dashboard_layout():
    """Dashboard-style visualization"""
    
    print("\n" + "="*60)
    print("📊 SEZIONE 5: DASHBOARD LAYOUT")
    print("="*60)
    
    # Generate comprehensive dataset
    np.random.seed(42)
    n_days = 30
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    
    # KPI data
    revenue = np.random.uniform(80000, 120000, n_days)
    costs = revenue * np.random.uniform(0.6, 0.8, n_days)
    profit = revenue - costs
    customers = np.random.randint(800, 1200, n_days)
    satisfaction = np.random.uniform(3.5, 5.0, n_days)
    
    dashboard_data = pd.DataFrame({
        'date': dates,
        'revenue': revenue,
        'costs': costs,
        'profit': profit,
        'customers': customers,
        'satisfaction': satisfaction
    })
    
    # Create dashboard
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Business Dashboard - January 2024', fontsize=16, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # KPI Cards (top row)
    kpi_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    
    kpis = [
        ('Total Revenue', f'${revenue.sum()/1000:.0f}K', 'green'),
        ('Avg Daily Profit', f'${profit.mean()/1000:.1f}K', 'blue'),
        ('Total Customers', f'{customers.sum():,}', 'orange'),
        ('Avg Satisfaction', f'{satisfaction.mean():.2f}/5.0', 'purple')
    ]
    
    for ax, (title, value, color) in zip(kpi_axes, kpis):
        ax.text(0.5, 0.7, value, ha='center', va='center',
               fontsize=20, fontweight='bold', color=color)
        ax.text(0.5, 0.3, title, ha='center', va='center',
               fontsize=10, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Revenue trend (middle left)
    ax_revenue = fig.add_subplot(gs[1, :2])
    ax_revenue.plot(dashboard_data['date'], dashboard_data['revenue']/1000,
                   'b-', linewidth=2, marker='o', markersize=4)
    ax_revenue.fill_between(dashboard_data['date'], 0, 
                           dashboard_data['revenue']/1000, alpha=0.3)
    ax_revenue.set_title('Daily Revenue Trend', fontweight='bold')
    ax_revenue.set_ylabel('Revenue ($K)')
    ax_revenue.grid(True, alpha=0.3)
    ax_revenue.tick_params(axis='x', rotation=45)
    
    # Profit margin (middle right)
    ax_margin = fig.add_subplot(gs[1, 2:])
    profit_margin = (profit / revenue) * 100
    ax_margin.bar(dashboard_data['date'], profit_margin, 
                 color='green', alpha=0.7, edgecolor='darkgreen')
    ax_margin.set_title('Daily Profit Margin', fontweight='bold')
    ax_margin.set_ylabel('Margin (%)')
    ax_margin.axhline(y=profit_margin.mean(), color='red', 
                      linestyle='--', label=f'Avg: {profit_margin.mean():.1f}%')
    ax_margin.legend()
    ax_margin.grid(True, alpha=0.3)
    ax_margin.tick_params(axis='x', rotation=45)
    
    # Customer satisfaction (bottom left)
    ax_satisfaction = fig.add_subplot(gs[2, :2])
    scatter = ax_satisfaction.scatter(dashboard_data['customers'],
                                     dashboard_data['satisfaction'],
                                     c=dashboard_data['revenue']/1000,
                                     cmap='viridis', s=100, alpha=0.6,
                                     edgecolors='black')
    ax_satisfaction.set_title('Customers vs Satisfaction', fontweight='bold')
    ax_satisfaction.set_xlabel('Number of Customers')
    ax_satisfaction.set_ylabel('Satisfaction Score')
    ax_satisfaction.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax_satisfaction)
    cbar.set_label('Revenue ($K)', rotation=270, labelpad=15)
    
    # Cost breakdown (bottom right)
    ax_costs = fig.add_subplot(gs[2, 2:])
    
    # Simulate cost categories
    cost_categories = {
        'Operations': 0.4,
        'Marketing': 0.25,
        'Salaries': 0.2,
        'Infrastructure': 0.15
    }
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    explode = (0.05, 0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = ax_costs.pie(
        cost_categories.values(),
        labels=cost_categories.keys(),
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=explode,
        shadow=True
    )
    
    ax_costs.set_title('Cost Distribution', fontweight='bold')
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Dashboard created successfully!")
    print("\n💡 Note: For interactive dashboards, consider:")
    print("   • Plotly/Dash for web-based interactive visualizations")
    print("   • Streamlit for rapid dashboard deployment")
    print("   • Bokeh for interactive plots in Jupyter")
    print("   • Panel for complex dashboard applications")

# ==============================================================================
# MAIN - Menu per le sezioni
# ==============================================================================

def main():
    """Menu principale per Data Visualization"""
    
    print("\n" + "="*60)
    print("📊 DATA VISUALIZATION - SCEGLI SEZIONE")
    print("="*60)
    
    sections = [
        ("Matplotlib Fundamentals", section1_matplotlib_basics),
        ("Seaborn Statistical Plots", section2_seaborn_advanced),
        ("Heatmaps & Correlation", section3_heatmaps_correlation),
        ("Time Series Visualization", section4_time_series),
        ("Dashboard Layout", section5_dashboard_layout)
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
    print("✅ PARTE 2 COMPLETATA!")
    print("Prossimo: session3_part3_machine_learning.py")
    print("="*60)

if __name__ == "__main__":
    main()
