"""
🚀 SESSIONE 3 - PARTE 3: MACHINE LEARNING MASTERY
=================================================
Scikit-learn, Model Training, Evaluation & Deployment
Durata: 90 minuti di machine learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_auc_score, 
                           mean_squared_error, r2_score, classification_report)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🤖 SESSIONE 3 PARTE 3: MACHINE LEARNING MASTERY")
print("="*80)

# ==============================================================================
# SEZIONE 1: SUPERVISED LEARNING - CLASSIFICATION
# ==============================================================================

def section1_classification():
    """Classification models and techniques"""
    
    print("\n" + "="*60)
    print("🎯 SEZIONE 1: CLASSIFICATION")
    print("="*60)
    
    # 1.1 DATA PREPARATION
    print("\n📊 1.1 DATA PREPARATION")
    print("-"*40)
    
    # Generate classification dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    X = np.random.randn(n_samples, 10)
    
    # Create target with some pattern
    y = (X[:, 0] + X[:, 1] * 0.5 - X[:, 2] * 0.3 + 
         np.random.randn(n_samples) * 0.5) > 0
    y = y.astype(int)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    print(f"\nFeature statistics:\n{df[feature_names].describe().T[['mean', 'std', 'min', 'max']]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 1.2 FEATURE SCALING
    print("\n⚡ 1.2 FEATURE SCALING")
    print("-"*40)
    
    # StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Before scaling:")
    print(f"  Train mean: {X_train[:, 0].mean():.3f}, std: {X_train[:, 0].std():.3f}")
    print("After scaling:")
    print(f"  Train mean: {X_train_scaled[:, 0].mean():.3f}, std: {X_train_scaled[:, 0].std():.3f}")
    
    # 1.3 MODEL TRAINING
    print("\n🤖 1.3 MODEL TRAINING")
    print("-"*40)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        # Train
        if name in ['SVM', 'Logistic Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
        print(f"  AUC-ROC:   {auc:.3f}")
    
    # 1.4 BEST MODEL ANALYSIS
    print("\n🏆 1.4 BEST MODEL ANALYSIS")
    print("-"*40)
    
    # Find best model by F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = models[best_model_name]
    
    print(f"Best model: {best_model_name}")
    print(f"F1 Score: {results[best_model_name]['f1']:.3f}")
    
    # Confusion Matrix
    if best_model_name in ['SVM', 'Logistic Regression']:
        y_pred_best = best_model.predict(X_test_scaled)
    else:
        y_pred_best = best_model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:5]
        
        print(f"\nTop 5 Important Features:")
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")
    
    # 1.5 CROSS-VALIDATION
    print("\n📏 1.5 CROSS-VALIDATION")
    print("-"*40)
    
    # Perform 5-fold cross-validation
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='f1')
    
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# ==============================================================================
# SEZIONE 2: SUPERVISED LEARNING - REGRESSION
# ==============================================================================

def section2_regression():
    """Regression models and techniques"""
    
    print("\n" + "="*60)
    print("📈 SEZIONE 2: REGRESSION")
    print("="*60)
    
    # 2.1 GENERATE REGRESSION DATA
    print("\n📊 2.1 REGRESSION DATA")
    print("-"*40)
    
    # Generate non-linear regression data
    np.random.seed(42)
    n_samples = 500
    
    X_reg = np.random.uniform(-3, 3, (n_samples, 5))
    y_reg = (X_reg[:, 0]**2 + 
             2 * X_reg[:, 1] - 
             X_reg[:, 2] + 
             0.5 * X_reg[:, 3] * X_reg[:, 4] + 
             np.random.randn(n_samples) * 2)
    
    # Add polynomial features for some columns
    X_reg = np.column_stack([
        X_reg,
        X_reg[:, 0]**2,  # Square of first feature
        X_reg[:, 1]**2,  # Square of second feature
        X_reg[:, 0] * X_reg[:, 1]  # Interaction term
    ])
    
    print(f"Features shape: {X_reg.shape}")
    print(f"Target stats: mean={y_reg.mean():.2f}, std={y_reg.std():.2f}")
    
    # Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # 2.2 TRAIN REGRESSION MODELS
    print("\n🤖 2.2 REGRESSION MODELS")
    print("-"*40)
    
    # Scale features
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)
    
    # Initialize regression models
    reg_models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0)
    }
    
    # Train and evaluate
    reg_results = {}
    
    for name, model in reg_models.items():
        if name == 'Gradient Boosting':
            continue  # Skip GB for regression in this example
            
        # Train
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVR']:
            model.fit(X_train_reg_scaled, y_train_reg)
            y_pred = model.predict(X_test_reg_scaled)
        else:
            model.fit(X_train_reg, y_train_reg)
            y_pred = model.predict(X_test_reg)
        
        # Evaluate
        mse = mean_squared_error(y_test_reg, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_reg, y_pred)
        
        reg_results[name] = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"\n{name}:")
        print(f"  MSE:  {mse:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²:   {r2:.3f}")
    
    # 2.3 RESIDUAL ANALYSIS
    print("\n📊 2.3 RESIDUAL ANALYSIS")
    print("-"*40)
    
    # Use best model for residual analysis
    best_reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    best_reg_model.fit(X_train_reg, y_train_reg)
    y_pred_best = best_reg_model.predict(X_test_reg)
    
    residuals = y_test_reg - y_pred_best
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred_best, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Residuals statistics:")
    print(f"  Mean: {residuals.mean():.3f}")
    print(f"  Std:  {residuals.std():.3f}")
    print(f"  Skewness: {stats.skew(residuals):.3f}")
    
    # 2.4 HYPERPARAMETER TUNING
    print("\n🎯 2.4 HYPERPARAMETER TUNING")
    print("-"*40)
    
    # Grid search for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf_reg, param_grid, cv=3, scoring='r2', n_jobs=-1
    )
    
    print("Running Grid Search...")
    grid_search.fit(X_train_reg, y_train_reg)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.3f}")
    
    # Test best model
    best_model_tuned = grid_search.best_estimator_
    y_pred_tuned = best_model_tuned.predict(X_test_reg)
    r2_tuned = r2_score(y_test_reg, y_pred_tuned)
    
    print(f"Test R² with best parameters: {r2_tuned:.3f}")

# ==============================================================================
# SEZIONE 3: UNSUPERVISED LEARNING
# ==============================================================================

def section3_unsupervised():
    """Clustering and dimensionality reduction"""
    
    print("\n" + "="*60)
    print("🔍 SEZIONE 3: UNSUPERVISED LEARNING")
    print("="*60)
    
    # 3.1 GENERATE CLUSTERING DATA
    print("\n📊 3.1 CLUSTERING DATA")
    print("-"*40)
    
    # Generate clustered data
    from sklearn.datasets import make_blobs, make_moons
    
    # Blobs for K-Means
    X_blobs, y_blobs = make_blobs(
        n_samples=300, centers=4, n_features=2,
        random_state=42, cluster_std=1.0
    )
    
    # Moons for DBSCAN
    X_moons, y_moons = make_moons(
        n_samples=300, noise=0.1, random_state=42
    )
    
    print(f"Blobs dataset: {X_blobs.shape}")
    print(f"Moons dataset: {X_moons.shape}")
    
    # 3.2 K-MEANS CLUSTERING
    print("\n🎯 3.2 K-MEANS CLUSTERING")
    print("-"*40)
    
    # Find optimal k using elbow method
    inertias = []
    silhouette_scores = []
    K_range = range(2, 10)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_blobs)
        inertias.append(kmeans.inertia_)
        
        from sklearn.metrics import silhouette_score
        score = silhouette_score(X_blobs, kmeans.labels_)
        silhouette_scores.append(score)
    
    # Plot elbow curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(K_range, inertias, 'bo-')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(K_range, silhouette_scores, 'ro-')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Apply K-Means with optimal k
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_blobs)
    
    print(f"\nK-Means with k={optimal_k}:")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    print(f"  Silhouette Score: {silhouette_score(X_blobs, kmeans_labels):.3f}")
    
    # 3.3 DBSCAN CLUSTERING
    print("\n🌙 3.3 DBSCAN CLUSTERING")
    print("-"*40)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_moons)
    
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    print(f"DBSCAN Results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")
    
    # Visualize clustering results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # K-Means visualization
    scatter1 = axes[0].scatter(X_blobs[:, 0], X_blobs[:, 1], 
                              c=kmeans_labels, cmap='viridis', alpha=0.6)
    axes[0].scatter(kmeans.cluster_centers_[:, 0], 
                   kmeans.cluster_centers_[:, 1],
                   c='red', marker='x', s=200, linewidths=3)
    axes[0].set_title('K-Means Clustering')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    plt.colorbar(scatter1, ax=axes[0])
    
    # DBSCAN visualization
    scatter2 = axes[1].scatter(X_moons[:, 0], X_moons[:, 1], 
                              c=dbscan_labels, cmap='viridis', alpha=0.6)
    axes[1].set_title('DBSCAN Clustering')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
    
    # 3.4 DIMENSIONALITY REDUCTION - PCA
    print("\n📉 3.4 PCA DIMENSIONALITY REDUCTION")
    print("-"*40)
    
    # Generate high-dimensional data
    np.random.seed(42)
    X_high = np.random.randn(500, 50)
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_high)
    
    # Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    print(f"Original dimensions: {X_high.shape[1]}")
    print(f"Components for 95% variance: {n_components_95}")
    print(f"Dimension reduction: {(1 - n_components_95/X_high.shape[1])*100:.1f}%")
    
    # Plot explained variance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(range(1, 21), explained_variance_ratio[:20])
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('PCA - Individual Explained Variance')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(range(1, 21), cumulative_variance[:20], 'bo-')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('PCA - Cumulative Explained Variance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==============================================================================
# SEZIONE 4: MODEL EVALUATION & VALIDATION
# ==============================================================================

def section4_model_evaluation():
    """Advanced model evaluation techniques"""
    
    print("\n" + "="*60)
    print("📊 SEZIONE 4: MODEL EVALUATION & VALIDATION")
    print("="*60)
    
    # 4.1 TRAIN-VALIDATION-TEST SPLIT
    print("\n📂 4.1 TRAIN-VALIDATION-TEST SPLIT")
    print("-"*40)
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(1000) * 0.5) > 0
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # 4.2 LEARNING CURVES
    print("\n📈 4.2 LEARNING CURVES")
    print("-"*40)
    
    from sklearn.model_selection import learning_curve
    
    # Generate learning curves
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', n_jobs=-1
    )
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'b-',
            label='Training score', marker='o')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'r-',
            label='Validation score', marker='s')
    
    # Add confidence bands
    plt.fill_between(train_sizes,
                    np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                    np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                    alpha=0.1, color='b')
    
    plt.fill_between(train_sizes,
                    np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                    np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                    alpha=0.1, color='r')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curves - Random Forest')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Diagnose bias/variance
    final_train_score = np.mean(train_scores[-1])
    final_val_score = np.mean(val_scores[-1])
    gap = final_train_score - final_val_score
    
    print(f"Final training score: {final_train_score:.3f}")
    print(f"Final validation score: {final_val_score:.3f}")
    print(f"Gap: {gap:.3f}")
    
    if gap > 0.1:
        print("→ High variance (overfitting) detected")
    elif final_val_score < 0.8:
        print("→ High bias (underfitting) detected")
    else:
        print("→ Good balance between bias and variance")
    
    # 4.3 ROC CURVES & AUC
    print("\n📉 4.3 ROC CURVES & AUC")
    print("-"*40)
    
    from sklearn.metrics import roc_curve, auc
    
    # Train multiple models
    models_roc = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models_roc.items():
        # Scale for SVM and LR
        if name in ['SVM', 'Logistic Regression']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            model.fit(X_train_scaled, y_train)
            y_proba = model.predict_proba(X_val_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_val, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, linewidth=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 4.4 FEATURE SELECTION
    print("\n🎯 4.4 FEATURE SELECTION")
    print("-"*40)
    
    # Select K best features
    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    # Get feature scores
    scores = selector.scores_
    indices = np.argsort(scores)[::-1]
    
    print("Top 10 features by ANOVA F-value:")
    for i in range(10):
        print(f"  {i+1}. Feature {indices[i]}: {scores[indices[i]]:.2f}")
    
    # Compare performance
    model_all = RandomForestClassifier(n_estimators=100, random_state=42)
    model_selected = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model_all.fit(X_train, y_train)
    model_selected.fit(X_train_selected, y_train)
    
    X_val_selected = selector.transform(X_val)
    
    score_all = model_all.score(X_val, y_val)
    score_selected = model_selected.score(X_val_selected, y_val)
    
    print(f"\nAccuracy with all features: {score_all:.3f}")
    print(f"Accuracy with selected features: {score_selected:.3f}")
    print(f"Dimension reduction: {(1 - 10/X_train.shape[1])*100:.1f}%")

# ==============================================================================
# SEZIONE 5: ML PIPELINE & DEPLOYMENT
# ==============================================================================

def section5_ml_pipeline():
    """Complete ML pipeline example"""
    
    print("\n" + "="*60)
    print("🚀 SEZIONE 5: ML PIPELINE & DEPLOYMENT")
    print("="*60)
    
    # 5.1 CREATE COMPLETE PIPELINE
    print("\n🔧 5.1 COMPLETE ML PIPELINE")
    print("-"*40)
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(1000) * 0.5) > 0
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.ensemble import RandomForestClassifier
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=10)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"Pipeline Performance:")
    print(f"  Train accuracy: {train_score:.3f}")
    print(f"  Test accuracy: {test_score:.3f}")
    
    # 5.2 HYPERPARAMETER OPTIMIZATION
    print("\n⚙️ 5.2 PIPELINE HYPERPARAMETER TUNING")
    print("-"*40)
    
    # Define parameter grid for pipeline
    param_grid = {
        'selector__k': [5, 10, 15],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, None]
    }
    
    # Grid search
    grid = GridSearchCV(
        pipeline, param_grid, cv=5, 
        scoring='accuracy', n_jobs=-1
    )
    
    print("Running grid search on pipeline...")
    grid.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.3f}")
    print(f"Test score: {grid.score(X_test, y_test):.3f}")
    
    # 5.3 MODEL PERSISTENCE
    print("\n💾 5.3 MODEL PERSISTENCE")
    print("-"*40)
    
    import pickle
    import joblib
    
    # Save model with pickle
    with open('model_pipeline.pkl', 'wb') as f:
        pickle.dump(grid.best_estimator_, f)
    
    # Save with joblib (better for large numpy arrays)
    joblib.dump(grid.best_estimator_, 'model_pipeline.joblib')
    
    # Load and test
    loaded_model = joblib.load('model_pipeline.joblib')
    loaded_score = loaded_model.score(X_test, y_test)
    
    print(f"Model saved successfully!")
    print(f"Loaded model test score: {loaded_score:.3f}")
    
    # 5.4 DEPLOYMENT READINESS
    print("\n🚀 5.4 DEPLOYMENT READINESS")
    print("-"*40)
    
    def predict_single_sample(model, sample):
        """Function to predict single sample"""
        # Ensure 2D array
        sample = np.array(sample).reshape(1, -1)
        
        # Get prediction and probability
        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0]
        
        return {
            'prediction': int(prediction),
            'probability_class_0': float(probability[0]),
            'probability_class_1': float(probability[1]),
            'confidence': float(max(probability))
        }
    
    # Test single prediction
    test_sample = X_test[0]
    result = predict_single_sample(loaded_model, test_sample)
    
    print(f"Single sample prediction:")
    print(f"  Input shape: {test_sample.shape}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Probabilities: [{result['probability_class_0']:.3f}, "
          f"{result['probability_class_1']:.3f}]")
    
    # 5.5 PRODUCTION METRICS
    print("\n📊 5.5 PRODUCTION METRICS")
    print("-"*40)
    
    # Simulate production monitoring
    def calculate_production_metrics(model, X_new, y_true=None):
        """Calculate metrics for production monitoring"""
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new)
        
        metrics = {
            'n_samples': len(X_new),
            'prediction_distribution': dict(zip(*np.unique(predictions, return_counts=True))),
            'mean_confidence': np.mean(np.max(probabilities, axis=1)),
            'low_confidence_ratio': np.mean(np.max(probabilities, axis=1) < 0.6)
        }
        
        if y_true is not None:
            metrics['accuracy'] = accuracy_score(y_true, predictions)
            metrics['precision'] = precision_score(y_true, predictions)
            metrics['recall'] = recall_score(y_true, predictions)
            metrics['f1'] = f1_score(y_true, predictions)
        
        return metrics
    
    # Calculate metrics
    prod_metrics = calculate_production_metrics(loaded_model, X_test, y_test)
    
    print("Production Metrics:")
    for key, value in prod_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Clean up
    import os
    os.remove('model_pipeline.pkl')
    os.remove('model_pipeline.joblib')
    
    print("\n✅ Pipeline ready for deployment!")

# ==============================================================================
# MAIN - Menu per le sezioni
# ==============================================================================

def main():
    """Menu principale per Machine Learning"""
    
    print("\n" + "="*60)
    print("🤖 MACHINE LEARNING - SCEGLI SEZIONE")
    print("="*60)
    
    sections = [
        ("Classification", section1_classification),
        ("Regression", section2_regression),
        ("Unsupervised Learning", section3_unsupervised),
        ("Model Evaluation", section4_model_evaluation),
        ("ML Pipeline & Deployment", section5_ml_pipeline)
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
    print("✅ PARTE 3 COMPLETATA!")
    print("Prossimo: session3_part4_projects.py")
    print("="*60)

if __name__ == "__main__":
    main()
