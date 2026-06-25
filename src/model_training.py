import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_logistic_regression(X_train, y_train, cv=5, random_state=42):
    """
    Train a Logistic Regression model using GridSearchCV.
    """
    print("Training Logistic Regression model with GridSearchCV...")
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l2']  # Using l2 as it is supported by default solvers
    }
    
    grid_search = GridSearchCV(lr, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_random_forest(X_train, y_train, cv=5, random_state=42):
    """
    Train a Random Forest model using GridSearchCV.
    """
    print("Training Random Forest model with GridSearchCV...")
    rf = RandomForestClassifier(random_state=random_state)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def save_model(model, save_path):
    """
    Save the trained model to disk.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Model saved successfully to {save_path}")
