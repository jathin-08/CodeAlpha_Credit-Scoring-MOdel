import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    df = pd.read_csv(file_path)
    print(f"Data successfully loaded. Shape: {df.shape}")
    return df

def clean_data(df):
    """
    Clean the dataset:
    - Handle outliers (e.g., person_age > 100, person_emp_length > 60)
    - Impute missing values:
        - person_emp_length: filled with overall median
        - loan_int_rate: filled with median interest rate corresponding to its loan_grade
    """
    df_clean = df.copy()
    
    # 1. Outlier removal / capping
    # Credit risk datasets sometimes contain erroneous values (e.g. age > 100 or emp_length > 60)
    original_len = len(df_clean)
    df_clean = df_clean[df_clean['person_age'] <= 90]
    df_clean = df_clean[df_clean['person_emp_length'].isna() | (df_clean['person_emp_length'] <= 50)]
    
    removed = original_len - len(df_clean)
    if removed > 0:
        print(f"Removed {removed} rows due to extreme age (>90) or employment length (>50) outliers.")
        
    # 2. Imputation of missing values
    # Impute employment length with median
    emp_length_median = df_clean['person_emp_length'].median()
    df_clean['person_emp_length'] = df_clean['person_emp_length'].fillna(emp_length_median)
    print(f"Imputed missing person_emp_length with median: {emp_length_median}")
    
    # Impute interest rate with the median of its loan grade
    if 'loan_grade' in df_clean.columns and 'loan_int_rate' in df_clean.columns:
        grade_medians = df_clean.groupby('loan_grade')['loan_int_rate'].transform('median')
        df_clean['loan_int_rate'] = df_clean['loan_int_rate'].fillna(grade_medians)
        
        # In case some grades are still NaN (unlikely), fill with overall median
        overall_rate_median = df_clean['loan_int_rate'].median()
        df_clean['loan_int_rate'] = df_clean['loan_int_rate'].fillna(overall_rate_median)
        print("Imputed missing loan_int_rate using median values per loan grade.")
        
    return df_clean

def split_data(df, target_col='loan_status', test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test
