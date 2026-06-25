import os
import argparse
import numpy as np
import pandas as pd

from src.data_preprocessing import load_data, clean_data, split_data
from src.feature_engineering import fit_transform_features, transform_features
from src.model_training import train_logistic_regression, train_random_forest, save_model
from src.model_evaluation import evaluate_model

# Function to generate sample data if not exists
def generate_sample_data(output_path, num_samples=2000, random_seed=42):
    np.random.seed(random_seed)
    
    person_age = np.random.randint(20, 70, size=num_samples)
    person_income = np.random.lognormal(mean=10.8, sigma=0.5, size=num_samples).astype(int)
    person_income = np.clip(person_income, 8000, 300000)
    
    person_emp_length = []
    for age in person_age:
        max_emp = min(age - 18, 40)
        if max_emp <= 0:
            emp = 0.0
        else:
            emp = float(np.random.randint(0, max_emp))
        if np.random.rand() < 0.05:
            person_emp_length.append(np.nan)
        else:
            person_emp_length.append(emp)
    person_emp_length = np.array(person_emp_length)
    
    home_options = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
    person_home_ownership = np.random.choice(home_options, size=num_samples, p=[0.5, 0.1, 0.38, 0.02])
    
    intent_options = ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
    loan_intent = np.random.choice(intent_options, size=num_samples, p=[0.18, 0.20, 0.18, 0.18, 0.12, 0.14])
    
    grade_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    loan_grade = np.random.choice(grade_options, size=num_samples, p=[0.32, 0.30, 0.20, 0.11, 0.05, 0.015, 0.005])
    
    loan_amnt = []
    for inc in person_income:
        max_loan = min(35000, int(inc * 0.4))
        max_loan = max(1000, max_loan)
        loan_amnt.append(np.random.randint(1000, max_loan))
    loan_amnt = np.array(loan_amnt)
    
    grade_int_rates = {
        'A': (5.0, 7.5), 'B': (7.5, 11.0), 'C': (11.0, 14.0),
        'D': (14.0, 17.0), 'E': (17.0, 20.0), 'F': (20.0, 22.0), 'G': (22.0, 25.0)
    }
    loan_int_rate = []
    for grade in loan_grade:
        low, high = grade_int_rates[grade]
        rate = np.random.uniform(low, high)
        if np.random.rand() < 0.08:
            loan_int_rate.append(np.nan)
        else:
            loan_int_rate.append(round(rate, 2))
    loan_int_rate = np.array(loan_int_rate)
    
    cb_person_default_on_file = np.random.choice(['Y', 'N'], size=num_samples, p=[0.18, 0.82])
    
    cb_person_cred_hist_length = []
    for age in person_age:
        max_hist = age - 18
        if max_hist <= 2:
            hist = 2
        else:
            hist = np.random.randint(2, max_hist)
        cb_person_cred_hist_length.append(hist)
    cb_person_cred_hist_length = np.array(cb_person_cred_hist_length)
    
    norm_income = person_income / 100000.0
    loan_to_income = loan_amnt / person_income
    
    log_odds = -1.6
    log_odds += loan_to_income * 3.0
    log_odds -= np.nan_to_num(person_emp_length, nan=2.0) * 0.05
    log_odds += (loan_grade == 'D').astype(int) * 0.5
    log_odds += (loan_grade == 'E').astype(int) * 1.0
    log_odds += (loan_grade == 'F').astype(int) * 1.5
    log_odds += (loan_grade == 'G').astype(int) * 2.0
    log_odds += (person_home_ownership == 'RENT').astype(int) * 0.4
    log_odds += (cb_person_default_on_file == 'Y').astype(int) * 0.8
    log_odds += (loan_intent == 'DEBTCONSOLIDATION').astype(int) * 0.3
    log_odds += (loan_intent == 'MEDICAL').astype(int) * 0.2
    log_odds -= norm_income * 0.3
    
    prob = 1 / (1 + np.exp(-log_odds))
    loan_status = np.random.binomial(1, prob)
    
    df = pd.DataFrame({
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_status': loan_status,
        'loan_percent_income': np.round(loan_to_income, 2),
        'cb_person_default_on_file': cb_person_default_on_file,
        'cb_person_cred_hist_length': cb_person_cred_hist_length
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample credit data generated at {output_path}")

def run_pipeline():
    parser = argparse.ArgumentParser(description="Credit Scoring Model Pipeline")
    parser.add_argument('--model', type=str, choices=['rf', 'lr'], default='rf',
                        help="Model algorithm: 'rf' for Random Forest or 'lr' for Logistic Regression")
    parser.add_argument('--test-size', type=float, default=0.2,
                        help="Test set size fraction (default: 0.2)")
    parser.add_argument('--random-state', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--cv', type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument('--data-path', type=str, default='data/sample_credit_data.csv',
                        help="Path to CSV dataset")
    parser.add_argument('--output-dir', type=str, default='output',
                        help="Directory to save artifacts and plots")
    
    args = parser.parse_args()
    
    # 1. Ensure sample data exists
    if not os.path.exists(args.data_path):
        print(f"Dataset {args.data_path} not found. Generating a synthetic one...")
        generate_sample_data(args.data_path, num_samples=2500, random_seed=args.random_state)
        
    # 2. Load data
    df = load_data(args.data_path)
    
    # 3. Clean data
    df_cleaned = clean_data(df)
    
    # 4. Split data
    X_train, X_test, y_train, y_test = split_data(
        df_cleaned, target_col='loan_status', test_size=args.test_size, random_state=args.random_state
    )
    
    # Define categorical and numerical columns for feature engineering
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    numerical_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
                      'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
    
    # 5. Feature Engineering
    preprocessor_path = os.path.join(args.output_dir, 'feature_preprocessor.joblib')
    X_train_processed, preprocessor = fit_transform_features(
        X_train, categorical_cols, numerical_cols, save_path=preprocessor_path
    )
    X_test_processed = transform_features(X_test, preprocessor, categorical_cols, numerical_cols)
    
    # 6. Train model
    model_path = os.path.join(args.output_dir, 'trained_model.joblib')
    if args.model == 'rf':
        model = train_random_forest(X_train_processed, y_train, cv=args.cv, random_state=args.random_state)
    else:
        model = train_logistic_regression(X_train_processed, y_train, cv=args.cv, random_state=args.random_state)
        
    # 7. Save model
    save_model(model, model_path)
    
    # 8. Evaluate model
    metrics = evaluate_model(model, X_test_processed, y_test, args.output_dir)
    print("\nPipeline execution complete! All output metrics and charts are saved under: ", args.output_dir)

if __name__ == '__main__':
    run_pipeline()
