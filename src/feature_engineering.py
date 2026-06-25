import os
# pyrefly: ignore [missing-import]
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(categorical_cols, numerical_cols):
    """
    Constructs a ColumnTransformer to perform scaling on numerical features 
    and one-hot encoding on categorical features.
    """
    # handle_unknown='ignore' ensures new categories in test set don't break the model.
    # We use sparse_output=False (or sparse=False for compatibility) to get dense array back.
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor

def fit_transform_features(X_train, categorical_cols, numerical_cols, save_path=None):
    """
    Fits preprocessor on X_train and transforms it. Optionally saves the preprocessor.
    """
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    
    print("Fitting and transforming feature preprocessor...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Get feature names after transformation
    ohe = preprocessor.named_transformers_['cat']
    encoded_cat_cols = list(ohe.get_feature_names_out(categorical_cols))
    all_feature_names = numerical_cols + encoded_cat_cols
    
    X_train_df = pd.DataFrame(X_train_transformed, columns=all_feature_names, index=X_train.index)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(preprocessor, save_path)
        print(f"Saved feature preprocessor to {save_path}")
        
    return X_train_df, preprocessor

def transform_features(X, preprocessor, categorical_cols, numerical_cols):
    """
    Transforms new feature data using a fitted preprocessor.
    """
    X_transformed = preprocessor.transform(X)
    
    ohe = preprocessor.named_transformers_['cat']
    encoded_cat_cols = list(ohe.get_feature_names_out(categorical_cols))
    all_feature_names = numerical_cols + encoded_cat_cols
    
    X_df = pd.DataFrame(X_transformed, columns=all_feature_names, index=X.index)
    return X_df
