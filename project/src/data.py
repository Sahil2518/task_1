import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import CONFIG

def generate_placemux_data(n_samples=1000, seed=42):
    np.random.seed(seed)
    
    # Generate mock dates
    base_date = pd.to_datetime('2026-01-01')
    registration_dates = base_date + pd.to_timedelta(np.random.randint(0, 180, n_samples), unit='d')
    last_login_dates = registration_dates + pd.to_timedelta(np.random.randint(1, 60, n_samples), unit='d')
    
    # Generate domain features
    domain_score = np.random.normal(65, 15, n_samples).clip(0, 100)
    aptitude_score = np.random.normal(70, 12, n_samples).clip(0, 100)
    projects_completed = np.random.poisson(2, n_samples)
    
    # Target variable: placed (1 or 0)
    # Probability depends on scores and projects
    placement_prob = 1 / (1 + np.exp(-(-5 + 0.05 * domain_score + 0.04 * aptitude_score + 0.5 * projects_completed)))
    placed = np.random.binomial(1, placement_prob)
    
    # Generate LEAKY features
    # has_offer_letter: Almost perfectly correlates with placed
    has_offer_letter = np.where(placed == 1, np.random.binomial(1, 0.95, n_samples), np.random.binomial(1, 0.01, n_samples))
    # days_to_placement: Only exists if placed
    days_to_placement = np.where(placed == 1, np.random.randint(10, 90, n_samples), np.nan)
    
    # Useless feature
    random_noise = np.random.rand(n_samples)
    
    df = pd.DataFrame({
        'registration_date': registration_dates,
        'last_login_date': last_login_dates,
        'domain_score': domain_score,
        'aptitude_score': aptitude_score,
        'projects_completed': projects_completed,
        'has_offer_letter': has_offer_letter,
        'days_to_placement': days_to_placement,
        'random_noise': random_noise,
        'placed': placed
    })
    
    return df

def feature_engineering(df, prune_leaks=True):
    """
    1. Re-confirm target: 'placed'
    2. Derive features from domain reasoning:
       - total_score = domain_score + aptitude_score
    3. Add aggregate/time-based features:
       - active_days = last_login_date - registration_date
       - registration_month
    4. Prune useless/leaky features.
    """
    df = df.copy()
    
    # 3. Time-based features
    df['active_days'] = (df['last_login_date'] - df['registration_date']).dt.days
    df['registration_month'] = df['registration_date'].dt.month
    
    # 2. Domain reasoning
    df['total_score'] = df['domain_score'] + df['aptitude_score']
    
    # 4. Prune leaks and useless columns
    if prune_leaks:
        cols_to_drop = [
            'registration_date', 'last_login_date', # Datetimes can't be fed directly
            'has_offer_letter',                     # LEAK: 95% correlation with target
            'days_to_placement',                    # LEAK: NaN when not placed
            'random_noise'                          # USELESS
        ]
        df = df.drop(columns=cols_to_drop)
        
    return df

def load_data():
    df_raw = generate_placemux_data(n_samples=1500, seed=CONFIG["seed"])
    
    # Lock baseline features by pruning leaks
    df_engineered = feature_engineering(df_raw, prune_leaks=True)
    
    X = df_engineered.drop(columns=['placed'])
    y = df_engineered['placed']

    # First split: separate out the test set (held out, untouched)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["seed"],
        stratify=y
    )

    # Second split: carve a validation set from the remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=CONFIG["val_size"],
        random_state=CONFIG["seed"],
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test