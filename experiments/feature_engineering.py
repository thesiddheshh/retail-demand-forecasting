import pandas as pd
import numpy as np

def add_date_features(df):
    """
    Add date-related features to the dataset.
    """
    # Ensure 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract various date components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = np.where(df['weekday'] >= 5, 1, 0)  # 1 for Saturday and Sunday, 0 otherwise
    
    return df


def aggregate_weekly_sales(df):
    """
    Aggregate daily sales data to weekly sales.
    """
    df['week'] = df['date'].dt.isocalendar().week
    weekly_sales = df.groupby(['year', 'week'])['monthly_sales'].sum().reset_index()
    return weekly_sales


def add_lag_features(df, lags=[1, 2, 3, 4, 5, 6, 12]):
    """
    Add lag features to the dataset.
    """
    for lag in lags:
        df[f'lag_{lag}'] = df['monthly_sales'].shift(lag)
    
    return df


def add_rolling_features(df, window_sizes=[3, 6, 12]):
    """
    Add rolling mean and rolling standard deviation features.
    """
    for window in window_sizes:
        df[f'rolling_mean_{window}'] = df['monthly_sales'].rolling(window).mean()
        df[f'rolling_std_{window}'] = df['monthly_sales'].rolling(window).std()
    
    return df


def add_expanding_features(df):
    """
    Add expanding mean and expanding standard deviation features.
    """
    df['expanding_mean'] = df['monthly_sales'].expanding().mean()
    df['expanding_std'] = df['monthly_sales'].expanding().std()
    
    return df


def feature_engineering(df):
    """
    Perform all feature engineering steps on the data.
    """
    # Step 1: Add date features
    df = add_date_features(df)
    
    # Step 2: Add lag features
    df = add_lag_features(df)
    
    # Step 3: Add rolling window features
    df = add_rolling_features(df)
    
    # Step 4: Add expanding window features
    df = add_expanding_features(df)
    
    return df
