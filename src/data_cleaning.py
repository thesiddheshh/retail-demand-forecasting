def clean(df):
    """
    Perform data cleaning operations.
    """
    # Drop rows with missing values
    df = df.dropna()
    # Reset index after dropping rows
    df = df.reset_index(drop=True)
    return df
