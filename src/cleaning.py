import pandas as pd


def clean_raw_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sanitizes raw transaction data based on EDA rules.
    
    Returns:
        tuple: (df_clean, canceled_df)
    """
    df_clean = df.copy()

    # Drop records missing Customer ID
    df_clean = df_clean.dropna(subset=["Customer ID"])
    
    # drop duplicates
    df_clean = df_clean.drop_duplicates()

    # Strip whitespace safely from string columns
    str_cols = df_clean.select_dtypes(include="object").columns
    for col in str_cols:
        df_clean[col] = df_clean[col].str.strip()

    # Type casting
    df_clean["Customer ID"] = df_clean["Customer ID"].astype("Int64").astype(str)
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])

    # Extract Canceled Invoices BEFORE filtering negative quantities
    is_canceled_mask = df_clean["Invoice"].astype(str).str.startswith("C", na=False)
    canceled_df = df_clean[is_canceled_mask].copy()

    # Filter invalid price/quantity entries/non-canceled
    df_clean = df_clean[(~is_canceled_mask) & (df_clean["Price"] > 0) & (df_clean["Quantity"] > 0)].copy()

    return df_clean, canceled_df