import pandas as pd
import numpy as np


def build_behavioral_features(df_clean: pd.DataFrame, canceled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data into customer-level behavioral features.

    Returns:
        pd.DataFrame
            One row per customer containing behavioral features.
    """
    df = df_clean.copy()

    # compute Total Price
    df['LineTotal'] = df['Price'] * df['Quantity']

    
    weekend_invoices = (
        df.loc[df["InvoiceDate"].dt.dayofweek >= 5]
        .groupby("Customer ID")["Invoice"]
        .nunique()
        .rename("WeekendInvoices")
    )


    # Compute Core Features
    customer_features = df.groupby("Customer ID").agg(
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
        Monetary=("LineTotal", "sum"),
        Frequency=("Invoice", "nunique"),
        TotalItemsPurchased=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique")
    ).reset_index()


    # Merge Weekend Invoice Counts
    customer_features = customer_features.merge(weekend_invoices, on='Customer ID', how='left')
    customer_features['WeekendInvoices'] = customer_features['WeekendInvoices'].fillna(0)


    # Compute Derived Time
    max_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    customer_features["Recency"] = (max_date - customer_features["LastPurchase"]).dt.total_seconds() / 86400.0
    customer_features["CustomerLifetime"] = (customer_features["LastPurchase"] - customer_features["FirstPurchase"]).dt.total_seconds() / 86400.0
    customer_features["AverageOrderValue"] = customer_features["Monetary"] / (customer_features["Frequency"])
    customer_features["AverageItemsPerOrder"] = customer_features["TotalItemsPurchased"] / customer_features["Frequency"]
    customer_features["PurchaseRate"] = (customer_features["Frequency"] - 1) / (customer_features["CustomerLifetime"] + 1.0)
    customer_features["WeekendPurchaseRatio"] = customer_features["WeekendInvoices"] / customer_features["Frequency"]


    # Compute Cancellation Rate
    clean_invoices = df.groupby("Customer ID")["Invoice"].nunique().rename("ValidInvoices")
    canceled_invoices = canceled_df.groupby("Customer ID")["Invoice"].nunique().rename("CanceledInvoices")
    cancellation_df = pd.concat([clean_invoices, canceled_invoices], axis=1).fillna(0).rename_axis("Customer ID").reset_index()

    cancellation_df["TotalInvoices"] = cancellation_df["ValidInvoices"] + cancellation_df["CanceledInvoices"]
    cancellation_df['CancellationRate'] = cancellation_df['CanceledInvoices'] / cancellation_df["TotalInvoices"]


    # Merge Cancellation Rate
    customer_features = customer_features.merge(
        cancellation_df[["Customer ID", "CancellationRate"]],
        on="Customer ID",
        how="left"
    )
    customer_features['CancellationRate'] = customer_features['CancellationRate'].fillna(0.0)


    customer_features = customer_features.drop(
        columns=['FirstPurchase', 'LastPurchase', 'WeekendInvoices']
    )

    return customer_features.reset_index(drop=True)