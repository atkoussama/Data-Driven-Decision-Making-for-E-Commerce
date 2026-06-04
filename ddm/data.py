import pandas as pd


def load_sales(path: str) -> pd.DataFrame:
    """Load the main ecommerce sales file and normalise common column names."""
    df = pd.read_excel(path)

    # normalize common datetime column names
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])
    elif 'InvoiceDate' in df.columns:
        df['order_date'] = pd.to_datetime(df['InvoiceDate'])
    elif 'InvoiceDate' in df.columns:
        df['order_date'] = pd.to_datetime(df['InvoiceDate'])

    return df


def load_retail(path: str) -> pd.DataFrame:
    """Load the legacy Online Retail file and return an aggregated country-level table."""
    col_names = ['InvoiceNo', 'StockCode', 'Description', 'Quantity',
                 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
    df2 = pd.read_excel(path, header=None, names=col_names)
    
    # Convert numeric columns safely
    df2['Quantity'] = pd.to_numeric(df2['Quantity'], errors='coerce')
    df2['UnitPrice'] = pd.to_numeric(df2['UnitPrice'], errors='coerce')
    
    # Drop rows with NaN values in critical columns
    df2 = df2.dropna(subset=['Quantity', 'UnitPrice'])
    
    df2 = df2[(df2['Quantity'] > 0) & (df2['UnitPrice'] > 0)]
    df2['TotalRevenue'] = df2['Quantity'] * df2['UnitPrice']

    retail_agg = df2.groupby('Country').agg(
        avg_unit_price=('UnitPrice', 'mean'),
        avg_basket_size=('Quantity', 'mean'),
        median_revenue_order=('TotalRevenue', 'median'),
        nb_unique_products=('StockCode', 'nunique'),
        total_transactions=('InvoiceNo', 'nunique')
    ).reset_index()

    return retail_agg
