import pandas as pd
from sklearn.preprocessing import LabelEncoder


def engineer_features(df: pd.DataFrame, retail_agg: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to the sales dataframe and merge retail aggregates."""
    # Merge retail aggregates by country; attempt to fill country from region mapping
    region_map = {
        'North': 'United Kingdom', 'South': 'Germany',
        'East': 'France', 'West': 'Australia',
        'Central': 'EIRE'
    }
    if 'region' in df.columns:
        df['Country'] = df['region'].map(region_map).fillna('United Kingdom')

    if retail_agg is not None and 'Country' in df.columns:
        df = df.merge(retail_agg, on='Country', how='left')

    # Date/time features
    if 'order_date' in df.columns:
        df['order_dayofweek'] = df['order_date'].dt.dayofweek
        df['order_month'] = df['order_date'].dt.month
        df['order_quarter'] = df['order_date'].dt.quarter
        df['is_weekend'] = (df['order_dayofweek'] >= 5).astype(int)

    # Numeric transformations
    if 'price' in df.columns and 'discount' in df.columns:
        df['net_price'] = df['price'] * (1 - df['discount'])
        df['discount_amount'] = df['price'] * df['discount']

    if 'shipping_cost' in df.columns and 'total_amount' in df.columns:
        df['shipping_ratio'] = df['shipping_cost'] / (df['total_amount'].replace(0, 0.01))

    if 'total_amount' in df.columns and 'quantity' in df.columns:
        df['revenue_per_item'] = df['total_amount'] / df['quantity'].replace(0, 1)

    if 'discount' in df.columns:
        df['is_high_discount'] = (df['discount'] > 0.2).astype(int)

    if 'delivery_time_days' in df.columns:
        df['is_slow_delivery'] = (df['delivery_time_days'] > 7).astype(int)

    if 'profit_margin' in df.columns:
        df['is_negative_margin'] = (df['profit_margin'] < 0).astype(int)

    if 'customer_age' in df.columns:
        df['age_group'] = pd.cut(df['customer_age'], bins=[0, 25, 35, 50, 65, 100],
                                  labels=['18-25', '26-35', '36-50', '51-65', '65+'])

    # Encode categorical columns safely
    le = LabelEncoder()
    for col in ['category', 'payment_method', 'region', 'customer_gender', 'age_group']:
        if col in df.columns:
            try:
                df[col + '_enc'] = le.fit_transform(df[col].astype(str))
            except Exception:
                df[col + '_enc'] = 0

    if 'returned' in df.columns:
        df['returned_num'] = (df['returned'] == 'Yes').astype(int)

    return df
