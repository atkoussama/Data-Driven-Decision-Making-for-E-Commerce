import os
import pandas as pd


def write_exports(df: pd.DataFrame, segment_profile: pd.DataFrame, fi_df: pd.DataFrame, eval_df: pd.DataFrame, monthly: pd.DataFrame, out_dir: str = '.'):
    os.makedirs(out_dir, exist_ok=True)

    paths = {}
    p1 = os.path.join(out_dir, 'orders_enriched.xlsx')
    df.to_excel(p1, index=False, sheet_name='Orders_Enriched')
    paths['orders_enriched'] = p1

    p2 = os.path.join(out_dir, 'customer_segments.xlsx')
    segment_profile.to_excel(p2, index=False, sheet_name='Customer_Segments')
    paths['customer_segments'] = p2

    p3 = os.path.join(out_dir, 'feature_importances.xlsx')
    fi_df.to_excel(p3, index=False, sheet_name='Feature_Importance')
    paths['feature_importances'] = p3

    p4 = os.path.join(out_dir, 'model_comparison.xlsx')
    eval_df.to_excel(p4, index=False, sheet_name='Model_Comparison')
    paths['model_comparison'] = p4

    p5 = os.path.join(out_dir, 'monthly_statistics.xlsx')
    monthly.to_excel(p5, index=False, sheet_name='Monthly_Statistics')
    paths['monthly_statistics'] = p5

    for k, v in paths.items():
        print(f"Wrote {k}: {v}")

    return paths
