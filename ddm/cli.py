import argparse
import os
from .data import load_sales, load_retail
from .features import engineer_features
from .models import cluster_customers, train_models, add_predictions
from .export import write_exports


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SALES = os.path.join(BASE_DIR, 'Datasets after cleaning', 'ecommerce_sales_34500.xlsx')
DEFAULT_RETAIL = os.path.join(BASE_DIR, 'Datasets after cleaning', 'silver_onlinee_retail.xlsx')
DEFAULT_OUTPUT = os.path.join(BASE_DIR, 'outputs')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run DDDM AI pipeline and export Excel reports')
    parser.add_argument('--sales', default=DEFAULT_SALES, help='Path to sales excel')
    parser.add_argument('--retail', default=DEFAULT_RETAIL, help='Path to online retail excel')
    parser.add_argument('--out', default=DEFAULT_OUTPUT, help='Output folder for generated excels')
    parser.add_argument('--only', choices=['all', 'enriched', 'segments', 'features', 'models', 'monthly'], default='all')
    args = parser.parse_args(argv)

    print(f"Loading sales from: {args.sales}")
    sales = load_sales(args.sales)
    print(f"Loading retail aggregates from: {args.retail}")
    retail_agg = load_retail(args.retail)

    print("Engineering features...")
    sales = engineer_features(sales, retail_agg)

    print("Clustering customers...")
    sales, seg_profile = cluster_customers(sales)

    FEATURES = [
        'price', 'discount', 'quantity', 'delivery_time_days', 'shipping_cost',
        'profit_margin', 'customer_age', 'total_amount',
        'order_dayofweek', 'order_month', 'order_quarter', 'is_weekend',
        'net_price', 'discount_amount', 'shipping_ratio', 'revenue_per_item',
        'is_high_discount', 'is_slow_delivery', 'is_negative_margin',
        'avg_unit_price', 'avg_basket_size', 'median_revenue_order',
        'nb_unique_products', 'total_transactions',
        'category_enc', 'payment_method_enc', 'region_enc',
        'customer_gender_enc', 'age_group_enc', 'Cluster'
    ]

    print("Training models and evaluating...")
    trained, eval_df, fi_df, best_model, scaler = train_models(sales, FEATURES)

    print("Adding predictions to dataset...")
    sales = add_predictions(sales, best_model, FEATURES)

    monthly = sales.groupby('order_month').agg(
        Nb_commandes=('order_id', 'count'),
        CA_mensuel=('total_amount', 'sum'),
        Taux_retour_pct=('returned_num', 'mean'),
        Marge_moyenne_pct=('profit_margin', 'mean'),
        Delai_moyen_jours=('delivery_time_days', 'mean')
    ).reset_index()
    if 'Taux_retour_pct' in monthly.columns:
        monthly['Taux_retour_pct'] = (monthly['Taux_retour_pct'] * 100).round(2)

    # Prepare segment profile in similar format
    seg_profile_out = seg_profile.copy()
    if 'return_rate' in seg_profile_out.columns:
        seg_profile_out['Taux_retour_pct'] = (seg_profile_out['return_rate'] * 100).round(2)

    print("Exporting reports...")
    paths = write_exports(sales, seg_profile_out, fi_df, eval_df, monthly, out_dir=args.out)
    print("Done.")
    return paths


if __name__ == '__main__':
    main()
