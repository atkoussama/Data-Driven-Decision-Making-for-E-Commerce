import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report, average_precision_score


def cluster_customers(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    cols = ['order_id', 'total_amount', 'discount', 'returned_num', 'delivery_time_days', 'profit_margin']
    agg = df.groupby('customer_id').agg(
        nb_orders=('order_id', 'count'),
        total_spend=('total_amount', 'sum'),
        avg_discount=('discount', 'mean'),
        return_rate=('returned_num', 'mean'),
        avg_delivery=('delivery_time_days', 'mean'),
        avg_margin=('profit_margin', 'mean')
    ).reset_index()

    sc = StandardScaler()
    Xk = sc.fit_transform(agg[['nb_orders', 'total_spend', 'avg_discount', 'return_rate', 'avg_delivery', 'avg_margin']].fillna(0))
    km = KMeans(n_clusters=4, random_state=random_state, n_init=10)
    agg['Cluster'] = km.fit_predict(Xk)
    names = {0: 'High-risk buyers', 1: 'Loyal premium', 2: 'Occasional buyers', 3: 'Discount-sensitive'}
    agg['Segment'] = agg['Cluster'].map(names)

    df = df.merge(agg[['customer_id', 'Cluster', 'Segment']], on='customer_id', how='left')
    df['Cluster'] = df['Cluster'].fillna(-1).astype(int)
    return df, agg


def train_models(df: pd.DataFrame, FEATURES, target_col='returned_num', threshold=0.35, random_state=42):
    X = df[FEATURES].fillna(0)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    # Try SMOTE, but degrade gracefully if not available
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X_train, y_train)
    except Exception:
        X_res, y_res = X_train, y_train

    sc = StandardScaler()
    X_res_sc = sc.fit_transform(X_res)
    X_test_sc = sc.transform(X_test)

    models = {
        'LogisticRegression': LogisticRegression(C=0.1, max_iter=1000, random_state=random_state),
        'RandomForest': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=random_state, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=random_state)
    }

    trained = {}
    eval_rows = []
    for name, model in models.items():
        # choose appropriate X for model (scaled for linear models)
        if name == 'LogisticRegression':
            Xtr = X_res_sc
            Xte = X_test_sc
        else:
            Xtr = X_res
            Xte = X_test

        model.fit(Xtr, y_res)
        yp = model.predict_proba(Xte)[:, 1]
        ypred = (yp >= threshold).astype(int)
        rep = classification_report(y_test, ypred, output_dict=True)
        trained[name] = {'model': model, 'y_proba': yp, 'y_pred': ypred}
        eval_rows.append({
            'Model': name,
            'AUC-ROC': round(roc_auc_score(y_test, yp), 4),
            'Avg_Precision': round(average_precision_score(y_test, yp), 4),
            'F1-Score': round(f1_score(y_test, ypred), 4),
            'Precision': round(rep['1']['precision'], 4) if '1' in rep else None,
            'Recall': round(rep['1']['recall'], 4) if '1' in rep else None,
            'Accuracy': round(rep['accuracy'], 4),
            'Decision_Threshold': threshold
        })
        print(f"Trained {name} — AUC={roc_auc_score(y_test, yp):.4f} | F1={f1_score(y_test, ypred):.4f}")

    eval_df = pd.DataFrame(eval_rows).sort_values('AUC-ROC', ascending=False)

    # feature importance for best model if available
    best_name = eval_df.iloc[0]['Model']
    best_model = trained[best_name]['model']
    if hasattr(best_model, 'feature_importances_'):
        fi = best_model.feature_importances_
    else:
        try:
            fi = np.abs(best_model.coef_[0])
        except Exception:
            fi = np.zeros(len(FEATURES))

    fi_df = pd.DataFrame({'Feature': FEATURES, 'Importance': fi}).sort_values('Importance', ascending=False).reset_index(drop=True)

    return trained, eval_df, fi_df, best_model, sc


def add_predictions(df: pd.DataFrame, best_model, FEATURES):
    X = df[FEATURES].fillna(0)
    if hasattr(best_model, 'predict_proba'):
        df['proba_return'] = best_model.predict_proba(X)[:, 1].round(4)
    else:
        df['proba_return'] = 0.0
    df['return_risk'] = pd.cut(df['proba_return'], bins=[0, 0.2, 0.4, 0.6, 1.0], labels=['Low', 'Medium', 'High', 'Very High'])
    return df
