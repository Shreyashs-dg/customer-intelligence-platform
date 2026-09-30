import numpy as np
import pandas as pd

KMEANS_FEATURE_ORDER = [
    "Recency", "Frequency_log", "Monetary_log", "Tenure",
    "AOV_log", "Purchase_Variability_log", "Product_Diversity_log",
]

CHURN_FEATURE_ORDER = [
    "Frequency_log", "Monetary_log", "Tenure",
    "AOV_log", "Purchase_Variability_log", "Product_Diversity_log", "Cluster",
]

SEGMENT_NAMES = {
    0: "New / Occasional Big-Ticket Buyers",
    1: "Loyal High-Value Customers",
    2: "Lost / Churned Customers",
    3: "At-Risk Regulars",
}

def apply_log_transforms(raw: dict) -> dict:
    out = dict(raw)
    out["Frequency_log"] = np.log1p(raw["Frequency"])
    out["Monetary_log"] = np.log1p(raw["Monetary"])
    out["AOV_log"] = np.log1p(raw["AOV"])
    out["Purchase_Variability_log"] = np.log1p(raw["Purchase_Variability"])
    out["Product_Diversity_log"] = np.log1p(raw["Product_Diversity"])
    return out


def build_kmeans_feature_row(raw_with_logs: dict) -> pd.DataFrame:
    return pd.DataFrame([[raw_with_logs[col] for col in KMEANS_FEATURE_ORDER]],
                         columns=KMEANS_FEATURE_ORDER)


def build_churn_feature_row(raw_with_logs: dict, cluster: int) -> pd.DataFrame:
    row = dict(raw_with_logs)
    row["Cluster"] = cluster
    return pd.DataFrame([[row[col] for col in CHURN_FEATURE_ORDER]],
                         columns=CHURN_FEATURE_ORDER)
