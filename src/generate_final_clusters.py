# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from cluster_library import RuleBasedCustomerClusterer

def generate_final_clusters():
    print("--- Generating Final Clusters (Matching Blog) ---")
    
    # Paths
    cleaned_data_path = os.path.join(project_root, "data/processed/cleaned_uk_data.csv")
    rules_path = os.path.join(project_root, "data/processed/rules_apriori_filtered.csv")
    output_path = os.path.join(project_root, "data/processed/customer_clusters_advanced.csv")
    
    # Init Clusterer
    print("Loading data...")
    clusterer = RuleBasedCustomerClusterer(df_clean=pd.read_csv(cleaned_data_path, parse_dates=["InvoiceDate"]))
    clusterer.build_customer_item_matrix(threshold=1)
    
    # Load Rules
    print("Loading rules...")
    clusterer.load_rules(rules_path, top_k=200, sort_by='lift')

    # Scenario: Hybrid (Weighted+RFM) K=3
    print("Building features (Weighted + RFM)...")
    X, meta = clusterer.build_final_features(
        weighting="lift",
        use_rfm=True,
        rfm_scale=True,
        rule_scale=False,
        min_antecedent_len=1
    )
    
    print("Fitting K-Means (K=3)...")
    labels = clusterer.fit_kmeans(X, n_clusters=3, random_state=42)
    
    # Save
    meta['cluster'] = labels
    meta.to_csv(output_path, index=False)
    
    print(f"Saved final clusters to: {output_path}")
    print("Distribution:")
    print(meta['cluster'].value_counts().sort_index())

if __name__ == "__main__":
    generate_final_clusters()
