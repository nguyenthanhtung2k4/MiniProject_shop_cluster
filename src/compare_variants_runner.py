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
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def run_comparison():
    print("--- Starting Systematic Comparison ---")
    
    # Paths
    cleaned_data_path = os.path.join(project_root, "data/processed/cleaned_uk_data.csv")
    rules_path = os.path.join(project_root, "data/processed/rules_apriori_filtered.csv")
    output_comparison_path = os.path.join(project_root, "data/processed/clustering_comparison_results.csv")
    
    # Init Clusterer
    print("Loading data...")
    clusterer = RuleBasedCustomerClusterer(df_clean=pd.read_csv(cleaned_data_path, parse_dates=["InvoiceDate"]))
    clusterer.build_customer_item_matrix(threshold=1)
    
    # Load Rules (Assumption: Rules file exists and is valid)
    if not os.path.exists(rules_path):
        print(f"Error: Rules file not found at {rules_path}")
        return
        
    print("Loading rules...")
    clusterer.load_rules(rules_path, top_k=200, sort_by='lift')

    # Define Scenarios
    scenarios = [
        {
            "name": "Binary Rules Only (K=3)",
            "weighting": "none",
            "use_rfm": False,
            "algo": "kmeans",
            "k": 3
        },
        {
            "name": "Weighted Rules (Lift) (K=3)",
            "weighting": "lift",
            "use_rfm": False,
            "algo": "kmeans",
            "k": 3
        },
        {
            "name": "[Proposed] Hybrid (Weighted+RFM) (K=3)",
            "weighting": "lift",
            "use_rfm": True,
            "rfm_scale": True,
            "rule_scale": False,
            "algo": "kmeans",
            "k": 3
        },
        {
            "name": "Hierarchical (Weighted+RFM) (K=3)",
            "weighting": "lift",
            "use_rfm": True,
            "rfm_scale": True,
            "rule_scale": False,
            "algo": "agglomerative",
            "k": 3
        },
         {
            "name": "Hierarchical (Weighted+RFM) (K=2)",
            "weighting": "lift",
            "use_rfm": True,
            "rfm_scale": True,
            "rule_scale": False,
            "algo": "agglomerative",
            "k": 2
        }
    ]
    
    results = []
    
    for sc in scenarios:
        print(f"Running scenario: {sc['name']}...")
        
        # Build Features
        X, _ = clusterer.build_final_features(
            weighting=sc['weighting'],
            use_rfm=sc['use_rfm'],
            rfm_scale=sc.get('rfm_scale', True),
            rule_scale=sc.get('rule_scale', False),
            min_antecedent_len=1
        )
        
        # Fit Model
        if sc['algo'] == 'kmeans':
            labels = clusterer.fit_kmeans(X, n_clusters=sc['k'], random_state=42)
        else:
            labels = clusterer.fit_agglomerative(X, n_clusters=sc['k'])
            
        # Calculate Metrics
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        
        # Cluster Distribution
        counts = pd.Series(labels).value_counts().sort_index()
        dist_str = ", ".join([f"C{i}:{c}" for i, c in counts.items()])
        
        # Actionability (Simple heuristic: is min cluster size > 1% of total?)
        min_size = counts.min()
        total = len(labels)
        actionable = "Yes" if min_size > (0.01 * total) else "No (Skewed)"
        
        results.append({
            "Scenario": sc['name'],
            "Algorithm": sc['algo'],
            "K": sc['k'],
            "Silhouette": round(sil, 3),
            "Calinski-Harabasz": round(ch, 1),
            "Distribution": dist_str,
            "Actionable?": actionable
        })
        
    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_comparison_path, index=False)
    
    print("\n--- RESULTS ---")
    print(results_df.to_markdown(index=False))
    print(f"\nSaved to: {output_comparison_path}")

if __name__ == "__main__":
    run_comparison()
