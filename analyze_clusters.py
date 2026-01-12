import pandas as pd
import numpy as np
import sys
import os

def analyze_clusters(cluster_file, rules_file, top_n_rules=5):
    if not os.path.exists(cluster_file) or not os.path.exists(rules_file):
        print(f"Files not found: {cluster_file} or {rules_file}")
        return

    print(f"Đang phân tích cụm từ file: {cluster_file}")
    
    # 1. Load data
    df_clusters = pd.read_csv(cluster_file) # CustomerID, cluster, Recency, Frequency, Monetary, ...
    rules_df = pd.read_csv(rules_file)      # ants, cons, lift, ...

    # Giả sử file rules_df có cột 'rule_str'. Nếu chưa, tạo nó.
    if 'rule_str' not in rules_df.columns:
        # Cố gắng tạo lại nếu có antecedents_str
        if 'antecedents_str' in rules_df.columns:
             rules_df['rule_str'] = rules_df['antecedents_str'].astype(str) + " -> " + rules_df['consequents_str'].astype(str)
        else:
             rules_df['rule_str'] = "Rule_" + rules_df.index.astype(str)

    # 2. Re-construct Rule Features để biết khách nào thoả luật nào
    # Vì file cluster output hiện tại không lưu cột feature của luật (nó chỉ lưu RFM và cluster label),
    # chúng ta phải join lại với cleaned data để tính toán lại tỷ lệ thoả luật,
    # HOẶC đơn giản hơn: load lại class RuleBasedCustomerClusterer để tái tạo ma trận X_rules.
    # Tuy nhiên, để script này độc lập và nhanh, ta sẽ làm cách đơn giản hoá:
    # "Với mỗi cụm, ta tính xem các sản phẩm trong Top Rules của toàn bộ tập dữ liệu xuất hiện tần suất ra sao?"
    # Nhưng đề bài yêu cầu "Luật nào chiếm ưu thế?".
    
    # Cách tốt nhất: Tận dụng lại class RuleBasedCustomerClusterer
    sys.path.append("src")
    from cluster_library import RuleBasedCustomerClusterer
    
    # Load cleaned data để tính lại feature mask checking
    cleaned_data_path = "data/processed/cleaned_uk_data.csv"
    if not os.path.exists(cleaned_data_path):
        print("Cần file data cleaned để verify rules.")
        return

    df_clean = pd.read_csv(cleaned_data_path)
    
    clusterer = RuleBasedCustomerClusterer(df_clean=df_clean)
    
    # Xây dựng lại customer_item matrix
    print("Re-building customer-item matrix...")
    clusterer.build_customer_item_matrix(threshold=1) # tốn chút thời gian
    
    # Load rules (Top 200 như lúc training)
    # LƯU Ý: Phải dùng đúng Top K và sort như lúc training thì index mới khớp.
    # Trong run_papermill ta dùng TOP_K=200, sort_by='lift'
    print("Loading rules to match training features...")
    loaded_rules = clusterer.load_rules(rules_file, top_k=200, sort_by='lift')
    
    # Tạo matrix 0/1 (không cần weight để đếm số lượng khách thoả luật cho dễ hiểu)
    print("Building rule feature matrix (binary)...")
    X_rules = clusterer.build_rule_feature_matrix(weighting='none')
    
    # X_rules là array (n_customers x n_rules)
    # Cần map lại CustomerID của X_rules với df_clusters
    
    # clusterer.customers_ là danh sách CustomerID tương ứng với các dòng của X_rules
    cust_ids = clusterer.customers_
    
    # Tạo DataFrame features
    rule_feat_df = pd.DataFrame(X_rules, columns=[f"Rule_{i}" for i in range(X_rules.shape[1])])
    rule_feat_df['CustomerID'] = cust_ids
    
    # Merge với cluster labels
    # Lưu ý: df_clusters['CustomerID'] cần ép kiểu str, zfill(6) nếu cần
    df_clusters['CustomerID'] = df_clusters['CustomerID'].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    
    merged = df_clusters.merge(rule_feat_df, on='CustomerID', how='inner')
    
    if merged.empty:
        print("Merge failed. Check CustomerID formats.")
        return

    # 3. Analyze per cluster
    n_clusters = merged['cluster'].nunique()
    print(f"\nTìm thấy {n_clusters} cụm khách hàng.\n")
    
    for c in sorted(merged['cluster'].unique()):
        subset = merged[merged['cluster'] == c]
        n_cust = len(subset)
        print(f"=== CỤM {c} (Số khách: {n_cust}) ===")
        
        # RFM trung bình
        if 'Recency' in subset.columns:
            r = subset['Recency'].mean()
            f = subset['Frequency'].mean()
            m = subset['Monetary'].mean()
            print(f"  [Trung bình RFM] Recency: {r:.1f} ngày | Freq: {f:.1f} lần | Money: {m:.1f}")
        
        # Tìm Top rules kích hoạt nhiều nhất
        # Các cột rule bắt đầu bằng "Rule_"
        rule_cols = [col for col in subset.columns if col.startswith("Rule_")]
        
        # Tính mean (tỷ lệ khách trong cụm thoả luật)
        rule_means = subset[rule_cols].mean().sort_values(ascending=False)
        
        print(f"  [Top {top_n_rules} Luật phổ biến nhất trong cụm]:")
        for i, (rule_col, score) in enumerate(rule_means.head(top_n_rules).items()):
            if score == 0: continue # Không in luật 0%
            
            # Lấy index luật từ tên cột "Rule_123" -> 123
            rule_idx = int(rule_col.split("_")[1])
            
            # Lấy nội dung luật
            if rule_idx < len(loaded_rules):
                r_str = loaded_rules.loc[rule_idx, 'rule_str']
                lift = loaded_rules.loc[rule_idx, 'lift']
                print(f"    {i+1}. {r_str} (Lift: {lift:.1f}) -> {score*100:.1f}% khách thoả luật này.")
            else:
                print(f"    {i+1}. Rule index {rule_idx} out of range??")
        print("\n")

if __name__ == "__main__":
    # Chọn file kết quả Advanced để phân tích
    CLUSTER_FILE = "data/processed/customer_clusters_advanced.csv"
    RULES_FILE = "data/processed/rules_apriori_filtered.csv"
    
    if os.path.exists(CLUSTER_FILE):
        analyze_clusters(CLUSTER_FILE, RULES_FILE)
    else:
        print(f"Chưa tìm thấy file {CLUSTER_FILE}. Hãy chạy pipeline trước.")
