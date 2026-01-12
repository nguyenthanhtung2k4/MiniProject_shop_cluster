import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path if needed (though running from root usually works)
if "src" not in sys.path:
    sys.path.append("src")

try:
    from cluster_library import RuleBasedCustomerClusterer
except ImportError:
    st.error("Không tìm thấy module cluster_library. Hãy chạy từ thư mục gốc dự án.")
    st.stop()

st.set_page_config(layout="wide", page_title="Customer Segmentation Dashboard")

st.title("Phân cụm khách hàng theo Luật kết hợp & RFM")

# --- PARAMETERS ---
CLUSTER_FILE = "/hdd3/nckh-AIAgent/tungtt/Datamining/MiniProject_shop_cluster/data/processed/customer_clusters_advanced.csv"
RULES_FILE = "/hdd3/nckh-AIAgent/tungtt/Datamining/MiniProject_shop_cluster/data/processed/rules_apriori_filtered.csv"
CLEANED_DATA_PATH = "/hdd3/nckh-AIAgent/tungtt/Datamining/MiniProject_shop_cluster/data/processed/cleaned_uk_data.csv"

# --- LOAD DATA ---
@st.cache_data
def load_data():
    if not os.path.exists(CLUSTER_FILE) or not os.path.exists(RULES_FILE) or not os.path.exists(CLEANED_DATA_PATH):
        return None, None, None
    
    cl_df = pd.read_csv(CLUSTER_FILE)
    cl_df['CustomerID'] = cl_df['CustomerID'].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    
    r_df = pd.read_csv(RULES_FILE)
    # Tạo cột rule_str nếu chưa có
    if 'rule_str' not in r_df.columns and 'antecedents_str' in r_df.columns:
        r_df['rule_str'] = r_df['antecedents_str'].astype(str) + " -> " + r_df['consequents_str'].astype(str)
        
    org_df = pd.read_csv(CLEANED_DATA_PATH, dtype={'InvoiceNo': str, 'CustomerID': str})
    return cl_df, r_df, org_df

cluster_df, rules_df, original_df = load_data()

if cluster_df is None:
    st.error("Chưa thấy các file dữ liệu output. Hãy chạy pipeline 'run_papermill.py' trước!")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("Bộ lọc")
selected_cluster = st.sidebar.selectbox(
    "Chọn Cụm Khách Hàng",
    options=sorted(cluster_df['cluster'].unique())
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"Tổng số khách hàng: {len(cluster_df)}\n\n"
    f"Tổng số cụm: {cluster_df['cluster'].nunique()}"
)

# --- MAIN CONTENT ---

# 1. Cluster Stats
st.subheader(f"Tổng quan Cụm {selected_cluster}")

subset = cluster_df[cluster_df['cluster'] == selected_cluster]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Số lượng khách", len(subset))
col2.metric("Recency TB (ngày)", f"{subset['Recency'].mean():.1f}")
col3.metric("Frequency TB (lần)", f"{subset['Frequency'].mean():.1f}")
col4.metric("Monetary TB (£)", f"{subset['Monetary'].mean():,.0f}")

# 2. Top Rules in this Cluster
st.subheader("🛒 Các luật mua sắm phổ biến nhất trong cụm")
st.markdown("Những cặp sản phẩm nào khách hàng trong cụm này thường mua?")

@st.cache_resource
def get_rule_features(_original_df, _rules_df):
    # Tính toán lại feature matrix (hơi tốn thời gian nên cache)
    clusterer = RuleBasedCustomerClusterer(df_clean=_original_df)
    clusterer.build_customer_item_matrix(threshold=1)
    
    # Load rules đúng như training
    # Lưu ý: cần khớp logic load_rules với training (Top 200, sort lift)
    loaded_rules = clusterer.load_rules(RULES_FILE, top_k=200, sort_by='lift') # Đây là giả định user dùng file path string, nhưng hàm load_rules cần path.
    # Sửa lại: load_rules nhận path. Ở đây ta truyền path dummy vì ta đã có dataframe rồi?
    # Không, clusterer.load_rules đọc file. Vậy ta truyền path RULES_FILE.
    
    X_rules = clusterer.build_rule_feature_matrix(weighting='none')
    
    feat_df = pd.DataFrame(X_rules, columns=[f"Rule_{i}" for i in range(X_rules.shape[1])])
    feat_df['CustomerID'] = clusterer.customers_
    return feat_df, loaded_rules

with st.spinner("Đang phân tích luật (có thể mất vài giây lần đầu)..."):
    feat_df, loaded_rules_meta = get_rule_features(original_df, rules_df)
    
    # Merge subset của cụm hiện tại với feature matrix
    merged_subset = subset[['CustomerID']].merge(feat_df, on='CustomerID', how='inner')
    
    if not merged_subset.empty:
        rule_cols = [c for c in merged_subset.columns if c.startswith("Rule_")]
        means = merged_subset[rule_cols].mean().sort_values(ascending=False).head(10)
        
        ranking_data = []
        for r_col, val in means.items():
            if val > 0:
                idx = int(r_col.split("_")[1])
                if idx < len(loaded_rules_meta):
                    row_rule = loaded_rules_meta.iloc[idx]
                    ranking_data.append({
                        "Luật": row_rule['rule_str'],
                        "Support (Toàn cục)": row_rule['support'],
                        "Lift": row_rule['lift'],
                        "% Khách cụm này thoả mãn": f"{val*100:.1f}%"
                    })
        
        st.table(pd.DataFrame(ranking_data))
    else:
        st.warning("Không merge được dữ liệu feature cho cụm này.")

# 3. Recomendation Strategy
st.subheader("💡 Đề xuất chiến lược")
r = subset['Recency'].mean()
f = subset['Frequency'].mean()
m = subset['Monetary'].mean()

if m > cluster_df['Monetary'].mean() * 1.5:
    st.success("**Chiến lược VIP:** Đây là nhóm khách hàng chi tiêu mạnh. Hãy cung cấp dịch vụ CSKH ưu tiên, quà tặng premium, và giới thiệu các sản phẩm high-end mới nhất.")
elif r > 100:
    st.warning("**Chiến lược Re-activation:** Nhóm này đã lâu không quay lại. Cần gửi email 'We miss you' kèm voucher giảm giá sâu hoặc free shipping để kéo họ lại.")
elif f > cluster_df['Frequency'].mean():
    st.info("**Chiến lược Loyalty:** Khách mua thường xuyên. Hãy khuyến khích họ tham gia chương trình tích điểm hoặc giới thiệu bạn bè (Referral).")
else:
    st.write("Nhóm khách hàng phổ thông. Nên tập trung vào các chương trình khuyến mãi đại trà hoặc Bundle các sản phẩm họ hay mua (xem bảng luật ở trên).")

