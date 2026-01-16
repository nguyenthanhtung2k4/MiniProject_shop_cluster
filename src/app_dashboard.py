import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# streamlit run app_dashboard.py

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

# A. DASHBOARD TỔNG QUAN (VISUALIZATION)
st.header("📊 Tổng quan Phân bố Khách hàng")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Tỷ lệ quy mô các cụm")
    # Pie chart
    cluster_counts = cluster_df['cluster'].value_counts().reset_index()
    cluster_counts.columns = ['cluster', 'count']
    cluster_counts['cluster_label'] = cluster_counts['cluster'].apply(lambda x: f"Cụm {x}")
    
    fig_pie = px.pie(cluster_counts, values='count', names='cluster_label', 
                     title='Tỷ lệ khách hàng theo cụm', hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Đặc điểm RFM trung bình theo cụm")
    # Group by cluster and calc mean
    rfm_mean = cluster_df.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
    rfm_mean_melted = rfm_mean.melt(id_vars='cluster', var_name='Metric', value_name='Value')
    
    # Bar chart (Normalized view is better usually, but raw is ok for massive diffs)
    # Vì Monetary chênh lệch quá lớn, ta nên vẽ riêng hoặc scale. Ở đây vẽ riêng Recency và Frequency.
    
    tab1, tab2 = st.tabs(["Recency & Frequency", "Monetary (Chi tiêu)"])
    
    with tab1:
        fig_bar1 = px.bar(rfm_mean_melted[rfm_mean_melted['Metric'].isin(['Recency', 'Frequency'])], 
                          x='cluster', y='Value', color='Metric', barmode='group',
                          title="So sánh R và F trung bình")
        st.plotly_chart(fig_bar1, use_container_width=True)
        
    with tab2:
        fig_bar2 = px.bar(rfm_mean_melted[rfm_mean_melted['Metric'] == 'Monetary'], 
                          x='cluster', y='Value', color='Metric', 
                          title="So sánh Chi tiêu (Monetary)", color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig_bar2, use_container_width=True)

st.markdown("---")

# B. CHI TIẾT CỤM
st.header("🔍 Phân tích Chi tiết từng Cụm")
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

# 3. Recomendation Strategy & Profiling
st.subheader("💡 Phân tích & Đề xuất chiến lược")

# Tính chỉ số trung bình của cụm hiện tại
r = subset['Recency'].mean()
f = subset['Frequency'].mean()
m = subset['Monetary'].mean()

# Tính chỉ số trung bình toàn cục
global_r = cluster_df['Recency'].mean()
global_f = cluster_df['Frequency'].mean()
global_m = cluster_df['Monetary'].mean()

r_ratio = r / global_r
f_ratio = f / global_f
m_ratio = m / global_m

st.markdown("#### So sánh với trung bình toàn sàn:")
col1, col2, col3 = st.columns(3)
col1.metric("So với R trung bình", f"{r_ratio:.2f}x", delta_color="inverse") # R càng thấp càng tốt
col2.metric("So với F trung bình", f"{f_ratio:.2f}x")
col3.metric("So với M trung bình", f"{m_ratio:.2f}x")

# Logic gán nhãn tự động
labels = []
strategies = []

if m_ratio > 1.5:
    labels.append("💰 Big Spender (Chi tiêu khủng)")
    strategies.append("- **VIP Care:** Cần chăm sóc đặc biệt, tặng quà tri ân.")
    strategies.append("- **Upsell:** Giới thiệu các bộ sưu tập giá trị cao (High-ticket items).")
elif m_ratio < 0.5:
    labels.append("💸 Low Spender (Chi tiêu thấp)")
    strategies.append("- **Price Sensitivity:** Tập trung vào các sản phẩm giảm giá, combo tiết kiệm.")

if f_ratio > 1.5:
    labels.append("🔄 Loyal Customer (Mua thường xuyên)")
    strategies.append("- **Loyalty Program:** Khuyến khích tham gia tích điểm, giới thiệu bạn bè.")
elif f_ratio < 0.8:
    labels.append("🛒 Occasional (Khách vãng lai)")

if r_ratio > 1.5:
    labels.append("💤 Dormant/Churn Risk (Nguy cơ rời bỏ)")
    strategies.append("- **Re-activation:** Gửi email 'We miss you' kèm voucher hạn chót để kéo khách quay lại ngay.")
elif r_ratio < 0.6:
    labels.append("🔥 Active (Đang hoạt động mạnh)")
    strategies.append("- **Engagement:** Duy trì tương tác qua thông báo sản phẩm mới.")

st.markdown(f"**Nhãn định danh:** {' | '.join(labels) if labels else 'Khách hàng trung bình'}")

if strategies:
    st.markdown("**Chiến lược đề xuất:**")
    for s in strategies:
        st.markdown(s)
else:
    st.info("Nhóm khách hàng này có chỉ số khá sát với trung bình. Nên áp dụng các chiến dịch marketing đại trà (Mass Marketing).")

