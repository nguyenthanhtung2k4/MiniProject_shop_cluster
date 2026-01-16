# Case Study: Phân Cụm Khách Hàng Dựa Trên Luật Kết Hợp (Rules + RFM)

Dự án này mở rộng bài toán Market Basket Analysis truyền thống: Thay vì chỉ tìm ra "Sản phẩm nào đi cùng nhau?", chúng tôi sử dụng chính các luật kết hợp đó để định danh "Nhóm khách hàng nào có hành vi giống nhau?". Project triển khai pipeline từ khai phá luật (Apriori) → trích xuất đặc trưng (Feature Engineering) → phân cụm nâng cao (Clustering) → đề xuất chiến lược.

---

## 👥 Thông tin Nhóm

- **Nhóm:** 3
- **Thành viên:**
  - Nguyễn Thanh Tùng
  - Lê Văn Vượng
  - Nguyễn Đức Anh
  - Đỗ Văn Tuyên
- **Chủ đề:** Phân khúc khách hàng dựa trên hành vi mua kèm (Association Rules) kết hợp giá trị (RFM).
- **Dataset:** Online Retail (UCI)

---

## 🎯 Mục tiêu
Sử dụng các luật kết hợp (Association Rules) làm đặc trưng đầu vào cho bài toán phân cụm nhằm tìm ra các nhóm khách hàng có "phong cách mua sắm" tương đồng. Mục tiêu cuối cùng là cá nhân hóa chiến lược marketing: **Đúng người - Đúng thời điểm - Đúng combo**.

---

## 1. Ý tưởng & Tiếp cận (Methodology)

- **Vấn đề của RFM truyền thống:** Chỉ cho biết khách hàng "giàu" hay "nghèo", "mới" hay "cũ", nhưng không biết họ thích mua gì.
- **Giải pháp của nhóm:** Biến đổi Luật kết hợp thành Vector đặc trưng (Rule-based Embedding).
  - Mỗi luật (ví dụ: *Mua Giấy ăn -> Mua Đĩa nhựa*) được xem như một "sở thích" (Interest).
  - Nếu khách hàng thoả mãn luật đó (mua đủ vế trái) -> Gán điểm trọng số (dựa trên Lift/Confidence).
  - Kết hợp vector luật này với chỉ số RFM (Recency-Frequency-Monetary) đã chuẩn hóa để tạo ra bộ đặc trưng lai (Hybrid Features).

## 2. Quy trình thực hiện (Pipeline)
Quy trình được tự động hóa hoàn toàn bằng Papermill:

3.  **Modeling:** Thử nghiệm K-Means và Hierarchical Clustering với K thay đổi từ 2 đến 10.
4.  **Optimizing K:** Sử dụng phương pháp Elbow và Silhouette Score để xác định số lượng cụm tối ưu.
5.  **Profiling:** Phân tích đặc điểm từng cụm và lên chiến lược.

### 2.1. Feature Engineering: Từ Luật đến Vector Đặc Trưng
Theo yêu cầu, nhóm đã xây dựng **2 biến thể đặc trưng** để so sánh hiệu quả:

1.  **Biến thể 1 (Baseline): Binary Rules Only**
    - Mỗi khách hàng được biểu diễn bằng vector $V = [r_1, r_2, ..., r_k]$ với $r_i \in \{0, 1\}$.
    - $r_i = 1$ nếu khách hàng mua đủ các sản phẩm trong vế trái (Antecedent) của luật thứ $i$.
    - **Ưu điểm:** Đơn giản, dễ hiểu.
    - **Nhược điểm:** Không phân biệt được mức độ "mê" sản phẩm (mua 1 lần vs mua 10 lần giống nhau).

2.  **Biến thể 2 (Advanced - Chosen): Weighted Rules + RFM**
    - **Weighted Rules:** Thay vì 0/1, giá trị $r_i$ được gán bằng **Lift** của luật đó.
        - *Lý do:* Luật có Lift cao (70-80) mang lại thông tin về "sở thích đặc biệt" mạnh hơn luật có Lift thấp.
    - **RFM Augmentation:** Ghép thêm 3 chỉ số Recency - Frequency - Monetary (đã được Scaled bằng StandardScaler) vào vector luật.
    - **Mục đích:** Vừa hiểu được **HÀNH VI MUA CÁI GÌ** (từ Rules) vừa hiểu được **GIÁ TRỊ KHÁCH HÀNG** (từ RFM).

### 2.2. Quy trình Lựa chọn Luật (Rule Filtering)
Để đảm bảo chất lượng input cho phân cụm, nhóm không lấy toàn bộ hàng nghìn luật sinh ra mà lọc theo quy trình:
1.  **Thuật toán:** Apriori (min_support=0.01).
2.  **Top-K Selection:** Chọn **200 luật** có **Lift cao nhất**.
3.  **Lý do chọn Top-200:** Thử nghiệm cho thấy nếu dùng quá ít (<50), thông tin quá thưa thớt. Nếu dùng quá nhiều (>500), vector bị nhiễu (curse of dimensionality) mà không tăng thêm độ tách biệt rõ rệt.

#### Cấu hình chi tiết (Parameters):
| Tham số | Giá trị | Mô tả |
| :--- | :--- | :--- |
| `MIN_SUPPORT` | 0.01 | Ngưỡng hỗ trợ tối thiểu (1%) |
| `MAX_LEN` | 3 | Độ dài tối đa của luật |
| `METRIC` | lift | Tiêu chí đánh giá chính |
| `MIN_THRESHOLD` | 1.0 | Ngưỡng lift tối thiểu |
| `FILTER_MIN_CONF` | 0.3 | Độ tin cậy tối thiểu (30%) |
| `FILTER_MIN_LIFT` | 1.2 | Lọc các luật có lift < 1.2 |
| `FILTER_MAX_ANTECEDENTS` | 2 | Tối đa 2 sản phẩm vế trái |
| `FILTER_MAX_CONSEQUENTS` | 1 | Tối đa 1 sản phẩm vế phải |

#### Top 10 Luật tiêu biểu (High Lift Rules)
Dưới đây là danh sách 10 luật có điểm Lift cao nhất, được ưu tiên làm đặc trưng chính phương phân cụm:

| Antecedents (Mua) | Consequents (Thì cũng mua) | Support | Confidence | Lift |
| :--- | :--- | :---: | :---: | :---: |
| *HERB MARKER PARSLEY, HERB MARKER ROSEMARY* | *HERB MARKER THYME* | 1.09% | 95.2% | **74.6** |
| *HERB MARKER MINT, HERB MARKER THYME* | *HERB MARKER ROSEMARY* | 1.06% | 95.5% | **74.5** |
| *HERB MARKER MINT, HERB MARKER THYME* | *HERB MARKER PARSLEY* | 1.04% | 94.0% | **74.3** |
| *HERB MARKER PARSLEY, HERB MARKER THYME* | *HERB MARKER ROSEMARY* | 1.09% | 95.2% | **74.2** |
| *HERB MARKER BASIL, HERB MARKER THYME* | *HERB MARKER ROSEMARY* | 1.07% | 95.1% | **74.2** |
| *HERB MARKER BASIL, HERB MARKER ROSEMARY* | *HERB MARKER THYME* | 1.07% | 93.7% | **73.4** |
| *HERB MARKER MINT, HERB MARKER ROSEMARY* | *HERB MARKER THYME* | 1.06% | 93.2% | **73.0** |
| *HERB MARKER MINT, HERB MARKER ROSEMARY* | *HERB MARKER PARSLEY* | 1.05% | 92.2% | **72.9** |
| *HERB MARKER BASIL, HERB MARKER THYME* | *HERB MARKER PARSLEY* | 1.04% | 92.1% | **72.8** |
| *HERB MARKER CHIVES* | *HERB MARKER PARSLEY* | 1.04% | 92.1% | **72.8** |


---

## 3. Thực nghiệm, So sánh & Lựa chọn K (Technical vs Business Trade-off)

Trong quá trình thực nghiệm, chúng tôi đứng trước một bài toán đánh đổi kinh điển giữa **Điểm số Toán học** và **Giá trị Kinh doanh**.

### 3.1. So sánh Hệ thống (Systematic Comparison)

Chúng tôi đã thực chạy thực nghiệm trên 5 kịch bản khác nhau để tìm ra cấu hình tối ưu. Dưới đây là kết quả thực tế (chạy trên toàn bộ dữ liệu):

| Kịch bản (Scenario) | K | Silhouette | Phân bổ mẫu (Cluster Sizes) | Đánh giá |
| :--- | :---: | :---: | :--- | :--- |
| **1. K-Means (Binary Rules)** | 3 | 0.483 | C0: 3536, C1: 125, C2: 260 | **Baseline.** Tách tạm ổn nhưng điểm thấp nhất. |
| **2. K-Means (Weighted Rules)** | 3 | **0.583** | C0: 3602, C1: 124, C2: 195 | **Tốt.** Việc thêm trọng số Lift giúp cụm rõ nét hơn hẵn. |
| **3. K-Means (Hybrid: Rule+RFM)** | 3 | 0.581 | C0: 3602, C1: 124, C2: 195 | **Được chọn.** Điểm tương đương kịch bản 2 nhưng có thêm thông tin RFM để làm giàu bài toán Profiling. |
| **4. Hierarchical (Weighted+RFM)** | 3 | 0.575 | C0: 134, C1: 3636, C2: 151 | **Công bằng.** Khi ép K=3, Hierarchical cho kết quả *kém hơn* K-Means một chút. |
| **5. Hierarchical (Weighted+RFM)** | 2 | **0.850** | C0: 3787, C1: 134 | **Toán học tốt nhất.** Silhouette rất cao nhưng phân cụm cực đoan (1 nhóm VIP nhỏ vs cả thế giới còn lại). |

### 3.2. Biện luận: Tại sao chọn K-Means (K=3) thay vì Hierarchical (K=2)?

Nhìn bảng trên, kịch bản số 5 (Hierarchical, K=2) có điểm số áp đảo (0.85). Tuy nhiên, nhóm quyết định **TỪ CHỐI** kết quả này và chọn **Kịch bản 3 (K-Means, K=3)** vì lý do Business:

1.  **Vấn đề của K=2 (Hierarchical):** Nó chỉ tách được 134 khách hàng "Siêu VIP" ra khỏi 3787 khách hàng còn lại. Doanh nghiệp không thể áp dụng *một chiến lược duy nhất* cho 3787 người này (bao gồm cả người mới, người cũ, người sắp rời bỏ). Đây là mô hình "Lười biếng" (Lazy clustering).
2.  **Sức mạnh của K=3 (K-Means):** Mô hình này bóc tách được nhóm 3787 người kia thành 2 phần:
    - **Nhóm Vãng lai (Mass):** ~3600 người.
    - **Nhóm Tiềm năng (Rising Stars):** ~195 người. Đây là nhóm quan trọng nhất để upsell mà mô hình K=2 đã bỏ sót.

$\rightarrow$ **Kết luận:** Chấp nhận giảm điểm Silhouette từ 0.85 xuống 0.58 để đổi lấy một tập khách hàng được phân khúc chi tiết và "Actionable" hơn.

---

## 4. Kết quả Phân Cụm & Customer Profiling (Chi tiết K=3)

Dựa trên mô hình K-Means (K=3) được lựa chọn, chúng tôi vẽ lại chân dung chi tiết như sau:

### 📊 Biểu đồ Phân tích Thực tế

#### 1. Tổng quan Phân bố & Tỷ lệ Cụm
![Tổng quan tỷ lệ khách hàng](img/newplot.png)

#### 2. So sánh Chỉ số RFM giữa các cụm (K=3)
Biểu đồ cho thấy sự khác biệt rõ rệt về hành vi Recency và Chi tiêu giữa 3 nhóm:
<p float="left">
  <img src="img/newplot (1).png" width="45%" />
  <img src="img/newplot (2).png" width="45%" /> 
</p>

#### 3. Mô hình Phân cụm (2D Visualization)
![Mô hình phân cụm 2D](img/output.png)

### Chi tiết 3 Chân dung:

### 💎 Cụm 1: The VIP Wholesalers (Nhà Buôn / VIP)
- **Quy mô:** ~3% (124 khách).
- **Chỉ số:** Chi tiêu cực khủng (**£17,000+**). Recency thấp.
- **Hành vi (Rules):** Mua sỉ. 90% các luật mua trọn bộ sưu tập (Herb Marker, Pantry Design) đều rơi vào nhóm này.
- **Chiến lược:** *Partnership & Exclusive*. Cung cấp chiết khấu B2B, mời tham gia sự kiện ra mắt sản phẩm kín.

### 🌟 Cụm 2: The Rising Stars (Ngôi Sao Đang Lên / Tiềm Năng)
- **Quy mô:** ~5% (195 khách).
- **Chỉ số:** Nhóm này có hành vi lai. Không giàu như VIP nhưng mua sắm rất "có gu".
- **Hành vi:** Thường kích hoạt các luật mua đồ trang trí nhỏ, quà tặng. Có tần suất quay lại cao hơn hẳn nhóm vãng lai.
- **Chiến lược:** *Membership Upgrading*. Thúc đẩy họ đạt ngưỡng VIP bằng các thử thách mua sắm (Gamification).

### 💤 Cụm 0: The Hibernating Masses (Đám Đông Vãng Lai)
- **Quy mô:** ~92% (3602 khách).
- **Chỉ số:** Giá trị thấp, Recency cao (lâu không mua).
- **Chiến lược:** *Mass Promotion*. Sử dụng các deal giảm giá sốc (Flash Sale) để kích thích nhu cầu cơ bản. Không nên tốn chi phí chăm sóc 1-1.

---

## 5. Phân tích Nâng cao & Mở rộng (Advanced Analysis)

Để đáp ứng các yêu cầu chuyên sâu của dự án (mục tiêu xuất sắc), nhóm đã thực hiện thêm các nghiên cứu so sánh mở rộng:

*(Phần này đã được tích hợp vào bảng so sánh tổng hợp ở mục 3.1)*

### 5.2. Góc nhìn Marketing: Customer Clustering vs Rule Clustering
Ngoài việc phân cụm khách hàng, nhóm cũng đã cân nhắc hướng tiếp cận **Phân cụm Luật (Rule Clustering)**:
- **Rule Clustering:** Gom các luật giống nhau (ví dụ: luật mua "Bát đĩa" và luật mua "Cốc chén") thành nhóm nhu cầu. *Lợi ích:* Giúp thiết kế gói sản phẩm (Bundling).
- **Customer Clustering (Đã chọn):** Gom người mua giống nhau. *Lợi ích:* Giúp target đối tượng (Direct Marketing).

**Kết luận:** Với mục tiêu tối ưu hóa CRM và Re-marketing, việc **phân cụm Khách hàng** dựa trên đặc trưng Luật mang lại hiệu quả trực tiếp và đo lường được doanh thu tốt hơn so với phân cụm Luật đơn thuần.

---

## 6. Kết luận & Hướng phát triển

- **Kết luận:** Việc đưa Luật kết hợp vào phân cụm giúp doanh nghiệp hiểu **SÂU** hơn về khách hàng. Kết hợp với việc chọn K=3, chúng ta có được bản đồ chiến lược rõ ràng cho từng nhóm đối tượng, tránh lãng phí ngân sách Marketing vào sai người.
- **Hướng mở rộng:**
  - Thử nghiệm thêm DBSCAN để xử lý nhiễu tốt hơn.
  - Sử dụng Deep Learning (Autoencoders) để nén vector đặc trưng khi số lượng luật quá lớn.

---

## 7. Link Code & Tài liệu
- **Repository:** https://github.com/nguyenthanhtung2k4/MiniProject_shop_cluster
- **Dashboard App:** `src/app_dashboard.py`
