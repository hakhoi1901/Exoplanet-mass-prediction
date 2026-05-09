# ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH
## TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN — KHOA CÔNG NGHỆ THÔNG TIN

---

# ĐỒ ÁN 2: Data Fitting và Phương Pháp OLS

**Môn học:** Toán Ứng Dụng và Thống Kê  
**Mã môn:** MTH00051  
**Học kỳ:** Học kỳ 2, 2025–2026

**GV Thực hành:** ThS. Võ Nam Thục Đoan, ThS. Lê Nhựt Nam  
**E-mail:** {vntdoan, lnnam}@fit.hcmus.edu.vn

> Tài liệu này dành riêng cho mục đích học thuật.

---

## Mục lục

- [Giới Thiệu Đồ Án](#giới-thiệu-đồ-án)
- [Phần 1: Lý Thuyết Data Fitting và Minh Họa](#phần-1-lý-thuyết-data-fitting-và-minh-họa)
  - [1.1 Bài Toán Data Fitting](#11-bài-toán-data-fitting)
  - [1.2 Phương Pháp Ordinary Least Squares (OLS)](#12-phương-pháp-ordinary-least-squares-ols)
  - [1.3 Đánh Giá Mô Hình](#13-đánh-giá-mô-hình)
  - [1.4 Các Vấn Đề Nâng Cao trong Data Fitting](#14-các-vấn-đề-nâng-cao-trong-data-fitting)
  - [1.5 Yêu Cầu Cài Đặt Python — Phần 1](#15-yêu-cầu-cài-đặt-python--phần-1)
  - [1.6 Tiêu Chí Đánh Giá — Phần 1](#16-tiêu-chí-đánh-giá--phần-1)
- [Phần 2: Ứng Dụng Data Fitting vào Dữ Liệu Thực Tế](#phần-2-ứng-dụng-data-fitting-vào-dữ-liệu-thực-tế)
  - [2.1 Tiêu Chí Chọn Bộ Dữ Liệu](#21-tiêu-chí-chọn-bộ-dữ-liệu)
  - [2.2 Tiền Xử Lý Dữ Liệu](#22-tiền-xử-lý-dữ-liệu)
  - [2.3 Xây Dựng và Đánh Giá Mô Hình](#23-xây-dựng-và-đánh-giá-mô-hình)
  - [2.4 Kỹ Thuật Nâng Cao (Tùy Chọn)](#24-kỹ-thuật-nâng-cao-tùy-chọn)
  - [2.5 Yêu Cầu Cài Đặt Python — Phần 2](#25-yêu-cầu-cài-đặt-python--phần-2)
  - [2.6 Tiêu Chí Đánh Giá — Phần 2](#26-tiêu-chí-đánh-giá--phần-2)
- [Phần 3: Yêu Cầu Chung và Hướng Dẫn Nộp Bài](#phần-3-yêu-cầu-chung-và-hướng-dẫn-nộp-bài)
- [Tài Liệu Tham Khảo](#tài-liệu-tham-khảo)

---

## Giới Thiệu Đồ Án

### Mục tiêu tổng quát

Đồ án này tập trung vào hai nhóm nhiệm vụ bổ sung cho nhau:

1. **Lý thuyết và minh họa** — Nắm vững nền tảng toán học của data fitting và phương pháp Ordinary Least Squares (OLS), sau đó minh họa các kết quả lý thuyết bằng code Python tự cài đặt.
2. **Ứng dụng thực tế** — Vận dụng data fitting để phân tích một bộ dữ liệu thực, bao gồm tiền xử lý, xây dựng mô hình hồi quy và đánh giá kết quả một cách có hệ thống.

Sau khi hoàn thành đồ án, sinh viên có khả năng:

- Giải thích và chứng minh các tính chất cốt lõi của OLS (unbiasedness, BLUE, Gauss–Markov).
- Cài đặt pipeline data fitting hoàn chỉnh từ đầu bằng Python, có thể so sánh được với `sklearn.LinearRegression`.
- Phân tích và xử lý bộ dữ liệu thực có missing values, outliers và các vấn đề thực tiễn.
- Đánh giá mô hình một cách toàn diện (hệ số R², residual analysis, cross-validation).

### Các công cụ cho phép sử dụng trong đồ án

- **Python 3.10+**: Ngôn ngữ cài đặt chính.
- **NumPy, SciPy**: Tính toán số; dùng để **kiểm chứng**, không dùng để thay thế cài đặt thuật toán.
- **Pandas**: Đọc, xử lý và thao tác dữ liệu.
- **Matplotlib, Seaborn**: Trực quan hóa dữ liệu và kết quả mô hình.
- **Scikit-learn**: Chỉ dùng để **so sánh** và kiểm chứng kết quả, **không** dùng để cài đặt OLS chính.
- **Jupyter Notebook**: Trình bày toàn bộ thực nghiệm.

> **Lưu ý:** Các hàm như `sklearn.linear_model.LinearRegression`, `numpy.linalg.lstsq` chỉ được dùng để **kiểm chứng** (verification). Phần cài đặt thuật toán chính phải được viết từ đầu dựa trên công thức toán học.

---

## Phần 1: Lý Thuyết Data Fitting và Minh Họa

> **Tóm tắt yêu cầu Phần 1:** Trình bày lại kiến thức đã học về data fitting và OLS. Với mỗi kết quả lý thuyết, sinh viên viết code Python để minh họa và kiểm chứng bằng dữ liệu giả lập (synthetic data).

### 1.1 Bài Toán Data Fitting

#### 1.1.1 Phát biểu bài toán tổng quát

**Định nghĩa 1.1** *(Bài toán Data Fitting).* Cho tập dữ liệu $D = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ với $\mathbf{x}_i \in \mathbb{R}^p$, $y_i \in \mathbb{R}$. Bài toán data fitting là tìm hàm $f : \mathbb{R}^p \to \mathbb{R}$ trong một lớp hàm cho trước sao cho $f$ xấp xỉ tốt nhất ánh xạ từ $\mathbf{x}_i$ đến $y_i$ theo một tiêu chí đã định.

Trong mô hình **hồi quy tuyến tính**, ta giả thiết:

$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \epsilon_i = \mathbf{x}_i^T \beta + \epsilon_i \tag{1}$$

với $\beta = (\beta_0, \beta_1, \ldots, \beta_p)^T \in \mathbb{R}^{p+1}$ là vector tham số cần ước lượng và $\epsilon_i$ là nhiễu ngẫu nhiên. Viết dưới dạng ma trận với $X \in \mathbb{R}^{n \times (p+1)}$ (ma trận design có cột đầu toàn 1):

$$\mathbf{y} = X\beta + \varepsilon \tag{2}$$

#### 1.1.2 Các Giả Thiết Gauss–Markov

| Ký hiệu | Nội dung |
|---------|----------|
| **GM1** | **Tuyến tính:** $\mathbf{y} = X\beta + \varepsilon$ |
| **GM2** | **Không hoàn hảo đa cộng tuyến:** $\text{rank}(X) = p + 1$ (các cột độc lập tuyến tính) |
| **GM3** | **Ngoại sinh:** $E[\varepsilon \mid X] = 0$, tức $E[\epsilon_i \mid \mathbf{x}_i] = 0$ |
| **GM4** | **Đồng phương sai:** $\text{Var}(\varepsilon \mid X) = \sigma^2 I_n$, tức $\text{Var}(\epsilon_i) = \sigma^2$ và $\text{Cov}(\epsilon_i, \epsilon_j) = 0$ với $i \neq j$ |
| **GM5** | **Phần dư Chuẩn:** $\varepsilon \mid X \sim \mathcal{N}(0, \sigma^2 I_n)$ |

---

### 1.2 Phương Pháp Ordinary Least Squares (OLS)

#### 1.2.1 Hàm mất mát và nghiệm OLS

OLS tìm $\hat{\beta}$ tối thiểu hóa tổng bình phương phần dư (Residual Sum of Squares):

$$\text{RSS}(\beta) = \|\mathbf{y} - X\beta\|_2^2 = \sum_{i=1}^n (y_i - \mathbf{x}_i^T \beta)^2 \tag{3}$$

**Định lý 1.1** *(Nghiệm OLS — Normal Equations).* Nếu $X^T X$ khả nghịch, nghiệm OLS duy nhất là:

$$\hat{\beta}_{\text{OLS}} = (X^T X)^{-1} X^T \mathbf{y} \tag{4}$$

**Chứng minh.** Tính đạo hàm và cho đạo hàm bằng không:

$$\nabla_\beta \text{RSS} = -2X^T(\mathbf{y} - X\beta) = 0$$

Sau đó giải ra: $X^T X \beta = X^T \mathbf{y}$. $\blacksquare$

#### 1.2.2 Ma Trận Chiếu và Hat Matrix

**Định nghĩa 1.2** *(Hat Matrix).* Ma trận chiếu (projection matrix hay hat matrix) là:

$$H = X(X^T X)^{-1} X^T \in \mathbb{R}^{n \times n} \tag{5}$$

**Mệnh đề 1.1** *(Tính chất của* $H$*)*:

(i) $H^2 = H$ (idempotent)  
(ii) $H^T = H$ (đối xứng)  
(iii) Giá trị riêng của $H$: chỉ là 0 hoặc 1  
(iv) $\text{rank}(H) = p + 1$  
(v) Giá trị fitted: $\hat{\mathbf{y}} = H\mathbf{y}$; phần dư: $\hat{\varepsilon} = (I - H)\mathbf{y}$

#### 1.2.3 Định Lý Gauss–Markov

**Định lý 1.2** *(Gauss–Markov).* Dưới các giả thiết GM1–GM4, ước lượng OLS $\hat{\beta}_{\text{OLS}}$ là ước lượng tuyến tính không chệch tốt nhất (**Best Linear Unbiased Estimator — BLUE**):

(i) **Không chệch:** $E[\hat{\beta}_{\text{OLS}}] = \beta$  
(ii) **Tốt nhất (phương sai nhỏ nhất):** Với mọi ước lượng tuyến tính không chệch $\tilde{\beta}$ khác, ta có $\text{Var}(\tilde{\beta}_j) \geq \text{Var}(\hat{\beta}_j^{\text{OLS}})$ với mọi $j$.

Ma trận hiệp phương sai của $\hat{\beta}_{\text{OLS}}$:

$$\text{Var}(\hat{\beta}_{\text{OLS}} \mid X) = \sigma^2 (X^T X)^{-1} \tag{6}$$

#### 1.2.4 Ước Lượng Phương Sai Nhiễu

Ước lượng không chệch của $\sigma^2$:

$$\hat{\sigma}^2 = \frac{\text{RSS}}{n - p - 1} = \frac{\|\mathbf{y} - X\hat{\beta}\|^2}{n - p - 1} \tag{7}$$

---

### 1.3 Đánh Giá Mô Hình

#### 1.3.1 Hệ số xác định R² và R² hiệu chỉnh

**Định nghĩa 1.3** *(Hệ số xác định).* Hệ số xác định R² được định nghĩa như sau:

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}, \quad R^2 \in [0, 1] \tag{8}$$

R² luôn tăng khi thêm biến. Để so sánh các mô hình có số biến khác nhau, ta hay dùng R² hiệu chỉnh:

$$\bar{R}^2 = 1 - \frac{n-1}{n-p-1}(1 - R^2) \tag{9}$$

#### 1.3.2 Kiểm Định Giả Thuyết

Dưới giả thiết chuẩn GM5, $\hat{\beta} \sim \mathcal{N}(\beta, \sigma^2(X^T X)^{-1})$.

**Kiểm định Student (t-test)** — Kiểm định ý nghĩa của từng đặc trưng đối với mô hình:

$$t_j = \frac{\hat{\beta}_j}{\hat{\sigma}\sqrt{[(X^T X)^{-1}]_{jj}}} \sim t_{n-p-1} \quad (\text{với } H_0 : \beta_j = 0) \tag{10}$$

**Kiểm định F cho mô hình tổng thể** — Kiểm định ý nghĩa của mô hình:

$$F = \frac{(\text{TSS} - \text{RSS})/p}{\text{RSS}/(n - p - 1)} \sim F_{p,\, n-p-1} \quad (\text{với } H_0 : \beta_1 = \cdots = \beta_p = 0) \tag{11}$$

---

### 1.4 Các Vấn Đề Nâng Cao trong Data Fitting

#### 1.4.1 Đa cộng tuyến (Multicollinearity)

Đa cộng tuyến xảy ra khi các cột của $X$ có tương quan cao, khiến $X^T X$ gần suy biến. Để phát hiện hiện tượng này, ta sử dụng hệ số phóng đại phương sai (Variance Inflation Factor):

$$\text{VIF}_j = \frac{1}{1 - R_j^2} \tag{12}$$

trong đó $R_j^2$ là R² khi hồi quy biến $X_j$ theo các biến còn lại. **VIF > 10** cho thấy đa cộng tuyến nghiêm trọng.

#### 1.4.2 Hồi Quy Ridge và Lasso (Regularization)

Khi dữ liệu có nhiều đặc trưng hoặc đa cộng tuyến, ta cần thêm thành phần chính quy hóa (regularization):

**Ridge Regression ($L_2$):**

$$\hat{\beta}_{\text{ridge}} = \arg\min_\beta \{\|\mathbf{y} - X\beta\|^2 + \lambda\|\beta\|_2^2\} = (X^T X + \lambda I)^{-1} X^T \mathbf{y} \tag{13}$$

**Lasso Regression ($L_1$):**

$$\hat{\beta}_{\text{lasso}} = \arg\min_\beta \{\|\mathbf{y} - X\beta\|^2 + \lambda\|\beta\|_1\} \tag{14}$$

Lasso không có nghiệm closed-form; giải bằng coordinate descent hoặc các phương pháp dưới gradient (subgradient methods).

#### 1.4.3 Phân Tích Phần Dư (Residual Analysis)

Sử dụng các công cụ thống kê mô tả để kiểm tra sai số của mô hình:

- **Residuals vs Fitted**: Kiểm tra tính tuyến tính và đồng phương sai.
- **Q-Q Plot**: Kiểm tra tính chuẩn của phần dư.
- **Scale-Location**: Kiểm tra phương sai đồng đều (homoscedasticity).
- **Cook's Distance**: Xác định các quan sát có ảnh hưởng lớn (influential points).

#### 1.4.4 Cross-Validation và Lựa Chọn Mô Hình

**k-Fold Cross-Validation:** Chia dữ liệu thành $k$ phần bằng nhau. Mỗi lần dùng $k-1$ phần để huấn luyện, 1 phần để kiểm tra. Lặp $k$ lần và lấy trung bình:

$$CV_{(k)} = \frac{1}{k} \sum_{i=1}^k \text{MSE}_i \tag{15}$$

**Tiêu chí lựa chọn mô hình** — AIC (Akaike Information Criterion) và BIC (Bayesian Information Criterion):

$$\text{AIC} = n\ln\!\left(\frac{\text{RSS}}{n}\right) + 2(p+2), \qquad \text{BIC} = n\ln\!\left(\frac{\text{RSS}}{n}\right) + (p+2)\ln n \tag{16}$$

---

### 1.5 Yêu Cầu Cài Đặt Python — Phần 1

Với mỗi mục dưới đây, sinh viên phải: **(a)** trình bày công thức toán học, **(b)** cài đặt Python từ đầu, **(c)** minh họa bằng dữ liệu giả lập, **(d)** kiểm chứng với NumPy/sklearn.

1. `ols_fit(X, y)` — Tính $\hat{\beta} = (X^T X)^{-1} X^T \mathbf{y}$ và $\hat{\sigma}^2$.
2. `hat_matrix(X)` — Tính $H = X(X^T X)^{-1} X^T$, kiểm tra idempotent.
3. `model_metrics(y, y_hat, p)` — Tính RSS, TSS, R², $\bar{R}^2$, kiểm định F.
4. `coef_inference(X, y, beta_hat, sigma2)` — Tính standard errors, t-statistics, p-values và khoảng tin cậy 95%.
5. `vif(X)` — Tính VIF cho từng biến.
6. `ridge_fit(X, y, lam)` — Cài đặt Ridge Regression, vẽ ridge trace.
7. `residual_plots(X, y, beta_hat)` — Vẽ 4 biểu đồ phân tích phần dư.
8. `kfold_cv(X, y, k)` — Cài đặt k-fold cross-validation, tính CV score.
9. **Minh họa định lý Gauss–Markov**: Mô phỏng Monte Carlo để kiểm chứng $E[\hat{\beta}] = \beta$ và OLS có phương sai nhỏ nhất.

---

### 1.6 Tiêu Chí Đánh Giá — Phần 1

| Tiêu chí | Mô tả | Điểm |
|----------|-------|------|
| Trình bày lý thuyết OLS | Đúng, đầy đủ công thức, có chứng minh | 1.0 |
| Cài đặt OLS từ đầu | Đúng, kiểm chứng với NumPy | 1.0 |
| Hat matrix và tính chất | Cài đặt, kiểm tra idempotent | 0.5 |
| Kiểm định hệ số (t, F) | Tính đúng t-stat, p-value | 0.5 |
| Regularization (Ridge/Lasso) | Cài đặt, vẽ ridge trace | 1.0 |
| Phân tích phần dư | 4 biểu đồ đầy đủ, nhận xét | 0.5 |
| Cross-validation | Cài k-fold CV, so sánh mô hình | 0.5 |
| Minh họa Gauss–Markov | Monte Carlo rõ ràng, nhận xét | 0.5 |
| Trình bày Notebook | Rõ ràng, có markdown giải thích | 0.5 |
| **Tổng Phần 1** | | **6.0** |

---

## Phần 2: Ứng Dụng Data Fitting vào Dữ Liệu Thực Tế

> **Tóm tắt yêu cầu Phần 2:** Chọn ít nhất một bộ dữ liệu thực có missing values, thực hiện tiền xử lý, áp dụng data fitting để giải bài toán hồi quy, đánh giá và phân tích kết quả.

### 2.1 Tiêu Chí Chọn Bộ Dữ Liệu

Bộ dữ liệu phải thỏa mãn đồng thời các điều kiện:

1. **Dữ liệu thực (real-world):** Thu thập từ quan sát thực tế, không phải dữ liệu tổng hợp (synthetic) hay dữ liệu toy (ví dụ: không dùng Iris, Boston Housing từ sklearn).
2. **Có missing values:** Dữ liệu gốc phải chứa ít nhất một cột có giá trị bị thiếu (≥ 5% dữ liệu bị thiếu để có ý nghĩa xử lý).
3. **Biến mục tiêu liên tục:** Bài toán hồi quy (regression), không phải phân loại (classification).
4. **Kích thước hợp lý:** $n \geq 200$ quan trắc, $p \geq 3$ biến đặc trưng.
5. **Nguồn đáng tin cậy:** Kaggle, UCI ML Repository, data.gov, World Bank, v.v.

**Gợi ý bộ dữ liệu tham khảo:**

- **Kaggle — House Prices** (kaggle.com/c/house-prices-advanced-regression-techniques): Dự đoán giá nhà với 79 biến, nhiều missing values.
- **UCI — Auto MPG**: Dự đoán mức tiêu hao nhiên liệu của xe hơi.
- **UCI — Bike Sharing Dataset**: Dự đoán số lượng xe đạp cho thuê.
- **World Bank Open Data**: Dữ liệu kinh tế vĩ mô theo quốc gia và năm.
- **WHO Global Health Observatory**: Dữ liệu sức khỏe toàn cầu.
- **OECD Data**: Dữ liệu giáo dục, lao động, kinh tế.

Sinh viên được khuyến khích tự chọn bộ dữ liệu phù hợp với sở thích và chuyên ngành.

---

### 2.2 Tiền Xử Lý Dữ Liệu

#### 2.2.1 Khảo Sát Dữ Liệu (Exploratory Data Analysis — EDA)

Trước khi xử lý, sinh viên phải thực hiện EDA bao gồm:

- Thống kê mô tả: mean, median, std, min, max, quartiles.
- Phân phối từng biến: histogram, boxplot.
- Ma trận tương quan: heatmap.
- Kiểm tra dữ liệu trùng lắp.
- Phân tích missing values: tỉ lệ thiếu theo từng cột.
- Phát hiện outliers: phương pháp IQR, z-score hoặc tự định nghĩa ngưỡng lọc outliers.

#### 2.2.2 Xử Lý Missing Values

| Phương pháp | Mô tả |
|-------------|-------|
| **MV1. Listwise deletion** | Xóa toàn bộ hàng có ít nhất một giá trị thiếu. Đơn giản nhưng gây mất thông tin. |
| **MV2. Mean/Median/Mode imputation** | Thay giá trị thiếu bằng thống kê của cột: $x_{ij}^{\text{imputed}} = \bar{x}_j$ hoặc $\text{median}(x_j)$ |
| **MV3. Regression imputation** | Dự đoán giá trị thiếu bằng cách hồi quy biến đó theo các biến còn lại. |
| **MV4. k-NN imputation** | Thay giá trị thiếu bằng trung bình của $k$ quan sát gần nhất theo khoảng cách Euclidean. |
| **MV5. Multiple Imputation (MICE)** | Tạo nhiều bản sao dữ liệu đã điền, phân tích từng bản, gộp kết quả theo quy tắc Rubin. |

> **Lưu ý:** Sinh viên cần giải thích lý do chọn phương pháp xử lý missing values cụ thể cho bộ dữ liệu của mình, dựa trên cơ chế thiếu dữ liệu: **MCAR** (Missing Completely At Random), **MAR** (Missing At Random) hay **MNAR** (Missing Not At Random).

#### 2.2.3 Các Bước Tiền Xử Lý Khác

- **Feature engineering:** Tạo biến mới, biến đổi phi tuyến (log, $\sqrt{\cdot}$, polynomial features).
- **Encoding biến phân loại:** One-hot encoding hoặc ordinal encoding.
- **Chuẩn hóa (normalization/standardization):** Z-score standardization:

$$x_j^{\text{std}} = \frac{x_j - \bar{x}_j}{s_j} \tag{17}$$

- **Phát hiện và xử lý outliers:** Winsorization hoặc loại bỏ có căn cứ.
- **Kiểm tra đa cộng tuyến:** VIF trước khi đưa vào mô hình.

---

### 2.3 Xây Dựng và Đánh Giá Mô Hình

#### 2.3.1 Quy trình xây dựng mô hình

```
EDA → Tiền xử lý → Train/Test Split → Xây dựng mô hình
     → Đánh giá → Tinh chỉnh → Báo cáo kết quả
                       ↑               |
                       └───────────────┘ (Điều chỉnh lại nếu cần)
```

#### 2.3.2 Các Mô Hình Cần Thử Nghiệm

Sinh viên xây dựng và so sánh **ít nhất 3 mô hình**:

| Mô hình | Loại | Mô tả |
|---------|------|-------|
| OLS cơ bản | Bắt buộc | Hồi quy với tất cả các biến (sau tiền xử lý) |
| OLS chọn biến | Bắt buộc | Loại bỏ biến dựa trên p-value hoặc VIF |
| Ridge / Lasso | Bắt buộc | Regularization, chọn $\lambda$ qua CV |
| Polynomial / Interaction | Tùy chọn | Thêm đặc trưng phi tuyến |
| Kernel / Bayesian | Nâng cao | Xem mục 2.4 |

#### 2.3.3 Tiêu Chí So Sánh Mô Hình

Mỗi mô hình được đánh giá trên tập test (không được dùng trong quá trình huấn luyện):

$$\text{MAE} = \frac{1}{n_{\text{test}}} \sum_i |y_i - \hat{y}_i|, \quad \text{RMSE} = \sqrt{\frac{1}{n_{\text{test}}} \sum_i (y_i - \hat{y}_i)^2}, \quad R^2_{\text{test}} = 1 - \frac{\text{RSS}_{\text{test}}}{\text{TSS}_{\text{test}}} \tag{18}$$

---

### 2.4 Kỹ Thuật Nâng Cao (Tùy Chọn)

#### Kernel Regression

Kernel regression mở rộng OLS sang không gian đặc trưng phi tuyến thông qua kernel trick:

$$\hat{y}(\mathbf{x}) = \mathbf{k}(\mathbf{x})^T (K + \lambda I)^{-1} \mathbf{y} \tag{19}$$

với $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ là ma trận Gram và $k(\cdot, \cdot)$ là hàm kernel (RBF, polynomial, v.v.):

$$k_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \exp\!\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2}\right) \tag{20}$$

Sinh viên cài đặt Kernel Ridge Regression và so sánh với OLS thông thường.

#### Bayesian Linear Regression

Bayesian approach đặt prior cho $\beta$:

$$\beta \sim \mathcal{N}(m_0, S_0), \qquad y \mid \mathbf{x}, \beta \sim \mathcal{N}(\mathbf{x}^T \beta, \sigma^2) \tag{21}$$

Phân phối hậu nghiệm (conjugate):

$$\beta \mid X, \mathbf{y} \sim \mathcal{N}(m_n, S_n) \tag{22}$$

$$S_n = \left(S_0^{-1} + \frac{1}{\sigma^2} X^T X\right)^{-1}, \qquad m_n = S_n\!\left(S_0^{-1} m_0 + \frac{1}{\sigma^2} X^T \mathbf{y}\right) \tag{23}$$

**Ưu điểm:** Cho thông tin về *uncertainty quantification* — khoảng tin cậy Bayesian (credible intervals) cho dự đoán. Sinh viên cài đặt và so sánh với OLS frequentist.

---

### 2.5 Yêu Cầu Cài Đặt Python — Phần 2

1. **Pipeline tiền xử lý:** Viết class `DataPipeline` xử lý missing values, encoding, chuẩn hóa theo thứ tự. Phải có thể `fit` trên train, `transform` trên test.
2. **So sánh 3+ mô hình:** Bảng tổng hợp MAE, RMSE, R² trên test set.
3. **Cross-validation:** Dùng k-fold (khuyến nghị $k = 5$ hoặc $k = 10$) để chọn siêu tham số $\lambda$ cho Ridge/Lasso.
4. **Phân tích phần dư:** Với mô hình tốt nhất, vẽ đầy đủ 4 biểu đồ chẩn đoán.
5. **Feature importance:** Vẽ biểu đồ hệ số hồi quy (sau chuẩn hóa) để giải thích mô hình.
6. **Nhận xét và kết luận:** Giải thích kết quả theo ngữ cảnh của bộ dữ liệu.

---

### 2.6 Tiêu Chí Đánh Giá — Phần 2

| Tiêu chí | Mô tả | Điểm |
|----------|-------|------|
| Chọn và mô tả dữ liệu | Đúng tiêu chí, mô tả rõ nguồn gốc | 0.5 |
| EDA | Đầy đủ thống kê mô tả, biểu đồ | 0.5 |
| Xử lý missing values | Đúng phương pháp, có giải thích | 1.0 |
| Tiền xử lý tổng thể | Pipeline đầy đủ, fit/transform đúng | 0.5 |
| Xây dựng ≥ 3 mô hình | OLS, Ridge/Lasso, một mô hình khác | 1.5 |
| Đánh giá trên test set | MAE, RMSE, R², phân tích phần dư | 1.0 |
| Nhận xét và kết luận | Phân tích có chiều sâu, liên hệ thực tế | 0.5 |
| Kỹ thuật nâng cao | Kernel / Bayesian (tùy chọn, **bonus**) | +0.5 |
| **Tổng Phần 2** | | **5.5 (+0.5)** |

> **Lưu ý:** Điểm kỹ thuật nâng cao (Kernel / Bayesian) là điểm **bonus**, tối đa cộng thêm 0.5 điểm vào tổng Phần 2. Điểm tổng đồ án vẫn quy về thang 10.

---

## Phần 3: Yêu Cầu Chung và Hướng Dẫn Nộp Bài

### 3.1 Cấu Trúc Báo Cáo

Báo cáo viết bằng LaTeX hoặc Markdown (xuất ra PDF), bao gồm:

1. **Trang bìa:** Họ và tên, MSSV, nhóm, giảng viên hướng dẫn.
2. **Mục lục.**
3. **Phần 1:** Lý thuyết và minh họa.
4. **Phần 2:** Ứng dụng thực tế.
5. **Kết luận:** Tóm tắt kết quả, bài học rút ra, hướng mở rộng.
6. **Tài liệu tham khảo:** Ít nhất 5 tài liệu.
7. **Phụ lục:** Bảng số liệu, biểu đồ bổ sung (nếu có).

### 3.2 Cấu Trúc Thư Mục Nộp Bài

```
Group_<ID>/
├── README.md
├── requirements.txt
├── report/
│   ├── report.pdf
│   └── report.tex
├── part1/
│   ├── ols_implementation.py   # OLS from scratch
│   ├── ridge_lasso.py
│   ├── residual_analysis.py
│   ├── cross_validation.py
│   └── part1_notebook.ipynb    # Theoretical demo
└── part2/
    ├── data/
    │   └── <ten_dataset>.csv   # Original data
    ├── data_pipeline.py        # Pre-processing
    ├── model_comparison.py     # Model compare
    ├── advanced_methods.py     # Kernel/Bayesian (if have)
    └── part2_notebook.ipynb    # Results analysis and discuss
```

### 3.3 Yêu Cầu Kỹ Thuật

- Sử dụng Python 3.10+, viết code rõ ràng (clean code), chú thích code nếu thật sự cần thiết.
- Tất cả biểu đồ phải có tiêu đề, nhãn trục, chú thích đầy đủ.
- Mọi quyết định (chọn $\lambda$, chọn $k$, xử lý outlier) phải được giải thích bằng lý luận, không phải thử-sai ngẫu nhiên.
- Kết quả phải tái lập được (reproducible): đặt `random_state` / seed cụ thể.
- Mỗi hàm có ít nhất 2 unit test kiểm tra kết quả trên dữ liệu đã biết.

### 3.4 Phân Công Nhóm và Đạo Đức Học Thuật

> **Lưu ý:**
> - Báo cáo phải ghi rõ **phân công công việc** của từng thành viên.
> - Giảng viên sẽ chọn lựa một số nhóm để vấn đáp nếu cần thiết.
> - **Nghiêm cấm** sao chép code hoặc báo cáo từ nhóm khác mà không trích dẫn nguồn.
> - Sử dụng AI (ChatGPT, Copilot, v.v.) để gợi ý là được phép, nhưng phải **hiểu và giải thích được toàn bộ code nộp**.
> - Vi phạm đạo đức học thuật dẫn đến **điểm 0 toàn bộ đồ án**.

### 3.5 Thang Điểm Tổng Hợp

| Phần | Nội dung | Điểm tối đa | Trọng số |
|------|----------|-------------|----------|
| 1 | Lý thuyết, minh họa, cài đặt OLS | 6.0 | 52% |
| 2 | Ứng dụng dữ liệu thực | 5.5 | 48% |
| Bonus | Kỹ thuật nâng cao (Kernel/Bayesian) | +0.5 | — |
| **Tổng cộng** | | **11.5 (+0.5)** | **100%** |

$$\text{Điểm cuối cùng} = \min\!\left(\frac{\text{Tổng}}{1.15},\ 10\right), \quad \text{quy về thang điểm 10}$$

---

### Tóm tắt sản phẩm nộp bài

- [ ] Báo cáo `report.pdf` (bắt buộc)
- [ ] Source code đầy đủ kèm `README.md` và `requirements.txt`
- [ ] Jupyter Notebooks: `part1_notebook.ipynb` và `part2_notebook.ipynb`
- [ ] Dữ liệu gốc: file `.csv` hoặc link download trong README
- [ ] Nộp qua: Moodle của Khoa
- [ ] **Hạn nộp: ngày 30/5/2026, trước 23:59**

---

## Tài Liệu Tham Khảo

[1] Gilbert Strang. *Introduction to Linear Algebra*, 6th ed. Wellesley-Cambridge Press, 2023.

[2] Gareth James, Daniela Witten, Trevor Hastie & Robert Tibshirani. *An Introduction to Statistical Learning*, 2nd ed. Springer, 2021. Truy cập miễn phí: https://www.statlearning.com

[3] Trevor Hastie, Robert Tibshirani & Jerome Friedman. *The Elements of Statistical Learning*, 2nd ed. Springer, 2009. Truy cập miễn phí: https://hastie.su.domains/ElemStatLearn/

[4] Christopher M. Bishop. *Pattern Recognition and Machine Learning*. Springer, 2006. (Chương 3: Linear Models for Regression)

[5] Kevin P. Murphy. *Probabilistic Machine Learning: An Introduction*. MIT Press, 2022. Truy cập miễn phí: https://probml.github.io/pml-book/book1.html

[6] Jake VanderPlas. *Python Data Science Handbook*. O'Reilly, 2016. https://jakevdp.github.io/PythonDataScienceHandbook/

[7] Wes McKinney. *Python for Data Analysis*, 3rd ed. O'Reilly, 2022.