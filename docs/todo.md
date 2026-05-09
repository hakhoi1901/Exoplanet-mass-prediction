# Danh sách Công việc Đồ án 2 - Data Fitting (MTH00051)

> **Lưu ý:** Danh sách này được tổng hợp dựa trên yêu cầu gốc của Giáo viên (`docs/project2.md`), đồng thời ánh xạ với các đầu việc (F1-F11, T1-T8) trong biên bản phân công của nhóm (`docs/HopLan1.md`). Ưu tiên các yêu cầu bắt buộc để tối đa hóa điểm số.

## Phần 1: Lý Thuyết và Minh Họa (Trọng số 52% - 6.0đ)

- [x] **0. Cài đặt các thuật toán Part 1 (Đã hoàn thành - Vượt 100% unit test)**
  - [x] **F1:** Cài đặt `ols_fit` (Normal equations, tính $\hat{\beta}, \hat{\sigma}^2$).
  - [x] **F2:** Cài đặt `hat_matrix` (Tính ma trận chiếu $H$, kiểm tra tính idempotent).
  - [x] **F3:** Cài đặt `model_metrics` (Tính R², Adj-R², F-stat, RSS, TSS, MAE, RMSE).
  - [x] **F4:** Cài đặt `coef_inference` (Tính standard error, t-stat, p-value, CI 95%).
  - [x] **F5:** Cài đặt `vif` (Tính Variance Inflation Factor phát hiện đa cộng tuyến).
  - [x] **F6:** Cài đặt `ridge_fit` (Giải Ridge Regression bằng closed-form).
  - [x] **F7:** Cài đặt `lasso_fit` (Giải Lasso Regression bằng Coordinate Descent).
  - [x] **F8:** Cài đặt `residual_plots` (Tính Cook's Distance và chuẩn bị dữ liệu phân tích phần dư).
  - [x] **F9:** Cài đặt `kfold_cv` (K-Fold Cross Validation từ đầu không dùng sklearn).
  - [x] **F10:** Cài đặt hàm mô phỏng Monte Carlo kiểm chứng định lý Gauss-Markov.

- [ ] **1. Hoàn thiện Notebook Phần 1 (`part1/part1_notebook.ipynb`) - Trọng số: 0.5đ (F11)**
  - [ ] **Mục tiêu:** Viết markdown trình bày lý thuyết, công thức cho từng hàm, sau đó gọi code minh họa trên synthetic data, so sánh với `sklearn`/`numpy` (hoặc lý thuyết).
  - [ ] **Demo F1-F5 (OLS cơ bản):**
    - [ ] `ols_fit`: In $\hat{\beta}$, $\hat{\sigma}^2$ và so sánh NumPy.
    - [ ] `hat_matrix`: Hiển thị Heatmap của ma trận chiếu $H$ (với $n \le 20$), vẽ Histogram của giá trị riêng (eigenvalues) để chứng minh chỉ chứa 0 và 1.
    - [ ] `model_metrics`: In bảng đầy đủ R², Adj-R², F-stat, RSS, TSS...
    - [ ] `coef_inference`: In bảng hệ số gồm t-stat, p-value, và CI 95%.
    - [ ] `vif`: Chạy hàm kiểm tra trên bộ dữ liệu có hiện tượng đa cộng tuyến, in bảng VIF.
  - [ ] **Demo F6-F9 (Nâng cao & CV):**
    - [ ] `ridge_fit`: Vẽ biểu đồ Ridge trace ($\lambda$ vs các hệ số).
    - [ ] `lasso_fit`: Vẽ biểu đồ Lasso path (hệ số bị co về 0).
    - [ ] `residual_plots`: Hiển thị 4 biểu đồ phân tích phần dư + dòng nhận xét.
    - [ ] `kfold_cv`: Vẽ đồ thị $\lambda$ theo CV score (log scale) để chọn $\lambda$ tối ưu.
  - [ ] **Demo Gauss-Markov (F10):**
    - [ ] Thực hiện mô phỏng Monte Carlo ($N=1000$).
    - [ ] Vẽ Histogram của $\hat{\beta}$ và so sánh phương sai của OLS với một estimator khác để chứng tỏ OLS là BLUE.

## Phần 2: Ứng Dụng Dữ Liệu Thực Tế (Trọng số 48% - 5.5đ + 0.5đ Bonus)

- [ ] **2. Chọn và Tải Dataset (T1) - Trọng số: 0.5đ**
  - [ ] Tìm bộ dữ liệu thực tế (Ví dụ trên Kaggle, UCI): $n \ge 200$, $p \ge 3$, biến mục tiêu liên tục (Regression).
  - [ ] **Bắt buộc:** Phải có ít nhất 1 cột chứa $\ge$ 5% dữ liệu bị thiếu (missing values).
  - [ ] Đặt file vào thư mục `part2/data/`.

- [ ] **3. Khảo Sát Dữ Liệu - EDA (T2) - Trọng số: 0.5đ**
  - [ ] *Thực hiện trong `part2_notebook.ipynb`:*
  - [ ] Thống kê mô tả (mean, median, quartile...).
  - [ ] Vẽ Histogram, Boxplot cho từng biến số.
  - [ ] Vẽ Heatmap ma trận tương quan và Scatter plot giữa Top 5 features với Target.
  - [ ] Kiểm tra giá trị thiếu (`missing_report`) và outliers (`detect_outliers_iqr`).

- [ ] **4. Xây Dựng Tiền Xử Lý - DataPipeline (T3) - Trọng số: 1.5đ**
  - [ ] *Thực hiện trong `part2/data_pipeline.py`:* Cài đặt class `DataPipeline` với `fit()`, `transform()`, `fit_transform()`.
  - [ ] **Xử lý Missing values (1.0đ):** Cài đặt thuật toán điền khuyết và **giải thích lý do** lựa chọn (ví dụ: MCAR, MAR, MNAR).
  - [ ] Xử lý Outlier, One-hot encoding, và Chuẩn hóa Z-score.
  - [ ] **Rất quan trọng:** Tránh Data Leakage (Các tham số scale, mean... chỉ được tính từ tập Train).

- [ ] **5. Xây Dựng và So Sánh Mô Hình (T4) - Trọng số: 1.5đ**
  - [ ] *Thực hiện trong `part2/model_comparison.py` và gọi trên Notebook:*
  - [ ] Tách Train/Test set (80/20, shuffle=True).
  - [ ] Áp dụng pipeline lên train và test.
  - [ ] Chạy và đánh giá ít nhất 3 mô hình:
    - [ ] (1) OLS với toàn bộ đặc trưng.
    - [ ] (2) OLS chọn biến (lọc bằng p-value > 0.05 hoặc VIF > 10).
    - [ ] (3) Ridge / Lasso Regression (siêu tham số $\lambda$ phải được chọn qua k-Fold CV).
  - [ ] Lập bảng so sánh kết quả (`compare_models`): Dựa trên MAE, RMSE, R² trên Test set.

- [ ] **6. Đánh giá chuyên sâu (T5)**
  - [ ] **Feature Importance:** Vẽ biểu đồ các hệ số (đã chuẩn hóa) để nhận xét biến nào ảnh hưởng mạnh nhất.
  - [ ] **Phân tích phần dư:** Dùng `residual_plots` cho mô hình TỐT NHẤT. Nhận xét phân tích 4 biểu đồ.

- [ ] **7. Kỹ Thuật Nâng Cao (T6) - Trọng số: +0.5đ Bonus**
  - [ ] *Thực hiện trong `part2/advanced_methods.py` (Chỉ cần chọn 1):*
  - [ ] Cài đặt Kernel Ridge Regression (RBF Kernel).
  - [ ] **HOẶC** Cài đặt Bayesian Linear Regression.
  - [ ] So sánh mô hình trên với OLS.

## Phần 3: Hoàn Thiện Project và Báo Cáo

- [ ] **8. Setup & Codebase**
  - [ ] Viết `README.md` rõ ràng (hướng dẫn tải data, chạy lệnh).
  - [ ] Cập nhật file `requirements.txt`.
  - [ ] Đặt Seed (`RANDOM_STATE = 42`) ở mọi nơi cần random để kết quả có thể tái lập (reproducible).

- [ ] **9. Báo Cáo Cuối Cùng (`report/report.pdf` & `report/report.tex`)**
  - [ ] Trang bìa chuẩn Form của trường/khoa. **Bắt buộc có phần Bảng Phân Công Công Việc nhóm.**
  - [ ] Mục lục đầy đủ.
  - [ ] Nêu bật các phương trình toán học và chứng minh lý thuyết liên quan ở Phần 1.
  - [ ] Chụp/copy biểu đồ kết quả EDA, Model Comparison vào Phần 2.
  - [ ] Đưa ra Kết luận cuối cùng và Bài học rút ra.
  - [ ] Trích dẫn tối thiểu 5 Tài liệu tham khảo theo đúng chuẩn.
  - [ ] Biên dịch LaTeX sang file `.pdf`.

- [ ] **10. Kiểm tra lần cuối (Trước 23:59 ngày 30/05/2026)**
  - [ ] Đảm bảo code clean, `RANDOM_STATE` = 42 cố định ở mọi nơi để reproducible.
  - [ ] Dọn dẹp các branch, hợp nhất vào `main`.
  - [ ] Đóng gói theo chuẩn thư mục `Group_<ID>`.
