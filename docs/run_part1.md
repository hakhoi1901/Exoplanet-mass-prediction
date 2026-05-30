# Hướng Dẫn Sinh Ảnh (Plots) Cho Phần 1

Để báo cáo Phần 1 thêm sinh động, bạn cần vẽ các biểu đồ phân tích (Gauss-Markov, Residuals, Ridge/Lasso, CV). Bạn có 2 cách để sinh ảnh:

## Cách 1: Chạy tự động tất cả
Mình đã tạo sẵn file `generate_plots.py` trong thư mục `part1`. File này gom toàn bộ các lệnh sinh ảnh lại với nhau. Bạn chỉ cần mở terminal ở thư mục gốc của project và chạy lệnh sau:

```bash
python part1/generate_plots.py
```

Code sẽ tự động chạy tất cả các hàm và lưu toàn bộ 5 ảnh (.png) vào thư mục `part1/output/`.

## Cách 2: Chạy riêng từng hàm (Dùng trong Jupyter Notebook)
Nếu bạn muốn dùng Jupyter Notebook để chạy và hiển thị ảnh, bạn có thể import và gọi hàm theo các đoạn code mẫu dưới đây:

### 1. Gauss-Markov (Biểu đồ phân phối của OLS và Alt)
```python
from part1.gauss_markov_demo import monte_carlo_gauss_markov, plot_beta_histograms

res = monte_carlo_gauss_markov(n_sim=1000, n_obs=100)
plot_beta_histograms(res["beta_ols_all"], res["beta_alt_all"], res["true_beta"], save_dir="part1/output")
```

### 2. Residual Analysis (4 biểu đồ phần dư)
```python
from test_utils import make_linear_data
from part1.ols_implementation import ols_fit
from part1.residual_analysis import residual_plots

X, y = make_linear_data(n=200, beta=[2.0, 3.0, -1.5], sigma=1.0)
res = ols_fit(X, y)
residual_plots(X, y, res["beta_hat"], save_dir="part1/output")
```

### 3. Ridge Trace & Lasso Path (Đường đi của hệ số hồi quy)
```python
from test_utils import make_linear_data
from part1.ridge_lasso import ridge_trace, lasso_trace

X, y = make_linear_data(n=100, beta=[1.0, 0.5, -0.5, 2.0], sigma=1.0)
ridge_trace(X, y, save_dir="part1/output")
lasso_trace(X, y, save_dir="part1/output")
```

### 4. Cross Validation (Chọn Lambda theo k-Fold CV)
```python
from test_utils import make_linear_data
from part1.ridge_lasso import ridge_fit, ridge_predict
from part1.cross_validation import cv_lambda_search

X, y = make_linear_data(n=100, beta=[1.0, 0.5, -0.5, 2.0], sigma=1.0)

def _predict(X_val, model):
    return ridge_predict(X_val, model["beta_hat"], model["mean_X"], model["std_X"])

cv_lambda_search(X, y, model_fn=ridge_fit, predict_fn=_predict, k=5, save_dir="part1/output")
```
