# Project 2 - Data Fitting

Đồ án gồm hai phần chính:

- `part1/`: cài đặt các thuật toán nền tảng từ đầu như OLS, Hat Matrix, kiểm định hệ số, VIF, Ridge, Lasso, Cross-Validation và Residual Analysis.
- `part2/`: áp dụng data fitting lên dữ liệu ngoại hành tinh, gồm pipeline tiền xử lý, so sánh mô hình tuyến tính và Kernel Ridge Regression bonus.

## Cấu Trúc Nhanh

```text
part1/                  Code lý thuyết và unit tests Phần 1
part1/output/           Hình minh họa sinh từ Phần 1
part2/data/             Dữ liệu đầu vào Phần 2
part2/output/           Output pipeline, model và biểu đồ Phần 2
report/                 Source LaTeX và PDF báo cáo
docs/project2.md        Đề bài gốc
```

## Cài Đặt Môi Trường

Mở terminal tại thư mục gốc project, sau đó kích hoạt virtual environment và cài thư viện:

```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
```

Nếu chưa tạo `venv`, có thể tạo bằng:

```powershell
python -m venv venv
```

## Part 1

Part 1 dùng dữ liệu giả lập để kiểm chứng công thức toán học và so sánh nhanh với thư viện chuẩn khi cần. Mỗi module chính đều có test nhúng ở cuối file.

### Chạy Toàn Bộ Test

```powershell
python -m pytest -q
```

Lệnh này gọi các test bridge trong `part1/test_ols_implementation.py`, rồi chạy các bộ test đã nhúng ở cuối từng file Part 1.
Nếu môi trường chưa cài `pytest`, có thể chạy trực tiếp từng file ở mục dưới; các file này đều tự gọi `run_tests()`.

### Chạy Test Riêng Từng File Part 1

Các file Part 1 đều có test ở cuối file và dùng `TestLogger` từ `test_utils.py`.

```powershell
python part1\ols_implementation.py
python part1\ridge_lasso.py
python part1\cross_validation.py
python part1\residual_analysis.py
python part1\gauss_markov_demo.py
```

Các hình minh họa của Part 1 được lưu trong:

```text
part1\output
```

## Part 2

Part 2 dùng file dữ liệu chính:

```text
part2\data\data.csv
```

Pipeline sẽ lọc schema, áp dụng domain restriction, chia train/test, xử lý missing bằng MICE, winsorization, chuẩn hóa và lọc VIF. Sau đó `model_comparison.py` huấn luyện OLS full, OLS selected, Ridge và Lasso; `advanced_methods.py` chạy Kernel Ridge Regression bonus.

### Chạy End-To-End

```powershell
python part2\data_pipeline.py --data part2\data\data.csv --outdir part2\output --random-state 42
python part2\model_comparison.py --preprocessed part2\output\preprocessed.pkl --outdir part2\output
python part2\advanced_methods.py --preprocessed part2\output\preprocessed.pkl --outdir part2\output --max-train 867
```

Output kiểm tra sẽ nằm trong:

```text
part2\output
```

Các file kết quả chính:

```text
part2\output\preprocessed.pkl
part2\output\model_results.json
part2\output\model_summary.csv
part2\output\advanced_results.json
part2\output\kernel_ridge_search.csv
```

## Notebook và Báo Cáo

Notebook chính:

```text
part1\part1_notebook.ipynb
part2\part2_notebook.ipynb
```

Báo cáo LaTeX nằm trong `report/`. Để build PDF, chạy trong thư mục `report`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```
