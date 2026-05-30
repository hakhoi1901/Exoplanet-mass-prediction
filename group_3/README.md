# Project 2 - Data Fitting

Mở terminal tại thư mục gốc project:

Dùng virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Part 1
### Chạy Toàn Bộ Test

```powershell
python -m pytest -q
```

Lệnh này gọi các test bridge trong `part1/test_ols_implementation.py`, rồi chạy các bộ test đã nhúng ở cuối từng file Part 1.

### Chạy Test Riêng Từng File Part 1

Các file Part 1 đều có test ở cuối file và dùng `TestLogger` từ `test_utils.py`.

```powershell
python part1\ols_implementation.py
python part1\ridge_lasso.py
python part1\cross_validation.py
python part1\residual_analysis.py
python part1\gauss_markov_demo.py
```

## Kiểm Tra Part 2 End-To-End

```powershell
python part2\data_pipeline.py --data part2\data\data.csv --outdir part2\output --random-state 42
python part2\model_comparison.py --preprocessed part2\output\preprocessed.pkl --outdir part2\output
python part2\advanced_methods.py --preprocessed part2\output\preprocessed.pkl --outdir part2\output --max-train 200
```

Output kiểm tra sẽ nằm trong:

```text
part2\output
```
