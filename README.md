```
.\venv\Scripts\activate
```

```
  py part1/cross_validation.py
  py part1/gauss_markov_demo.py
  py part1/ols_implementation.py
  py part1/residual_analysis.py
  py part1/ridge_lasso.py
```

```
python part2/data_pipeline.py --data part2/data/planet.csv --outdir part2/output
python part2/model_comparison.py --preprocessed part2/output/preprocessed.pkl --outdir part2/output
python part2/advanced_methods.py --preprocessed part2/output/preprocessed.pkl --outdir part2/output --max-train 500
```