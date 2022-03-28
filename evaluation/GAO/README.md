# GAO Benchmark

## Evaluation

1. Add your model predictions to `configs/methods.yaml` as follows:
```python
# Method name should NOT contain string "random"
'Your Method': PATH_TO_PREDICTIONS
```
2. Evaluate the predictions using

```bash
python evaluate.py --split val/test --config configs/methods.yaml --verbose
```