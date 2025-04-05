# Synthetic Image Generation for Enhancing Polyp Detection

### Summary of Experiments
| model_name | transformations / architecture | batch_size | dropout | criterion         | optimizer | learning_rate | weight_decay | other techniques | weighted F1-score |
|------------|---------------------------------|------------|---------|-------------------|-----------|---------------|--------------|------------------|-------------------|
| model1     | Resize, Normalize, n_features=256 | 16       | 0.3     | CrossEntropyLoss  | Adam      | 0.0001        | None         | -                | 53,64%            |
| model2     | Resize, Normalize                 | 16       | 0.3     | CrossEntropyLoss  | Adam      | 0.001         | None         | -                | 59,56%            |
| model3     | Resize, Normalize                 | 16       | 0.5     | CrossEntropyLoss  | Adam      | 0.001         | None         | -                | 60,3%             |
| model4     | Resize, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation (30) | 16 | 0.5 | CrossEntropyLoss  | Adam      | 0.001         | None         | -                | 56,87%            |
| model5     | Resize, Normalize, RandomHorizontalFlip, RandomRotation (30) | 16       | 0.5     | CrossEntropyLoss  | Adam      | 0.001         | None         | -                | 59,73%            |
| model6     | Resize, Normalize                 | 16       | 0.5     | CrossEntropyLoss  | Adam      | 0.001         | 1e-5         | -                | 70,10%            |
| model7     | Resize, Normalize                 | 16       | 0.5     | CrossEntropyLoss  | Adam      | 0.001         | 1e-3         | -                | 62,58%            |
| model8     | Resize, Normalize                 | 16       | 0.5     | CrossEntropyLoss  | AdamW     | 0.001         | 1e-5         | -                | 54,8%             |
| model9     | Resize, Normalize                 | 32       | 0.5     | CrossEntropyLoss  | Adam      | 0.001         | 1e-5         | -                | 61,28%            |
| model10    | Resize, Normalize, n_features=128 | 16       | 0.5     | CrossEntropyLoss  | Adam      | 0.001         | 1e-5         | -                | 58,06%            |
| model11    | Resize, Normalize                 | 8        | 0.5     | CrossEntropyLoss  | Adam      | 0.001         | 1e-3         | -                | 59,79%            |
| model12    | Resize, Normalize                 | 16       | 0.5     | CrossEntropyLoss  | Adam      | 0.001         | 1e-3         | Weighted loss    | 72,09%            |
| model13    | Resize, Normalize                 | 16       | 0.5     | CrossEntropyLoss  | Adam      | 0.001         | 1e-3         | Focal loss       | -                 |
| model14    | Resize, Normalize                 | 16       | 0.5     | CrossEntropyLoss  | Adam      | 0.001         | 1e-3         | Weighted sampling | 49,94%           |
| model15    | Resize, Normalize                 | 16       | 0.5     | CrossEntropyLoss  | Adam      | 0.001         | 1e-3         | AD vs rest       | -           |

Best models:
- model7: 62,58%
- model12: 72,09%