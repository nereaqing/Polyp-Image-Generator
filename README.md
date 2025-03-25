# Synthetic Image Generation for Enhancing Polyp Detection

### Summary of Experiments
| model_name | transformations | batch_size | dropout | criterion | optimizer | learning_rate | weight_decay | Weighted F1-score |
|------------|-----------------|------------|---------|-----------|-----------|---------------|--------------|----------|
| model1     | Resize, Normalize | 16       |  0.3    | CrossEntropyLoss  | Adam  | 0.0001    | None  | 53,64% |
| model2     | Resize, Normalize | 16       |  0.3    | CrossEntropyLoss  | Adam  | 0.001    | None  | 59,56% |
| model3     | Resize, Normalize | 16       |  0.5    | CrossEntropyLoss  | Adam  | 0.001    | None  | 60,3% |
| model4     | Resize, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation (30) | 16       |  0.5    | CrossEntropyLoss  | Adam  | 0.001    | None  | 56,87% |
| model5     | Resize, Normalize, RandomHorizontalFlip, RandomRotation (30) | 16       |  0.5    | CrossEntropyLoss  | Adam  | 0.001    | None  | - |
| model6     | Resize, Normalize, RandomHorizontalFlip, RandomRotation (30) | 16       |  0.5    | CrossEntropyLoss  | Adam  | 0.001    | 1e-5  | - |
| model7     | Resize, Normalize, RandomHorizontalFlip, RandomRotation (30) | 16       |  0.5    | CrossEntropyLoss  | Adam  | 0.001    | 1e-3  | - |
| model8     | Resize, Normalize, RandomHorizontalFlip, RandomRotation (30) | 32       |  0.5    | CrossEntropyLoss  | Adam  | 0.001    | 1e-3  | - |
| model8     | Resize, Normalize, RandomHorizontalFlip, RandomRotation (30) | 32       |  0.5    | CrossEntropyLoss  | AdamW  | 0.001    | 1e-3  | - |