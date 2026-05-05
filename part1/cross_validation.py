import numpy as np

def kfold_cv(X: list[list[float]], y: list[float], k: int, model_fn, **model_kwargs):
    """
    F9: k-Fold Cross-Validation từ đầu.
    """
    X_np = np.array(X)
    y_np = np.array(y)
    n = len(y)
    
    # Shuffle indices
    indices = np.arange(n)
    np.random.seed(42) # RANDOM_STATE
    np.random.shuffle(indices)
    
    folds = np.array_split(indices, k)
    cv_scores = []
    
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        X_train, y_train = X_np[train_idx], y_np[train_idx]
        X_val, y_val = X_np[val_idx], y_np[val_idx]
        
        # Fit model
        model = model_fn(X_train.tolist(), y_train.tolist(), **model_kwargs)
        
        # Predict (Cần viết thêm hàm predict phụ thuộc vào model_fn)
        # y_pred = predict(X_val, model) 
        # score = np.mean((y_val - y_pred)**2)
        # cv_scores.append(score)
        
    return {
        'mean_cv_score': np.mean(cv_scores) if cv_scores else 0.0,
        'cv_scores': cv_scores
    }