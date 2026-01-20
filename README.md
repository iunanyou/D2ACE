# D2ACE
D2ACE: Multi-Label Batch Selection Guided by Dual Dynamics and Adaptive Correlation Enhancement

# Package:
Python==3.12.4 numpy==1.26.4 scikit-multilearn==0.2.0 torch==2.4.1

# Usage:
## datasets
Multi-label datasets can be downloaded from https://mulan.sourceforge.net/datasets-mlc.html.
Create a data folder in the current directory, and place the downloaded dataset files into it.

## run main.py
In detail, we can train on different datasets:

**dataset used**
```python
path_to_arff_files = ["cal500", "birds", "enron", "scene", "yeast", "Corel5k", "rcv1subset1", "rcv1subset2", "rcv1subset3", "bibtex", "yahoo-Arts1", "yahoo-Business1", "mediamill"]
```

**label count of corresponding dataset:**
```python
label_counts = [174, 19, 53, 6, 14, 374, 101, 101, 101, 159, 25, 28, 101]
```

**feature retention ratio of the corresponding dataset:**
```python
select_feature = [1, 1, 1, 1, 1, 1, 0.02, 0.02, 0.02, 1, 0.05, 0.05, 1]
```

# Key code:
1. uncertainty metric

```python
def update_H(H, y_pred, ids, max_history_length=5):
    y_pred_numpy = y_pred.detach().cpu().numpy()
    for i, idx in enumerate(ids):
        if idx not in H:
            H[idx] = deque(maxlen=max_history_length) 
        H[idx].append(y_pred_numpy[i])  
    return H

def update_E(H,E,ids,label_dim):
    for idx in ids:
        current_predictions_history = np.array(H[idx])
        last_row_index = len(current_predictions_history) - 1
        for j in range(label_dim): 
            diffs = np.abs(np.diff(current_predictions_history[:, j]))
            mean_diffs = np.sum(diffs)/len(diffs)
            current_entropy = -1 / np.log(2) * (current_predictions_history[last_row_index][j]  * np.log(current_predictions_history[last_row_index][j]) + (1 - current_predictions_history[last_row_index][j] ) * np.log(1 -current_predictions_history[last_row_index][j] ))
            E[idx,j]=1/2*mean_diffs+1/2* current_entropy
    return E
```



2. hardness metric

```python
def update_ema_flip(ema_flip, H, ids, label_dim, alpha=0.7):
    """
    update the ema_flip matrix (with dimensions ins_dim x label_dim)
    H: a dictionary mapping an index (idx) to a deque containing recent prediction vectors (either probabilities or 0/1 binary values)
    alpha: the EMA smoothing factor
    """
    for idx in ids:
        hist = np.array(H[idx])  # shape (history_len, label_dim)
        if hist.shape[0] >= 2:
            prev = hist[-2]
            last = hist[-1]
            flip = np.abs(last - prev)  # in {0,1} if preds are 0/1, but can be float if probs
        else:
            flip = np.zeros(label_dim)
        # ema update
        ema_flip[idx, :] = alpha * flip + (1 - alpha) * ema_flip[idx, :]
    return ema_flip

h_mat = loss * (1.0 - ema_flip_np)
```

