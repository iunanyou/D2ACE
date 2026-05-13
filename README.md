# 1. D2ACE

D2ACE: Multi-Label Batch Selection Guided by Dual Dynamics and Adaptive Correlation Enhancement
Accpeted by IJCAI-ECAI 2026
https://arxiv.org/abs/2605.09400


# 2. Package:

Python==3.12.4 numpy==1.26.4 scikit-multilearn==0.2.0 torch==2.4.1



# 3. Usage:

## 3.1 Datasets

Multi-label datasets can be downloaded from https://mulan.sourceforge.net/datasets-mlc.html.
Create a data folder in the current directory, and place the downloaded dataset files into it.



## 3.2 Run main.py

In detail, we can train on different datasets:

#### **dataset used**

```python
path_to_arff_files = ["cal500", "birds", "enron", "scene", "yeast", "Corel5k", "rcv1subset1", "rcv1subset2", "rcv1subset3", "bibtex", "yahoo-Arts1", "yahoo-Business1", "mediamill"]
```

#### **label count of corresponding dataset:**

```python
label_counts = [174, 19, 53, 6, 14, 374, 101, 101, 101, 159, 25, 28, 101]
```

#### **feature retention ratio of the corresponding dataset:**

```python
select_feature = [1, 1, 1, 1, 1, 1, 0.02, 0.02, 0.02, 1, 0.05, 0.05, 1]
```



# 4. Key code:

## 4.1 Uncertainty metric

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



## 4.2 Hardness metric

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



## 4.3 Dynamic Label Weight

```python
mu_h = np.mean(h_mat, axis=0)
sigma_h = np.std(h_mat, axis=0)
v_h_raw = np.exp(0.5 * mu_h + 0.5 * sigma_h)

mu_e = np.mean(custom_dataloader.E, axis=0)
sigma_e = np.std(custom_dataloader.E, axis=0)
v_e_raw = np.exp(0.5 * mu_e + 0.5 * sigma_e)
```



## 4.4 Local Context-Aware Label Correlation Enhancement

```python
P_neighbors_fixed = P_fixed[nbrs_idx]  # shape: (ins_dim, n_nbrs, label_dim)
Z_fixed = np.any(P_neighbors_fixed, axis=1).astype(np.float32)  # shape: (ins_dim, label_dim)

P = P_fixed                # (ins_dim, label_dim)
Z = Z_fixed.astype(np.float32)

# ----- mask E/h -----
E_mat = custom_dataloader.E.astype(np.float64)
H_mat = h_mat.astype(np.float64)

E_masked = E_mat * P
H_masked = H_mat * P
```



## 4.5 Sampling

```python
 sample_score_from_E = S1_uncertain + S2_uncertain
 sample_score_from_h = S1_hard + S2_hard

# ---- sample mixing schedule ----
start_epoch = 10
end_epoch = 70

t = np.clip(epoch, start_epoch, end_epoch)
frac = (t - start_epoch) / max(1, (end_epoch - start_epoch))

p_beta_epoch = 0.7 + frac * (0.3 - 0.7)
p_from_E = calculate_probabilities(sample_score_from_E, epoch, start_epoch, num_epochs, 8)
p_from_h = calculate_probabilities(sample_score_from_h, epoch, start_epoch, num_epochs, 8)
```



# 5. Contact:

If you have any questions or suggestions, feel free to contact me: uaena_lee@163.com
