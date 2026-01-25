
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.data import random_split, DataLoader,TensorDataset
import torch
from sklearn.metrics import (
    jaccard_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.neighbors import NearestNeighbors
import random
import os
from skmultilearn.dataset import load_from_arff
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from skmultilearn.model_selection import IterativeStratification
import warnings
warnings.filterwarnings("ignore")
import math
from collections import deque
from util import *
from layers import *
from model import *
from function import *
import time

#####seed####
def seed_all(seed): 
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed = 55
seed_all(seed)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class CFG:
    def __init__(self, name, X, y):
        self.name = name      
        self.X_train = X
        self.y_train = y
        self.configs = {} 
    
    def getconfig(self):
        self.configs['label_matrix'] = np.array(self.y_train)
        self.configs['num_classes'] = self.y_train.shape[1] 
        self.configs['num_ins'] = self.X_train.shape[0] 
        self.configs['seed'] = 55
        self.configs['batch_size'] = 128
        self.configs['epoch'] = 100 
        self.configs['lr'] = 1e-4
        self.configs['device'] = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.configs['weight'] = limb(np.array(self.X_train), np.array(self.y_train))
        self.configs['extra_sample'] = int(self.X_train.shape[0] * 0.1)
        self.configs['min_ins_idx'] = minority_instance(np.array(self.y_train))
        self.configs['minority_label_indices'], _ = Labeltype(np.array(self.y_train))
        self.configs['weight_list'] = calweight(np.array(self.X_train), np.array(self.y_train))
        self.configs['card'], _ = CardAndDens(np.array(self.X_train), np.array(self.y_train))
        
        # DELA
        if self.name == 'DELA':
            self.configs['in_features'] = self.X_train.shape[1] 
            self.configs['latent_dim'] = math.ceil(self.X_train.shape[1] / 2)    
            self.configs['lr_ratio'] = 0.8
            self.configs['drop_ratio'] = 0.2
            self.configs['tau'] = 2 / 3
            self.configs['beta'] = 1e-4
            self.configs['out_index'] = -1
        
        # CLIF
        if self.name == 'CLIF':
            self.configs['class_emb_size'] = self.y_train.shape[1]  
            self.configs['input_x_size'] = self.X_train.shape[1] 
            self.configs['num_layers'] = 2 
            self.configs['in_layers'] = 3 
            self.configs['hidden_list'] = [math.ceil(self.y_train.shape[1] / 2)]  
            self.configs['out_index'] = 0        
        
        # PACA
        if self.name == 'PACA':
            self.configs['drop_ratio'] = 0.1
            self.configs['latent_dim'] = math.ceil(self.X_train.shape[1] / 2)
            self.configs['in_features'] = self.X_train.shape[1]  # 输入x的维度
            self.configs['rand_seed'] = self.configs['seed']
            self.configs['eps'] = 1e-8    
            self.configs['lr_scheduler'] = 'fix'
            self.configs['binary_data'] = False
            self.configs['weight_decay'] = 1e-5
            self.configs['alpha'] = 2
            self.configs['gamma'] = 10
            self.configs['scheduler_warmup_epoch'] = 5
            self.configs['scheduler_decay_epoch'] = 10
            self.configs['scheduler_decay_rate'] = 1e-5 
            self.configs['out_index'] = -2
        
        return self.configs

seed_all(seed)
device = torch.device('cuda')
def FeatureSelect(X, p):
    if p == 1:
        return X.toarray(), feature_names
    else:
        featurecount = int(X.shape[1] * p)
        Selectfeatureindex = [x[0] for x in (sorted(enumerate(X.sum(axis=0).tolist()[0]), key=lambda x: x[1], reverse=True))][:featurecount]
        Allfeatureindex = [i for i in range(X.shape[1])]
        featureindex = [i for i in Allfeatureindex if i not in Selectfeatureindex]
        new_x = np.delete(X.toarray(), featureindex, axis=1)
        new_featurename = [feature_names[i] for i in Selectfeatureindex] 
        return new_x, new_featurename

def LabelSelect(y):
    b = []
    new_labelname = [i for i in label_names]
    for i in range(y.shape[1]):
        if y[:, i].sum() <= 20:
            b.append(i)
            new_labelname.remove(label_names[i])
    new_y = np.delete(y.toarray(), b, axis=1)
    return new_y, new_labelname

def macro_averaging_auc(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2
    p = np.zeros(l)
    q = np.sum(Y, 0)

    r, c = np.nonzero(Y)
    for i, j in zip(r, c):
        p[j] += np.sum((Y[:, j] < 0.5) * (O[:, j] <= O[i, j]))
    i = (q > 0) * (q < n)

    return np.sum(p[i] / (q[i] * (n - q[i]))) / l

def hamming_loss(Y, P, O):
    n = (Y.shape[0] + P.shape[0]) // 2
    l = (Y.shape[1] + P.shape[1]) // 2

    s1 = np.sum(Y, 1)
    s2 = np.sum(P, 1)
    ss = np.sum(Y * P, 1)

    return np.sum(s1 + s2 - 2 * ss) / (n * l)
def micro_auc(Y, P, O):
    try:
        return roc_auc_score(Y, O, average='micro')
    except ValueError:
        return np.nan

def macro_auc(Y, P, O):
    n_labels = Y.shape[1]
    aucs = []
    for j in range(n_labels):
        y_j = Y[:, j]
        o_j = O[:, j]
        if np.unique(y_j).size == 2:
            aucs.append(roc_auc_score(y_j, o_j))
    if len(aucs) == 0:
        return np.nan
    return np.mean(aucs)
    # return roc_auc_score(Y, O, average='macro')
def subset_accuracy(Y, P, O):
    return np.mean(np.all(Y == P, axis=1))

def example_precision(Y, P, O):
    precisions = []
    for y, p in zip(Y, P):
        tp = np.sum(y * p)
        pred_pos = np.sum(p)
        if pred_pos > 0:
            precisions.append(tp / pred_pos)
    return np.mean(precisions) if precisions else 0.0

def example_recall(Y, P, O):
    recalls = []
    for y, p in zip(Y, P):
        tp = np.sum(y * p)
        true_pos = np.sum(y)
        if true_pos > 0:
            recalls.append(tp / true_pos)
    return np.mean(recalls) if recalls else 0.0

def jaccard_index(Y, P, O):
    return jaccard_score(Y, P, average='samples')

def micro_aupr(Y, P, O):
    return average_precision_score(Y, O, average='micro')

def macro_aupr(Y, P, O):
    return average_precision_score(Y, O, average='macro')

def average_precision(Y, P, O):
    return average_precision_score(Y, O, average='samples')

def coverage(Y, P, O):
    n_samples, n_labels = Y.shape
    sorted_indices = np.argsort(-O, axis=1)
    total_coverage = 0.0
    for i in range(n_samples):
        true_labels = np.where(Y[i] == 1)[0]
        if len(true_labels) == 0:
            continue
        max_pos = -1
        for label in true_labels:
            pos = np.where(sorted_indices[i] == label)[0][0]
            if pos > max_pos:
                max_pos = pos
        total_coverage += (max_pos + 1)
    return total_coverage / n_samples

def macro_acc(Y,P,O):
    return np.mean([accuracy_score(Y[:, j], P[:, j]) for j in range(P.shape[1])])

def micro_acc(Y,P,O):
    return accuracy_score(Y.ravel(), P.ravel())

def macro_recall(Y,P,O):
    return recall_score(Y, P, average='macro',zero_division=0)

def micro_recall(Y,P,O):
    return recall_score(Y, P, average='micro',zero_division=0)

def macro_precision(Y,P,O):
    return precision_score(Y, P, average='macro',zero_division=0)

def micro_precision(Y,P,O):
    return precision_score(Y, P, average='micro',zero_division=0)

    
def one_error(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2
    i = np.argmax(O, 1)
    return np.sum(1 - Y[range(n), i]) / n

def ranking_loss(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2

    p = np.zeros(n)
    q = np.sum(Y, 1)

    r, c = np.nonzero(Y)
    for i, j in zip(r, c): 
        p[i] += np.sum((Y[i, :] < 0.5) * (O[i, :] >= O[i, j]))

    i = (q > 0) * (q < l)

    return np.sum(p[i] / (q[i] * (l - q[i]))) / n

def micro_f1(Y, P, O):
    return f1_score(Y, P, average='micro')

def macro_f1(Y, P, O):
    return f1_score(Y, P, average='macro')

def eval_metrics(mod, metrics, datasets, idx, batch_size, device):
    mod.eval()
    y_true_list = []
    y_scores_list = []
    test_dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True, num_workers=0)
    for x, y in test_dataloader:
        _, y_pred = mod.predict(x)
        y_true_list.append(y.cpu().numpy())
        y_scores_list.append(y_pred.cpu().numpy())
    y_true = np.vstack(y_true_list)
    y_prob = np.vstack(y_scores_list)
    y_pred = np.round(y_prob).astype(int)
    res_dict1 = {metric.__name__: metric(y_true, y_pred, y_prob) for metric in metrics}
    return res_dict1

def update_L(L, loss_matrix, ids, max_history_length=5):
    """
    Update the per-sample-per-label loss history dictionary L
    loss_matrix: Tensor or ndarray with shape (batch_size, label_dim)
                 Each element represents the loss value of a sample for a specific label.
    ids: List of dataset indices
    
    """
    if isinstance(loss_matrix, torch.Tensor):
        loss_numpy = loss_matrix.detach().cpu().numpy()
    else:
        loss_numpy = np.array(loss_matrix)
    for i, idx in enumerate(ids):
        if idx not in L:
            L[idx] = deque(maxlen=max_history_length)
        L[idx].append(loss_numpy[i])
    return L

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

def calculate_probabilities(U, epoch, e_start, e_end, s_start):
    # Delta is set to 1/N
    N = len(U)
    Delta = 1 / N 
    # Exponentially decay the selection pressure as training progresses
    if epoch < e_start:
        s = s_start
    else:
        decay_factor = np.exp(np.log(1 / s_start) / (e_end - e_start))
        s = s_start * (decay_factor ** (epoch - e_start))
        s = max(s, 1.0)  # Ensure it decays to at least 1
    # Normalize U to [0,1]
    U_min = np.min(U)
    U_max = np.max(U)
    if U_max - U_min > 0:
        U_normalized = (U - U_min) / (U_max - U_min)
    else:
        U_normalized = np.zeros_like(U)
    quantized_indices = np.ceil((1 - U_normalized) / Delta).astype(int)
    # Clip to avoid negative or extreme values
    quantized_indices = np.clip(quantized_indices, 1, N)
    # Calculate the exponent for the sampling probability
    exponent = - (np.log(s) / N) * quantized_indices 
    # Calculate the unnormalized probabilities
    unnormalized_probabilities = np.exp(exponent)
    return unnormalized_probabilities


class CustomDataLoader:
    def __init__(self, dataset, net, configs, H, E, U, sample_probabilities, warm_epoch):
        self.shuffle = False
        self.dataset = dataset
        self.net = net
        self.batch_size = configs['batch_size']
        self.sample_probabilities = sample_probabilities
        self.dataset_indices = list(range(len(self.dataset)))
        self.warm_epoch = warm_epoch

        self.H = H
        self.E = E
        self.U = U

    def __iter__(self):
        # warm-up
        if self.current_epoch < self.warm_epoch:
            batch_counter = 0
            for start_idx in range(0, len(self.dataset_indices), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(self.dataset_indices))
                batch_indices = self.dataset_indices[start_idx:end_idx]
                batch_counter += 1
                yield self.get_data_from_indices(batch_indices)
            return

        # adopt mixed probability sampling
        all_indices = np.array(self.dataset_indices)
        all_probabilities = np.array(self.sample_probabilities, dtype=float)
        total_sum = np.sum(all_probabilities)
        if total_sum <= 0:
            all_probabilities = np.ones_like(all_probabilities) / len(all_probabilities)
        else:
            all_probabilities = all_probabilities / total_sum

        total_samples = len(self.dataset)
        num_batches = int(np.ceil(total_samples / self.batch_size))

        for _ in range(num_batches):
            num_required = min(self.batch_size, total_samples - _ * self.batch_size)
            if num_required > 0:
                if np.any(np.isnan(all_probabilities)):
                    all_probabilities = np.nan_to_num(all_probabilities, nan=1e-10)
                    all_probabilities = all_probabilities / np.sum(all_probabilities)
                chosen_indices = np.random.choice(all_indices, size=num_required, replace=False, p=all_probabilities)
            else:
                chosen_indices = []
            batch_indices = list(chosen_indices)
            yield self.get_data_from_indices(batch_indices)
        return

    def get_data_from_indices(self, indices):
        x, y = zip(*[self.dataset[i] for i in indices])
        return indices, torch.stack(x), torch.stack(y)

    def set_epoch(self, epoch):
        self.current_epoch = epoch


def deque_to_ema(deq, alpha=0.7):
    arr = np.array(deq)
    if arr.shape[0] == 0:
        return np.zeros(arr.shape[1]) if arr.ndim>1 else np.zeros(1)
    ema = arr[0].astype(float)
    for t in range(1, arr.shape[0]):
        ema = alpha * arr[t] + (1 - alpha) * ema
    return ema

def training(configs, train_dataset, test_dataset):

    net = CLIFModel(configs).to(device)
    num_epochs = configs['epoch']
    batch_size = configs['batch_size']
    lr = 1e-4
    weight_decay = 1e-4
    betas = (0.9, 0.999)
    ins_dim = configs['num_ins']
    label_dim = configs['num_classes']

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay
    )

    # ---- 1. Train/Validation Split ----
    total_train = len(train_dataset)
    val_size = int(0.1 * total_train)
    train_size = total_train - val_size

    train_subset, validation_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(55)
    )

    # ---- 2. Rebuild Metrics for train_subset ----
    ins_dim = len(train_subset)
    sample_indices = list(range(ins_dim))

    H = {idx: deque(maxlen=5) for idx in sample_indices}
    E = np.zeros((ins_dim, label_dim))
    L = {idx: deque(maxlen=5) for idx in sample_indices}
    ema_flip = np.zeros((ins_dim, label_dim), dtype=float)
    U = np.zeros(ins_dim)

    sample_probabilities = np.ones(ins_dim) / ins_dim

    X_train_subset = np.array(
        [train_subset[i][0].cpu().numpy() for i in range(len(train_subset))]
    )

    # ---- kNN neighbors ----
    k_neighbors = 6
    include_self = True
    n_nbrs = k_neighbors + (1 if include_self else 0)

    nn = NearestNeighbors(n_neighbors=n_nbrs, metric='euclidean', n_jobs=-1)
    nn.fit(X_train_subset)
    nbrs_dist, nbrs_idx = nn.kneighbors(X_train_subset, return_distance=True)

    # ---- 3. DataLoaders ----
    custom_dataloader = CustomDataLoader(
        train_subset,
        net=net,
        configs=configs,
        H=H,
        E=E,
        U=U,
        sample_probabilities=sample_probabilities,
        warm_epoch=10
    )

    validation_dataloader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    best_auc = 0
    best_model_state = None
    epoch_losses_train = []

    warm_epoch = 10
    warmup_epochs = warm_epoch

    warmup_steps = warmup_epochs * int(configs['num_ins'] / batch_size)
    global_step = 0

    def update_learning_rate(optimizer, global_step, warmup_steps=warmup_steps, base_lr=lr):
        if global_step < warmup_steps:
            lr = base_lr * (global_step / warmup_steps)
        else:
            lr = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    # ----------------------------------------------------------------------
    #                               Training Loop
    # ----------------------------------------------------------------------
    for epoch in range(num_epochs):

        net.train()
        batch_counter = 0
        loss_tracker = 0.0
        custom_dataloader.set_epoch(epoch)

        # ---- Standard Training ----
        for idx, x, y in custom_dataloader:

            optimizer.zero_grad()
            outputs = net(x)
            probs = torch.sigmoid(outputs[0])
            loss_dict = net.loss_function_train(outputs, y)
            loss = loss_dict['Loss']
            loss.backward()
            optimizer.step()
            global_step += 1
            loss_matrix_batch = loss_dict['loss_matrix']
            update_learning_rate(optimizer, global_step)
            L = update_L(L, loss_matrix_batch, idx, max_history_length=5)
            custom_dataloader.H = update_H(
                custom_dataloader.H,
                torch.sigmoid(outputs[0]),
                idx
            )

            ema_flip = update_ema_flip(
                ema_flip, custom_dataloader.H, idx, label_dim, alpha=0.7
            )

            custom_dataloader.E = update_E(
                custom_dataloader.H, custom_dataloader.E, idx, label_dim
            )

            batch_counter += 1


        # ------------------ construct loss / hard matrices ------------------
        loss_matrix_current = np.zeros((ins_dim, label_dim))

        for i in range(ins_dim):
            if len(L[i]) > 0:
                loss_matrix_current[i, :] = L[i][-1]

        loss_norm = 1 - np.expm1(-loss_matrix_current)

        if isinstance(ema_flip, torch.Tensor):
            ema_flip_np = ema_flip.detach().cpu().numpy().astype(np.float32)
        else:
            ema_flip_np = np.array(ema_flip, dtype=np.float32)

        h_mat = loss_norm * (1.0 - ema_flip_np)

        # ---------- label weights v_j ----------
        mu_h = np.mean(h_mat, axis=0)
        sigma_h = np.std(h_mat, axis=0)
        v_h_raw = np.exp(0.5 * mu_h + 0.5 * sigma_h)

        mu_e = np.mean(custom_dataloader.E, axis=0)
        sigma_e = np.std(custom_dataloader.E, axis=0)
        v_e_raw = np.exp(0.5 * mu_e + 0.5 * sigma_e)

        y_train_subset = np.stack([train_subset[i][1].cpu().numpy() for i in range(len(train_subset))], axis=0)
        P_fixed = (y_train_subset > 0.5).astype(np.int8)   # shape: (ins_dim, label_dim)

        # Precompute P_neighbors and Z using neighbor indices
        P_neighbors_fixed = P_fixed[nbrs_idx]  # shape: (ins_dim, n_nbrs, label_dim)
        Z_fixed = np.any(P_neighbors_fixed, axis=1).astype(np.float32)  # shape: (ins_dim, label_dim)

        P = P_fixed                # (ins_dim, label_dim)
        Z = Z_fixed.astype(np.float32)

        # ----- mask E/h -----
        E_mat = custom_dataloader.E.astype(np.float64)
        H_mat = h_mat.astype(np.float64)

        E_masked = E_mat * P
        H_masked = H_mat * P

        # ---- cosine similarity ----
        eps_cos = 1e-8
        E_col_norms = np.linalg.norm(E_masked, axis=0) + eps_cos
        H_col_norms = np.linalg.norm(H_masked, axis=0) + eps_cos

        E_normed = E_masked / E_col_norms[np.newaxis, :]
        H_normed = H_masked / H_col_norms[np.newaxis, :]

        C_unc = E_normed.T @ E_normed
        C_hard = H_normed.T @ H_normed

        C_unc = C_unc / (C_unc.sum(axis=1, keepdims=True) + eps_cos)
        C_hard = C_hard / (C_hard.sum(axis=1, keepdims=True) + eps_cos)

        S_U = Z @ C_unc.T
        S_H = Z @ C_hard.T

        # ---- weighted mats ----
        v_e_vec = v_e_raw.astype(np.float64)
        v_h_vec = v_h_raw.astype(np.float64)

        uncertain_label_weighted = E_mat * v_e_vec[np.newaxis, :]
        hard_label_weighted = H_mat * v_h_vec[np.newaxis, :]

        S1_uncertain = uncertain_label_weighted.sum(axis=1)
        S1_hard = hard_label_weighted.sum(axis=1)

        masked_uncertain_weighted = uncertain_label_weighted * P
        masked_hard_weighted = hard_label_weighted * P

        uncertain_c_weighted = masked_uncertain_weighted * S_U
        hard_c_weighted = masked_hard_weighted * S_H

        S2_uncertain = uncertain_c_weighted.sum(axis=1)
        S2_hard = hard_c_weighted.sum(axis=1)

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
   
        # ---- create pools ----
        p_unc_norm = p_from_E / (np.sum(p_from_E) + 1e-12)
        p_hard_norm = p_from_h / (np.sum(p_from_h) + 1e-12)

        # ---- mixture ----
        overall_p = p_beta_epoch * p_unc_norm + (1 - p_beta_epoch) * p_hard_norm
        # ---- write back to dataloader ----
        custom_dataloader.sample_probabilities = overall_p
        # ---- record train loss ----
        epoch_losses_train.append(loss_tracker / batch_counter)

        # ----------------------------------------------------------------------
        #                            Validation
        # ----------------------------------------------------------------------
        net.eval()
        y_true_list = []
        y_scores_list = []
        loss_tracker = 0.0
        batch_counter = 0

        with torch.no_grad():
            for x, y in validation_dataloader:
                _, y_pred = net.predict(x)
                y_true_list.append(y.cpu().numpy())
                y_scores_list.append(y_pred.cpu().numpy())

                outputs = net(x)
                loss_dict = net.loss_function_train(outputs, y)
                loss_tracker += loss_dict['Loss'].item()
                batch_counter += 1

        y_true = np.vstack(y_true_list)
        y_scores = np.vstack(y_scores_list)

        auc = macro_averaging_auc(y_true, y_scores, y_scores)

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_model_state = net.state_dict().copy()

    # ----------------------------------------------------------------------
    #                 Test with Best Checkpoint
    # ----------------------------------------------------------------------
    net.load_state_dict(best_model_state)

    mets = eval_metrics(
        net,
        [
            macro_f1, micro_f1, macro_auc, micro_auc,
            subset_accuracy, example_precision, example_recall,
            jaccard_index, micro_aupr, macro_aupr,
            macro_acc, micro_acc, macro_recall, micro_recall,
            macro_precision, micro_precision, average_precision,
            ranking_loss, hamming_loss, coverage, one_error
        ],
        test_dataset,
        configs['out_index'],
        configs['batch_size'],
        device
    )

    return mets


path_to_arff_files = ["emotions", "scene", "yeast", "Corel5k", "rcv1subset1", "rcv1subset2", "rcv1subset3", 
                      "yahoo-Business1", "yahoo-Arts1", "bibtex", 'tmc2007', 'enron', 'cal500', 'LLOG-F','genbase','birds','mediamill']
label_counts = [6, 6, 14, 374, 101, 101, 101, 28, 25, 159, 22, 53, 174, 75,27,19,101]
select_feature = [1, 1, 1, 1, 0.02, 0.02, 0.02, 0.05, 0.05, 1, 0.01, 1, 1, 1,1, 1,1]

sum_time = 0
k_fold = IterativeStratification(n_splits=5, order=1, random_state=55)

for idx, dataname in enumerate(path_to_arff_files):
    start_time = time.time()
    path_to_arff_file = f"./data/{dataname}.arff"
    X, y, feature_names, label_names = load_from_arff(
        path_to_arff_file,
        label_count=label_counts[idx],
        label_location="end",
        load_sparse=False,
        return_attribute_definitions=True
    )
    
    X, feature_names = FeatureSelect(X, select_feature[idx])  
    y, label_names = LabelSelect(y)
    non_zero_mask = np.any(y!=0,axis=1)
    X=X[non_zero_mask]
    y=y[non_zero_mask]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    print(f"\n{'='*30}")
    print(f"Processing dataset: {dataname}")
    print(f"{'='*30}")
 
    dicts = []
    training_times = []
    
    for fold_idx, (train, test) in enumerate(k_fold.split(X, y)):
        # Create the dataset
        train_dataset = TensorDataset(torch.tensor(X[train], device=device,dtype=torch.float), 
                                     torch.tensor(y[train], device=device,dtype=torch.float))
        test_dataset = TensorDataset(torch.tensor(X[test], device=device,dtype=torch.float), 
                                    torch.tensor(y[test], device=device,dtype=torch.float))
        
        # Get the model configuration
        configs = CFG('CLIF', X[train], y[train]).getconfig()
        
        # Train and evaluate the model
        dict_1 = training(configs,train_dataset,test_dataset)   
        print(f"\nFold {fold_idx+1} Results:")
        for metric_name, value in dict_1.items():
            print(f"{metric_name}: {value:.4f}")
        dicts.append(dict_1)
    
        # Calculate the average results
    averages_and_stds = {}
    for key in dicts[0].keys():
        values = [d[key] for d in dicts]
        averages_and_stds[key] = {
            'average': round(np.mean(values), 4),
            'std': round(np.std(values), 4)
        }
    
    print(f"\n{'='*30}")
    print(f"Final Results for {dataname}:")
    print(averages_and_stds)
    print(f"{'='*30}\n\n")
    
    end_time = time.time()
    fold_time = end_time - start_time
    sum_time += fold_time

print(f"Total training time: {sum_time:.2f} seconds")