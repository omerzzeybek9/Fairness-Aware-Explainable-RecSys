import random
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from metrics import (
    evaluate_ranking, compute_group_metrics,
    compute_all_ilap_metrics, print_ilap_report
)

def _build_index_maps(train_interactions, movies_sub):
    user_ids  = sorted(train_interactions["userId"].unique())
    movie_ids = sorted(movies_sub["movieId"].unique())

    uid2idx = {u: i for i, u in enumerate(user_ids)}
    mid2idx = {m: i for i, m in enumerate(movie_ids)}

    # movieId -> title
    mid2title = dict(zip(movies_sub["movieId"], movies_sub["title"]))
    idx2title = {i: mid2title[m] for m, i in mid2idx.items() if m in mid2title}
    title2mid = {v: k for k, v in mid2title.items()}

    return uid2idx, mid2idx, idx2title, title2mid, user_ids, movie_ids


def _build_interaction_matrix(train_interactions, uid2idx, mid2idx):
    n_users = len(uid2idx)
    n_items = len(mid2idx)
    mat = np.zeros((n_users, n_items), dtype=np.float32)
    for _, row in train_interactions.iterrows():
        u = uid2idx.get(int(row["userId"]))
        m = mid2idx.get(int(row["movieId"]))
        if u is not None and m is not None:
            mat[u, m] = 1.0
    return mat


def _user_node_to_id(user_node):
    return int(user_node.split("_")[1])


def _pack_results(user_node, ground_truths, recs):
    return {
        "user":          user_node,
        "ground_truths": ground_truths,
        "top_k_recs":    recs,
        "pattern_types": ["cf"] * len(recs),
        "paths":         [[]] * len(recs),
        "num_gt":        len(ground_truths),
    }


def run_user_cf(train_interactions, test_set_dict, movies_sub,
                k=10, n_neighbors=20, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 55)
    print("  UserCF: User-Based Collaborative Filtering")
    print("=" * 55)

    uid2idx, mid2idx, idx2title, title2mid, user_ids, movie_ids = \
        _build_index_maps(train_interactions, movies_sub)

    mat = _build_interaction_matrix(train_interactions, uid2idx, mid2idx)

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat_norm = mat / norms
    sim = mat_norm @ mat_norm.T          

    results = []
    for user_node, ground_truths in tqdm(test_set_dict.items(), desc="UserCF"):
        uid = _user_node_to_id(user_node)
        if uid not in uid2idx:
            results.append(_pack_results(user_node, ground_truths, []))
            continue

        u_idx = uid2idx[uid]
        sims  = sim[u_idx].copy()
        sims[u_idx] = -1          

        nn_idx = np.argpartition(sims, -n_neighbors)[-n_neighbors:]
        nn_idx = nn_idx[np.argsort(sims[nn_idx])[::-1]]

        seen = set(np.where(mat[u_idx] > 0)[0])
        scores = defaultdict(float)
        for nb in nn_idx:
            w = float(sims[nb])
            if w <= 0:
                continue
            for m_idx in np.where(mat[nb] > 0)[0]:
                if m_idx not in seen:
                    scores[m_idx] += w

        top_items = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
        recs = [idx2title[i] for i in top_items if i in idx2title]

        results.append(_pack_results(user_node, ground_truths, recs))

    return results


def run_svd_mf(train_interactions, test_set_dict, movies_sub,
               k=10, n_factors=50, n_epochs=20, lr=0.01,
               reg=0.01, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 55)
    print("  SVD-MF: Matrix Factorization (BPR)")
    print("=" * 55)

    uid2idx, mid2idx, idx2title, title2mid, user_ids, movie_ids = \
        _build_index_maps(train_interactions, movies_sub)

    n_users = len(uid2idx)
    n_items = len(mid2idx)

    P = np.random.normal(0, 0.01, (n_users, n_factors)).astype(np.float32)
    Q = np.random.normal(0, 0.01, (n_items, n_factors)).astype(np.float32)

    user_pos = defaultdict(list)
    for _, row in train_interactions.iterrows():
        u = uid2idx.get(int(row["userId"]))
        m = mid2idx.get(int(row["movieId"]))
        if u is not None and m is not None:
            user_pos[u].append(m)

    item_set = list(range(n_items))

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_samples  = 0
        for u, pos_items in user_pos.items():
            for pos in pos_items:
                neg = random.choice(item_set)
                while neg in user_pos[u]:
                    neg = random.choice(item_set)

                diff = float(np.dot(P[u], Q[pos] - Q[neg]))
                grad = -1.0 / (1.0 + math.exp(diff)) if diff < 50 else -math.exp(-diff)

                P[u]    -= lr * (grad * (Q[pos] - Q[neg]) + reg * P[u])
                Q[pos]  -= lr * (grad * P[u]              + reg * Q[pos])
                Q[neg]  -= lr * (-grad * P[u]             + reg * Q[neg])

                total_loss -= math.log(1.0 / (1.0 + math.exp(-diff)) + 1e-10)
                n_samples  += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  BPR loss: {total_loss/max(n_samples,1):.4f}")

    mat = _build_interaction_matrix(train_interactions, uid2idx, mid2idx)

    scores_all = P @ Q.T 

    results = []
    for user_node, ground_truths in tqdm(test_set_dict.items(), desc="SVD-MF"):
        uid = _user_node_to_id(user_node)
        if uid not in uid2idx:
            results.append(_pack_results(user_node, ground_truths, []))
            continue

        u_idx  = uid2idx[uid]
        scores = scores_all[u_idx].copy()

        seen_mask = mat[u_idx] > 0
        scores[seen_mask] = -np.inf

        top_items = np.argpartition(scores, -k)[-k:]
        top_items = top_items[np.argsort(scores[top_items])[::-1]]
        recs = [idx2title[i] for i in top_items if i in idx2title]

        results.append(_pack_results(user_node, ground_truths, recs))

    return results


class _MLPModel(nn.Module):
    def __init__(self, n_users, n_items, n_factors=32, hidden=(64, 32)):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)

        layers = []
        in_dim = n_factors * 2
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = torch.cat([u, i], dim=1)
        return self.mlp(x).squeeze(1)


def run_mlp_cf(train_interactions, test_set_dict, movies_sub,
               k=10, n_factors=32, hidden=(64, 32),
               n_epochs=20, lr=1e-3, batch_size=512,
               neg_ratio=4, seed=42, device="cpu"):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 55)
    print("  MLP-CF: Neural Collaborative Filtering")
    print("=" * 55)

    uid2idx, mid2idx, idx2title, title2mid, user_ids, movie_ids = \
        _build_index_maps(train_interactions, movies_sub)

    n_users = len(uid2idx)
    n_items = len(mid2idx)

    pos_pairs = []
    user_pos  = defaultdict(set)
    for _, row in train_interactions.iterrows():
        u = uid2idx.get(int(row["userId"]))
        m = mid2idx.get(int(row["movieId"]))
        if u is not None and m is not None:
            pos_pairs.append((u, m))
            user_pos[u].add(m)

    item_list = list(range(n_items))

    def _build_dataset():
        users, items, labels = [], [], []
        for u, m in pos_pairs:
            users.append(u); items.append(m); labels.append(1.0)
            for _ in range(neg_ratio):
                neg = random.choice(item_list)
                while neg in user_pos[u]:
                    neg = random.choice(item_list)
                users.append(u); items.append(neg); labels.append(0.0)
        return (
            torch.tensor(users,  dtype=torch.long),
            torch.tensor(items,  dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
        )

    model_mlp = _MLPModel(n_users, n_items, n_factors, hidden).to(device)
    optimizer = optim.Adam(model_mlp.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        model_mlp.train()
        u_t, i_t, l_t = _build_dataset()

        # Shuffle
        perm = torch.randperm(len(u_t))
        u_t, i_t, l_t = u_t[perm], i_t[perm], l_t[perm]

        total_loss = 0.0
        n_batches  = 0
        for start in range(0, len(u_t), batch_size):
            ub = u_t[start:start+batch_size].to(device)
            ib = i_t[start:start+batch_size].to(device)
            lb = l_t[start:start+batch_size].to(device)

            optimizer.zero_grad()
            out  = model_mlp(ub, ib)
            loss = criterion(out, lb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  BCE loss: {total_loss/max(n_batches,1):.4f}")

    mat = _build_interaction_matrix(train_interactions, uid2idx, mid2idx)

    model_mlp.eval()
    all_item_idx = torch.arange(n_items, dtype=torch.long).to(device)

    results = []
    for user_node, ground_truths in tqdm(test_set_dict.items(), desc="MLP-CF"):
        uid = _user_node_to_id(user_node)
        if uid not in uid2idx:
            results.append(_pack_results(user_node, ground_truths, []))
            continue

        u_idx = uid2idx[uid]
        u_tensor = torch.tensor([u_idx] * n_items, dtype=torch.long).to(device)

        with torch.no_grad():
            scores = model_mlp(u_tensor, all_item_idx).cpu().numpy()

        # Mask seen items
        seen_mask = mat[u_idx] > 0
        scores[seen_mask] = -np.inf

        top_items = np.argpartition(scores, -k)[-k:]
        top_items = top_items[np.argsort(scores[top_items])[::-1]]
        recs = [idx2title[i] for i in top_items if i in idx2title]

        results.append(_pack_results(user_node, ground_truths, recs))

    return results


def run_all(train_interactions, test_set_dict, movies_sub,
            k=10, seed=42, device="cpu",
            ucf_neighbors=20,
            svd_factors=50, svd_epochs=20,
            mlp_factors=32, mlp_epochs=20):
    results_ucf = run_user_cf(
        train_interactions, test_set_dict, movies_sub,
        k=k, n_neighbors=ucf_neighbors, seed=seed,
    )
    results_svd = run_svd_mf(
        train_interactions, test_set_dict, movies_sub,
        k=k, n_factors=svd_factors, n_epochs=svd_epochs, seed=seed,
    )
    results_mlp = run_mlp_cf(
        train_interactions, test_set_dict, movies_sub,
        k=k, n_factors=mlp_factors, n_epochs=mlp_epochs, seed=seed,
        device=device,
    )
    return results_ucf, results_svd, results_mlp



def compare_all(models_dict, user_gender_map, adj=None, k_values=(1, 3, 5, 10)):
    k10 = 10
    sep = "=" * 100

    print(sep)
    print(f"  MODEL COMPARISON — Accuracy & Fairness @ K={k10}")
    print(sep)

    print(f"\n{'Model':<18} {'HR@10':>7} {'MRR@10':>7} {'NDCG@10':>8} "
          f"{'HR_M':>7} {'HR_F':>7} {'Gap':>7} "
          f"{'DI':>7} {'EO':>7} {'DP':>7} {'CF':>7}")
    print("-" * 100)

    all_rows = {}
    for label, res in models_dict.items():
        rm   = evaluate_ranking(res, k_values=k_values)
        gm   = compute_group_metrics(res, user_gender_map, k_values)

        hr_m = gm["M"][k10]["HR"]
        hr_f = gm["F"][k10]["HR"]
        gap  = hr_f - hr_m

        print(f"{label:<18} {rm[k10]['HR']:>7.4f} {rm[k10]['MRR']:>7.4f} "
              f"{rm[k10]['NDCG']:>8.4f} "
              f"{hr_m:>7.4f} {hr_f:>7.4f} {gap:>+7.4f} ")

        all_rows[label] = {
            "ranking": rm, "group": gm,
        }

    print(sep)

    print(f"\n{'Model':<18}", end="")
    for k in k_values:
        print(f"  HR@{k:>2}", end="")
    for k in k_values:
        print(f"  NDCG@{k:>2}", end="")
    print()
    print("-" * (18 + 8 * len(k_values) * 2))

    for label, data in all_rows.items():
        print(f"{label:<18}", end="")
        for k in k_values:
            print(f"  {data['ranking'][k]['HR']:>5.4f}", end="")
        for k in k_values:
            print(f"  {data['ranking'][k]['NDCG']:>7.4f}", end="")
        print()

    print()

    print(f"\n{'ILAP Fairness @ K=10':^60}")
    print("-" * 60)
    ilap_all = {
        label: compute_all_ilap_metrics(res, user_gender_map, k=k10)
        for label, res in models_dict.items()
    }
    ilap_keys = ["DF", "VU", "AU", "UU", "OU", "NU", "GCE"]
    print(f"{'Metric':<8}", end="")
    for label in models_dict:
        print(f"  {label:>14}", end="")
    print()
    print("-" * (8 + 16 * len(models_dict)))
    for key in ilap_keys:
        print(f"{key:<8}", end="")
        for label in models_dict:
            v = ilap_all[label].get(key, float("nan"))
            print(f"  {v:>14.4f}", end="")
        print()

    print()
    return all_rows