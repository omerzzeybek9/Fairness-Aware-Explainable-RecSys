"""
baselines.py — Baseline recommenders for comparison against the KG-Path model.

Baselines:
    run_random(test_set_dict, all_items, k_values, seed)
        → Random recommendations (sanity check, expected HR@10 ~ 0.15)

    run_popularity(test_set_dict, user_item_edges, all_items, k_values)
        → Most-popular items (strong non-personalised baseline, expected HR@10 ~ 0.52)

    run_gpt2_gender(test_set_dict, adj, all_users, movie_titles_set,
                    base_rels, device, k_values, ...)
        → GPT-2 path model WITH gender-aware paths included.
          Proves the demographic-blind approach works: if this performs
          similarly or worse than the main model, demographic info adds
          no value (and risks fairness violations).

All three return a results list in the same format as the main model:
    [{"user": "User_<id>", "ground_truths": [...], "top_k_recs": [...],
      "pattern_types": [...], "num_gt": int}, ...]

Usage (in notebook):
    import importlib
    import scripts.baselines as baselines
    importlib.reload(baselines)

    results_random     = baselines.run_random(test_set_dict, all_items)
    results_popularity = baselines.run_popularity(test_set_dict, user_item_edges, all_items)
    results_gender     = baselines.run_gpt2_gender(
                             test_set_dict, adj, all_users, movie_titles_set,
                             BASE_RELS, device)
"""

import random
from collections import Counter

from tqdm import tqdm

from paths import sample_guided_paths, BASE_RELS as _BASE_RELS
from model import (build_vocab, create_path_dataset, create_model,
                   train_model, generate_topk)
from metrics import evaluate_ranking, compute_group_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 1. Random Baseline
# ══════════════════════════════════════════════════════════════════════════════

def run_random(test_set_dict, all_items, k_values=(1, 3, 5, 10), seed=42):
    """
    Recommend K random items per user (excluding training likes when adj is
    unavailable — purely random from the full item pool).

    Parameters
    ----------
    test_set_dict : dict  {"User_<id>" -> [ground_truth_movie, ...]}
    all_items     : list  all movie title strings
    k_values      : tuple cutoff values
    seed          : int   random seed for reproducibility

    Returns
    -------
    results : list of dicts (same schema as main model results)
    """
    random.seed(seed)
    max_k = max(k_values)
    all_items = list(all_items)

    results = []
    for user_node, ground_truths in test_set_dict.items():
        recs = random.sample(all_items, min(max_k, len(all_items)))
        results.append({
            "user": user_node,
            "ground_truths": ground_truths,
            "top_k_recs": recs,
            "pattern_types": ["random"] * len(recs),
            "num_gt": len(ground_truths),
        })

    metrics = evaluate_ranking(results, k_values=k_values)
    _print_summary("Random Baseline", metrics, k_values)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. Popularity Baseline
# ══════════════════════════════════════════════════════════════════════════════

def run_popularity(test_set_dict, user_item_edges, all_items,
                   k_values=(1, 3, 5, 10)):
    """
    Recommend the globally most-popular items (by training interaction count),
    excluding items the user already liked in training.

    Parameters
    ----------
    test_set_dict    : dict  {"User_<id>" -> [ground_truth_movie, ...]}
    user_item_edges  : list  [("User_<id>", "likes", movie_title), ...]
    all_items        : list  all movie title strings
    k_values         : tuple cutoff values

    Returns
    -------
    results : list of dicts
    """
    max_k = max(k_values)

    # Count training interactions per item
    item_counts = Counter(movie for _, _, movie in user_item_edges)
    popular_items = [item for item, _ in item_counts.most_common()]

    # Build per-user training set for exclusion
    user_train_items = {}
    for user, _, movie in user_item_edges:
        user_train_items.setdefault(user, set()).add(movie)

    results = []
    for user_node, ground_truths in test_set_dict.items():
        seen = user_train_items.get(user_node, set())
        recs = [m for m in popular_items if m not in seen][:max_k]
        # Pad with unseen items if not enough popular ones
        if len(recs) < max_k:
            remaining = [m for m in all_items if m not in seen and m not in recs]
            recs += remaining[:max_k - len(recs)]
        results.append({
            "user": user_node,
            "ground_truths": ground_truths,
            "top_k_recs": recs,
            "pattern_types": ["popularity"] * len(recs),
            "num_gt": len(ground_truths),
        })

    metrics = evaluate_ranking(results, k_values=k_values)
    _print_summary("Popularity Baseline", metrics, k_values)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. GPT-2 + Gender Baseline
# ══════════════════════════════════════════════════════════════════════════════

def _sample_gender_cf_path(user_node, adj):
    """
    Gender-aware CF path:
    User → hasGender → Gender_X → rev_hasGender → UserB → likes → Movie

    Bridges users of the same gender, explicitly using demographic info.
    This is exactly what the main model avoids.
    """
    genders = list(adj[user_node].get("hasGender", set()))
    if not genders:
        return None
    gender_node = genders[0]
    same_gender_users = list(
        adj[gender_node].get("rev_hasGender", set()) - {user_node}
    )
    if not same_gender_users:
        return None
    user_b = random.choice(same_gender_users)
    liked_by_user = set(adj[user_node].get("likes", set()))
    candidates = list(adj[user_b].get("likes", set()) - liked_by_user)
    if not candidates:
        return None
    movie = random.choice(candidates)
    return [user_node, "hasGender", gender_node, "rev_hasGender", user_b, "likes", movie]


def run_gpt2_gender(test_set_dict, adj, all_users, movie_titles_set,
                    base_rels, device,
                    k_values=(1, 3, 5, 10),
                    paths_per_user=150,
                    gender_weight=0.20,
                    epochs=10, lr=3e-4, patience=2,
                    seed=42):
    """
    Train and evaluate a GPT-2 path model that INCLUDES gender-aware paths.

    The gender path pattern (User→hasGender→Gender→rev_hasGender→UserB→likes→Movie)
    explicitly uses demographic info during training, which is what the main
    demographic-blind model avoids.

    Parameters
    ----------
    test_set_dict   : dict   {"User_<id>" -> [ground_truth, ...]}
    adj             : dict   KG adjacency graph (must contain hasGender edges)
    all_users       : list   training user node strings
    movie_titles_set: set    all valid movie title strings
    base_rels       : set    base relation tokens (from paths.py)
    device          : torch.device
    k_values        : tuple  cutoff values
    paths_per_user  : int    path attempts per user during sampling
    gender_weight   : float  sampling weight for the gender pattern (0–1)
    epochs          : int    training epochs
    lr              : float  learning rate
    patience        : int    early stopping patience
    seed            : int    random seed

    Returns
    -------
    results : list of dicts
    """
    random.seed(seed)

    # ── Step 1: Sample paths WITH gender pattern ──────────────────────────────
    print("=" * 60)
    print("GPT-2 + GENDER BASELINE")
    print("=" * 60)
    print("\n[1/4] Sampling paths (with gender-aware pattern)...")

    # Redistribute weights: scale existing patterns down to fit gender_weight
    base_patterns = {
        "genre": 0.25, "director": 0.20, "cf": 0.20,
        "cast": 0.15, "composer": 0.10, "writer": 0.10,
    }
    scale = 1.0 - gender_weight
    pattern_weights = {k: v * scale for k, v in base_patterns.items()}
    pattern_weights["gender"] = gender_weight

    # Temporarily register the gender sampler into paths._SAMPLERS
    import paths as _paths_mod
    _paths_mod._SAMPLERS["gender"] = _sample_gender_cf_path

    gender_base_rels = set(base_rels) | {"hasGender"}

    paths = sample_guided_paths(
        users=all_users,
        adj=adj,
        paths_per_user=paths_per_user,
        pattern_weights=pattern_weights,
    )

    # Clean up: remove gender sampler so it doesn't affect other code
    del _paths_mod._SAMPLERS["gender"]

    # ── Step 2: Build vocab & dataset ─────────────────────────────────────────
    print("\n[2/4] Building vocabulary and dataset...")
    vocab, id2tok, PAD, BOS, EOS, UNK = build_vocab(paths, gender_base_rels)
    print(f"Vocab size: {len(vocab)}")

    train_loader, val_loader, MAX_LEN, _ = create_path_dataset(
        paths, vocab, gender_base_rels, PAD, BOS, EOS, UNK,
        batch_size=64, val_ratio=0.1,
    )

    # ── Step 3: Train model ───────────────────────────────────────────────────
    print("\n[3/4] Training GPT-2 + Gender model...")
    model = create_model(
        vocab_size=len(vocab), max_len=MAX_LEN,
        BOS=BOS, EOS=EOS, device=device,
        n_embd=192, n_layer=4, n_head=4, dropout=0.1,
    )
    model = train_model(
        model, train_loader, val_loader, device=device,
        epochs=epochs, lr=lr, patience=patience,
    )

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    print("\n[4/4] Evaluating...")
    max_k = max(k_values)
    results = []

    for user_node, ground_truths in tqdm(test_set_dict.items(), desc="Evaluating"):
        if user_node not in adj:
            continue
        topk_with_type = generate_topk(
            user_node, model, vocab, id2tok, adj, gender_base_rels,
            movie_titles_set, PAD, BOS, EOS, UNK, MAX_LEN,
            device=device, K=max_k,
        )
        ranked_list = [c for c, t in topk_with_type]
        pattern_types = [t for c, t in topk_with_type]
        results.append({
            "user": user_node,
            "ground_truths": ground_truths,
            "top_k_recs": ranked_list,
            "pattern_types": pattern_types,
            "num_gt": len(ground_truths),
        })

    metrics = evaluate_ranking(results, k_values=k_values)
    _print_summary("GPT-2 + Gender Baseline", metrics, k_values)
    return results, model, vocab, id2tok, PAD, BOS, EOS, UNK, MAX_LEN


# ══════════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(label, metrics, k_values):
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  {'K':>4} | {'HR@K':>8} | {'MRR@K':>8} | {'NDCG@K':>8}")
    print(f"  {'-'*40}")
    for k in k_values:
        m = metrics[k]
        print(f"  {k:>4} | {m['HR']:>8.4f} | {m['MRR']:>8.4f} | {m['NDCG']:>8.4f}")
    print()
