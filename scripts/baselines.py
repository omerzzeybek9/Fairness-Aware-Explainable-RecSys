"""
baselines.py — GPT-2 + Gender baseline for comparison against the main model.

run_gpt2_gender(test_set_dict, adj, all_users, movie_titles_set,
                base_rels, device, ...)
    → GPT-2 path model WITH gender-aware paths included.
      Proves the demographic-blind approach works: removing demographic info
      improves both accuracy and fairness.

Returns a results list in the same format as the main model:
    [{"user": "User_<id>", "ground_truths": [...], "top_k_recs": [...],
      "pattern_types": [...], "num_gt": int}, ...]

Usage (in notebook):
    import importlib
    import scripts.baselines as baselines
    importlib.reload(baselines)

    results_gender, *_ = baselines.run_gpt2_gender(
        test_set_dict, adj, all_users, movie_titles_set,
        BASE_RELS, device, k_values=K_VALUES)
"""

import random

from tqdm import tqdm

from paths import sample_guided_paths
from model import (build_vocab, create_path_dataset, create_model,
                   train_model, generate_topk)
from metrics import evaluate_ranking


# ══════════════════════════════════════════════════════════════════════════════
# Gender-aware path sampler
# ══════════════════════════════════════════════════════════════════════════════

def _sample_gender_cf_path(user_node, adj):
    """
    User → hasGender → Gender_X → rev_hasGender → UserB → likes → Movie

    Bridges users of the same gender, explicitly using demographic info.
    This is exactly what the main model avoids.
    """
    if user_node not in adj:
        return None
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


# ══════════════════════════════════════════════════════════════════════════════
# GPT-2 + Gender Baseline
# ══════════════════════════════════════════════════════════════════════════════

def run_gpt2_gender(test_set_dict, adj, all_users, movie_titles_set,
                    base_rels, device,
                    k_values=(1, 3, 5, 10),
                    paths_per_user=200,
                    gender_weight=0.20,
                    epochs=10, lr=3e-4, patience=2,
                    seed=42,
                    lambda_div=0.03,
                    path_balance_free_repeats=2,
                    max_paths_per_movie=4,
                    eval_batch_size=512,
):
    """
    Train and evaluate a GPT-2 path model that INCLUDES gender-aware paths.

    Parameters
    ----------
    test_set_dict    : dict   {"User_<id>" -> [ground_truth, ...]}
    adj              : dict   KG adjacency graph (must contain hasGender edges)
    all_users        : list   training user node strings
    movie_titles_set : set    all valid movie title strings
    base_rels        : set    base relation tokens
    device           : torch.device
    k_values         : tuple  cutoff values
    paths_per_user   : int    path attempts per user
    gender_weight    : float  sampling weight for gender pattern (0–1)
    epochs           : int    training epochs
    lr               : float  learning rate
    patience         : int    early stopping patience
    seed             : int    random seed

    Returns
    -------
    results, model, vocab, id2tok, PAD, BOS, EOS, UNK, MAX_LEN
    """
    random.seed(seed)

    print("=" * 60)
    print("GPT-2 + GENDER BASELINE")
    print("=" * 60)

    # Step 1: Sample paths with gender pattern mixed in
    print("\n[1/4] Sampling paths (with gender-aware pattern)...")
    base_patterns = {
        "genre": 0.25, "director": 0.20, "cf": 0.25,
        "cast": 0.15, "writer": 0.15,
    }
    scale = 1.0 - gender_weight
    pattern_weights = {k: v * scale for k, v in base_patterns.items()}
    pattern_weights["gender"] = gender_weight

    import paths as _paths_mod
    _paths_mod._SAMPLERS["gender"] = _sample_gender_cf_path
    try:
        gender_base_rels = set(base_rels) | {"hasGender"}
        paths = sample_guided_paths(
            users=all_users, adj=adj,
            paths_per_user=paths_per_user,
            pattern_weights=pattern_weights,
        )
    finally:
        _paths_mod._SAMPLERS.pop("gender", None)

    # Step 2: Build vocab & dataset
    print("\n[2/4] Building vocabulary and dataset...")
    vocab, id2tok, PAD, BOS, EOS, UNK = build_vocab(paths, gender_base_rels)
    print(f"Vocab size: {len(vocab)}")
    train_loader, val_loader, MAX_LEN, _ = create_path_dataset(
        paths, vocab, gender_base_rels, PAD, BOS, EOS, UNK,
        batch_size=64, val_ratio=0.1,
    )

    # Step 3: Train
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

    # Step 4: Evaluate
    print("\n[4/4] Evaluating...")
    max_k = max(k_values)
    results = []
    for user_node, ground_truths in tqdm(test_set_dict.items(), desc="Evaluating"):
        if user_node not in adj:
            continue
        # generate_topk will skip users with < 2 likes and use adaptive
        # max_total_attempts (clamp(liked_count * 20, K*10, 500)) automatically
        topk_with_type = generate_topk(
            user_node, model, vocab, id2tok, adj, gender_base_rels,
            movie_titles_set, PAD, BOS, EOS, UNK, MAX_LEN,
            device=device, K=max_k,
            max_total_attempts=None,   # deterministic candidate enumeration
            include_gender=True,       # GPT-2+Gender control baseline
            max_paths_per_movie=max_paths_per_movie,
            lambda_div=lambda_div,
            path_balance_free_repeats=path_balance_free_repeats,
            eval_batch_size=eval_batch_size,
        )
        if not topk_with_type:
            continue
        results.append({
            "user": user_node,
            "ground_truths": ground_truths,
            "top_k_recs": [c for c, t, p in topk_with_type],
            "pattern_types": [t for c, t, p in topk_with_type],
            "paths": [p for c, t, p in topk_with_type],
            "num_gt": len(ground_truths),
        })

    metrics = evaluate_ranking(results, k_values=k_values)
    print(f"\n{'─'*50}")
    print("  GPT-2 + Gender Baseline")
    print(f"{'─'*50}")
    print(f"  {'K':>4} | {'HR@K':>8} | {'MRR@K':>8} | {'NDCG@K':>8}")
    print(f"  {'-'*40}")
    for k in k_values:
        m = metrics[k]
        print(f"  {k:>4} | {m['HR']:>8.4f} | {m['MRR']:>8.4f} | {m['NDCG']:>8.4f}")
    print()

    return results, model, vocab, id2tok, PAD, BOS, EOS, UNK, MAX_LEN