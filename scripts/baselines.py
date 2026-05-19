import random

from tqdm import tqdm

from paths import sample_guided_paths
from model import (build_vocab, create_path_dataset, create_model,
                   train_model, generate_topk)
from metrics import evaluate_ranking

def _sample_gender_cf_path(user_node, adj):
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

    print("\n[2/4] Building vocabulary and dataset...")
    vocab, id2tok, PAD, BOS, EOS, UNK = build_vocab(paths, gender_base_rels)
    print(f"Vocab size: {len(vocab)}")
    train_loader, val_loader, MAX_LEN, _ = create_path_dataset(
        paths, vocab, gender_base_rels, PAD, BOS, EOS, UNK,
        batch_size=64, val_ratio=0.1,
    )

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