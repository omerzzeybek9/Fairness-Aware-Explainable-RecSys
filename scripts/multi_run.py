"""
multi_run.py — Run the full pipeline multiple times with different seeds
and report mean ± std with statistical significance tests.

Usage (in notebook):
    import importlib
    import scripts.multi_run as multi_run
    importlib.reload(multi_run)

    all_results = multi_run.run(
        adj, all_users, test_set_dict, movie_titles_set,
        user_gender_map, BASE_RELS, device,
        paths_per_user=200, k_values=[1, 3, 5, 10],
        seeds=[42, 123, 456, 789, 2024],
    )
    multi_run.report(all_results)
"""

import random
import numpy as np
from scipy import stats
from tqdm import tqdm

from paths import sample_guided_paths
from model import (build_vocab, create_path_dataset, create_model,
                   train_model, generate_topk)
from metrics import (evaluate_ranking, compute_group_metrics,
                     disparate_impact, equalized_opportunity,
                     demographic_parity, counterfactual_fairness_score,
                     compute_all_ilap_metrics)


def run(adj, all_users, test_set_dict, movie_titles_set,
        user_gender_map, base_rels, device,
        paths_per_user=200, k_values=(1, 3, 5, 10),
        seeds=(42, 123, 456, 789, 2024),
        epochs=10, lr=3e-4, patience=2,
        pattern_weights=None):
    """
    Run full pipeline once per seed. Returns list of per-run result dicts.

    Parameters
    ----------
    adj              : KG adjacency graph
    all_users        : list of training user node strings
    test_set_dict    : dict {"User_<id>" -> [ground_truths]}
    movie_titles_set : set of all valid movie title strings
    user_gender_map  : dict {"User_<id>" -> 'M'/'F'}
    base_rels        : set of base relation tokens
    device           : torch.device
    paths_per_user   : int
    k_values         : tuple of cutoff values
    seeds            : tuple of random seeds
    epochs           : int training epochs per run
    lr               : float learning rate
    patience         : int early stopping patience
    pattern_weights  : dict pattern→weight for path sampling (None = default 6-pattern set)

    Returns
    -------
    list of dicts, one per seed:
        {
          "seed": int,
          "results": list,
          "ranking": {k: {HR, MRR, NDCG}},
          "group":   {gender: {k: {HR, MRR, NDCG}}},
          "ilap":    dict,
          "DI": float, "EO": float, "DP": float, "CF": float,
        }
    """
    import torch
    all_runs = []
    k_eval = max(k_values)
    if pattern_weights is None:
        pattern_weights = {
            "genre": 0.25, "director": 0.20, "cf": 0.20,
            "cast": 0.15, "composer": 0.10, "writer": 0.10,
        }

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}  ({seeds.index(seed)+1}/{len(seeds)})")
        print(f"{'='*60}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Sample paths
        paths = sample_guided_paths(
            users=all_users, adj=adj,
            paths_per_user=paths_per_user,
            pattern_weights=pattern_weights,
        )

        # Build vocab & dataset
        vocab, id2tok, PAD, BOS, EOS, UNK = build_vocab(paths, base_rels)
        train_loader, val_loader, MAX_LEN, _ = create_path_dataset(
            paths, vocab, base_rels, PAD, BOS, EOS, UNK,
            batch_size=64, val_ratio=0.1,
        )

        # Train
        model = create_model(
            vocab_size=len(vocab), max_len=MAX_LEN,
            BOS=BOS, EOS=EOS, device=device,
            n_embd=192, n_layer=4, n_head=4, dropout=0.1,
        )
        model = train_model(
            model, train_loader, val_loader, device=device,
            epochs=epochs, lr=lr, patience=patience,
        )

        # Evaluate
        results = []
        for user_node, ground_truths in tqdm(
            test_set_dict.items(), desc=f"Seed {seed} — evaluating"
        ):
            if user_node not in adj:
                continue
            topk_with_type = generate_topk(
                user_node, model, vocab, id2tok, adj, base_rels,
                movie_titles_set, PAD, BOS, EOS, UNK, MAX_LEN,
                device=device, K=k_eval,
            )
            results.append({
                "user": user_node,
                "ground_truths": ground_truths,
                "top_k_recs": [c for c, t, p in topk_with_type],
                "pattern_types": [t for c, t, p in topk_with_type],
                "paths": [p for c, t, p in topk_with_type],
                "num_gt": len(ground_truths),
            })

        ranking     = evaluate_ranking(results, k_values=k_values)
        group       = compute_group_metrics(results, user_gender_map, k_values)
        ilap        = compute_all_ilap_metrics(results, user_gender_map, k=10)
        cf, _       = counterfactual_fairness_score(results, user_gender_map, adj, 10)

        run_dict = {
            "seed":    seed,
            "results": results,
            "ranking": ranking,
            "group":   group,
            "ilap":    ilap,
            "DI":      disparate_impact(group, 10),
            "EO":      equalized_opportunity(group, 10),
            "DP":      demographic_parity(results, user_gender_map, 10),
            "CF":      cf,
        }
        all_runs.append(run_dict)

        # Quick per-seed summary
        print(f"  HR@10={ranking[10]['HR']:.4f}  MRR@10={ranking[10]['MRR']:.4f}  "
              f"DI={run_dict['DI']:.4f}  Gap={group['F'][10]['HR']-group['M'][10]['HR']:+.4f}")

    return all_runs


def report(all_runs, k_values=(1, 3, 5, 10)):
    """
    Print mean ± std for all metrics and run t-tests against the
    GPT-2+Gender baseline (if provided in all_runs as 'gender_runs').

    Parameters
    ----------
    all_runs  : list of run dicts from run()
    k_values  : tuple of cutoff values
    """
    print("\n" + "="*70)
    print("  MULTI-RUN RESULTS")
    print(f"  {len(all_runs)} seeds: {[r['seed'] for r in all_runs]}")
    print("="*70)

    # Accuracy metrics
    print(f"\n{'Metric':<14}", end="")
    for k in k_values:
        print(f"  {'@'+str(k):>10}", end="")
    print()
    print("-" * (14 + 12 * len(k_values)))

    for metric in ["HR", "MRR", "NDCG"]:
        print(f"{metric:<14}", end="")
        for k in k_values:
            vals = [r["ranking"][k][metric] for r in all_runs]
            print(f"  {np.mean(vals):.4f}±{np.std(vals):.4f}", end="")
        print()

    # Gender breakdown at K=10
    print(f"\n{'Gender HR@10':<14}  {'Male':>12}  {'Female':>12}  {'Gap':>10}")
    print("-" * 54)
    male_hrs   = [r["group"]["M"][10]["HR"] for r in all_runs]
    female_hrs = [r["group"]["F"][10]["HR"] for r in all_runs]
    gaps       = [f - m for f, m in zip(female_hrs, male_hrs)]
    print(f"{'Mean ± Std':<14}  {np.mean(male_hrs):.4f}±{np.std(male_hrs):.4f}  "
          f"{np.mean(female_hrs):.4f}±{np.std(female_hrs):.4f}  "
          f"{np.mean(gaps):+.4f}±{np.std(gaps):.4f}")

    # Fairness metrics
    print(f"\n{'Metric':<8}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print("-" * 54)
    for key in ["DI", "EO", "DP", "CF"]:
        vals = [r[key] for r in all_runs]
        print(f"{key:<8}  {np.mean(vals):>10.4f}  {np.std(vals):>10.4f}"
              f"  {np.min(vals):>10.4f}  {np.max(vals):>10.4f}")

    # Per-run table
    print(f"\n{'Seed':>6}  {'HR@10':>8}  {'MRR@10':>8}  {'NDCG@10':>8}  "
          f"{'DI':>7}  {'Gap':>7}")
    print("-" * 56)
    for r in all_runs:
        gap = r["group"]["F"][10]["HR"] - r["group"]["M"][10]["HR"]
        print(f"{r['seed']:>6}  {r['ranking'][10]['HR']:>8.4f}  "
              f"{r['ranking'][10]['MRR']:>8.4f}  "
              f"{r['ranking'][10]['NDCG']:>8.4f}  "
              f"{r['DI']:>7.4f}  {gap:>+7.4f}")

    # Statistical summary with t-distribution CI (correct for small N)
    hr_vals = [r["ranking"][10]["HR"] for r in all_runs]
    di_vals = [r["DI"] for r in all_runs]
    n = len(hr_vals)
    hr_ci = stats.t.interval(0.95, df=n - 1,
                              loc=np.mean(hr_vals),
                              scale=stats.sem(hr_vals)) if n > 1 else (np.nan, np.nan)
    di_ci = stats.t.interval(0.95, df=n - 1,
                              loc=np.mean(di_vals),
                              scale=stats.sem(di_vals)) if n > 1 else (np.nan, np.nan)
    print(f"\n{'─'*55}")
    print(f"  N={n} seeds  (95% CI uses t-distribution, df={n-1})")
    print(f"  HR@10:  {np.mean(hr_vals):.4f} ± {np.std(hr_vals, ddof=1):.4f}  "
          f"(95% CI: [{hr_ci[0]:.4f}, {hr_ci[1]:.4f}])")
    print(f"  DI@10:  {np.mean(di_vals):.4f} ± {np.std(di_vals, ddof=1):.4f}  "
          f"(95% CI: [{di_ci[0]:.4f}, {di_ci[1]:.4f}])")
    print(f"{'─'*55}\n")

    return {
        "HR_mean":  np.mean(hr_vals),
        "HR_std":   np.std(hr_vals),
        "DI_mean":  np.mean(di_vals),
        "DI_std":   np.std(di_vals),
    }


def compare(our_runs, gender_runs, k=10):
    """
    Paired t-test: GPT-2 (Ours) vs GPT-2+Gender across seeds.

    Parameters
    ----------
    our_runs    : list of run dicts from run() for our model
    gender_runs : list of run dicts from run() for GPT-2+Gender
    k           : cutoff value
    """
    print("\n" + "="*60)
    print("  STATISTICAL COMPARISON: GPT-2 (Ours) vs GPT-2+Gender")
    print("="*60)

    metrics = {
        "HR@10":    ([r["ranking"][k]["HR"]           for r in our_runs],
                     [r["ranking"][k]["HR"]           for r in gender_runs]),
        "HR_M@10":  ([r["group"]["M"][k]["HR"]        for r in our_runs],
                     [r["group"]["M"][k]["HR"]        for r in gender_runs]),
        "HR_F@10":  ([r["group"]["F"][k]["HR"]        for r in our_runs],
                     [r["group"]["F"][k]["HR"]        for r in gender_runs]),
        "MRR@10":   ([r["ranking"][k]["MRR"]          for r in our_runs],
                     [r["ranking"][k]["MRR"]          for r in gender_runs]),
        "DI":       ([r["DI"]                         for r in our_runs],
                     [r["DI"]                         for r in gender_runs]),
    }

    print(f"\n{'Metric':<10}  {'Ours':>12}  {'Gender':>12}  {'Δ':>8}  {'p-value':>10}  {'Sig?':>6}")
    print("-" * 68)

    for name, (ours, gender) in metrics.items():
        t_stat, p_val = stats.ttest_rel(ours, gender)
        delta = np.mean(ours) - np.mean(gender)
        sig = "YES ✓" if p_val < 0.05 else "no"
        print(f"{name:<10}  {np.mean(ours):>6.4f}±{np.std(ours):.4f}  "
              f"{np.mean(gender):>6.4f}±{np.std(gender):.4f}  "
              f"{delta:>+8.4f}  {p_val:>10.4f}  {sig:>6}")
    print()
