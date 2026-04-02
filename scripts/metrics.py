"""
metrics.py — Evaluation metrics for KG-based fair recommendation.

Sections:
    1. Standard ranking metrics      — HR@K, MRR@K, NDCG@K
    2. Group evaluation              — per-gender metric breakdown
    3. ILAP fairness metrics         — DF, VU, AU, UU, OU, NU, KS, GCE
    4. Additional fairness metrics   — Disparate Impact, Equalized Opportunity,
                                       Demographic Parity, Counterfactual Fairness
    5. Path fairness metrics         — path type distribution, cross-genre access,
                                       path length fairness, path score KS
"""

import math
from collections import Counter, defaultdict

import numpy as np
from scipy import stats as scipy_stats


# ══════════════════════════════════════════════════════════════════════════════
# 1. Standard Ranking Metrics
# ══════════════════════════════════════════════════════════════════════════════

def hit_at_k(ranked_list, ground_truths, k):
    return 1.0 if set(ranked_list[:k]) & set(ground_truths) else 0.0


def reciprocal_rank_at_k(ranked_list, ground_truths, k):
    gt_set = set(ground_truths)
    for i, item in enumerate(ranked_list[:k]):
        if item in gt_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_list, ground_truths, k):
    gt_set = set(ground_truths)
    dcg    = sum(1.0 / math.log2(i + 2)
                 for i, item in enumerate(ranked_list[:k]) if item in gt_set)
    ideal  = min(len(gt_set), k)
    idcg   = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(results, k_values=(1, 3, 5, 10)):
    """
    Compute HR, MRR, NDCG at multiple K values.

    Args:
        results: list of dicts with keys 'top_k_recs' and 'ground_truths'
        k_values: iterable of K values

    Returns:
        dict {k: {'HR': float, 'MRR': float, 'NDCG': float}}
    """
    scores = {k: {"HR": [], "MRR": [], "NDCG": []} for k in k_values}
    for res in results:
        ranked, gt = res["top_k_recs"], res["ground_truths"]
        for k in k_values:
            scores[k]["HR"].append(hit_at_k(ranked, gt, k))
            scores[k]["MRR"].append(reciprocal_rank_at_k(ranked, gt, k))
            scores[k]["NDCG"].append(ndcg_at_k(ranked, gt, k))

    return {k: {m: float(np.mean(v)) for m, v in scores[k].items()}
            for k in k_values}


# ══════════════════════════════════════════════════════════════════════════════
# 2. Group Evaluation (per gender)
# ══════════════════════════════════════════════════════════════════════════════

def build_user_gender_map(users_df):
    """
    Build a mapping from 'User_<id>' node string to 'M' or 'F'.

    Args:
        users_df: DataFrame with columns ['userId', 'gender']
    """
    mapping = {}
    for _, row in users_df.iterrows():
        g = row["gender"]
        if g in ("M", "F"):
            mapping[f"User_{int(row['userId'])}"] = g
    return mapping


def compute_group_metrics(results, gender_map, k_values=(1, 3, 5, 10)):
    """
    Per-gender HR@K, MRR@K, NDCG@K.

    Returns:
        dict {gender: {k: {'HR', 'MRR', 'NDCG', 'n'}}}
    """
    scores = {g: {k: {"HR": [], "MRR": [], "NDCG": []}
                  for k in k_values} for g in ("M", "F")}

    for res in results:
        g = gender_map.get(res["user"])
        if g not in ("M", "F"):
            continue
        ranked, gt = res["top_k_recs"], res["ground_truths"]
        for k in k_values:
            scores[g][k]["HR"].append(hit_at_k(ranked, gt, k))
            scores[g][k]["MRR"].append(reciprocal_rank_at_k(ranked, gt, k))
            scores[g][k]["NDCG"].append(ndcg_at_k(ranked, gt, k))

    out = {}
    for g in ("M", "F"):
        out[g] = {}
        for k in k_values:
            n = len(scores[g][k]["HR"])
            out[g][k] = {
                "HR":   float(np.mean(scores[g][k]["HR"]))   if n else 0.0,
                "MRR":  float(np.mean(scores[g][k]["MRR"]))  if n else 0.0,
                "NDCG": float(np.mean(scores[g][k]["NDCG"])) if n else 0.0,
                "n":    n,
            }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. ILAP Fairness Metrics
# ══════════════════════════════════════════════════════════════════════════════
#
# Notation (shared across all ILAP metrics):
#   g   = disadvantaged group (default: Female)
#   ¬g  = advantaged group    (default: Male)
#   y_j = avg predicted score for item j from group
#   r_j = avg actual relevance for item j from group
#
# For a top-K recommender:
#   predicted score  = 1 if item appears in user's top-K list, else 0
#   actual relevance = 1 if item is in user's test ground truth, else 0


def _build_item_score_tables(results, gender_map, k,
                              disadv="F", adv="M"):
    """
    For every item that appears in any user's top-K or ground truth,
    compute per-group average predicted score (y) and actual relevance (r).

    Returns:
        items        : sorted list of all items
        y_disadv     : dict[item → avg predicted score, disadvantaged group]
        y_adv        : dict[item → avg predicted score, advantaged group]
        r_disadv     : dict[item → avg actual relevance, disadvantaged group]
        r_adv        : dict[item → avg actual relevance, advantaged group]
    """
    pred   = {disadv: defaultdict(list), adv: defaultdict(list)}
    actual = {disadv: defaultdict(list), adv: defaultdict(list)}

    all_items = set()
    for res in results:
        g = gender_map.get(res["user"])
        if g not in (disadv, adv):
            continue
        top_k  = set(res["top_k_recs"][:k])
        gt_set = set(res["ground_truths"])
        items  = top_k | gt_set
        all_items |= items
        for item in items:
            pred[g][item].append(1.0 if item in top_k  else 0.0)
            actual[g][item].append(1.0 if item in gt_set else 0.0)

    items = sorted(all_items)

    def avg(d, item):
        vals = d.get(item, [])
        return float(np.mean(vals)) if vals else 0.0

    y_d = {i: avg(pred[disadv],   i) for i in items}
    y_a = {i: avg(pred[adv],      i) for i in items}
    r_d = {i: avg(actual[disadv], i) for i in items}
    r_a = {i: avg(actual[adv],    i) for i in items}

    return items, y_d, y_a, r_d, r_a


def differential_fairness(results, gender_map, k, disadv="F", adv="M", eps=1e-9):
    """
    Differential Fairness (DF).

    ε = mean_item |log P(rec | adv) - log P(rec | disadv)|
    Perfect fairness = ε = 0; smaller is fairer.

    Matches ILAP reference: epsilon_values.mean() (not max).
    """
    items, y_d, y_a, _, _ = _build_item_score_tables(
        results, gender_map, k, disadv, adv)

    if not items:
        return 0.0

    epsilons = []
    for item in items:
        p_d = max(y_d[item], eps)
        p_a = max(y_a[item], eps)
        epsilons.append(abs(math.log(p_d) - math.log(p_a)))

    return float(np.mean(epsilons))


def value_unfairness(results, gender_map, k, disadv="F", adv="M"):
    """
    Value Unfairness (VU).

    Uval = (1/n) Σ_j |(y_d_j - r_d_j) - (y_a_j - r_a_j)|

    Matches ILAP reference: mean of absolute per-item differences.
    Zero = fair; higher = more unfair.
    """
    items, y_d, y_a, r_d, r_a = _build_item_score_tables(
        results, gender_map, k, disadv, adv)
    if not items:
        return 0.0
    vals = [abs((y_d[i] - r_d[i]) - (y_a[i] - r_a[i])) for i in items]
    return float(np.mean(vals))


def absolute_unfairness(results, gender_map, k, disadv="F", adv="M"):
    """
    Absolute Unfairness (AU).

    Uabs = (1/n) Σ_j ||y_d_j - r_d_j| - |y_a_j - r_a_j||

    Matches ILAP reference: mean of absolute differences of absolute errors.
    Zero = fair; higher = more unfair.
    """
    items, y_d, y_a, r_d, r_a = _build_item_score_tables(
        results, gender_map, k, disadv, adv)
    if not items:
        return 0.0
    vals = [abs(abs(y_d[i] - r_d[i]) - abs(y_a[i] - r_a[i])) for i in items]
    return float(np.mean(vals))


def underestimation_unfairness(results, gender_map, k, disadv="F", adv="M"):
    """
    Underestimation Unfairness (UU).

    Uunder = (1/n) Σ_j |max(0, r_d_j - y_d_j) - max(0, r_a_j - y_a_j)|

    Matches ILAP reference: mean of absolute per-item underestimation gaps.
    Zero = fair; higher = more unfair.
    """
    items, y_d, y_a, r_d, r_a = _build_item_score_tables(
        results, gender_map, k, disadv, adv)
    if not items:
        return 0.0
    vals = [abs(max(0.0, r_d[i] - y_d[i]) - max(0.0, r_a[i] - y_a[i]))
            for i in items]
    return float(np.mean(vals))


def overestimation_unfairness(results, gender_map, k, disadv="F", adv="M"):
    """
    Overestimation Unfairness (OU).

    Uover = (1/n) Σ_j |max(0, y_d_j - r_d_j) - max(0, y_a_j - r_a_j)|

    Matches ILAP reference: mean of absolute per-item overestimation gaps.
    Zero = fair; higher = more unfair.
    """
    items, y_d, y_a, r_d, r_a = _build_item_score_tables(
        results, gender_map, k, disadv, adv)
    if not items:
        return 0.0
    vals = [abs(max(0.0, y_d[i] - r_d[i]) - max(0.0, y_a[i] - r_a[i]))
            for i in items]
    return float(np.mean(vals))


def nonparity_unfairness(results, gender_map, k, disadv="F", adv="M"):
    """
    NonParity Unfairness (NU).

    Upar = |avg_y_disadv - avg_y_adv|

    Difference in average predicted scores between groups.
    """
    items, y_d, y_a, _, _ = _build_item_score_tables(
        results, gender_map, k, disadv, adv)
    if not items:
        return 0.0
    return float(abs(np.mean(list(y_d.values())) - np.mean(list(y_a.values()))))


def ks_statistic(results, gender_map, k, disadv="F", adv="M"):
    """
    Kolmogorov-Smirnov Statistic.

    Computes the KS statistic between the per-user utility (NDCG@K)
    distributions of the two gender groups.

    KS = max |CDF_M(x) - CDF_F(x)|

    Returns: (ks_stat, p_value)
    """
    utilities = {"M": [], "F": []}
    for res in results:
        g = gender_map.get(res["user"])
        if g not in ("M", "F"):
            continue
        utilities[g].append(ndcg_at_k(res["top_k_recs"], res["ground_truths"], k))

    if not utilities["M"] or not utilities["F"]:
        return 0.0, 1.0

    ks_stat, p_val = scipy_stats.ks_2samp(utilities[adv], utilities[disadv])
    return float(ks_stat), float(p_val)


def generalized_cross_entropy(results, gender_map, k,
                               disadv="F", adv="M", alpha=0.5):
    """
    Generalized Cross Entropy (GCE) — Hellinger distance (α=0.5).

    GCE(M, a) = (1 / α(1-α)) * [Σ_aj p_f^α(aj) * p^(1-α)(aj) - 1]

    With uniform fair distribution p_f = [0.5, 0.5] for binary groups.

    Returns: gce (float), lower is fairer.
    """
    # p(group) = fraction of total recommendations that went to each group
    counts = {"M": 0, "F": 0}
    for res in results:
        g = gender_map.get(res["user"])
        if g in ("M", "F"):
            counts[g] += len(res["top_k_recs"][:k])

    total = counts["M"] + counts["F"]
    if total == 0:
        return 0.0

    p   = {g: counts[g] / total for g in ("M", "F")}
    p_f = {"M": 0.5, "F": 0.5}

    inner = sum(
        (p_f[g] ** alpha) * (p[g] ** (1.0 - alpha))
        for g in ("M", "F")
    )
    gce = (1.0 / (alpha * (1.0 - alpha))) * (inner - 1.0)
    return float(gce)


def compute_all_ilap_metrics(results, gender_map, k=10,
                              disadv="F", adv="M"):
    """
    Compute all 7 ILAP fairness metrics at once.

    Returns:
        dict with keys: DF, VU, AU, UU, OU, NU, KS, KS_pval, GCE
    """
    ks_stat, ks_p = ks_statistic(results, gender_map, k, disadv, adv)
    return {
        "DF":      differential_fairness(results,       gender_map, k, disadv, adv),
        "VU":      value_unfairness(results,             gender_map, k, disadv, adv),
        "AU":      absolute_unfairness(results,          gender_map, k, disadv, adv),
        "UU":      underestimation_unfairness(results,   gender_map, k, disadv, adv),
        "OU":      overestimation_unfairness(results,    gender_map, k, disadv, adv),
        "NU":      nonparity_unfairness(results,         gender_map, k, disadv, adv),
        "KS":      ks_stat,
        "KS_pval": ks_p,
        "GCE":     generalized_cross_entropy(results,   gender_map, k, disadv, adv),
    }


def print_ilap_report(ilap_dict, k=10, label="Model"):
    print(f"\n{'─'*50}")
    print(f"ILAP Fairness Metrics — {label} @ K={k}")
    print(f"{'─'*50}")
    descs = {
        "DF":  ("Differential Fairness ε",      "lower = fairer"),
        "VU":  ("Value Unfairness",              "≈0 = fair"),
        "AU":  ("Absolute Unfairness",           "≈0 = fair"),
        "UU":  ("Underestimation Unfairness",    "≈0 = fair"),
        "OU":  ("Overestimation Unfairness",     "≈0 = fair"),
        "NU":  ("NonParity Unfairness",          "≈0 = fair"),
        "KS":  ("KS Statistic",                  f"p={ilap_dict.get('KS_pval', 0):.3f}"),
        "GCE": ("Generalized Cross Entropy",     "lower = fairer"),
    }
    for key, (name, hint) in descs.items():
        if key in ilap_dict:
            print(f"  {name:<35} {ilap_dict[key]:>8.4f}  ({hint})")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Additional Fairness Metrics
# ══════════════════════════════════════════════════════════════════════════════

def disparate_impact(group_metrics, k):
    """
    min(HR_M, HR_F) / max(HR_M, HR_F).  Target ≥ 0.8 (80% rule).
    """
    hr_m = group_metrics["M"][k]["HR"]
    hr_f = group_metrics["F"][k]["HR"]
    denom = max(hr_m, hr_f)
    return min(hr_m, hr_f) / denom if denom > 0 else 1.0


def equalized_opportunity(group_metrics, k):
    """|HR_M@K − HR_F@K|.  Target ≈ 0."""
    return abs(group_metrics["M"][k]["HR"] - group_metrics["F"][k]["HR"])


def demographic_parity(results, gender_map, k):
    """
    Jaccard similarity of aggregate top-K recommendation sets for M vs F.
    High = both groups are recommended from the same catalog space.
    """
    rec_sets = {"M": set(), "F": set()}
    for res in results:
        g = gender_map.get(res["user"])
        if g in ("M", "F"):
            rec_sets[g].update(res["top_k_recs"][:k])
    inter = rec_sets["M"] & rec_sets["F"]
    union = rec_sets["M"] | rec_sets["F"]
    return len(inter) / len(union) if union else 1.0


def counterfactual_fairness_score(results, gender_map, adj, k):
    """
    For each user u, find the most similar user of OPPOSITE gender by Jaccard
    similarity of their training liked movies.

    Score = mean overlap fraction of top-K recs between u and its closest
    opposite-gender neighbour.  Higher → more counterfactually fair.  Target > 0.5.

    Returns: (score, n_pairs)
    """
    result_lookup = {res["user"]: res["top_k_recs"] for res in results}
    user_liked    = {res["user"]: frozenset(adj[res["user"]].get("likes", set()))
                     for res in results}

    male_users   = [r["user"] for r in results if gender_map.get(r["user"]) == "M"]
    female_users = [r["user"] for r in results if gender_map.get(r["user"]) == "F"]

    def jaccard(a, b):
        u = a | b
        return len(a & b) / len(u) if u else 0.0

    overlaps = []
    for res in results:
        u      = res["user"]
        gender = gender_map.get(u)
        if gender not in ("M", "F"):
            continue
        pool     = female_users if gender == "M" else male_users
        liked_u  = user_liked[u]
        best_sim = -1.0
        best_v   = None
        for v in pool:
            sim = jaccard(liked_u, user_liked[v])
            if sim > best_sim:
                best_sim, best_v = sim, v
        if best_v is None:
            continue
        recs_u = set(result_lookup[u][:k])
        recs_v = set(result_lookup[best_v][:k])
        overlaps.append(len(recs_u & recs_v) / k if k > 0 else 0.0)

    score = float(np.mean(overlaps)) if overlaps else 0.0
    return score, len(overlaps)


def print_additional_fairness_report(results, group_metrics, gender_map,
                                     adj, k_values=(5, 10), label="Model"):
    print(f"\n{'─'*50}")
    print(f"Additional Fairness Metrics — {label}")
    print(f"{'─'*50}")
    print(f"  {'K':>3} | {'ΔHR':>7} | {'ΔMRR':>7} | {'ΔNDCG':>8} | "
          f"{'DI':>8} | {'EqOpp':>8} | {'DP':>8} | {'CF':>8}")
    print(f"  {'-'*72}")
    for k in k_values:
        gm, gf = group_metrics["M"][k], group_metrics["F"][k]
        dhr    = abs(gm["HR"]   - gf["HR"])
        dmrr   = abs(gm["MRR"]  - gf["MRR"])
        dndcg  = abs(gm["NDCG"] - gf["NDCG"])
        di     = disparate_impact(group_metrics, k)
        eo     = equalized_opportunity(group_metrics, k)
        dp     = demographic_parity(results, gender_map, k)
        cf, _  = counterfactual_fairness_score(results, gender_map, adj, k)
        print(f"  {k:>3} | {dhr:>7.4f} | {dmrr:>7.4f} | {dndcg:>8.4f} | "
              f"{di:>8.4f} | {eo:>8.4f} | {dp:>8.4f} | {cf:>8.4f}")
        di_flag = "PASS" if di >= 0.8 else "FAIL"
        print(f"  {'':>3}   DI {di_flag} (≥0.8 required)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Path Fairness Metrics  (novel KG-specific contribution)
# ══════════════════════════════════════════════════════════════════════════════

def path_type_distribution(results, gender_map, disadv="F", adv="M"):
    """
    For each gender group, compute what fraction of recommendation paths
    use each relation type as the first KG hop (index 1 in the path list).

    Reveals which relation types are gender-skewed.

    Args:
        results: list of dicts with key 'paths' (list of path token lists)
                 and 'user'

    Returns:
        dist: dict {gender: Counter of relation → fraction}
    """
    rel_counts = {"M": Counter(), "F": Counter()}

    for res in results:
        g = gender_map.get(res["user"])
        if g not in ("M", "F"):
            continue
        for path in res.get("paths", []):
            if len(path) >= 2:
                first_rel = path[1]   # User → [first_rel] → ...
                rel_counts[g][first_rel] += 1

    dist = {}
    for g in ("M", "F"):
        total = sum(rel_counts[g].values())
        dist[g] = {rel: count / total
                   for rel, count in rel_counts[g].items()} if total else {}

    return dist


def cross_genre_access(results, gender_map, adj, genre_key="hasGenre",
                        k=10, disadv="F", adv="M"):
    """
    Among users who both like at least one Action/Drama/etc. movie in training,
    compare the fraction whose top-K recommendations include movies of that genre.

    This tests proxy bias: do female Action fans get Action recommendations
    as often as male Action fans?

    Args:
        results:    list of dicts with 'user', 'top_k_recs'
        gender_map: dict user_node → 'M'|'F'
        adj:        KG adjacency (to get user's liked genres)

    Returns:
        DataFrame-like list of dicts per genre:
            {genre, access_M, access_F, gap, gap_direction}
    """
    # Build: genre → {user_node → has_genre_in_liked_movies}
    genre_to_users = defaultdict(lambda: {"M": [], "F": []})

    for res in results:
        u = res["user"]
        g = gender_map.get(u)
        if g not in ("M", "F"):
            continue
        liked_movies  = adj[u].get("likes", set())
        liked_genres  = set()
        for movie in liked_movies:
            liked_genres |= adj[movie].get(genre_key, set())

        top_k_genres = set()
        for movie in res["top_k_recs"][:k]:
            top_k_genres |= adj.get(movie, {}).get(genre_key, set())

        genre_to_users["__all__"][g].append(
            {"user": u, "liked_genres": liked_genres, "rec_genres": top_k_genres}
        )

    # Collect all genres seen in liked movies
    all_genres = set()
    for g in ("M", "F"):
        for entry in genre_to_users["__all__"][g]:
            all_genres |= entry["liked_genres"]

    rows = []
    for genre in sorted(all_genres):
        counts = {"M": {"users": 0, "got_rec": 0},
                  "F": {"users": 0, "got_rec": 0}}
        for g in ("M", "F"):
            for entry in genre_to_users["__all__"][g]:
                if genre in entry["liked_genres"]:
                    counts[g]["users"] += 1
                    if genre in entry["rec_genres"]:
                        counts[g]["got_rec"] += 1

        access_m = (counts["M"]["got_rec"] / counts["M"]["users"]
                    if counts["M"]["users"] > 0 else 0.0)
        access_f = (counts["F"]["got_rec"] / counts["F"]["users"]
                    if counts["F"]["users"] > 0 else 0.0)
        gap = access_m - access_f

        rows.append({
            "genre":          genre,
            "users_M":        counts["M"]["users"],
            "users_F":        counts["F"]["users"],
            "access_M":       round(access_m, 4),
            "access_F":       round(access_f, 4),
            "gap_M_minus_F":  round(gap, 4),
        })

    rows.sort(key=lambda x: abs(x["gap_M_minus_F"]), reverse=True)
    return rows


def path_length_fairness(results, gender_map, disadv="F", adv="M"):
    """
    Compare average recommendation path length between gender groups.

    Longer paths = weaker evidence = less confident recommendations.
    A gap here means the KG has sparser coverage of one group's preferences.

    Returns:
        dict {gender: {'mean_len', 'std_len', 'n'}}
    """
    lengths = {"M": [], "F": []}

    for res in results:
        g = gender_map.get(res["user"])
        if g not in ("M", "F"):
            continue
        for path in res.get("paths", []):
            lengths[g].append(len(path))

    out = {}
    for g in ("M", "F"):
        arr = lengths[g]
        out[g] = {
            "mean_len": float(np.mean(arr))   if arr else 0.0,
            "std_len":  float(np.std(arr))    if arr else 0.0,
            "n":        len(arr),
        }
    out["gap"] = abs(out["M"]["mean_len"] - out["F"]["mean_len"])
    return out


def path_score_ks(results, gender_map, disadv="F", adv="M"):
    """
    KS test on per-user average path score distributions by gender.

    A significant KS stat means the model assigns systematically
    different confidence scores to one gender's recommendations.

    Returns: (ks_stat, p_value)
    """
    scores = {"M": [], "F": []}

    for res in results:
        g = gender_map.get(res["user"])
        if g not in ("M", "F"):
            continue
        user_scores = [s for _, _, s in res.get("recs_with_scores", [])]
        if user_scores:
            scores[g].append(float(np.mean(user_scores)))

    if not scores["M"] or not scores["F"]:
        return 0.0, 1.0

    ks_stat, p_val = scipy_stats.ks_2samp(scores[adv], scores[disadv])
    return float(ks_stat), float(p_val)


def print_path_fairness_report(results, gender_map, adj, k=10, label="Model"):
    print(f"\n{'─'*50}")
    print(f"Path Fairness Metrics — {label} @ K={k}")
    print(f"{'─'*50}")

    # Path length
    pl = path_length_fairness(results, gender_map)
    print(f"\n  Path Length by Gender:")
    print(f"    Male   : mean={pl['M']['mean_len']:.2f}  std={pl['M']['std_len']:.2f}  n={pl['M']['n']}")
    print(f"    Female : mean={pl['F']['mean_len']:.2f}  std={pl['F']['std_len']:.2f}  n={pl['F']['n']}")
    print(f"    Gap    : {pl['gap']:.2f}")

    # Path type distribution
    dist = path_type_distribution(results, gender_map)
    all_rels = sorted(set(dist["M"]) | set(dist["F"]))
    if all_rels:
        print(f"\n  Path Type Distribution (first hop):")
        print(f"    {'Relation':<25} {'Male':>8} {'Female':>8} {'Gap':>8}")
        print(f"    {'-'*52}")
        for rel in all_rels:
            pm = dist["M"].get(rel, 0.0)
            pf = dist["F"].get(rel, 0.0)
            print(f"    {rel:<25} {pm:>8.3f} {pf:>8.3f} {pm - pf:>+8.3f}")

    # Cross-genre access
    cga = cross_genre_access(results, gender_map, adj, k=k)
    if cga:
        print(f"\n  Cross-Genre Access (top-5 most biased genres):")
        print(f"    {'Genre':<20} {'Users_M':>8} {'Users_F':>8} {'Access_M':>10} {'Access_F':>10} {'Gap':>8}")
        print(f"    {'-'*68}")
        for row in cga[:5]:
            print(f"    {row['genre']:<20} {row['users_M']:>8} {row['users_F']:>8} "
                  f"{row['access_M']:>10.3f} {row['access_F']:>10.3f} "
                  f"{row['gap_M_minus_F']:>+8.3f}")
