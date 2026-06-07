"""
metrics.py — Evaluation metrics for KG-based fair recommendation.

Sections:
    1. Standard ranking metrics      — HR@K, MRR@K, NDCG@K
    2. Group evaluation              — per-gender metric breakdown
    3. ILAP fairness metrics         — DF, VU, AU, UU, OU, NU, KS, GCE
    4. Additional fairness metrics   — HR gap, Demographic Parity,
                                       Counterfactual Fairness
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


def differential_fairness(results, gender_map, k, disadv="F", adv="M", alpha=1.0):
    from collections import Counter
    import math
    import numpy as np

    group_counts = {
        disadv: Counter(),
        adv: Counter(),
    }

    group_users = {
        disadv: 0,
        adv: 0,
    }

    all_items = set()

    for res in results:
        user = res["user"]
        g = gender_map.get(user)

        if g not in (disadv, adv):
            continue

        group_users[g] += 1

        recs = res.get("top_k_recs", [])[:k]
        group_counts[g].update(recs)
        all_items.update(recs)

    if not all_items:
        return 0.0

    total_d = group_users[disadv] * k
    total_a = group_users[adv] * k

    if total_d == 0 or total_a == 0:
        return 0.0

    n_items = len(all_items)

    epsilons = []

    for item in all_items:
        p_d = (group_counts[disadv][item] + alpha) / (
            total_d + alpha * n_items
        )
        p_a = (group_counts[adv][item] + alpha) / (
            total_a + alpha * n_items
        )

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

    Measures divergence between observed per-group HR@K distribution and
    the fair (equal HR) distribution. Uses per-user hit rates so the metric
    is independent of group size, making it comparable across models.

    Returns: gce (float), lower is fairer (0 = perfectly fair HR across groups).
    """
    hits = {"M": [], "F": []}
    for res in results:
        g = gender_map.get(res["user"])
        if g in ("M", "F"):
            hit = 1.0 if set(res["top_k_recs"][:k]) & set(res["ground_truths"]) else 0.0
            hits[g].append(hit)

    hr = {}
    for g in ("M", "F"):
        hr[g] = float(np.mean(hits[g])) if hits[g] else 0.0

    total_hr = hr["M"] + hr["F"]
    if total_hr == 0:
        return 0.0

    # Observed distribution over groups (by HR contribution)
    p   = {g: hr[g] / total_hr for g in ("M", "F")}
    # Fair distribution: equal HR share
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
    Counterfactual fairness via demographic-flip simulation.

    For each user u, we construct a counterfactual user u' that is identical
    to u in every way except their gender node is swapped (M↔F).  We then
    measure how much u's recommendations would change if the system "saw"
    the opposite gender — using the overlap between:
      - u's actual top-K recommendations
      - top-K recommendations of the most taste-similar opposite-gender user

    A high score means the model's recommendations are stable across gender
    (i.e., gender doesn't drive the output) → fairer.

    Score = 1 - mean_gender_sensitivity, where gender_sensitivity per user
    is the fraction of their top-K recs that differ from their closest
    opposite-gender twin.  Score closer to 1.0 = more counterfactually fair.

    Returns: (score, n_pairs)
    """
    result_lookup = {res["user"]: res["top_k_recs"] for res in results}
    user_liked    = {res["user"]: frozenset(adj.get(res["user"], {}).get("likes", set()))
                     for res in results}

    male_users   = [r["user"] for r in results if gender_map.get(r["user"]) == "M"]
    female_users = [r["user"] for r in results if gender_map.get(r["user"]) == "F"]

    def jaccard(a, b):
        u = a | b
        return len(a & b) / len(u) if u else 0.0

    sensitivities = []
    for res in results:
        u      = res["user"]
        gender = gender_map.get(u)
        if gender not in ("M", "F"):
            continue
        pool    = female_users if gender == "M" else male_users
        liked_u = user_liked[u]

        # Find taste-closest opposite-gender user as counterfactual proxy
        best_sim, best_v = -1.0, None
        for v in pool:
            sim = jaccard(liked_u, user_liked[v])
            if sim > best_sim:
                best_sim, best_v = sim, v
        if best_v is None:
            continue

        recs_u = set(result_lookup[u][:k])
        recs_v = set(result_lookup[best_v][:k])
        # Sensitivity = fraction of recs that differ when gender is flipped
        sensitivity = 1.0 - (len(recs_u & recs_v) / k) if k > 0 else 0.0
        sensitivities.append(sensitivity)

    # Score = 1 - mean_sensitivity  (higher = more stable = fairer)
    mean_sensitivity = float(np.mean(sensitivities)) if sensitivities else 0.0
    score = 1.0 - mean_sensitivity
    return score, len(sensitivities)


def print_additional_fairness_report(results, group_metrics, gender_map,
                                     adj, k_values=(5, 10), label="Model"):
    """Print a compact group-level fairness summary.

    ILAP metrics are the main fairness block in this project. This report is
    intentionally short and avoids extra threshold-based metrics.
    """
    print(f"\n{'─'*50}")
    print(f"Group-level Fairness Summary — {label}")
    print(f"{'─'*50}")
    print(f"  {'K':>3} | {'HR_M':>7} | {'HR_F':>7} | {'HR Gap':>7} | "
          f"{'DP':>8} | {'CF':>8}")
    print(f"  {'-'*58}")
    for k in k_values:
        gm, gf = group_metrics["M"][k], group_metrics["F"][k]
        dhr    = abs(gm["HR"] - gf["HR"])
        dp     = demographic_parity(results, gender_map, k)
        cf, _  = counterfactual_fairness_score(results, gender_map, adj, k)
        print(f"  {k:>3} | {gm['HR']:>7.4f} | {gf['HR']:>7.4f} | {dhr:>7.4f} | "
              f"{dp:>8.4f} | {cf:>8.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Path Fairness Metrics  (novel KG-specific contribution)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_entropy_from_counts(counter):
    """Entropy of a Counter as a simple explanation-diversity score."""
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = [c / total for c in counter.values() if c > 0]
    return float(-sum(p * math.log(p) for p in probs))


def path_type_counter(results, k=None):
    """
    Count explanation path types from the stored ``pattern_types`` field.

    This is preferable to reading the first relation in the raw path, because
    most recommendation paths start with ``User -> likes``. The meaningful
    explanation type is the extracted pattern type: genre/director/cf/cast/writer.
    """
    counter = Counter()
    for res in results:
        pts = res.get("pattern_types", [])
        if k is not None:
            pts = pts[:k]
        counter.update([pt for pt in pts if pt])
    return counter


def path_diversity_summary(results, k=10):
    """
    Summarize explanation diversity for the top-K recommendations.

    Returns dominant path type, dominant path ratio, entropy and raw counts.
    Higher entropy and lower dominant ratio mean less explanation concentration.
    """
    counter = path_type_counter(results, k=k)
    total = sum(counter.values())
    if total == 0:
        return {
            "total": 0,
            "dominant_type": None,
            "dominant_ratio": 0.0,
            "entropy": 0.0,
            "counts": counter,
        }
    dominant_type, dominant_count = counter.most_common(1)[0]
    return {
        "total": total,
        "dominant_type": dominant_type,
        "dominant_ratio": float(dominant_count / total),
        "entropy": _safe_entropy_from_counts(counter),
        "counts": counter,
    }


def path_type_distribution(results, gender_map, disadv="F", adv="M", k=None):
    """
    For each gender group, compute the fraction of recommendation explanation
    path types (genre/director/cf/cast/writer) in the top-K list.

    Important: this uses ``result['pattern_types']`` instead of raw path[1],
    because raw paths usually begin with ``likes`` and would hide the actual
    explanation type.
    """
    rel_counts = {"M": Counter(), "F": Counter()}

    for res in results:
        g = gender_map.get(res["user"])
        if g not in ("M", "F"):
            continue
        pts = res.get("pattern_types", [])
        if k is not None:
            pts = pts[:k]
        rel_counts[g].update([pt for pt in pts if pt])

    dist = {}
    for g in ("M", "F"):
        total = sum(rel_counts[g].values())
        dist[g] = {rel: count / total
                   for rel, count in rel_counts[g].items()} if total else {}

    return dist


def print_path_diversity_report(results, k=10, label="Model"):
    """Print overall explanation path diversity for a result list."""
    summary = path_diversity_summary(results, k=k)
    print(f"\n{'─'*50}")
    print(f"Path-Type Diversity — {label} @ K={k}")
    print(f"{'─'*50}")
    print(f"  Total explanations : {summary['total']}")
    print(f"  Dominant type      : {summary['dominant_type']}")
    print(f"  Dominant ratio     : {summary['dominant_ratio']:.4f}")
    print(f"  Path entropy       : {summary['entropy']:.4f}")
    print("\n  Distribution:")
    total = max(1, summary['total'])
    for typ, cnt in summary['counts'].most_common():
        print(f"    {typ:<12} {cnt:>6} ({cnt / total * 100:>5.1f}%)")


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


def _sid_score(pattern_types, k=None):
    """
    Simpson's Index of Diversity (SID) for a single user's top-K path types.

    SID = 1 - Σ n_i*(n_i-1) / N*(N-1)

    where R = number of distinct path pattern types,
          n_i = count of paths of type i,
          N   = total number of paths.

    SID ∈ [0, 1]. Higher = more diverse path explanations.
    SID = 0 when all paths are the same type (no diversity).
    SID = 1 when all paths are different types (maximum diversity).

    Args:
        pattern_types : list of path type strings for one user's recs
        k             : if set, only consider top-k recommendations

    Returns:
        float in [0, 1]
    """
    pts = pattern_types[:k] if k else pattern_types
    N = len(pts)
    if N <= 1:
        return 0.0
    from collections import Counter
    counts = Counter(pts)
    numerator = sum(n * (n - 1) for n in counts.values())
    return 1.0 - numerator / (N * (N - 1))


def compute_path_fairness_metrics(results, gender_map, k=10,
                                  disadv="F", adv="M"):
    """
    Compute all four path fairness metrics from Fu et al. (SIGIR 2020).

    Metrics:
        GRU  — Group Recommendation Unfairness
               Mean HR difference between advantaged and disadvantaged groups.
        GEDU — Group Explanation Diversity Unfairness
               Mean SID difference between advantaged and disadvantaged groups.
        IRU  — Individual Recommendation Unfairness
               Gini coefficient of per-user HR scores (0 = equal, 1 = max unequal).
        IEDU — Individual Explanation Diversity Unfairness
               Gini coefficient of per-user SID scores.

    Args:
        results    : list of result dicts from generate_topk evaluation
        gender_map : dict {"User_<id>" -> "M" or "F"}
        k          : cutoff for top-K recommendations
        disadv     : disadvantaged group label (default "F")
        adv        : advantaged group label (default "M")

    Returns:
        dict with keys: GRU, GEDU, IRU, IEDU,
                        sid_adv, sid_disadv, hr_adv, hr_disadv
    """
    import math

    hr_scores  = {}   # user -> HR@k (1 if any GT in top-k, else 0)
    sid_scores = {}   # user -> SID@k

    for res in results:
        user    = res["user"]
        gt_set  = set(res["ground_truths"])
        recs    = res["top_k_recs"][:k]
        pts     = res.get("pattern_types", [])[:k]

        hr_scores[user]  = 1.0 if any(m in gt_set for m in recs) else 0.0
        sid_scores[user] = _sid_score(pts, k)

    adv_users    = [r["user"] for r in results if gender_map.get(r["user"]) == adv]
    disadv_users = [r["user"] for r in results if gender_map.get(r["user"]) == disadv]

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    hr_adv    = mean([hr_scores[u]  for u in adv_users    if u in hr_scores])
    hr_disadv = mean([hr_scores[u]  for u in disadv_users if u in hr_scores])
    sid_adv   = mean([sid_scores[u] for u in adv_users    if u in sid_scores])
    sid_disadv= mean([sid_scores[u] for u in disadv_users if u in sid_scores])

    # GRU: |mean_HR_adv - mean_HR_disadv| (Eq. 2)
    GRU  = abs(hr_adv - hr_disadv)

    # GEDU: |mean_SID_adv - mean_SID_disadv| (Eq. 3)
    GEDU = abs(sid_adv - sid_disadv)

    # IRU: Gini coefficient of HR scores (Eq. 4)
    all_hr  = [hr_scores[r["user"]]  for r in results if r["user"] in hr_scores]
    all_sid = [sid_scores[r["user"]] for r in results if r["user"] in sid_scores]

    def gini(vals):
        n = len(vals)
        if n == 0 or sum(vals) == 0:
            return 0.0
        vals_sorted = sorted(vals)
        total = sum(vals_sorted)
        numerator = sum(abs(vals_sorted[i] - vals_sorted[j])
                        for i in range(n) for j in range(n))
        return numerator / (2 * n * total)

    IRU  = gini(all_hr)
    IEDU = gini(all_sid)

    return {
        "GRU":        GRU,
        "GEDU":       GEDU,
        "IRU":        IRU,
        "IEDU":       IEDU,
        "hr_adv":     hr_adv,
        "hr_disadv":  hr_disadv,
        "sid_adv":    sid_adv,
        "sid_disadv": sid_disadv,
        "adv":        adv,
        "disadv":     disadv,
    }


def print_path_fairness_report(results, gender_map,
                               results_baseline=None,
                               label="GPT-2 (Ours)",
                               label_baseline="GPT-2 + Gender",
                               k=10):
    """
    Print path-level fairness report using Fu et al. (SIGIR 2020) metrics.

    Shows GRU, GEDU, IRU, IEDU for the main model and optionally
    compares with a baseline.
    """
    sep = "─" * 62

    m = compute_path_fairness_metrics(results, gender_map, k=k)

    print(f"\n{sep}")
    print(f"  Path-Level Fairness — Fu et al. SIGIR 2020 @ K={k}")
    print(f"  Reference: Fairness-Aware Explainable Rec. over KGs")
    print(sep)

    # Per-group SID
    print(f"\n  Simpson's Index of Diversity (SID) — explanation diversity:")
    print(f"    {m['adv']} (advantaged) : {m['sid_adv']:.4f}")
    print(f"    {m['disadv']} (disadvantaged): {m['sid_disadv']:.4f}")
    print(f"    Gap                : {m['sid_adv'] - m['sid_disadv']:+.4f}")

    # Per-group HR
    print(f"\n  HR@{k} by group:")
    print(f"    {m['adv']} (advantaged) : {m['hr_adv']:.4f}")
    print(f"    {m['disadv']} (disadvantaged): {m['hr_disadv']:.4f}")
    print(f"    Gap                : {m['hr_adv'] - m['hr_disadv']:+.4f}")

    # Four metrics
    print(f"\n  {'Metric':<8} {'Value':>8}  {'Description'}")
    print(f"  {'-'*56}")
    print(f"  {'GRU':<8} {m['GRU']:>8.4f}  Group Recommendation Unfairness (lower=fairer)")
    print(f"  {'GEDU':<8} {m['GEDU']:>8.4f}  Group Explanation Diversity Unfairness (lower=fairer)")
    print(f"  {'IRU':<8} {m['IRU']:>8.4f}  Individual Rec. Unfairness/Gini (lower=fairer)")
    print(f"  {'IEDU':<8} {m['IEDU']:>8.4f}  Individual Explanation Diversity Unfairness (lower=fairer)")

    # Baseline comparison
    if results_baseline is not None:
        mb = compute_path_fairness_metrics(results_baseline, gender_map, k=k)
        print(f"\n  {'Metric':<8} {label:>18} {label_baseline:>18} {'Δ':>8}")
        print(f"  {'-'*58}")
        for key in ("GRU", "GEDU", "IRU", "IEDU"):
            delta = m[key] - mb[key]
            better = "✓" if delta < 0 else ("=" if delta == 0 else "✗")
            print(f"  {key:<8} {m[key]:>18.4f} {mb[key]:>18.4f} {delta:>+8.4f} {better}")

    print(f"\n{sep}\n")
    return m


def disparate_impact(group_metrics, k):
    """Disparate impact: min(HR_M, HR_F) / max(HR_M, HR_F). >=0.8 is fair."""
    hr_m = group_metrics["M"][k]["HR"]
    hr_f = group_metrics["F"][k]["HR"]
    denom = max(hr_m, hr_f)
    return min(hr_m, hr_f) / denom if denom > 0 else 1.0


# ---------------------------------------------------------------------------
# Sampled Evaluation (literature-standard protocol)
# ---------------------------------------------------------------------------

def evaluate_ranking_sampled(results, all_movies, adj, k_values=(10,),
                              n_negatives=99, seed=42):
    """
    Sampled evaluation: 1 positive + n_negatives random negatives.
    Computes MRR and NDCG only. HR uses full-catalog evaluation.
    Standard protocol in PGPR, PEARLM, CAFE.
    """
    import random as _random
    import math as _math
    rng = _random.Random(seed)
    all_movies_list = list(all_movies)
    metrics = {k: {"MRR": 0.0, "NDCG": 0.0} for k in k_values}
    n_pairs = 0

    for res in results:
        user_node     = res["user"]
        ground_truths = res["ground_truths"]
        top_k_recs    = res["top_k_recs"]
        liked_all = set(adj.get(user_node, {}).get("likes", set()))
        liked_all.update(ground_truths)
        rec_rank = {movie: len(top_k_recs) - rank
                    for rank, movie in enumerate(top_k_recs)}
        for pos_item in ground_truths:
            negatives_pool = [m for m in all_movies_list
                              if m not in liked_all and m != pos_item]
            sampled_negs = (rng.sample(negatives_pool, n_negatives)
                            if len(negatives_pool) >= n_negatives
                            else negatives_pool)
            pool = [pos_item] + sampled_negs
            pool_scored = sorted(pool, key=lambda m: rec_rank.get(m, -1), reverse=True)
            pos_rank = pool_scored.index(pos_item) + 1
            for k in k_values:
                if pos_rank <= k:
                    metrics[k]["MRR"]  += 1.0 / pos_rank
                    metrics[k]["NDCG"] += 1.0 / _math.log2(pos_rank + 1)
            n_pairs += 1

    if n_pairs > 0:
        for k in k_values:
            for m in ("MRR", "NDCG"):
                metrics[k][m] /= n_pairs
    return metrics


def evaluate_ranking_sampled_group(results, all_movies, adj, user_gender_map,
                                   k_values=(10,), n_negatives=99, seed=42):
    """Sampled evaluation split by gender group."""
    male_results   = [r for r in results if user_gender_map.get(r["user"]) == "M"]
    female_results = [r for r in results if user_gender_map.get(r["user"]) == "F"]
    return {
        "M": evaluate_ranking_sampled(male_results, all_movies, adj, k_values, n_negatives, seed),
        "F": evaluate_ranking_sampled(female_results, all_movies, adj, k_values, n_negatives, seed),
    }
