import math
from collections import Counter, defaultdict

import numpy as np
from scipy import stats as scipy_stats


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
    scores = {k: {"HR": [], "MRR": [], "NDCG": []} for k in k_values}
    for res in results:
        ranked, gt = res["top_k_recs"], res["ground_truths"]
        for k in k_values:
            scores[k]["HR"].append(hit_at_k(ranked, gt, k))
            scores[k]["MRR"].append(reciprocal_rank_at_k(ranked, gt, k))
            scores[k]["NDCG"].append(ndcg_at_k(ranked, gt, k))

    return {k: {m: float(np.mean(v)) for m, v in scores[k].items()}
            for k in k_values}


def build_user_gender_map(users_df):
    mapping = {}
    for _, row in users_df.iterrows():
        g = row["gender"]
        if g in ("M", "F"):
            mapping[f"User_{int(row['userId'])}"] = g
    return mapping


def compute_group_metrics(results, gender_map, k_values=(1, 3, 5, 10)):
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


def _build_item_score_tables(results, gender_map, k,
                              disadv="F", adv="M"):
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
    items, y_d, y_a, r_d, r_a = _build_item_score_tables(
        results, gender_map, k, disadv, adv)
    if not items:
        return 0.0
    vals = [abs((y_d[i] - r_d[i]) - (y_a[i] - r_a[i])) for i in items]
    return float(np.mean(vals))


def absolute_unfairness(results, gender_map, k, disadv="F", adv="M"):
    items, y_d, y_a, r_d, r_a = _build_item_score_tables(
        results, gender_map, k, disadv, adv)
    if not items:
        return 0.0
    vals = [abs(abs(y_d[i] - r_d[i]) - abs(y_a[i] - r_a[i])) for i in items]
    return float(np.mean(vals))


def underestimation_unfairness(results, gender_map, k, disadv="F", adv="M"):
    items, y_d, y_a, r_d, r_a = _build_item_score_tables(
        results, gender_map, k, disadv, adv)
    if not items:
        return 0.0
    vals = [abs(max(0.0, r_d[i] - y_d[i]) - max(0.0, r_a[i] - y_a[i]))
            for i in items]
    return float(np.mean(vals))


def overestimation_unfairness(results, gender_map, k, disadv="F", adv="M"):
    items, y_d, y_a, r_d, r_a = _build_item_score_tables(
        results, gender_map, k, disadv, adv)
    if not items:
        return 0.0
    vals = [abs(max(0.0, y_d[i] - r_d[i]) - max(0.0, y_a[i] - r_a[i]))
            for i in items]
    return float(np.mean(vals))


def nonparity_unfairness(results, gender_map, k, disadv="F", adv="M"):
    items, y_d, y_a, _, _ = _build_item_score_tables(
        results, gender_map, k, disadv, adv)
    if not items:
        return 0.0
    return float(abs(np.mean(list(y_d.values())) - np.mean(list(y_a.values()))))


def ks_statistic(results, gender_map, k, disadv="F", adv="M"):
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

    p   = {g: hr[g] / total_hr for g in ("M", "F")}
    p_f = {"M": 0.5, "F": 0.5}

    inner = sum(
        (p_f[g] ** alpha) * (p[g] ** (1.0 - alpha))
        for g in ("M", "F")
    )
    gce = (1.0 / (alpha * (1.0 - alpha))) * (inner - 1.0)
    return float(gce)


def compute_all_ilap_metrics(results, gender_map, k=10,
                              disadv="F", adv="M"):
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

def equalized_opportunity(group_metrics, k):
    """|HR_M@K − HR_F@K|.  Target ≈ 0."""
    return abs(group_metrics["M"][k]["HR"] - group_metrics["F"][k]["HR"])


def demographic_parity(results, gender_map, k):
    rec_sets = {"M": set(), "F": set()}
    for res in results:
        g = gender_map.get(res["user"])
        if g in ("M", "F"):
            rec_sets[g].update(res["top_k_recs"][:k])
    inter = rec_sets["M"] & rec_sets["F"]
    union = rec_sets["M"] | rec_sets["F"]
    return len(inter) / len(union) if union else 1.0


def counterfactual_fairness_score(results, gender_map, adj, k):
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

        best_sim, best_v = -1.0, None
        for v in pool:
            sim = jaccard(liked_u, user_liked[v])
            if sim > best_sim:
                best_sim, best_v = sim, v
        if best_v is None:
            continue

        recs_u = set(result_lookup[u][:k])
        recs_v = set(result_lookup[best_v][:k])
        sensitivity = 1.0 - (len(recs_u & recs_v) / k) if k > 0 else 0.0
        sensitivities.append(sensitivity)

    mean_sensitivity = float(np.mean(sensitivities)) if sensitivities else 0.0
    score = 1.0 - mean_sensitivity
    return score, len(sensitivities)


def print_additional_fairness_report(results, group_metrics, gender_map,
                                     adj, k_values=(5, 10), label="Model"):
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


def _safe_entropy_from_counts(counter):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = [c / total for c in counter.values() if c > 0]
    return float(-sum(p * math.log(p) for p in probs))


def path_type_counter(results, k=None):
    counter = Counter()
    for res in results:
        pts = res.get("pattern_types", [])
        if k is not None:
            pts = pts[:k]
        counter.update([pt for pt in pts if pt])
    return counter


def path_diversity_summary(results, k=10):
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

    pl = path_length_fairness(results, gender_map)
    print(f"\n  Path Length by Gender:")
    print(f"    Male   : mean={pl['M']['mean_len']:.2f}  std={pl['M']['std_len']:.2f}  n={pl['M']['n']}")
    print(f"    Female : mean={pl['F']['mean_len']:.2f}  std={pl['F']['std_len']:.2f}  n={pl['F']['n']}")
    print(f"    Gap    : {pl['gap']:.2f}")

    dist = path_type_distribution(results, gender_map, k=k)
    all_rels = sorted(set(dist["M"]) | set(dist["F"]))
    if all_rels:
        print(f"\n  Path Type Distribution (explanation pattern):")
        print(f"    {'Relation':<25} {'Male':>8} {'Female':>8} {'Gap':>8}")
        print(f"    {'-'*52}")
        for rel in all_rels:
            pm = dist["M"].get(rel, 0.0)
            pf = dist["F"].get(rel, 0.0)
            print(f"    {rel:<25} {pm:>8.3f} {pf:>8.3f} {pm - pf:>+8.3f}")

    cga = cross_genre_access(results, gender_map, adj, k=k)
    if cga:
        print(f"\n  Cross-Genre Access (top-5 most biased genres):")
        print(f"    {'Genre':<20} {'Users_M':>8} {'Users_F':>8} {'Access_M':>10} {'Access_F':>10} {'Gap':>8}")
        print(f"    {'-'*68}")
        for row in cga[:5]:
            print(f"    {row['genre']:<20} {row['users_M']:>8} {row['users_F']:>8} "
                  f"{row['access_M']:>10.3f} {row['access_F']:>10.3f} "
                  f"{row['gap_M_minus_F']:>+8.3f}")