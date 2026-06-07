"""
kg_builder.py — Knowledge Graph construction for movie recommendation.

Pipeline:
    IMDb offline pipeline:
        - uses local IMDb TSV files
        - no network calls, no rate limits

Main entry points:
    load_or_build_kg_imdb(movies_sub, imdb_dir="data/imdb",
                          cache_path="cache/kg_cache_imdb.pkl")
        -> IMDb KG

    build_adj(readable_triples, user_item_edges, user_info)
        -> adjacency graph
"""

import os
import pickle
import re
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Relation definitions
# ---------------------------------------------------------------------------

BASE_RELS = {
    "likes",
    "hasGenre",
    "directedBy",
    "year",
    "country",
    "hasCast",
    "hasComposer",
    "writtenBy",
    "hasProducer",
    "hasCinematographer",
    "hasGender",
}

_KG_RELATIONS = {
    "hasGenre",
    "directedBy",
    "year",
    "country",
    "hasCast",
    "hasComposer",
    "writtenBy",
    "hasProducer",
    "hasCinematographer",
}


# ---------------------------------------------------------------------------
# Adjacency Graph
# ---------------------------------------------------------------------------


def build_adj(readable_triples, user_item_edges, user_info):
    """
    Build KG adjacency graph from readable triples, user-item edges,
    and user demographic info.

    Args:
        readable_triples : list of (head, relation, tail)
        user_item_edges  : list of ('User_<id>', 'likes', movie_title)
        user_info        : dict {userId -> {'gender': 'M'|'F', ...}}

    Returns:
        adj : defaultdict[node -> defaultdict[relation -> set(tails)]]
    """
    def rev_rel(r):
        return f"rev_{r}"

    adj = defaultdict(lambda: defaultdict(set))

    # User-item training edges.
    for h, r, t in user_item_edges:
        adj[h][r].add(t)
        adj[t][rev_rel(r)].add(h)

    # KG triples.
    qcode = re.compile(r"^Q\d+$")

    for h, r, t in readable_triples:
        if r not in _KG_RELATIONS:
            continue

        if qcode.match(str(h)) or qcode.match(str(t)):
            continue

        adj[h][r].add(t)
        adj[t][rev_rel(r)].add(h)

    # Gender edges for fairness auditing and the optional gender baseline.
    for uid, info in user_info.items():
        g = info.get("gender")

        if g in ("M", "F"):
            u = f"User_{int(uid)}"
            gender_node = f"Gender_{g}"
            adj[u]["hasGender"].add(gender_node)
            adj[gender_node]["rev_hasGender"].add(u)

    node_count = len(adj)
    edge_count = sum(len(v) for h in adj for v in adj[h].values())
    rel_counts = Counter(r for h in adj for r in adj[h])

    print(f"Adj → nodes: {node_count}, total edge-instances: {edge_count}")
    for r, c in rel_counts.most_common():
        print(f"  {r}: {c}")

    return adj


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def save_kg_cache(cache_path, qid_map, label_cache, kg_triples, readable_triples):
    """Save KG data to pickle cache."""
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    kg_data = {
        "qid_map": qid_map,
        "label_cache": label_cache,
        "kg_triples": kg_triples,
        "readable_triples": readable_triples,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(kg_data, f)

    size_kb = os.path.getsize(cache_path) / 1024
    print(f"KG saved to {cache_path} ({size_kb:.1f} KB)")


def load_kg_cache(cache_path):
    """Load KG data from pickle cache."""
    with open(cache_path, "rb") as f:
        kg_data = pickle.load(f)

    return (
        kg_data["qid_map"],
        kg_data["label_cache"],
        kg_data["kg_triples"],
        kg_data["readable_triples"],
    )


# ---------------------------------------------------------------------------
# IMDb Pipeline — offline, no rate limits
# ---------------------------------------------------------------------------


def load_imdb_data(imdb_dir="data/imdb"):
    """
    Load the four IMDb TSV files into memory.

    Required files in imdb_dir:
        title.basics.tsv
        title.crew.tsv
        title.principals.tsv
        name.basics.tsv

    Returns:
        basics_df, crew_df, principals_df, names_dict
    """
    import pandas as pd

    print("Loading IMDb datasets...")

    basics_df = pd.read_csv(
        os.path.join(imdb_dir, "title.basics.tsv"),
        sep="\t",
        na_values=r"\N",
        dtype=str,
        low_memory=False,
        usecols=[
            "tconst",
            "titleType",
            "primaryTitle",
            "originalTitle",
            "startYear",
            "genres",
        ],
    )

    basics_df = basics_df[basics_df["titleType"].isin(["movie", "tvMovie"])].copy()
    basics_df["startYear"] = pd.to_numeric(basics_df["startYear"], errors="coerce")

    crew_df = pd.read_csv(
        os.path.join(imdb_dir, "title.crew.tsv"),
        sep="\t",
        na_values=r"\N",
        dtype=str,
    )

    principals_df = pd.read_csv(
        os.path.join(imdb_dir, "title.principals.tsv"),
        sep="\t",
        na_values=r"\N",
        dtype=str,
        low_memory=False,
        usecols=["tconst", "ordering", "nconst", "category"],
    )
    principals_df["ordering"] = pd.to_numeric(principals_df["ordering"], errors="coerce")

    names_df = pd.read_csv(
        os.path.join(imdb_dir, "name.basics.tsv"),
        sep="\t",
        na_values=r"\N",
        dtype=str,
        low_memory=False,
        usecols=["nconst", "primaryName"],
    )

    names_dict = dict(zip(names_df["nconst"], names_df["primaryName"]))

    print(f"  IMDb movies:       {len(basics_df):,}")
    print(f"  Crew entries:      {len(crew_df):,}")
    print(f"  Principal entries: {len(principals_df):,}")
    print(f"  Person names:      {len(names_dict):,}")

    return basics_df, crew_df, principals_df, names_dict


def match_movies_to_imdb(movies_sub, basics_df):
    """
    Match MovieLens movie titles, e.g. 'Toy Story (1995)', to IMDb tconst IDs.

    Matching strategy:
        1. primaryTitle + year
        2. originalTitle + year
        3. primaryTitle only

    Returns:
        tconst_map : dict {movieId -> tconst or None}
    """
    basics_idx = basics_df.copy()
    basics_idx["norm_primary"] = basics_idx["primaryTitle"].str.lower().str.strip()
    basics_idx["norm_original"] = basics_idx["originalTitle"].fillna("").str.lower().str.strip()

    primary_year_map = {}
    original_year_map = {}
    primary_only_map = {}

    for _, row in basics_idx.iterrows():
        tc = row["tconst"]
        pt = row["norm_primary"]
        ot = row["norm_original"]
        yr = row["startYear"]

        key_py = (pt, yr)
        key_oy = (ot, yr)

        if key_py not in primary_year_map:
            primary_year_map[key_py] = tc
        if ot and key_oy not in original_year_map:
            original_year_map[key_oy] = tc
        if pt not in primary_only_map:
            primary_only_map[pt] = tc

    tconst_map = {}
    matched = 0

    for _, row in movies_sub.iterrows():
        mid = int(row["movieId"])
        title = row["title"]

        yr_m = re.search(r"\((\d{4})\)\s*$", title)
        year = float(yr_m.group(1)) if yr_m else float("nan")
        clean = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip().lower()

        tconst = (
            primary_year_map.get((clean, year))
            or original_year_map.get((clean, year))
            or primary_only_map.get(clean)
        )

        tconst_map[mid] = tconst
        if tconst:
            matched += 1

    total = len(movies_sub)
    pct = matched / total * 100 if total else 0.0
    print(f"IMDb match: {matched}/{total} movies ({pct:.1f}%)")

    return tconst_map


def build_kg_from_imdb(movies_sub, tconst_map, basics_df, crew_df,
                       principals_df, names_dict, max_cast=5):
    """
    Build human-readable KG triples directly from IMDb local data.

    Relations produced:
        hasGenre, directedBy, writtenBy, hasCast,
        hasComposer, hasProducer, hasCinematographer, year

    Args:
        max_cast : maximum top-billed actors per movie

    Returns:
        readable_triples : list of (movie_title, relation, entity_name)
    """
    import pandas as pd

    mid_to_title = dict(zip(movies_sub["movieId"].astype(int), movies_sub["title"]))
    mid_to_tconst = {mid: tc for mid, tc in tconst_map.items() if tc is not None}
    tconst_to_title = {tc: mid_to_title[mid] for mid, tc in mid_to_tconst.items()}
    valid_tconsts = set(tconst_to_title)

    basics_f = basics_df[basics_df["tconst"].isin(valid_tconsts)]
    crew_f = crew_df[crew_df["tconst"].isin(valid_tconsts)]
    principals_f = principals_df[principals_df["tconst"].isin(valid_tconsts)]

    readable_triples = []

    # Genre triples.
    for _, row in basics_f.iterrows():
        title = tconst_to_title.get(row["tconst"])
        if not title or pd.isna(row.get("genres")):
            continue

        for genre in str(row["genres"]).split(","):
            genre = genre.strip()
            if genre:
                readable_triples.append((title, "hasGenre", genre))

    # Director and writer triples.
    for _, row in crew_f.iterrows():
        title = tconst_to_title.get(row["tconst"])
        if not title:
            continue

        if pd.notna(row.get("directors")):
            for nconst in str(row["directors"]).split(","):
                name = names_dict.get(nconst.strip())
                if name:
                    readable_triples.append((title, "directedBy", name))

        if pd.notna(row.get("writers")):
            for nconst in str(row["writers"]).split(","):
                name = names_dict.get(nconst.strip())
                if name:
                    readable_triples.append((title, "writtenBy", name))

    # Cast, composer, producer, cinematographer triples.
    cat_to_rel = {
        "actor":           "hasCast",
        "actress":         "hasCast",
        "composer":        "hasComposer",
        "producer":        "hasProducer",
        "cinematographer": "hasCinematographer",
    }

    for _, row in principals_f.iterrows():
        cat = row.get("category")
        rel = cat_to_rel.get(cat)

        if not rel:
            continue

        # Limit actors/actresses to top billing; keep all others.
        if rel == "hasCast" and (pd.isna(row["ordering"]) or row["ordering"] > max_cast):
            continue

        title = tconst_to_title.get(row["tconst"])
        name = names_dict.get(row.get("nconst"))

        if title and name:
            readable_triples.append((title, rel, name))

    # Year triples from MovieLens title strings.
    for _, row in movies_sub.iterrows():
        m = re.search(r"\((\d{4})\)", row["title"])
        if m:
            readable_triples.append((row["title"], "year", m.group(1)))

    readable_triples = list(dict.fromkeys(readable_triples))

    rel_counts = Counter(r for _, r, _ in readable_triples)
    print(f"IMDb KG — {len(readable_triples)} readable triples:")
    for rel, cnt in rel_counts.most_common():
        print(f"  {rel}: {cnt}")

    return readable_triples


def load_or_build_kg_imdb(movies_sub, imdb_dir="data/imdb",
                          cache_path="cache/kg_cache_imdb.pkl", max_cast=5):
    """
    Build the KG from local IMDb TSV files.

    No network calls, no rate limits.

    Args:
        movies_sub : DataFrame with ['movieId', 'title']
        imdb_dir   : directory containing IMDb TSV files
        cache_path : path to cache file
        max_cast   : maximum top-billed actors per movie

    Returns:
        tconst_map, {}, [], readable_triples, movies_out
    """
    needed_ids = set(movies_sub["movieId"].astype(int))

    if os.path.exists(cache_path):
        print(f"Loading KG from cache: {cache_path}")
        tconst_map, _, _, readable_triples = load_kg_cache(cache_path)

        cached_ids = set(int(k) for k in tconst_map.keys())
        missing_ids = needed_ids - cached_ids

        if not missing_ids:
            print(f"  Cache fully covers requested {len(needed_ids)} movies.")
        else:
            print(f"  {len(missing_ids)} new movies need processing...")

            missing_df = movies_sub[
                movies_sub["movieId"].astype(int).isin(missing_ids)
            ].copy()

            basics_df, crew_df, principals_df, names_dict = load_imdb_data(imdb_dir)
            new_tconst_map = match_movies_to_imdb(missing_df, basics_df)
            new_readable = build_kg_from_imdb(
                missing_df,
                new_tconst_map,
                basics_df,
                crew_df,
                principals_df,
                names_dict,
                max_cast=max_cast,
            )

            tconst_map.update(new_tconst_map)

            existing_set = set(map(tuple, readable_triples))
            for t in new_readable:
                if tuple(t) not in existing_set:
                    readable_triples.append(t)
                    existing_set.add(tuple(t))

            save_kg_cache(cache_path, tconst_map, {}, [], readable_triples)
            print(
                f"  Cache updated: {len(tconst_map)} movies, "
                f"{len(readable_triples)} triples."
            )

        movies_out = movies_sub.copy()
        movies_out["qid"] = movies_out["movieId"].astype(int).map(tconst_map)
        movies_out = movies_out.dropna(subset=["qid"]).copy()

        print(
            f"  movies with tconst: {len(movies_out)}, "
            f"readable_triples: {len(readable_triples)}"
        )

        return tconst_map, {}, [], readable_triples, movies_out

    # No IMDb KG cache exists — full build.
    print(f"No cache at '{cache_path}' — full IMDb build for {len(movies_sub)} movies.")

    basics_df, crew_df, principals_df, names_dict = load_imdb_data(imdb_dir)
    tconst_map = match_movies_to_imdb(movies_sub, basics_df)
    readable_triples = build_kg_from_imdb(
        movies_sub,
        tconst_map,
        basics_df,
        crew_df,
        principals_df,
        names_dict,
        max_cast=max_cast,
    )

    save_kg_cache(cache_path, tconst_map, {}, [], readable_triples)

    movies_out = movies_sub.copy()
    movies_out["qid"] = movies_out["movieId"].astype(int).map(tconst_map)
    movies_out = movies_out.dropna(subset=["qid"]).copy()

    print(
        f"  movies with tconst: {len(movies_out)}, "
        f"readable_triples: {len(readable_triples)}"
    )

    return tconst_map, {}, [], readable_triples, movies_out
