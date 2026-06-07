"""
kg_builder.py — Knowledge Graph construction for movie recommendation.

Pipelines included:

1. Wikidata pipeline, safer version:
   - searches Wikidata QIDs with retry/backoff
   - caches QID progress while running
   - fetches KG triples with smaller SPARQL batches
   - resolves labels in safer smaller batches
   - saves and reuses KG cache

2. IMDb offline pipeline:
   - uses local IMDb TSV files
   - no network calls
   - no Wikidata rate limits

Main entry points:
    load_or_build_kg(movies_sub, cache_path="cache/kg_cache_wikidata.pkl")
        -> Wikidata KG

    load_or_build_kg_imdb(movies_sub, imdb_dir="data/imdb",
                          cache_path="cache/kg_cache_imdb.pkl")
        -> IMDb KG, recommended when full ML-100K causes Wikidata rate limits

    build_adj(readable_triples, user_item_edges, user_info)
        -> adjacency graph
"""

import os
import pickle
import random
import re
import time
from collections import Counter, defaultdict

import requests


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

DEFAULT_USER_AGENT = "KGProject/1.0 academic-research contact: student-project"


# ---------------------------------------------------------------------------
# Safe request helper for Wikidata
# ---------------------------------------------------------------------------


def safe_get(url, params=None, headers=None, timeout=20,
             max_retries=6, base_sleep=2.0):
    """
    Safe requests.get wrapper for Wikidata.

    Handles:
      - 429 Too Many Requests
      - temporary 5xx server errors
      - connection/time-out errors

    Uses exponential backoff and respects Retry-After when available.
    """
    headers = headers or {"User-Agent": DEFAULT_USER_AGENT}

    for attempt in range(max_retries):
        try:
            r = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")

                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = base_sleep * (2 ** attempt)
                else:
                    wait = base_sleep * (2 ** attempt)

                wait += random.uniform(0, 1.0)
                print(f"  429 Too Many Requests. Waiting {wait:.1f}s...")
                time.sleep(wait)
                continue

            if 500 <= r.status_code < 600:
                wait = base_sleep * (2 ** attempt) + random.uniform(0, 1.0)
                print(f"  Wikidata server error {r.status_code}. Waiting {wait:.1f}s...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r

        except requests.exceptions.RequestException as e:
            wait = base_sleep * (2 ** attempt) + random.uniform(0, 1.0)
            print(f"  Request error: {e}. Waiting {wait:.1f}s...")
            time.sleep(wait)

    print("  Request failed after maximum retries.")
    return None


# ---------------------------------------------------------------------------
# Wikidata QID Search
# ---------------------------------------------------------------------------


def wikidata_search(title, sleep_s=1.0):
    """
    Search Wikidata for a movie QID by title.
    Prefers results whose description contains 'film'.

    Returns:
        QID string, for example 'Q12345', or None.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": title,
        "language": "en",
        "format": "json",
        "limit": 5,
    }
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    r = safe_get(url, params=params, headers=headers, timeout=20)

    if r is None:
        print(f"  wikidata_search failed: {title}")
        time.sleep(sleep_s)
        return None

    try:
        data = r.json()
    except Exception as e:
        print(f"  JSON decode error for {title}: {e}")
        time.sleep(sleep_s)
        return None

    if "search" not in data:
        time.sleep(sleep_s)
        return None

    for item in data["search"]:
        desc = item.get("description", "").lower()
        if "film" in desc or "movie" in desc:
            time.sleep(sleep_s)
            return item.get("id")

    if data["search"]:
        time.sleep(sleep_s)
        return data["search"][0].get("id")

    time.sleep(sleep_s)
    return None


def fetch_qid_map(movies_sub, sleep_s=1.0, cache_path="cache/qid_map.pkl"):
    """
    Fetch Wikidata QIDs with progress caching.

    If interrupted or rate-limited, already fetched QIDs are saved and reused.

    Args:
        movies_sub : DataFrame with ['movieId', 'title']
        sleep_s    : delay between QID search requests
        cache_path : separate progress cache for QID lookup

    Returns:
        qid_map    : dict {movieId -> QID or None}
        movies_out : movies_sub copy with 'qid' column, rows without QID dropped
    """
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            qid_map = pickle.load(f)
        print(f"Loaded cached QID map: {len(qid_map)} movies")
    else:
        qid_map = {}

    df = movies_sub.copy()
    df["clean_title"] = df["title"].str.replace(
        r"\s*\(\d{4}\)\s*$",
        "",
        regex=True,
    ).str.strip()

    total = len(df)

    for _, row in df.iterrows():
        mid = int(row["movieId"])
        title = row["clean_title"]

        if mid in qid_map:
            continue

        qid = wikidata_search(title, sleep_s=sleep_s)
        qid_map[mid] = qid

        print(f"  {len(qid_map)}/{total}: {mid} → {title} → {qid}")

        with open(cache_path, "wb") as f:
            pickle.dump(qid_map, f)

        time.sleep(sleep_s)

    df["qid"] = df["movieId"].astype(int).map(qid_map)
    df = df.dropna(subset=["qid"]).copy()

    print(f"Movies with QID: {len(df)} / {len(movies_sub)}")

    return qid_map, df


# ---------------------------------------------------------------------------
# Wikidata SPARQL Property Fetch
# ---------------------------------------------------------------------------


def _init_sparql():
    """Initialize SPARQLWrapper for Wikidata."""
    from SPARQLWrapper import JSON, SPARQLWrapper

    sparql = SPARQLWrapper(
        "https://query.wikidata.org/sparql",
        agent=DEFAULT_USER_AGENT,
    )
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(90)
    return sparql


def get_film_properties(qid, sparql):
    """
    Fetch genre, director, country, cast, composer, and screenwriter for one movie QID.

    Returns:
        list of (film_qid, relation, tail_qid) triples
    """
    query = f"""
    SELECT ?genre ?director ?country ?cast ?composer ?screenwriter WHERE {{
        OPTIONAL {{ wd:{qid} wdt:P136 ?genre . }}
        OPTIONAL {{ wd:{qid} wdt:P57  ?director . }}
        OPTIONAL {{ wd:{qid} wdt:P495 ?country . }}
        OPTIONAL {{ wd:{qid} wdt:P161 ?cast . }}
        OPTIONAL {{ wd:{qid} wdt:P86  ?composer . }}
        OPTIONAL {{ wd:{qid} wdt:P58  ?screenwriter . }}
    }}
    """
    sparql.setQuery(query)
    results = sparql.query().convert()

    triples = []
    for row in results["results"]["bindings"]:
        if "genre" in row:
            triples.append((qid, "hasGenre", row["genre"]["value"].split("/")[-1]))
        if "director" in row:
            triples.append((qid, "directedBy", row["director"]["value"].split("/")[-1]))
        if "country" in row:
            triples.append((qid, "country", row["country"]["value"].split("/")[-1]))
        if "cast" in row:
            triples.append((qid, "hasCast", row["cast"]["value"].split("/")[-1]))
        if "composer" in row:
            triples.append((qid, "hasComposer", row["composer"]["value"].split("/")[-1]))
        if "screenwriter" in row:
            triples.append((qid, "writtenBy", row["screenwriter"]["value"].split("/")[-1]))

    return list(dict.fromkeys(triples))


def fetch_kg_triples(qid_map, sleep_s=4.0, batch_size=5):
    """
    Fetch KG triples for all movies that have QIDs.

    Uses smaller SPARQL batches to reduce rate-limit/server-pressure problems.
    If a batch fails, it retries movies individually.

    Args:
        qid_map    : dict {movieId -> QID or None}
        sleep_s    : delay between SPARQL calls
        batch_size : number of QIDs per SPARQL query

    Returns:
        kg_triples : list of (qid, relation, tail_qid)
    """
    sparql = _init_sparql()
    kg_triples = []

    movie_to_qid = {k: v for k, v in qid_map.items() if v is not None}
    qids = list(dict.fromkeys(movie_to_qid.values()))

    batches = [qids[i:i + batch_size] for i in range(0, len(qids), batch_size)]
    print(f"  Fetching {len(qids)} movies in {len(batches)} SPARQL batches of {batch_size}...")

    for b_idx, batch in enumerate(batches):
        values_clause = " ".join(f"wd:{q}" for q in batch)
        query = f"""
        SELECT ?film ?genre ?director ?country ?cast ?composer ?screenwriter WHERE {{
            VALUES ?film {{ {values_clause} }}
            OPTIONAL {{ ?film wdt:P136 ?genre . }}
            OPTIONAL {{ ?film wdt:P57  ?director . }}
            OPTIONAL {{ ?film wdt:P495 ?country . }}
            OPTIONAL {{ ?film wdt:P161 ?cast . }}
            OPTIONAL {{ ?film wdt:P86  ?composer . }}
            OPTIONAL {{ ?film wdt:P58  ?screenwriter . }}
        }}
        """

        try:
            sparql.setQuery(query)
            results = sparql.query().convert()

            batch_triples = []
            for row in results["results"]["bindings"]:
                film_qid = row["film"]["value"].split("/")[-1]

                if "genre" in row:
                    batch_triples.append((film_qid, "hasGenre", row["genre"]["value"].split("/")[-1]))
                if "director" in row:
                    batch_triples.append((film_qid, "directedBy", row["director"]["value"].split("/")[-1]))
                if "country" in row:
                    batch_triples.append((film_qid, "country", row["country"]["value"].split("/")[-1]))
                if "cast" in row:
                    batch_triples.append((film_qid, "hasCast", row["cast"]["value"].split("/")[-1]))
                if "composer" in row:
                    batch_triples.append((film_qid, "hasComposer", row["composer"]["value"].split("/")[-1]))
                if "screenwriter" in row:
                    batch_triples.append((film_qid, "writtenBy", row["screenwriter"]["value"].split("/")[-1]))

            batch_triples = list(dict.fromkeys(batch_triples))
            kg_triples.extend(batch_triples)

            print(f"  Batch {b_idx + 1}/{len(batches)}: {len(batch_triples)} triples")

        except Exception as e:
            print(f"  Batch {b_idx + 1} error: {e} — retrying individually...")

            for qid in batch:
                try:
                    triples = get_film_properties(qid, sparql)
                    kg_triples.extend(triples)
                    print(f"    {qid}: {len(triples)} triples")
                except Exception as e2:
                    print(f"    Skip {qid}: {e2}")

                time.sleep(sleep_s)

        time.sleep(sleep_s)

    kg_triples = list(dict.fromkeys(kg_triples))
    print(f"Total KG triples: {len(kg_triples)}")

    return kg_triples


# ---------------------------------------------------------------------------
# Wikidata Label Resolution
# ---------------------------------------------------------------------------


def get_label(qid, label_cache):
    """
    Resolve a single Wikidata QID to its English label.
    Updates label_cache in-place.
    """
    if qid in label_cache:
        return label_cache[qid]

    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    r = safe_get(url, headers=headers, timeout=15)

    if r is None:
        label_cache[qid] = qid
        time.sleep(0.5)
        return qid

    try:
        data = r.json()
        ent = data.get("entities", {}).get(qid, {})
        labels = ent.get("labels", {})
        label_cache[qid] = labels.get("en", {}).get("value", qid)
    except Exception:
        label_cache[qid] = qid

    time.sleep(0.5)
    return label_cache[qid]


def get_labels_batch(qids, label_cache, batch_size=25, sleep_s=1.0):
    """
    Resolve multiple Wikidata QIDs to English labels in safe smaller batches.
    """
    missing = [
        q for q in qids
        if q not in label_cache and isinstance(q, str) and q.startswith("Q")
    ]

    if not missing:
        return

    batches = [missing[i:i + batch_size] for i in range(0, len(missing), batch_size)]

    url = "https://www.wikidata.org/w/api.php"
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    for batch in batches:
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels",
            "languages": "en",
            "format": "json",
        }

        r = safe_get(url, params=params, headers=headers, timeout=30)

        if r is None:
            print("  Label batch failed — falling back to single lookups")
            for qid in batch:
                get_label(qid, label_cache)
            time.sleep(sleep_s)
            continue

        try:
            data = r.json()
            for qid, ent in data.get("entities", {}).items():
                label = ent.get("labels", {}).get("en", {}).get("value", qid)
                label_cache[qid] = label
        except Exception as e:
            print(f"  Label batch parse error: {e} — falling back to single lookups")
            for qid in batch:
                get_label(qid, label_cache)

        time.sleep(sleep_s)


def build_readable_triples(kg_triples, movies_sub, label_cache=None):
    """
    Convert QID-based triples to human-readable triples.
    Also adds year triples extracted from movie titles.

    Args:
        kg_triples  : list of (qid, relation, tail_qid)
        movies_sub  : DataFrame with 'qid' and 'title' columns
        label_cache : dict to reuse/update

    Returns:
        readable_triples : list of (movie_title, relation, label)
        label_cache      : updated dict
    """
    if label_cache is None:
        label_cache = {}

    qid_to_title = dict(zip(movies_sub["qid"], movies_sub["title"]))

    tail_qids = [
        t for _, _, t in kg_triples
        if isinstance(t, str) and t.startswith("Q") and t not in label_cache
    ]

    unique_tail_qids = list(dict.fromkeys(tail_qids))
    if unique_tail_qids:
        print(f"  Resolving {len(unique_tail_qids)} unique QID labels in batches...")
        get_labels_batch(unique_tail_qids, label_cache, batch_size=25, sleep_s=1.0)

    readable_triples = []

    for h, r, t in kg_triples:
        if isinstance(h, str) and h.startswith("Q"):
            h_label = qid_to_title.get(h, get_label(h, label_cache))
        else:
            h_label = h

        if isinstance(t, str) and t.startswith("Q"):
            t_label = label_cache.get(t, t)
        else:
            t_label = t

        readable_triples.append((h_label, r, t_label))

    # Add year triples from MovieLens title strings.
    for _, row in movies_sub.iterrows():
        m = re.search(r"\((\d{4})\)", row["title"])
        if m:
            readable_triples.append((row["title"], "year", m.group(1)))

    readable_triples = list(dict.fromkeys(readable_triples))

    rel_counts = Counter(r for _, r, _ in readable_triples)
    print(f"Readable triples: {len(readable_triples)}")
    for rel, cnt in rel_counts.most_common():
        print(f"  {rel}: {cnt}")

    return readable_triples, label_cache


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
# Main Wikidata Entry Point
# ---------------------------------------------------------------------------


def load_or_build_kg(movies_sub, cache_path="cache/kg_cache_wikidata.pkl",
                     qid_cache_path="cache/ml100k_qid_map.pkl",
                     qid_sleep_s=1.5,
                     sparql_sleep_s=4.0,
                     sparql_batch_size=5):
    """
    Load Wikidata KG from cache if available, otherwise build and save.

    Incremental behavior:
        If cache exists but some requested movies are missing, only the new
        movies are fetched and merged into the existing cache.

    Args:
        movies_sub          : DataFrame with ['movieId', 'title']
        cache_path          : full KG cache file
        qid_cache_path      : progress cache for QID lookup
        qid_sleep_s         : delay between QID search requests
        sparql_sleep_s      : delay between SPARQL requests
        sparql_batch_size   : QIDs per SPARQL query

    Returns:
        qid_map          : dict {movieId -> QID}
        label_cache      : dict {QID -> label}
        kg_triples       : list of QID triples
        readable_triples : list of readable triples
        movies_out       : movies_sub with qid column, rows without QID dropped
    """
    needed_ids = set(movies_sub["movieId"].astype(int))

    if os.path.exists(cache_path):
        print(f"Loading KG from cache: {cache_path}")
        qid_map, label_cache, kg_triples, readable_triples = load_kg_cache(cache_path)

        cached_ids = set(int(k) for k in qid_map.keys())
        missing_ids = needed_ids - cached_ids

        if missing_ids:
            print(
                f"  Cache has {len(cached_ids)} movies; "
                f"{len(missing_ids)} new movies need fetching..."
            )

            missing_df = movies_sub[
                movies_sub["movieId"].astype(int).isin(missing_ids)
            ].copy()

            new_qid_map, new_movies_df = fetch_qid_map(
                missing_df,
                sleep_s=qid_sleep_s,
                cache_path=qid_cache_path,
            )

            new_kg_triples = fetch_kg_triples(
                new_qid_map,
                sleep_s=sparql_sleep_s,
                batch_size=sparql_batch_size,
            )

            new_readable, label_cache = build_readable_triples(
                new_kg_triples,
                new_movies_df,
                label_cache,
            )

            qid_map.update(new_qid_map)
            kg_triples.extend(new_kg_triples)
            kg_triples = list(dict.fromkeys(kg_triples))

            existing_set = set(readable_triples)
            for t in new_readable:
                if t not in existing_set:
                    readable_triples.append(t)
                    existing_set.add(t)

            save_kg_cache(
                cache_path,
                qid_map,
                label_cache,
                kg_triples,
                readable_triples,
            )

            print(
                f"  Cache updated: now {len(qid_map)} movies, "
                f"{len(readable_triples)} readable triples."
            )
        else:
            print(f"  Cache fully covers requested {len(needed_ids)} movies.")

        movies_out = movies_sub.copy()
        movies_out["qid"] = movies_out["movieId"].astype(int).map(qid_map)
        movies_out = movies_out.dropna(subset=["qid"]).copy()

        print(
            f"  movies with QID: {len(movies_out)}, "
            f"readable_triples: {len(readable_triples)}"
        )

        return qid_map, label_cache, kg_triples, readable_triples, movies_out

    # No full KG cache exists.
    n = len(movies_sub)
    n_batches = (n + sparql_batch_size - 1) // sparql_batch_size

    print(f"No cache at '{cache_path}' — full Wikidata build for {n} movies.")
    print(f"  QID lookup sleep: {qid_sleep_s}s per movie")
    print(
        f"  SPARQL: {n_batches} batches of {sparql_batch_size}, "
        f"sleep {sparql_sleep_s}s per batch"
    )

    qid_map, movies_out = fetch_qid_map(
        movies_sub,
        sleep_s=qid_sleep_s,
        cache_path=qid_cache_path,
    )

    kg_triples = fetch_kg_triples(
        qid_map,
        sleep_s=sparql_sleep_s,
        batch_size=sparql_batch_size,
    )

    label_cache = {}
    readable_triples, label_cache = build_readable_triples(
        kg_triples,
        movies_out,
        label_cache,
    )

    save_kg_cache(
        cache_path,
        qid_map,
        label_cache,
        kg_triples,
        readable_triples,
    )

    return qid_map, label_cache, kg_triples, readable_triples, movies_out


# ---------------------------------------------------------------------------
# IMDb Pipeline, offline and no rate limits
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
        basics_df
        crew_df
        principals_df
        names_dict
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
    Match MovieLens movie titles, for example 'Toy Story (1995)', to IMDb tconst IDs.

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
        hasGenre
        directedBy
        writtenBy
        hasCast
        hasComposer
        year

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

    # Cast and composer triples.
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

        # Limit actors/actresses to top billing; keep all composers.
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

    No network calls, no Wikidata rate limits.

    Return signature matches load_or_build_kg. The qid_map field stores
    movieId -> tconst.

    Args:
        movies_sub : DataFrame with ['movieId', 'title']
        imdb_dir   : directory containing IMDb TSV files
        cache_path : path to cache file
        max_cast   : maximum top-billed actors per movie
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

    # No IMDb KG cache exists.
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