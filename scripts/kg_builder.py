"""
kg_builder.py — Knowledge Graph construction for movie recommendation.

Wikidata pipeline (original):
    1. Search Wikidata for movie QIDs  (wikidata_search)
    2. Fetch film properties via SPARQL (get_film_properties)
    3. Resolve QID labels to human-readable names (get_label)
    4. Build adjacency graph from readable triples + user interactions (build_adj)
    5. Cache / load everything (load_or_build_kg)

IMDb pipeline (offline, no rate limits):
    1. Load local TSV files from data/imdb/  (load_imdb_data)
    2. Match movie titles to IMDb tconst IDs (match_movies_to_imdb)
    3. Build readable triples directly       (build_kg_from_imdb)
    4. Cache / load everything               (load_or_build_kg_imdb)

Main entry points:
    load_or_build_kg(movies_sub, cache_path, ...)       — Wikidata
    load_or_build_kg_imdb(movies_sub, imdb_dir, ...)    — IMDb (recommended)

    build_adj(readable_triples, user_item_edges, user_info)
        → adj (defaultdict)
"""

import os
import pickle
import re
import time
from collections import Counter, defaultdict

import requests

BASE_RELS = {
    "likes", "hasGenre", "directedBy", "year", "country",
    "hasCast", "hasComposer", "writtenBy", "hasGender",
}

_KG_RELATIONS = {"hasGenre", "directedBy", "year", "country",
                 "hasCast", "hasComposer", "writtenBy"}


# ── Wikidata QID Search ────────────────────────────────────────────────────────

def wikidata_search(title, sleep_s=0.2):
    """
    Search Wikidata for a movie QID by title.
    Prefers results whose description contains 'film'.

    Returns: QID string (e.g. 'Q12345') or None
    """
    try:
        url    = "https://www.wikidata.org/w/api.php"
        params = {
            "action":   "wbsearchentities",
            "search":   title,
            "language": "en",
            "format":   "json",
            "limit":    5,
        }
        headers = {"User-Agent": "KG/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        if "search" not in data:
            return None
        for item in data["search"]:
            desc = item.get("description", "").lower()
            if "film" in desc:
                time.sleep(sleep_s)
                return item["id"]
        if data["search"]:
            time.sleep(sleep_s)
            return data["search"][0]["id"]
        return None
    except Exception as e:
        print(f"  wikidata_search error ({title}): {e}")
        return None


def fetch_qid_map(movies_sub, sleep_s=0.2):
    """
    Fetch Wikidata QIDs for all movies in movies_sub.

    Args:
        movies_sub: DataFrame with columns ['movieId', 'title']
        sleep_s:    sleep between requests

    Returns:
        qid_map     : dict {movieId (int) → QID string or None}
        movies_sub  : DataFrame with 'qid' column added, rows without QID dropped
    """
    df = movies_sub.copy()
    df["clean_title"] = df["title"].str.replace(
        r"\s*\(\d{4}\)\s*$", "", regex=True
    ).str.strip()

    qid_map = {}
    for _, row in df.iterrows():
        mid   = int(row["movieId"])
        title = row["clean_title"]
        if mid not in qid_map:
            qid = wikidata_search(title, sleep_s=sleep_s)
            print(f"  {mid} → {title} → {qid}")
            qid_map[mid] = qid
            time.sleep(sleep_s)

    df["qid"] = df["movieId"].map(qid_map)
    df = df.dropna(subset=["qid"]).copy()
    print(f"Movies with QID: {len(df)} / {len(movies_sub)}")
    return qid_map, df


# ── SPARQL Property Fetch ──────────────────────────────────────────────────────

def _init_sparql():
    from SPARQLWrapper import JSON, SPARQLWrapper
    sparql = SPARQLWrapper(
        "https://query.wikidata.org/sparql",
        agent="KGProject/1.0",
    )
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(60)
    return sparql


def get_film_properties(qid, sparql):
    """
    Fetch genre, director, country, cast, composer, screenwriter for a movie QID.

    Returns: list of (qid, relation, tail_qid) triples
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
        if "genre"        in row:
            triples.append((qid, "hasGenre",   row["genre"]["value"].split("/")[-1]))
        if "director"     in row:
            triples.append((qid, "directedBy", row["director"]["value"].split("/")[-1]))
        if "country"      in row:
            triples.append((qid, "country",    row["country"]["value"].split("/")[-1]))
        if "cast"         in row:
            triples.append((qid, "hasCast",    row["cast"]["value"].split("/")[-1]))
        if "composer"     in row:
            triples.append((qid, "hasComposer",row["composer"]["value"].split("/")[-1]))
        if "screenwriter" in row:
            triples.append((qid, "writtenBy",  row["screenwriter"]["value"].split("/")[-1]))
    return list(dict.fromkeys(triples))


def fetch_kg_triples(qid_map, sleep_s=1.5, batch_size=20):
    """
    Fetch KG triples for all movies that have a QID.
    Uses batched SPARQL queries (batch_size movies per query) to reduce
    the number of round-trips to Wikidata from N → N/batch_size.

    Returns: list of (qid, relation, tail_qid) triples
    """
    sparql       = _init_sparql()
    kg_triples   = []
    movie_to_qid = {k: v for k, v in qid_map.items() if v is not None}
    qids         = list(set(movie_to_qid.values()))   # unique QIDs only

    batches = [qids[i:i+batch_size] for i in range(0, len(qids), batch_size)]
    print(f"  Fetching {len(qids)} movies in {len(batches)} batches of {batch_size}...")

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
                if "genre"        in row:
                    batch_triples.append((film_qid, "hasGenre",    row["genre"]["value"].split("/")[-1]))
                if "director"     in row:
                    batch_triples.append((film_qid, "directedBy",  row["director"]["value"].split("/")[-1]))
                if "country"      in row:
                    batch_triples.append((film_qid, "country",     row["country"]["value"].split("/")[-1]))
                if "cast"         in row:
                    batch_triples.append((film_qid, "hasCast",     row["cast"]["value"].split("/")[-1]))
                if "composer"     in row:
                    batch_triples.append((film_qid, "hasComposer", row["composer"]["value"].split("/")[-1]))
                if "screenwriter" in row:
                    batch_triples.append((film_qid, "writtenBy",   row["screenwriter"]["value"].split("/")[-1]))
            # Deduplicate within batch
            batch_triples = list(dict.fromkeys(batch_triples))
            kg_triples.extend(batch_triples)
            print(f"  Batch {b_idx+1}/{len(batches)}: {len(batch_triples)} triples")
        except Exception as e:
            print(f"  Batch {b_idx+1} error: {e} — retrying individually...")
            # Fallback: fetch one by one for failed batch
            for qid in batch:
                try:
                    triples = get_film_properties(qid, sparql)
                    kg_triples.extend(triples)
                except Exception as e2:
                    print(f"    Skip {qid}: {e2}")
                time.sleep(sleep_s)
        time.sleep(sleep_s)

    print(f"Total KG triples: {len(kg_triples)}")
    return kg_triples


# ── Label Resolution ──────────────────────────────────────────────────────────

def get_label(qid, label_cache):
    """
    Resolve a single Wikidata QID to its English label.
    Updates label_cache in-place. Used as a fallback for single lookups.
    """
    if qid in label_cache:
        return label_cache[qid]
    url     = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    headers = {"User-Agent": "KGProject/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data   = r.json()
        ent    = data.get("entities", {}).get(qid, {})
        labels = ent.get("labels", {})
        label_cache[qid] = labels.get("en", {}).get("value", qid)
    except Exception:
        label_cache[qid] = qid
    time.sleep(0.1)
    return label_cache[qid]


def get_labels_batch(qids, label_cache, batch_size=50):
    """
    Resolve multiple Wikidata QIDs to English labels in batches.
    Uses the wbgetentities API endpoint (50 IDs per request).
    Updates label_cache in-place.
    """
    missing = [q for q in qids if q not in label_cache and isinstance(q, str) and q.startswith("Q")]
    if not missing:
        return

    batches = [missing[i:i+batch_size] for i in range(0, len(missing), batch_size)]
    url     = "https://www.wikidata.org/w/api.php"
    headers = {"User-Agent": "KGProject/1.0"}

    for batch in batches:
        params = {
            "action":    "wbgetentities",
            "ids":       "|".join(batch),
            "props":     "labels",
            "languages": "en",
            "format":    "json",
        }
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            for qid, ent in data.get("entities", {}).items():
                label = ent.get("labels", {}).get("en", {}).get("value", qid)
                label_cache[qid] = label
        except Exception as e:
            print(f"  Label batch error: {e} — falling back to single lookups")
            for qid in batch:
                get_label(qid, label_cache)
        time.sleep(0.3)


def build_readable_triples(kg_triples, movies_sub, label_cache=None):
    """
    Convert QID-based triples to human-readable label triples.
    Also adds year triples extracted from movie titles.

    Args:
        kg_triples  : list of (qid, relation, tail_qid)
        movies_sub  : DataFrame with 'qid' and 'title' columns
        label_cache : dict to reuse/update (created fresh if None)

    Returns:
        readable_triples : list of (movie_title, relation, label)
        label_cache      : updated dict
    """
    if label_cache is None:
        label_cache = {}

    qid_to_title = dict(zip(movies_sub["qid"], movies_sub["title"]))

    # Batch-resolve all unknown tail QIDs before looping (50 per API call)
    tail_qids = [t for _, _, t in kg_triples
                 if isinstance(t, str) and t.startswith("Q") and t not in label_cache]
    if tail_qids:
        print(f"  Resolving {len(set(tail_qids))} unique QID labels in batches...")
        get_labels_batch(list(set(tail_qids)), label_cache)

    readable_triples = []
    for h, r, t in kg_triples:
        h_label = qid_to_title.get(h, get_label(h, label_cache)) \
                  if isinstance(h, str) and h.startswith("Q") else h
        t_label = label_cache.get(t, t) \
                  if isinstance(t, str) and t.startswith("Q") else t
        readable_triples.append((h_label, r, t_label))

    # Year triples from title strings
    for _, row in movies_sub.iterrows():
        m = re.search(r"\((\d{4})\)", row["title"])
        if m:
            readable_triples.append((row["title"], "year", m.group(1)))

    rel_counts = Counter(r for _, r, _ in readable_triples)
    print(f"Readable triples: {len(readable_triples)}")
    for rel, cnt in rel_counts.most_common():
        print(f"  {rel}: {cnt}")

    return readable_triples, label_cache


# ── Adjacency Graph ───────────────────────────────────────────────────────────

def build_adj(readable_triples, user_item_edges, user_info):
    """
    Build the KG adjacency graph from readable triples, user-item edges,
    and user demographic info (for hasGender edges).

    Args:
        readable_triples : list of (head, relation, tail) with human-readable labels
        user_item_edges  : list of ('User_<id>', 'likes', movie_title)
        user_info        : dict {userId (int) → {'gender': 'M'|'F', ...}}

    Returns:
        adj : defaultdict[node → defaultdict[relation → set(tails)]]
    """
    def rev_rel(r):
        return f"rev_{r}"

    adj = defaultdict(lambda: defaultdict(set))

    # User–item edges (training likes)
    for h, r, t in user_item_edges:
        adj[h][r].add(t)
        adj[t][rev_rel(r)].add(h)

    # KG triples (genre, director, cast, etc.) — exclude gender and unresolved Q-codes
    _qcode = re.compile(r"^Q\d+$")
    for h, r, t in readable_triples:
        if r not in _KG_RELATIONS:
            continue
        if _qcode.match(str(h)) or _qcode.match(str(t)):
            continue   # drop triples with unresolved Wikidata IDs
        adj[h][r].add(t)
        adj[t][rev_rel(r)].add(h)

    # Gender edges (audit / fairness monitoring — blacklisted in PGPR action space)
    for uid, info in user_info.items():
        g = info.get("gender")
        if g in ("M", "F"):
            u = f"User_{int(uid)}"
            adj[u]["hasGender"].add(f"Gender_{g}")
            adj[f"Gender_{g}"]["rev_hasGender"].add(u)

    node_count = len(adj)
    edge_count = sum(len(v) for h in adj for v in adj[h].values())
    rel_counts = Counter(r for h in adj for r in adj[h])

    print(f"Adj → nodes: {node_count}, total edge-instances: {edge_count}")
    for r, c in rel_counts.most_common():
        print(f"  {r}: {c}")

    return adj


# ── Cache helpers ─────────────────────────────────────────────────────────────

def save_kg_cache(cache_path, qid_map, label_cache, kg_triples, readable_triples):
    kg_data = {
        "qid_map":          qid_map,
        "label_cache":      label_cache,
        "kg_triples":       kg_triples,
        "readable_triples": readable_triples,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(kg_data, f)
    size_kb = os.path.getsize(cache_path) / 1024
    print(f"KG saved to {cache_path} ({size_kb:.1f} KB)")


def load_kg_cache(cache_path):
    with open(cache_path, "rb") as f:
        kg_data = pickle.load(f)
    return (kg_data["qid_map"], kg_data["label_cache"],
            kg_data["kg_triples"], kg_data["readable_triples"])


# ── Main Entry Point ──────────────────────────────────────────────────────────

def load_or_build_kg(movies_sub, cache_path="kg_cache_v4.pkl"):
    """
    Load KG from cache if available, otherwise fetch from Wikidata and save.

    Incremental update: if cache exists but some requested movies are missing,
    only fetches the new movies and merges them into the existing cache.
    This means going from 200 → 500 movies only fetches 300 new ones.

    Args:
        movies_sub  : DataFrame with ['movieId', 'title'] columns
        cache_path  : path to the pickle cache file

    Returns:
        qid_map          : dict {movieId → QID}
        label_cache      : dict {QID → human-readable label}
        kg_triples       : list of (qid, relation, tail_qid)
        readable_triples : list of (title, relation, label)
        movies_sub       : DataFrame (may have rows dropped if QID not found)
    """
    needed_ids = set(movies_sub["movieId"].astype(int))

    if os.path.exists(cache_path):
        print(f"Loading KG from cache: {cache_path}")
        qid_map, label_cache, kg_triples, readable_triples = load_kg_cache(cache_path)
        cached_ids = set(qid_map.keys())
        missing_ids = needed_ids - cached_ids

        if missing_ids:
            print(f"  Cache has {len(cached_ids)} movies; "
                  f"{len(missing_ids)} new movies need fetching...")

            missing_df = movies_sub[movies_sub["movieId"].astype(int).isin(missing_ids)].copy()

            # Step 1: QID lookup for new movies only
            new_qid_map, new_movies_df = fetch_qid_map(missing_df)

            # Step 2: Batch SPARQL for new movies only
            new_kg_triples = fetch_kg_triples(new_qid_map)

            # Step 3: Resolve labels (batch, reuses existing label_cache)
            new_readable, label_cache = build_readable_triples(
                new_kg_triples, new_movies_df, label_cache
            )

            # Merge into existing cache
            qid_map.update(new_qid_map)
            kg_triples.extend(new_kg_triples)

            # Deduplicate readable_triples before merging
            existing_set = set(readable_triples)
            for t in new_readable:
                if t not in existing_set:
                    readable_triples.append(t)
                    existing_set.add(t)

            save_kg_cache(cache_path, qid_map, label_cache, kg_triples, readable_triples)
            print(f"  Cache updated: now {len(qid_map)} movies, "
                  f"{len(readable_triples)} readable triples.")
        else:
            print(f"  Cache fully covers requested {len(needed_ids)} movies.")

        movies_sub = movies_sub.copy()
        movies_sub["qid"] = movies_sub["movieId"].astype(int).map(qid_map)
        movies_sub = movies_sub.dropna(subset=["qid"]).copy()

        print(f"  movies with QID: {len(movies_sub)}, "
              f"readable_triples: {len(readable_triples)}")
        return qid_map, label_cache, kg_triples, readable_triples, movies_sub

    # ── No cache exists: full build ────────────────────────────────────────
    n = len(movies_sub)
    n_batches = (n + 19) // 20
    print(f"No cache at '{cache_path}' — full build for {n} movies.")
    print(f"  QID lookup:  ~{n * 0.4 / 60:.1f} min")
    print(f"  SPARQL:      ~{n_batches * 1.5 / 60:.1f} min ({n_batches} batches of 20)")
    print(f"  Labels:      ~{n // 50 * 0.3 / 60:.1f} min (batch API)")

    qid_map, movies_sub = fetch_qid_map(movies_sub)
    kg_triples = fetch_kg_triples(qid_map)
    label_cache = {}
    readable_triples, label_cache = build_readable_triples(
        kg_triples, movies_sub, label_cache
    )
    save_kg_cache(cache_path, qid_map, label_cache, kg_triples, readable_triples)

    return qid_map, label_cache, kg_triples, readable_triples, movies_sub


# ═══════════════════════════════════════════════════════════════════════════════
# IMDb Pipeline (offline — no API calls, no rate limits)
# ═══════════════════════════════════════════════════════════════════════════════

def load_imdb_data(imdb_dir="data/imdb"):
    """
    Load the four IMDb TSV files into memory.

    Required files in imdb_dir:
        title.basics.tsv, title.crew.tsv,
        title.principals.tsv, name.basics.tsv

    Returns:
        basics_df     : DataFrame (movies only)
        crew_df       : DataFrame
        principals_df : DataFrame
        names_dict    : dict {nconst -> primaryName}
    """
    import pandas as pd

    print("Loading IMDb datasets...")

    basics_df = pd.read_csv(
        os.path.join(imdb_dir, "title.basics.tsv"),
        sep="\t", na_values=r"\N", dtype=str, low_memory=False,
        usecols=["tconst", "titleType", "primaryTitle",
                 "originalTitle", "startYear", "genres"],
    )
    basics_df = basics_df[basics_df["titleType"].isin(["movie", "tvMovie"])].copy()
    basics_df["startYear"] = pd.to_numeric(basics_df["startYear"], errors="coerce")

    crew_df = pd.read_csv(
        os.path.join(imdb_dir, "title.crew.tsv"),
        sep="\t", na_values=r"\N", dtype=str,
    )

    principals_df = pd.read_csv(
        os.path.join(imdb_dir, "title.principals.tsv"),
        sep="\t", na_values=r"\N", dtype=str, low_memory=False,
        usecols=["tconst", "ordering", "nconst", "category"],
    )
    principals_df["ordering"] = pd.to_numeric(
        principals_df["ordering"], errors="coerce"
    )

    names_df = pd.read_csv(
        os.path.join(imdb_dir, "name.basics.tsv"),
        sep="\t", na_values=r"\N", dtype=str, low_memory=False,
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
    Match ML-100K/1M movie titles (e.g. "Toy Story (1995)") to IMDb tconst IDs.

    Matching strategy (in order of priority):
        1. primaryTitle + year   (exact, case-insensitive)
        2. originalTitle + year
        3. primaryTitle only     (no year — last resort)

    Returns:
        tconst_map : dict {movieId (int) -> tconst str or None}
    """
    import pandas as pd

    basics_idx = basics_df.copy()
    basics_idx["norm_primary"]  = basics_idx["primaryTitle"].str.lower().str.strip()
    basics_idx["norm_original"] = basics_idx["originalTitle"].fillna("").str.lower().str.strip()

    primary_year_map  = {}
    original_year_map = {}
    primary_only_map  = {}

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
        mid   = int(row["movieId"])
        title = row["title"]

        yr_m  = re.search(r"\((\d{4})\)\s*$", title)
        year  = float(yr_m.group(1)) if yr_m else float("nan")
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
    print(f"IMDb match: {matched}/{total} movies ({matched/total*100:.1f}%)")
    return tconst_map


def build_kg_from_imdb(movies_sub, tconst_map, basics_df, crew_df,
                        principals_df, names_dict, max_cast=5):
    """
    Build human-readable KG triples directly from IMDb local data.
    No QID step — names are resolved immediately via names_dict.

    Relations produced: hasGenre, directedBy, writtenBy, hasCast, hasComposer, year

    Args:
        max_cast : max top-billed actors per movie (ordering <= max_cast)

    Returns:
        readable_triples : list of (movie_title, relation, entity_name)
    """
    import pandas as pd

    mid_to_title    = dict(zip(movies_sub["movieId"].astype(int), movies_sub["title"]))
    mid_to_tconst   = {mid: tc for mid, tc in tconst_map.items() if tc is not None}
    tconst_to_title = {tc: mid_to_title[mid] for mid, tc in mid_to_tconst.items()}
    valid_tconsts   = set(tconst_to_title)

    basics_f     = basics_df[basics_df["tconst"].isin(valid_tconsts)]
    crew_f       = crew_df[crew_df["tconst"].isin(valid_tconsts)]
    principals_f = principals_df[principals_df["tconst"].isin(valid_tconsts)]

    readable_triples = []

    # Genre triples
    for _, row in basics_f.iterrows():
        title = tconst_to_title.get(row["tconst"])
        if not title or pd.isna(row.get("genres")):
            continue
        for genre in str(row["genres"]).split(","):
            genre = genre.strip()
            if genre:
                readable_triples.append((title, "hasGenre", genre))

    # Director and writer triples (crew file — all directors/writers)
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

    # Cast (top max_cast billed) and composer triples (principals file)
    cat_to_rel = {"actor": "hasCast", "actress": "hasCast", "composer": "hasComposer"}
    for _, row in principals_f.iterrows():
        cat = row.get("category")
        rel = cat_to_rel.get(cat)
        if not rel:
            continue
        # Limit actors/actresses to top billing; keep all composers
        if rel == "hasCast" and (pd.isna(row["ordering"]) or row["ordering"] > max_cast):
            continue
        title = tconst_to_title.get(row["tconst"])
        name  = names_dict.get(row.get("nconst"))
        if title and name:
            readable_triples.append((title, rel, name))

    # Year triples from title strings
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
                           cache_path="kg_cache_imdb.pkl", max_cast=5):
    """
    Build the KG from local IMDb TSV files (no network calls, no rate limits).

    Incremental: if the cache already exists and only some movies are new,
    only those new movies are processed and merged in.

    Return signature matches load_or_build_kg so the notebook needs no
    structural changes. The 'qid_map' field stores movieId -> tconst.

    Args:
        movies_sub : DataFrame with ['movieId', 'title'] columns
        imdb_dir   : directory containing the four IMDb TSV files
        cache_path : path to the pickle cache file
        max_cast   : max top-billed actors per movie
    """
    needed_ids = set(movies_sub["movieId"].astype(int))

    if os.path.exists(cache_path):
        print(f"Loading KG from cache: {cache_path}")
        tconst_map, _, _, readable_triples = load_kg_cache(cache_path)
        cached_ids  = set(tconst_map.keys())
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
            new_readable   = build_kg_from_imdb(
                missing_df, new_tconst_map,
                basics_df, crew_df, principals_df, names_dict, max_cast,
            )

            tconst_map.update(new_tconst_map)
            existing_set = set(map(tuple, readable_triples))
            for t in new_readable:
                if tuple(t) not in existing_set:
                    readable_triples.append(t)
                    existing_set.add(tuple(t))

            save_kg_cache(cache_path, tconst_map, {}, [], readable_triples)
            print(f"  Cache updated: {len(tconst_map)} movies, "
                  f"{len(readable_triples)} triples.")

        movies_out = movies_sub.copy()
        movies_out["qid"] = movies_out["movieId"].astype(int).map(tconst_map)
        movies_out = movies_out.dropna(subset=["qid"]).copy()
        print(f"  movies with tconst: {len(movies_out)}, "
              f"readable_triples: {len(readable_triples)}")
        return tconst_map, {}, [], readable_triples, movies_out

    # ── No cache: full build ───────────────────────────────────────────────
    print(f"No cache at '{cache_path}' — full IMDb build for {len(movies_sub)} movies.")
    basics_df, crew_df, principals_df, names_dict = load_imdb_data(imdb_dir)
    tconst_map       = match_movies_to_imdb(movies_sub, basics_df)
    readable_triples = build_kg_from_imdb(
        movies_sub, tconst_map,
        basics_df, crew_df, principals_df, names_dict, max_cast,
    )
    save_kg_cache(cache_path, tconst_map, {}, [], readable_triples)

    movies_out = movies_sub.copy()
    movies_out["qid"] = movies_out["movieId"].astype(int).map(tconst_map)
    movies_out = movies_out.dropna(subset=["qid"]).copy()
    print(f"  movies with tconst: {len(movies_out)}, "
          f"readable_triples: {len(readable_triples)}")
    return tconst_map, {}, [], readable_triples, movies_out