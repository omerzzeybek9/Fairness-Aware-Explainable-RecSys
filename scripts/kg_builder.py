"""
kg_builder.py — Knowledge Graph construction for movie recommendation.

Pipeline:
    1. Search Wikidata for movie QIDs  (wikidata_search)
    2. Fetch film properties via SPARQL (get_film_properties)
    3. Resolve QID labels to human-readable names (get_label)
    4. Build adjacency graph from readable triples + user interactions (build_adj)
    5. Cache / load everything (load_or_build_kg)

Main entry point:
    load_or_build_kg(movies_sub, cache_path, ...)
        → qid_map, label_cache, kg_triples, readable_triples

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


def fetch_kg_triples(qid_map, sleep_s=2.0):
    """
    Fetch KG triples for all movies that have a QID.

    Returns: list of (qid, relation, tail_qid) triples
    """
    sparql       = _init_sparql()
    kg_triples   = []
    movie_to_qid = {k: v for k, v in qid_map.items() if v is not None}

    for mid, qid in movie_to_qid.items():
        try:
            triples = get_film_properties(qid, sparql)
            kg_triples.extend(triples)
            print(f"  {mid} ({qid}): {len(triples)} triples")
        except Exception as e:
            print(f"  Error {mid} ({qid}): {e}")
        time.sleep(sleep_s)

    print(f"Total KG triples: {len(kg_triples)}")
    return kg_triples


# ── Label Resolution ──────────────────────────────────────────────────────────

def get_label(qid, label_cache):
    """
    Resolve a Wikidata QID to its English label.
    Updates label_cache in-place.
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

    readable_triples = []
    for h, r, t in kg_triples:
        h_label = qid_to_title.get(h, get_label(h, label_cache)) \
                  if isinstance(h, str) and h.startswith("Q") else h
        t_label = get_label(t, label_cache) \
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
    if os.path.exists(cache_path):
        print(f"Loading KG from cache: {cache_path}")
        qid_map, label_cache, kg_triples, readable_triples = load_kg_cache(cache_path)

        movies_sub = movies_sub.copy()
        movies_sub["qid"] = movies_sub["movieId"].map(qid_map)
        movies_sub = movies_sub.dropna(subset=["qid"]).copy()

        print(f"  movies with QID: {len(movies_sub)}, "
              f"readable_triples: {len(readable_triples)}")
        return qid_map, label_cache, kg_triples, readable_triples, movies_sub

    print(f"No cache found at '{cache_path}' — fetching from Wikidata.")
    print(f"Estimated time: ~{len(movies_sub) * 2 // 60} min")

    # Step 1: QID lookup
    qid_map, movies_sub = fetch_qid_map(movies_sub)

    # Step 2: SPARQL property fetch
    kg_triples = fetch_kg_triples(qid_map)

    # Step 3: Label resolution + year triples
    label_cache      = {}
    readable_triples, label_cache = build_readable_triples(
        kg_triples, movies_sub, label_cache
    )

    # Step 4: Save
    save_kg_cache(cache_path, qid_map, label_cache, kg_triples, readable_triples)

    return qid_map, label_cache, kg_triples, readable_triples, movies_sub
