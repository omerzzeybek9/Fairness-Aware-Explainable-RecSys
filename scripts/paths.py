"""
paths.py — KG path sampling for PGPR training data generation.

Usage:
    from paths import sample_guided_paths, path_is_faithful, is_relation
"""

import random
from collections import defaultdict

BASE_RELS = {
    "likes", "hasGenre", "directedBy", "year", "country",
    "hasCast", "hasComposer", "writtenBy", "hasGender",
}


def is_relation(tok):
    return isinstance(tok, str) and (tok in BASE_RELS or tok.startswith("rev_"))


def path_is_faithful(path, adj):
    """Verify every (h, r, t) triple in the path exists in adj."""
    for i in range(0, len(path) - 2, 2):
        h, r, t = path[i], path[i + 1], path[i + 2]
        if t not in adj[h].get(r, set()):
            return False
    return True


# ── Individual path samplers ───────────────────────────────────────────────────

def sample_genre_path(user_node, adj):
    """User → likes → MovieA → hasGenre → Genre → rev_hasGenre → MovieB"""
    liked = list(adj[user_node].get("likes", set()))
    if not liked:
        return None
    movie_a = random.choice(liked)
    genres = list(adj[movie_a].get("hasGenre", set()))
    if not genres:
        return None
    genre = random.choice(genres)
    candidates = list(adj[genre].get("rev_hasGenre", set()) - {movie_a})
    if not candidates:
        return None
    movie_b = random.choice(candidates)
    return [user_node, "likes", movie_a, "hasGenre", genre, "rev_hasGenre", movie_b]


def sample_director_path(user_node, adj):
    """User → likes → MovieA → directedBy → Director → rev_directedBy → MovieB"""
    liked = list(adj[user_node].get("likes", set()))
    if not liked:
        return None
    movie_a = random.choice(liked)
    directors = list(adj[movie_a].get("directedBy", set()))
    if not directors:
        return None
    director = random.choice(directors)
    candidates = list(adj[director].get("rev_directedBy", set()) - {movie_a})
    if not candidates:
        return None
    movie_b = random.choice(candidates)
    return [user_node, "likes", movie_a, "directedBy", director, "rev_directedBy", movie_b]


def sample_cf_path(user_node, adj):
    """User → likes → MovieA → rev_likes → UserB → likes → MovieB"""
    liked = list(adj[user_node].get("likes", set()))
    if not liked:
        return None
    movie_a = random.choice(liked)
    other_users = list(adj[movie_a].get("rev_likes", set()) - {user_node})
    if not other_users:
        return None
    user_b = random.choice(other_users)
    their_movies = list(adj[user_b].get("likes", set()) - {movie_a} - set(liked))
    if not their_movies:
        return None
    movie_b = random.choice(their_movies)
    return [user_node, "likes", movie_a, "rev_likes", user_b, "likes", movie_b]


def sample_cast_path(user_node, adj):
    """User → likes → MovieA → hasCast → Actor → rev_hasCast → MovieB"""
    liked = list(adj[user_node].get("likes", set()))
    if not liked:
        return None
    movie_a = random.choice(liked)
    actors = list(adj[movie_a].get("hasCast", set()))
    if not actors:
        return None
    actor = random.choice(actors)
    candidates = list(adj[actor].get("rev_hasCast", set()) - {movie_a})
    if not candidates:
        return None
    movie_b = random.choice(candidates)
    return [user_node, "likes", movie_a, "hasCast", actor, "rev_hasCast", movie_b]


def sample_composer_path(user_node, adj):
    """User → likes → MovieA → hasComposer → Composer → rev_hasComposer → MovieB"""
    liked = list(adj[user_node].get("likes", set()))
    if not liked:
        return None
    movie_a = random.choice(liked)
    composers = list(adj[movie_a].get("hasComposer", set()))
    if not composers:
        return None
    composer = random.choice(composers)
    candidates = list(adj[composer].get("rev_hasComposer", set()) - {movie_a})
    if not candidates:
        return None
    movie_b = random.choice(candidates)
    return [user_node, "likes", movie_a, "hasComposer", composer, "rev_hasComposer", movie_b]


def sample_writer_path(user_node, adj):
    """User → likes → MovieA → writtenBy → Writer → rev_writtenBy → MovieB"""
    liked = list(adj[user_node].get("likes", set()))
    if not liked:
        return None
    movie_a = random.choice(liked)
    writers = list(adj[movie_a].get("writtenBy", set()))
    if not writers:
        return None
    writer = random.choice(writers)
    candidates = list(adj[writer].get("rev_writtenBy", set()) - {movie_a})
    if not candidates:
        return None
    movie_b = random.choice(candidates)
    return [user_node, "likes", movie_a, "writtenBy", writer, "rev_writtenBy", movie_b]


# ── Main sampling function ─────────────────────────────────────────────────────

DEFAULT_PATTERN_WEIGHTS = {
    "genre":    0.25,
    "director": 0.20,
    "cf":       0.20,
    "cast":     0.15,
    "composer": 0.10,
    "writer":   0.10,
}

_SAMPLERS = {
    "genre":    sample_genre_path,
    "director": sample_director_path,
    "cf":       sample_cf_path,
    "cast":     sample_cast_path,
    "composer": sample_composer_path,
    "writer":   sample_writer_path,
}


def sample_guided_paths(users, adj, paths_per_user=150,
                        pattern_weights=None, deduplicate=True):
    """
    Sample KG paths for a list of users.

    Args:
        users:            list of user node strings (e.g. ['User_1', ...])
        adj:              KG adjacency dict  adj[node][relation] = set(tails)
        paths_per_user:   number of path attempts per user
        pattern_weights:  dict of pattern_name → sampling weight
        deduplicate:      remove duplicate paths

    Returns:
        list of paths, each path is a list of tokens
    """
    if pattern_weights is None:
        pattern_weights = DEFAULT_PATTERN_WEIGHTS

    pattern_names = list(pattern_weights.keys())
    weights = [pattern_weights[p] for p in pattern_names]
    pattern_counts = {p: 0 for p in pattern_names}
    paths = []

    for u in users:
        for _ in range(paths_per_user):
            pattern = random.choices(pattern_names, weights=weights, k=1)[0]
            path = _SAMPLERS[pattern](u, adj)

            if path is None:
                for fallback in pattern_names:
                    if fallback != pattern:
                        path = _SAMPLERS[fallback](u, adj)
                        if path is not None:
                            pattern = fallback
                            break

            if path is not None:
                paths.append(path)
                pattern_counts[pattern] += 1

    total = sum(pattern_counts.values())
    print("Pattern distribution:")
    for p, c in pattern_counts.items():
        print(f"  {p}: {c} ({c / total * 100:.1f}%)")
    print(f"Total paths before dedupe: {len(paths)}")

    if deduplicate:
        seen = set()
        unique = []
        for p in paths:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                unique.append(p)
        paths = unique
        print(f"After dedupe: {len(paths)}")

    return paths
