import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def build_vocab(paths, base_rels):
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}

    for p in paths:
        for tok in p:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    id2tok = {i: t for t, i in vocab.items()}
    PAD, BOS, EOS, UNK = [vocab[t] for t in SPECIAL_TOKENS]

    return vocab, id2tok, PAD, BOS, EOS, UNK


def is_relation(tok, base_rels):
    return isinstance(tok, str) and (tok in base_rels or tok.startswith("rev_"))



def _encode_path(tokens, vocab, UNK, BOS, EOS):
    ids = [vocab.get(tok, UNK) for tok in tokens]
    return [BOS] + ids + [EOS]


def _pad_to_len(ids, length, pad_id):
    return ids[:length] + [pad_id] * max(0, length - len(ids))


def _corrupt_path(path, all_entities, base_rels):
    last_entity_idx = None

    for i in range(len(path) - 1, -1, -1):
        if not is_relation(path[i], base_rels):
            last_entity_idx = i
            break

    if last_entity_idx is None:
        return path[:]

    neg = path[:]
    original = neg[last_entity_idx]
    candidates = [e for e in all_entities if e != original]

    if not candidates:
        return path[:]

    neg[last_entity_idx] = random.choice(candidates)
    return neg


class PathDataset(Dataset):

    def __init__(self, paths, max_len, vocab, all_entities, base_rels,
                 PAD, BOS, EOS, UNK):
        self.positives = []
        self.negatives = []

        for p in paths:
            pos_ids = _pad_to_len(
                _encode_path(p, vocab, UNK, BOS, EOS),
                max_len,
                PAD,
            )

            neg = _corrupt_path(p, all_entities, base_rels)
            neg_ids = _pad_to_len(
                _encode_path(neg, vocab, UNK, BOS, EOS),
                max_len,
                PAD,
            )

            self.positives.append(pos_ids)
            self.negatives.append(neg_ids)

    def __len__(self):
        return len(self.positives)

    def __getitem__(self, idx):
        pos = torch.tensor(self.positives[idx], dtype=torch.long)
        neg = torch.tensor(self.negatives[idx], dtype=torch.long)

        pos_labels = pos.clone()
        neg_labels = neg.clone()

        # PAD token id is 0 because SPECIAL_TOKENS starts with <pad>.
        pos_labels[pos_labels == 0] = -100
        neg_labels[neg_labels == 0] = -100

        return pos, pos_labels, neg, neg_labels


def create_path_dataset(paths, vocab, base_rels, PAD, BOS, EOS, UNK,
                        batch_size=64, val_ratio=0.1):
    if not paths:
        raise ValueError("No paths were provided to create_path_dataset().")

    max_len = max(len(p) for p in paths) + 2

    all_entities = sorted({
        tok
        for p in paths
        for tok in p
        if not is_relation(tok, base_rels)
    })

    shuffled = paths[:]
    random.shuffle(shuffled)

    split = int(len(shuffled) * (1 - val_ratio))
    if len(shuffled) > 1:
        split = max(1, min(split, len(shuffled) - 1))
    else:
        split = 1

    train_paths = shuffled[:split]
    val_paths = shuffled[split:]

    if not val_paths:
        val_paths = train_paths[:]

    train_ds = PathDataset(
        train_paths, max_len, vocab, all_entities, base_rels,
        PAD, BOS, EOS, UNK,
    )
    val_ds = PathDataset(
        val_paths, max_len, vocab, all_entities, base_rels,
        PAD, BOS, EOS, UNK,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"MAX_LEN: {max_len}")
    print(f"Train paths: {len(train_paths)}, Val paths: {len(val_paths)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    return train_loader, val_loader, max_len, all_entities

def create_model(vocab_size, max_len, BOS, EOS, device="cpu",
                 n_embd=256, n_layer=4, n_head=4, dropout=0.1):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_len,
        n_ctx=max_len,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        bos_token_id=BOS,
        eos_token_id=EOS,
    )

    model = GPT2LMHeadModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    return model

def _compute_combined_loss(model, pos_ids, pos_labels, neg_ids, neg_labels,
                           margin=0.5, lambda_neg=0.3):
    pos_out = model(input_ids=pos_ids, labels=pos_labels)
    lm_loss = pos_out.loss

    neg_out = model(input_ids=neg_ids, labels=neg_labels)
    neg_nll = neg_out.loss

    contrastive = torch.clamp(lm_loss - neg_nll + margin, min=0.0)

    return lm_loss + lambda_neg * contrastive


def train_model(model, train_loader, val_loader, device="cpu",
                epochs=8, lr=3e-4, patience=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs * len(train_loader)),
        eta_min=1e-5,
    )

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for ep in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep + 1}/{epochs}")

        for pos, pos_labels, neg, neg_labels in pbar:
            pos = pos.to(device)
            pos_labels = pos_labels.to(device)
            neg = neg.to(device)
            neg_labels = neg_labels.to(device)

            loss = _compute_combined_loss(
                model,
                pos,
                pos_labels,
                neg,
                neg_labels,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        model.eval()
        val_total = 0.0
        val_n = 0

        with torch.no_grad():
            for pos, pos_labels, _neg, _neg_labels in val_loader:
                pos = pos.to(device)
                pos_labels = pos_labels.to(device)

                out = model(input_ids=pos, labels=pos_labels)
                val_total += out.loss.item() * pos.size(0)
                val_n += pos.size(0)

        val_loss = val_total / max(1, val_n)
        print(f"  Epoch {ep + 1} val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {ep + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"Restored best model (val_loss: {best_val_loss:.4f})")

    return model

def _allowed_next_tokens(prefix_tokens, adj, base_rels):
    if len(prefix_tokens) == 0:
        return set(adj.keys())

    last = prefix_tokens[-1]

    if not is_relation(last, base_rels):
        return set(adj[last].keys()) if last in adj else set()

    if len(prefix_tokens) < 2:
        return set()

    prev_entity = prefix_tokens[-2]
    return set(adj.get(prev_entity, {}).get(last, set()))


def constrained_generate(start_tokens, model, vocab, id2tok, adj, base_rels,
                         PAD, BOS, EOS, UNK, max_len, device="cpu",
                         max_new_tokens=20, temperature=1.0, topk=30):
    model.eval()

    ids = torch.tensor(
        [_encode_path(start_tokens, vocab, UNK, BOS, EOS)],
        dtype=torch.long,
    ).to(device)

    ids = ids[:, :-1]

    def _strip(tokens):
        return [t for t in tokens if t not in ("<bos>", "<eos>", "<pad>")]

    def _ids_to_tokens(id_list):
        return [id2tok.get(int(i), "<unk>") for i in id_list]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=ids)
            logits = out.logits[:, -1, :] / max(temperature, 1e-6)

            prefix = _strip(_ids_to_tokens(ids[0].tolist()))
            allowed = _allowed_next_tokens(prefix, adj, base_rels)

            if not allowed:
                ids = torch.cat(
                    [ids, torch.tensor([[EOS]], device=device)],
                    dim=1,
                )
                break

            mask = torch.zeros(len(vocab), dtype=torch.bool, device=device)

            for t in allowed:
                if t in vocab:
                    mask[vocab[t]] = True

            mask[EOS] = True
            logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)

            if topk and topk > 0:
                top_vals, top_idx = torch.topk(
                    logits,
                    k=min(topk, logits.size(-1)),
                    dim=-1,
                )
                probs = torch.softmax(top_vals, dim=-1)
                choice = torch.multinomial(probs, num_samples=1)
                next_token_id = top_idx.gather(-1, choice)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            ids = torch.cat([ids, next_token_id], dim=1)

            if int(next_token_id.item()) == EOS or ids.size(1) >= max_len:
                break

    return _strip(_ids_to_tokens(ids[0].tolist()))

def extract_patterns(tokens, movie_titles_set):
    """Detect explanation patterns in a generated or enumerated KG path."""

    def _is_user(x):
        return isinstance(x, str) and x.startswith("User_")

    def _is_movie(x):
        return x in movie_titles_set

    patterns = {}

    for i in range(0, len(tokens) - 6):
        w = tokens[i:i + 7]

        if (
            _is_user(w[0])
            and w[1] == "likes"
            and _is_movie(w[2])
            and w[3] == "rev_likes"
            and _is_user(w[4])
            and w[5] == "likes"
            and _is_movie(w[6])
        ):
            patterns["cf"] = {"seed": w[2], "candidate": w[6]}
            break

    bridge_patterns = [
        ("genre", "hasGenre", "rev_hasGenre", "genre"),
        ("director", "directedBy", "rev_directedBy", "director"),
        ("cast", "hasCast", "rev_hasCast", "actor"),
        ("composer", "hasComposer", "rev_hasComposer", "composer"),
        ("writer", "writtenBy", "rev_writtenBy", "writer"),
    ]

    for pat_name, fwd_rel, rev_rel, entity_key in bridge_patterns:
        if pat_name in patterns:
            continue

        for i in range(0, len(tokens) - 4):
            w = tokens[i:i + 5]

            if (
                _is_movie(w[0])
                and w[1] == fwd_rel
                and w[3] == rev_rel
                and _is_movie(w[4])
            ):
                patterns[pat_name] = {
                    "seed": w[0],
                    entity_key: w[2],
                    "candidate": w[4],
                }
                break

    for i in range(0, len(tokens) - 6):
        w = tokens[i:i + 7]

        if (
            _is_user(w[0])
            and w[1] == "hasGender"
            and w[3] == "rev_hasGender"
            and _is_user(w[4])
            and w[5] == "likes"
            and _is_movie(w[6])
        ):
            patterns["gender"] = {"candidate": w[6]}
            break

    return patterns


PATTERN_PRIORITY = [
    "genre",
    "cf",
    "director",
    "cast",
    "writer",
    "composer",
    "gender",
]


DEFAULT_PATTERN_SCORES = {
    "genre": 0.88,
    "cf": 0.87,
    "director": 0.80,
    "cast": 0.75,
    "writer": 0.75,
    "composer": 0.70,
    "gender": 0.50,
}


def compute_pattern_specificity(adj, movie_titles_set):
    pattern_rels = {
        "director": ("directedBy", "rev_directedBy"),
        "cast": ("hasCast", "rev_hasCast"),
        "composer": ("hasComposer", "rev_hasComposer"),
        "writer": ("writtenBy", "rev_writtenBy"),
        "genre": ("hasGenre", "rev_hasGenre"),
    }

    raw_scores = {}
    avg_info = {}

    for pat, (_fwd_rel, rev_rel) in pattern_rels.items():
        counts = []

        for node, rels in adj.items():
            if rev_rel not in rels:
                continue

            films = [m for m in rels[rev_rel] if m in movie_titles_set]

            if len(films) >= 2:
                counts.append(len(films))

        avg = sum(counts) / len(counts) if counts else 1.0
        avg_info[pat] = avg
        raw_scores[pat] = 1.0 / avg

    max_s = max(raw_scores.values()) if raw_scores else 1.0
    min_s = min(raw_scores.values()) if raw_scores else 0.0

    scores = {}

    for k, v in raw_scores.items():
        scores[k] = 0.5 + 0.5 * (v - min_s) / (max_s - min_s + 1e-9)

    scores["cf"] = 0.87
    scores["gender"] = 0.50

    print("Pattern specificity scores (data-driven):")
    print(f"  {'Pattern':<10}  {'Avg movies/entity':>18}  {'Score':>7}")
    print(f"  {'-' * 40}")

    for k in PATTERN_PRIORITY:
        if k in scores:
            avg = avg_info.get(k, 0)
            print(f"  {k:<10}  {avg:>18.2f}  {scores[k]:>7.4f}")

    return scores


def compute_cf_score(user_node, candidate_movie, adj, liked, max_neighbors=30):
    candidate_likers = adj.get(candidate_movie, {}).get("rev_likes", set())

    if not candidate_likers or not liked:
        return 0.0

    sims = []

    for other_user in candidate_likers:
        if other_user == user_node:
            continue

        other_liked = adj.get(other_user, {}).get("likes", set())

        if not other_liked:
            continue

        inter = len(liked & other_liked)
        union = len(liked | other_liked)

        if union > 0:
            sims.append(inter / union)

    if not sims:
        return 0.0

    sims.sort(reverse=True)
    top_sims = sims[:max_neighbors]

    return sum(top_sims) / len(top_sims)


def compute_user_popularity_preference(user_node, adj, max_popularity):
    liked = adj.get(user_node, {}).get("likes", set())

    if not liked or max_popularity <= 0:
        return 0.5

    pops = []
    for m in liked:
        pop = len(adj.get(m, {}).get("rev_likes", set()))
        pops.append(pop / max_popularity)

    if not pops:
        return 0.5

    return sum(pops) / len(pops)


def compute_popularity_calibration_score(user_node, candidate_movie, adj, max_popularity):
    if max_popularity <= 0:
        return 0.5

    user_pref = compute_user_popularity_preference(
        user_node=user_node,
        adj=adj,
        max_popularity=max_popularity,
    )

    cand_pop = len(adj.get(candidate_movie, {}).get("rev_likes", set()))
    cand_pop_norm = cand_pop / max_popularity
    cand_pop_norm = max(0.0, min(1.0, cand_pop_norm))

    return 1.0 - abs(user_pref - cand_pop_norm)


def score_path(gen_tokens, user_node, model, vocab, adj, movie_titles_set,
               PAD, BOS, EOS, UNK, base_rels, device="cpu",
               _movie_genres_cache=None,
               _pattern_scores=None):
    ids = torch.tensor(
        [_encode_path(gen_tokens, vocab, UNK, BOS, EOS)],
        dtype=torch.long,
    ).to(device)

    ids = ids[:, :model.config.n_positions]

    with torch.no_grad():
        logits = model(input_ids=ids).logits

    log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
    targets = ids[0, 1:]
    mean_lp = float(log_probs[range(len(targets)), targets].mean().item())

    # Map mean log-prob into [0, 1]. The same mapping the batched scorer uses.
    return float(min(1.0, max(0.0, 1.0 + mean_lp / 10.0)))

def enumerate_candidates(user_node, adj, movie_titles_set, liked,
                         include_gender=False,
                         max_paths_per_movie=2):
    all_paths = {}
    priority = {pat: i for i, pat in enumerate(PATTERN_PRIORITY)}

    def _add(movie_b, pat_type, path):
        if movie_b not in movie_titles_set or movie_b in liked:
            return

        if movie_b not in all_paths:
            all_paths[movie_b] = []

        existing_types = {pt for pt, _p in all_paths[movie_b]}
        if pat_type in existing_types:
            return

        all_paths[movie_b].append((pat_type, path))
        all_paths[movie_b].sort(key=lambda x: priority.get(x[0], 999))

        if len(all_paths[movie_b]) > max_paths_per_movie:
            all_paths[movie_b] = all_paths[movie_b][:max_paths_per_movie]

    bridge_rels = [
        ("genre", "hasGenre", "rev_hasGenre"),
        ("director", "directedBy", "rev_directedBy"),
        ("cast", "hasCast", "rev_hasCast"),
        ("writer", "writtenBy", "rev_writtenBy"),
        ("composer", "hasComposer", "rev_hasComposer"),
    ]

    for movie_a in liked:
        for pat_type, fwd_rel, rev_rel in bridge_rels:
            for entity in adj.get(movie_a, {}).get(fwd_rel, set()):
                for movie_b in adj.get(entity, {}).get(rev_rel, set()):
                    _add(
                        movie_b,
                        pat_type,
                        [
                            user_node,
                            "likes",
                            movie_a,
                            fwd_rel,
                            entity,
                            rev_rel,
                            movie_b,
                        ],
                    )

        for user_b in adj.get(movie_a, {}).get("rev_likes", set()):
            if user_b == user_node:
                continue

            for movie_b in adj.get(user_b, {}).get("likes", set()):
                _add(
                    movie_b,
                    "cf",
                    [
                        user_node,
                        "likes",
                        movie_a,
                        "rev_likes",
                        user_b,
                        "likes",
                        movie_b,
                    ],
                )

    if include_gender:
        for gender_node in adj.get(user_node, {}).get("hasGender", set()):
            for user_b in adj.get(gender_node, {}).get("rev_hasGender", set()):
                if user_b == user_node:
                    continue

                for movie_b in adj.get(user_b, {}).get("likes", set()):
                    _add(
                        movie_b,
                        "gender",
                        [
                            user_node,
                            "hasGender",
                            gender_node,
                            "rev_hasGender",
                            user_b,
                            "likes",
                            movie_b,
                        ],
                    )

    return all_paths


def _batch_score_conf(path_list, model, vocab, UNK, BOS, EOS, device,
                      batch_size=256):
    if not path_list:
        return []

    max_pos = model.config.n_positions

    encoded = [
        _encode_path(path_tokens, vocab, UNK, BOS, EOS)[:max_pos]
        for path_tokens in path_list
    ]

    s_conf_list = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            batch_enc = encoded[start:start + batch_size]
            max_len_batch = max(len(e) for e in batch_enc)

            padded = [
                e + [0] * (max_len_batch - len(e))
                for e in batch_enc
            ]

            ids_t = torch.tensor(padded, dtype=torch.long).to(device)
            logits = model(input_ids=ids_t).logits

            for i, orig_ids in enumerate(batch_enc):
                seq_len = len(orig_ids)

                if seq_len <= 1:
                    s_conf_list.append(0.0)
                    continue

                log_probs = torch.log_softmax(logits[i, :seq_len - 1], dim=-1)
                targets = ids_t[i, 1:seq_len]
                mean_lp = float(
                    log_probs[range(seq_len - 1), targets].mean().item()
                )

                s_conf = float(min(1.0, max(0.0, 1.0 + mean_lp / 10.0)))
                s_conf_list.append(s_conf)

    return s_conf_list

def _dedupe_best_per_movie(scored):
    best = {}
    for score, movie, pat_type, path_tokens in scored:
        if movie not in best or score > best[movie][0]:
            best[movie] = (score, pat_type, path_tokens)
    return [
        (score, movie, pat_type, path_tokens)
        for movie, (score, pat_type, path_tokens) in best.items()
    ]


def _diversify_topk(scored, K, lambda_div=0.0, free_repeats=2):
    if not scored:
        return []

    if lambda_div <= 0:
        unique_scored = _dedupe_best_per_movie(scored)
        scored_sorted = sorted(unique_scored, key=lambda x: x[0], reverse=True)
        return [(m, t, p) for _s, m, t, p in scored_sorted[:K]]

    pool = sorted(scored, key=lambda x: x[0], reverse=True)
    selected = []
    selected_movies = set()
    type_counts = {}

    while pool and len(selected) < K:
        best_idx, best_adj = -1, -float("inf")

        for i, (score, movie, pat_type, path_tokens) in enumerate(pool):
            if movie in selected_movies:
                continue
            excess_repeats = max(0, type_counts.get(pat_type, 0) - free_repeats)
            penalty = lambda_div * excess_repeats
            adjusted = score - penalty
            if adjusted > best_adj:
                best_adj, best_idx = adjusted, i

        if best_idx < 0:
            break

        score, movie, pat_type, path_tokens = pool.pop(best_idx)
        selected.append((score, movie, pat_type, path_tokens))
        selected_movies.add(movie)
        type_counts[pat_type] = type_counts.get(pat_type, 0) + 1

    return [(m, t, p) for _s, m, t, p in selected]


def generate_topk(user_node, model, vocab, id2tok, adj, base_rels,
                  movie_titles_set, PAD, BOS, EOS, UNK, max_len,
                  device="cpu", K=10, max_total_attempts=None,
                  include_gender=False, _pattern_scores=None,
                  max_paths_per_movie=4, lambda_div=0.0, path_balance_free_repeats=2,
                  eval_batch_size=512):
    if user_node not in adj:
        return []

    liked = set(adj[user_node].get("likes", set()))

    if not liked:
        return []

    movie_genres_cache = {
        m: set(adj.get(m, {}).get("hasGenre", set()))
        for m in movie_titles_set
    }

    if max_total_attempts == "random":
        attempts_cap = 600
        scored_candidates = []
        seen_candidates = set()
        total_attempts = 0

        while len(scored_candidates) < K * 5 and total_attempts < attempts_cap:
            total_attempts += 1

            gen = constrained_generate(
                [user_node],
                model,
                vocab,
                id2tok,
                adj,
                base_rels,
                PAD,
                BOS,
                EOS,
                UNK,
                max_len,
                device,
                max_new_tokens=15,
                temperature=1.0,
                topk=20,
            )

            pats = extract_patterns(gen, movie_titles_set)

            if not pats:
                continue

            meta = None

            for pt in PATTERN_PRIORITY:
                if pt in pats:
                    meta = pats[pt]
                    meta["type"] = pt
                    break

            if not meta:
                continue

            candidate = meta.get("candidate")

            if candidate and candidate not in liked and candidate not in seen_candidates:
                seen_candidates.add(candidate)

                score = score_path(
                    gen,
                    user_node,
                    model,
                    vocab,
                    adj,
                    movie_titles_set,
                    PAD,
                    BOS,
                    EOS,
                    UNK,
                    base_rels,
                    device,
                    _movie_genres_cache=movie_genres_cache,
                    _pattern_scores=_pattern_scores,
                )

                scored_candidates.append(
                    (score, candidate, meta.get("type", "unknown"), gen)
                )

        return _diversify_topk(
            scored_candidates, K, lambda_div=lambda_div,
            free_repeats=path_balance_free_repeats,
        )

    all_candidates = enumerate_candidates(
        user_node,
        adj,
        movie_titles_set,
        liked,
        include_gender=include_gender,
        max_paths_per_movie=max_paths_per_movie,
    )

    if not all_candidates:
        return []

    flat_movies = []
    flat_pat_types = []
    flat_paths = []

    for movie, path_list in all_candidates.items():
        for pat_type, path_tokens in path_list:
            flat_movies.append(movie)
            flat_pat_types.append(pat_type)
            flat_paths.append(path_tokens)

    s_conf_all = _batch_score_conf(
        flat_paths,
        model,
        vocab,
        UNK,
        BOS,
        EOS,
        device,
        batch_size=eval_batch_size,
    )

    liked_set = set(adj[user_node].get("likes", set()))

    cf_overlap: dict[str, int] = {}
    for movie_b in set(flat_movies):
        if movie_b in liked_set:
            continue
        co_users_b = set(adj.get(movie_b, {}).get("rev_likes", set()))
        overlap = 0
        for seed in liked_set:
            co_users_seed = set(adj.get(seed, {}).get("rev_likes", set()))
            overlap += len(co_users_b & co_users_seed)
        cf_overlap[movie_b] = overlap

    max_cf = max(cf_overlap.values(), default=1)
    if max_cf == 0:
        max_cf = 1

    scored = []
    for s_conf, movie, pat_type, path_tokens in zip(
        s_conf_all, flat_movies, flat_pat_types, flat_paths
    ):
        s_cf = cf_overlap.get(movie, 0) / max_cf
        score = 0.5 * s_conf + 0.5 * s_cf
        scored.append((score, movie, pat_type, path_tokens))

    return _diversify_topk(
        scored, K, lambda_div=lambda_div,
        free_repeats=path_balance_free_repeats,
    )