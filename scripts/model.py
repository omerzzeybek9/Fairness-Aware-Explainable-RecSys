"""
model.py - GPT-2 based KG path model for explainable recommendation.

The model learns to generate valid KG paths (e.g. User -> likes -> Movie_A ->
hasGenre -> Genre -> rev_hasGenre -> Movie_B) using a small GPT-2 language model.
At inference time, constrained generation ensures every generated path is a
valid walk in the knowledge graph.

Functions:
    build_vocab(paths, base_rels)           -> vocab, id2tok, special token ids
    create_path_dataset(paths, ...)         -> train_loader, val_loader
    create_model(vocab_size, max_len, ...)  -> GPT2LMHeadModel
    train_model(model, train_loader, ...)   -> trained model
    constrained_generate(model, ...)        -> list of tokens
    extract_patterns(tokens, ...)           -> dict of detected patterns
    score_path(tokens, model, ...)          -> float score
    generate_topk(user_node, model, ...)    -> list of (movie, pattern_type)
"""

import random
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def build_vocab(paths, base_rels):
    """
    Build token vocabulary from sampled KG paths.

    Returns:
        vocab   : dict {token_str -> int}
        id2tok  : dict {int -> token_str}
        PAD, BOS, EOS, UNK : int ids for special tokens
    """
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for p in paths:
        for tok in p:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    id2tok = {i: t for t, i in vocab.items()}
    PAD, BOS, EOS, UNK = [vocab[t] for t in SPECIAL_TOKENS]
    return vocab, id2tok, PAD, BOS, EOS, UNK


def is_relation(tok, base_rels):
    """Check if a token is a relation (forward or reverse)."""
    return isinstance(tok, str) and (tok in base_rels or tok.startswith("rev_"))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _encode_path(tokens, vocab, UNK, BOS, EOS):
    """Convert a path (list of strings) to a list of token ids with BOS/EOS."""
    ids = [vocab.get(tok, UNK) for tok in tokens]
    return [BOS] + ids + [EOS]


def _pad_to_len(ids, length, pad_id):
    """Pad or truncate a list of ids to a fixed length."""
    return ids[:length] + [pad_id] * max(0, length - len(ids))


def _corrupt_path(path, all_entities, base_rels):
    """Create a negative path by replacing the last entity with a random one."""
    last_entity_idx = None
    for i in range(len(path) - 1, -1, -1):
        if not is_relation(path[i], base_rels):
            last_entity_idx = i
            break
    if last_entity_idx is None:
        return path[:]
    neg = path[:]
    original = neg[last_entity_idx]
    replacement = random.choice(all_entities)
    while replacement == original:
        replacement = random.choice(all_entities)
    neg[last_entity_idx] = replacement
    return neg


class PathDataset(Dataset):
    """Dataset of positive KG paths paired with corrupted negatives."""

    def __init__(self, paths, max_len, vocab, all_entities, base_rels, PAD, BOS, EOS, UNK):
        self.positives = []
        self.negatives = []
        for p in paths:
            pos_ids = _pad_to_len(_encode_path(p, vocab, UNK, BOS, EOS), max_len, PAD)
            neg = _corrupt_path(p, all_entities, base_rels)
            neg_ids = _pad_to_len(_encode_path(neg, vocab, UNK, BOS, EOS), max_len, PAD)
            self.positives.append(pos_ids)
            self.negatives.append(neg_ids)

    def __len__(self):
        return len(self.positives)

    def __getitem__(self, idx):
        pos = torch.tensor(self.positives[idx], dtype=torch.long)
        neg = torch.tensor(self.negatives[idx], dtype=torch.long)
        pos_labels = pos.clone()
        pos_labels[pos_labels == 0] = -100  # mask padding in loss
        neg_labels = neg.clone()
        neg_labels[neg_labels == 0] = -100
        return pos, pos_labels, neg, neg_labels


def create_path_dataset(paths, vocab, base_rels, PAD, BOS, EOS, UNK,
                        batch_size=64, val_ratio=0.1):
    """
    Shuffle paths, split into train/val, create DataLoaders.

    Returns:
        train_loader, val_loader, max_len, all_entities
    """
    max_len = max(len(p) for p in paths) + 2  # +2 for BOS/EOS
    all_entities = sorted({tok for p in paths for tok in p if not is_relation(tok, base_rels)})

    shuffled = paths[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_ratio))
    train_paths = shuffled[:split]
    val_paths = shuffled[split:]

    train_ds = PathDataset(train_paths, max_len, vocab, all_entities, base_rels, PAD, BOS, EOS, UNK)
    val_ds = PathDataset(val_paths, max_len, vocab, all_entities, base_rels, PAD, BOS, EOS, UNK)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"MAX_LEN: {max_len}")
    print(f"Train paths: {len(train_paths)}, Val paths: {len(val_paths)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    return train_loader, val_loader, max_len, all_entities


# ---------------------------------------------------------------------------
# Model Creation
# ---------------------------------------------------------------------------

def create_model(vocab_size, max_len, BOS, EOS, device="cpu",
                 n_embd=192, n_layer=4, n_head=4, dropout=0.1):
    """Create a small GPT-2 model for KG path generation."""
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _compute_combined_loss(model, pos_ids, pos_labels, neg_ids, neg_labels,
                           margin=0.5, lambda_neg=0.3):
    """LM loss on positive paths + margin-based contrastive loss."""
    pos_out = model(input_ids=pos_ids, labels=pos_labels)
    lm_loss = pos_out.loss
    neg_out = model(input_ids=neg_ids, labels=neg_labels)
    neg_nll = neg_out.loss
    contrastive = torch.clamp(lm_loss - neg_nll + margin, min=0.0)
    return lm_loss + lambda_neg * contrastive


def train_model(model, train_loader, val_loader, device="cpu",
                epochs=10, lr=3e-4, patience=2):
    """
    Train the GPT-2 path model with early stopping.

    Returns the model restored to its best validation state.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for ep in range(epochs):
        # --- train ---
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}")
        for pos, pos_labels, neg, neg_labels in pbar:
            pos, pos_labels = pos.to(device), pos_labels.to(device)
            neg, neg_labels = neg.to(device), neg_labels.to(device)

            loss = _compute_combined_loss(model, pos, pos_labels, neg, neg_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # --- validate ---
        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for pos, pos_labels, neg, neg_labels in val_loader:
                pos, pos_labels = pos.to(device), pos_labels.to(device)
                out = model(input_ids=pos, labels=pos_labels)
                val_total += out.loss.item() * pos.size(0)
                val_n += pos.size(0)
        val_loss = val_total / max(1, val_n)
        print(f"  Epoch {ep+1} val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {ep+1}")
                break

    model.load_state_dict(best_state)
    print(f"Restored best model (val_loss: {best_val_loss:.4f})")
    return model


# ---------------------------------------------------------------------------
# Constrained Generation
# ---------------------------------------------------------------------------

def _allowed_next_tokens(prefix_tokens, adj, base_rels):
    """Given a generated prefix, return valid next tokens from the KG."""
    if len(prefix_tokens) == 0:
        return set(adj.keys())
    last = prefix_tokens[-1]
    if not is_relation(last, base_rels):
        # last token is an entity -> next must be a relation
        return set(adj[last].keys()) if last in adj else set()
    # last token is a relation -> next must be a tail entity
    if len(prefix_tokens) < 2:
        return set()
    prev_entity = prefix_tokens[-2]
    return set(adj[prev_entity].get(last, set()))


def constrained_generate(start_tokens, model, vocab, id2tok, adj, base_rels,
                         PAD, BOS, EOS, UNK, max_len, device="cpu",
                         max_new_tokens=20, temperature=1.0, topk=30):
    """
    Generate a KG path starting from start_tokens, constrained so every
    step follows a valid edge in the knowledge graph.

    Returns:
        list of token strings (without special tokens)
    """
    model.eval()
    ids = torch.tensor([_encode_path(start_tokens, vocab, UNK, BOS, EOS)], dtype=torch.long).to(device)
    ids = ids[:, :-1]  # remove EOS, we'll generate it

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
                ids = torch.cat([ids, torch.tensor([[EOS]], device=device)], dim=1)
                break

            # Mask logits to only allow valid next tokens
            mask = torch.zeros(len(vocab), dtype=torch.bool)
            for t in allowed:
                if t in vocab:
                    mask[vocab[t]] = True
            mask[EOS] = True
            logits = logits.masked_fill(~mask.unsqueeze(0).to(device), -1e9)

            # Top-k sampling
            if topk and topk > 0:
                top_vals, top_idx = torch.topk(logits, k=min(topk, logits.size(-1)), dim=-1)
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


# ---------------------------------------------------------------------------
# Pattern Extraction & Path Scoring
# ---------------------------------------------------------------------------

def extract_patterns(tokens, movie_titles_set):
    """
    Detect recommendation patterns in a generated path.

    Supported patterns:
        cf       : User -> likes -> Movie -> rev_likes -> User -> likes -> Movie
        genre    : Movie -> hasGenre -> Genre -> rev_hasGenre -> Movie
        director : Movie -> directedBy -> Dir -> rev_directedBy -> Movie
        cast     : Movie -> hasCast -> Actor -> rev_hasCast -> Movie
        composer : Movie -> hasComposer -> Comp -> rev_hasComposer -> Movie
        writer   : Movie -> writtenBy -> Writer -> rev_writtenBy -> Movie

    Returns:
        dict {pattern_name -> {seed, candidate, ...}}
    """
    def _is_user(x):
        return isinstance(x, str) and x.startswith("User_")

    def _is_movie(x):
        return x in movie_titles_set

    patterns = {}

    # CF pattern (7 tokens)
    for i in range(0, len(tokens) - 6):
        w = tokens[i:i+7]
        if (_is_user(w[0]) and w[1] == "likes" and _is_movie(w[2]) and
                w[3] == "rev_likes" and _is_user(w[4]) and w[5] == "likes" and _is_movie(w[6])):
            patterns["cf"] = {"seed": w[2], "candidate": w[6]}
            break

    # Bridge patterns (5 tokens): Movie -> rel -> Entity -> rev_rel -> Movie
    bridge_patterns = [
        ("genre",    "hasGenre",     "rev_hasGenre",     "genre"),
        ("director", "directedBy",   "rev_directedBy",   "director"),
        ("cast",     "hasCast",      "rev_hasCast",      "actor"),
        ("composer", "hasComposer",  "rev_hasComposer",  "composer"),
        ("writer",   "writtenBy",    "rev_writtenBy",    "writer"),
    ]
    for pat_name, fwd_rel, rev_rel, entity_key in bridge_patterns:
        if pat_name in patterns:
            continue
        for i in range(0, len(tokens) - 4):
            w = tokens[i:i+5]
            if (_is_movie(w[0]) and w[1] == fwd_rel and w[3] == rev_rel and _is_movie(w[4])):
                patterns[pat_name] = {"seed": w[0], entity_key: w[2], "candidate": w[4]}
                break

    return patterns


# Pattern priority: prefer more specific patterns
PATTERN_PRIORITY = ["director", "cast", "composer", "writer", "genre", "cf"]


def score_path(gen_tokens, user_node, model, vocab, adj, movie_titles_set,
               PAD, BOS, EOS, UNK, base_rels, device="cpu"):
    """
    Score a generated path based on model likelihood, faithfulness,
    pattern type, genre overlap, and novelty.
    """
    score = 0.0

    # 1) Model likelihood (lower NLL = better)
    ids = torch.tensor([_encode_path(gen_tokens, vocab, UNK, BOS, EOS)], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
        nll = out.loss.item()
    score += max(0, 5.0 - nll)

    # 2) Faithfulness bonus
    def _path_is_faithful(path):
        for i in range(0, len(path) - 2, 2):
            h, r, t = path[i], path[i+1], path[i+2]
            if t not in adj.get(h, {}).get(r, set()):
                return False
        return True

    if _path_is_faithful(gen_tokens):
        score += 2.0

    # 3) Pattern-specific bonus
    pats = extract_patterns(gen_tokens, movie_titles_set)
    pattern_bonus = {"director": 1.5, "cast": 1.2, "composer": 1.1,
                     "writer": 1.1, "genre": 1.0, "cf": 0.8}
    for pt in PATTERN_PRIORITY:
        if pt in pats:
            score += pattern_bonus[pt]
            break

    # 4) Genre overlap with user preferences
    candidate = None
    for pt in PATTERN_PRIORITY:
        if pt in pats:
            candidate = pats[pt].get("candidate")
            break

    if candidate and candidate in adj:
        liked = adj[user_node].get("likes", set())
        user_genres = set()
        for m in liked:
            user_genres.update(adj.get(m, {}).get("hasGenre", set()))
        cand_genres = set(adj[candidate].get("hasGenre", set()))
        if user_genres and cand_genres:
            overlap = len(user_genres & cand_genres) / len(user_genres | cand_genres)
            score += overlap * 2.0

        # 5) Novelty bonus (prefer less popular items)
        popularity = len(adj[candidate].get("rev_likes", set()))
        max_pop = max((len(adj.get(m, {}).get("rev_likes", set())) for m in liked), default=1)
        novelty = 1.0 - min(popularity / max(max_pop, 1), 1.0)
        score += novelty * 0.5

    return score


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def generate_topk(user_node, model, vocab, id2tok, adj, base_rels,
                  movie_titles_set, PAD, BOS, EOS, UNK, max_len,
                  device="cpu", K=10, max_total_attempts=800):
    """
    Generate top-K movie recommendations for a user by repeatedly sampling
    constrained paths and scoring them.

    Returns:
        list of (movie_title, pattern_type) tuples, sorted by score
    """
    liked = set(adj[user_node].get("likes", set()))
    scored_candidates = []
    seen_candidates = set()
    total_attempts = 0

    while len(scored_candidates) < K * 3 and total_attempts < max_total_attempts:
        total_attempts += 1

        gen = constrained_generate(
            [user_node], model, vocab, id2tok, adj, base_rels,
            PAD, BOS, EOS, UNK, max_len, device,
            max_new_tokens=30, temperature=1.0, topk=20,
        )
        pats = extract_patterns(gen, movie_titles_set)
        if not pats:
            continue

        # Pick the best pattern
        meta = None
        for pt in PATTERN_PRIORITY:
            if pt in pats:
                meta = pats[pt]
                meta["type"] = pt
                break

        if meta:
            candidate = meta.get("candidate")
            if candidate and candidate not in liked and candidate not in seen_candidates:
                seen_candidates.add(candidate)
                path_score = score_path(
                    gen, user_node, model, vocab, adj, movie_titles_set,
                    PAD, BOS, EOS, UNK, base_rels, device,
                )
                scored_candidates.append((path_score, candidate, meta.get("type", "unknown")))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return [(c, t) for _, c, t in scored_candidates[:K]]
