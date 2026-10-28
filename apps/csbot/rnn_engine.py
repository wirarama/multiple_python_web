"""
rnn_engine.py — GRU Recurrent Neural Network classifier, implemented from
scratch in NumPy (no TensorFlow/PyTorch), that learns to match a user
message to a Knowledge Base entry — the same job nlp_engine.py's decision
tree already does, but via a trained neural model instead of hand-tuned
TF-IDF/keyword-overlap scoring.

Trained per-SLM: every SLM (see app.py's multi-SLM registry) can have zero
or one trained RNN model attached, built straight from that SLM's own `kb`
data — no separate question/answer dataset is required. The resulting
model (embeddings, GRU weights, dense layer, vocabulary, label map) is one
JSON-serializable dict, so it can be exported and re-imported exactly like
an SLM export (see /api/rnn/* routes in app.py) — including onto a
*different* SLM/machine, as long as the KB entries it was trained on are
still present there (matched back by entry id; entries no longer found
are reported as "stale" instead of crashing).

Architecture
------------
    token ids -> Embedding -> GRU (single layer, manual BPTT)
              -> final hidden state -> Dense -> Softmax over KB entries

Training data is bootstrapped from the KB itself: for every entry we
generate several short training texts from its keywords (individually and
joined) plus a slice of its answer text, all labeled with that entry's
class index — mirroring what the decision tree already treats as "this
text belongs to this entry". This lets the RNN be (re)trained with one
click, with zero extra data-entry work for the user.
"""

import math
import random
import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from nlp_engine import tokenize, remove_stopwords

FORMAT_TAG     = 'csbot-rnn-model'
FORMAT_VERSION = 1

PAD, UNK = '<pad>', '<unk>'

# ── Safety caps — this trains synchronously inside a Flask request ─────────
MAX_EPOCHS  = 300
MAX_HIDDEN  = 128
MAX_EMBED   = 96
MAX_VOCAB   = 4000
MAX_SEQ_LEN = 32
MAX_BATCH   = 128


def _now() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _preprocess(text: str) -> List[str]:
    """Same tokenizer family as the decision tree (nlp_engine.tokenize +
    stopword removal), kept unstemmed so the RNN can still learn surface
    morphology on its own."""
    return remove_stopwords(tokenize(text or ''))


# ══════════════════════════════════════════════════════════════════════════
# Vocabulary
# ══════════════════════════════════════════════════════════════════════════
class Vocab:
    def __init__(self, tokens: Optional[List[str]] = None):
        self.itos: List[str] = [PAD, UNK]
        self.stoi: Dict[str, int] = {PAD: 0, UNK: 1}
        if tokens:
            for t in tokens:
                self.add(t)

    def add(self, tok: str) -> int:
        if tok not in self.stoi:
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)
        return self.stoi[tok]

    def encode(self, tok: str) -> int:
        return self.stoi.get(tok, 1)  # UNK

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, texts: List[str], max_vocab: int = MAX_VOCAB) -> 'Vocab':
        freq: Dict[str, int] = {}
        for t in texts:
            for tok in _preprocess(t):
                freq[tok] = freq.get(tok, 0) + 1
        ranked = sorted(freq, key=lambda k: (-freq[k], k))[:max_vocab]
        return cls(ranked)

    def to_list(self) -> List[str]:
        """Exported form — PAD/UNK are implicit (always re-added first)."""
        return self.itos[2:]

    @classmethod
    def from_list(cls, tokens: List[str]) -> 'Vocab':
        return cls(tokens)


def encode_text(vocab: Vocab, text: str, max_len: int) -> Tuple[np.ndarray, int, float]:
    """Returns (ids, actual_length, unk_ratio). unk_ratio is the fraction of
    tokens the vocabulary didn't recognize — a cheap out-of-vocabulary/
    out-of-scope signal used to guard against over-confident predictions
    on gibberish or off-topic input (see LoadedRNNModel.predict)."""
    toks = _preprocess(text)[:max_len]
    ids = [vocab.encode(t) for t in toks]
    unk_ratio = (sum(1 for i in ids if i == 1) / len(ids)) if ids else 1.0
    length = max(len(ids), 1)
    if not ids:
        ids = [1]  # UNK
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64), length, unk_ratio


# ══════════════════════════════════════════════════════════════════════════
# GRU classifier — manual forward + backward (BPTT), pure NumPy
# ══════════════════════════════════════════════════════════════════════════
class GRUClassifier:
    PARAM_NAMES = ['E', 'Wz', 'Wr', 'Wh', 'Uz', 'Ur', 'Uh',
                   'bz', 'br', 'bh', 'Wout', 'bout']

    def __init__(self, vocab_size: int, n_classes: int,
                 embed_dim: int = 32, hidden_dim: int = 48, seed: int = 42):
        self.vocab_size = vocab_size
        self.n_classes  = n_classes
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim
        self.rng = np.random.default_rng(seed)

        D, H, C, V = embed_dim, hidden_dim, n_classes, vocab_size
        gate_scale = 1.0 / math.sqrt(max(D, 1))
        rec_scale  = 1.0 / math.sqrt(max(H, 1))
        out_scale  = 1.0 / math.sqrt(max(H, 1))

        self.E = (self.rng.standard_normal((V, D)) * 0.1).astype(np.float64)
        self.E[0] = 0.0  # PAD stays at zero, never trained

        self.Wz = (self.rng.standard_normal((D, H)) * gate_scale).astype(np.float64)
        self.Wr = (self.rng.standard_normal((D, H)) * gate_scale).astype(np.float64)
        self.Wh = (self.rng.standard_normal((D, H)) * gate_scale).astype(np.float64)
        self.Uz = (self.rng.standard_normal((H, H)) * rec_scale).astype(np.float64)
        self.Ur = (self.rng.standard_normal((H, H)) * rec_scale).astype(np.float64)
        self.Uh = (self.rng.standard_normal((H, H)) * rec_scale).astype(np.float64)
        self.bz = np.zeros(H)
        self.br = np.zeros(H)
        self.bh = np.zeros(H)

        self.Wout = (self.rng.standard_normal((H, C)) * out_scale).astype(np.float64)
        self.bout = np.zeros(C)

    def _params(self) -> Dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in self.PARAM_NAMES}

    # ── forward ─────────────────────────────────────────────────────────
    def forward(self, X: np.ndarray, lengths: np.ndarray):
        """X: (B,T) token ids. lengths: (B,) actual (unpadded) lengths."""
        B, T = X.shape
        Xemb = self.E[X]  # (B,T,D)
        h = np.zeros((B, self.hidden_dim))
        cache = []
        for t in range(T):
            x_t = Xemb[:, t, :]
            z  = _sigmoid(x_t @ self.Wz + h @ self.Uz + self.bz)
            r  = _sigmoid(x_t @ self.Wr + h @ self.Ur + self.br)
            hh = np.tanh(x_t @ self.Wh + (r * h) @ self.Uh + self.bh)
            h_new = (1 - z) * h + z * hh
            mask_t = (t < lengths).astype(np.float64).reshape(-1, 1)
            h_prev = h
            h = mask_t * h_new + (1 - mask_t) * h_prev
            cache.append((x_t, h_prev, z, r, hh, mask_t))
        logits = h @ self.Wout + self.bout
        return logits, h, cache

    def predict_proba(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        logits, _, _ = self.forward(X, lengths)
        return _softmax(logits)

    # ── loss + full backprop-through-time ──────────────────────────────
    def loss_and_grads(self, X: np.ndarray, lengths: np.ndarray, y: np.ndarray):
        B, T = X.shape
        logits, h_final, cache = self.forward(X, lengths)
        probs = _softmax(logits)
        clipped = np.clip(probs[np.arange(B), y], 1e-9, 1.0)
        loss = float(-np.mean(np.log(clipped)))
        acc  = float(np.mean(np.argmax(probs, axis=1) == y))

        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1
        dlogits /= B

        grads = {k: np.zeros_like(v) for k, v in self._params().items()}
        grads['Wout'] = h_final.T @ dlogits
        grads['bout'] = dlogits.sum(axis=0)
        dh = dlogits @ self.Wout.T

        for t in reversed(range(T)):
            x_t, h_prev, z, r, hh, mask_t = cache[t]

            dz  = dh * (hh - h_prev) * mask_t
            dhh = dh * z * mask_t
            # pass-through unchanged for padded rows (mask_t == 0)
            dh_prev = dh * (1 - z) * mask_t + dh * (1 - mask_t)

            dhh_pre = dhh * (1 - hh ** 2)
            grads['Wh'] += x_t.T @ dhh_pre
            grads['Uh'] += (r * h_prev).T @ dhh_pre
            grads['bh'] += dhh_pre.sum(axis=0)
            dx_h  = dhh_pre @ self.Wh.T
            d_rh  = dhh_pre @ self.Uh.T
            dr    = d_rh * h_prev
            dh_prev += d_rh * r

            dr_pre = dr * r * (1 - r)
            grads['Wr'] += x_t.T @ dr_pre
            grads['Ur'] += h_prev.T @ dr_pre
            grads['br'] += dr_pre.sum(axis=0)
            dx_r = dr_pre @ self.Wr.T
            dh_prev += dr_pre @ self.Ur.T

            dz_pre = dz * z * (1 - z)
            grads['Wz'] += x_t.T @ dz_pre
            grads['Uz'] += h_prev.T @ dz_pre
            grads['bz'] += dz_pre.sum(axis=0)
            dx_z = dz_pre @ self.Wz.T
            dh_prev += dz_pre @ self.Uz.T

            dx_t = dx_h + dx_r + dx_z
            np.add.at(grads['E'], X[:, t], dx_t)

            dh = dh_prev

        grads['E'][0] = 0.0  # keep PAD gradient-free
        return loss, acc, grads

    # ── training loop (Adam) ───────────────────────────────────────────
    def fit(self, X: np.ndarray, lengths: np.ndarray, y: np.ndarray,
            epochs: int = 40, batch_size: int = 32, lr: float = 0.01,
            seed: int = 42) -> List[Dict[str, float]]:
        n = X.shape[0]
        params = self._params()
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0

        rng = np.random.default_rng(seed)
        history = []
        for ep in range(epochs):
            order = rng.permutation(n)
            ep_loss, ep_acc, nb = 0.0, 0.0, 0
            for start in range(0, n, batch_size):
                bidx = order[start:start + batch_size]
                loss, acc, grads = self.loss_and_grads(X[bidx], lengths[bidx], y[bidx])
                step += 1
                for k in params:
                    g = np.clip(grads[k], -5.0, 5.0)
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v[k] = b2 * v[k] + (1 - b2) * (g * g)
                    mhat = m[k] / (1 - b1 ** step)
                    vhat = v[k] / (1 - b2 ** step)
                    params[k] -= lr * mhat / (np.sqrt(vhat) + eps)
                ep_loss += loss
                ep_acc  += acc
                nb += 1
            history.append({'epoch': ep + 1,
                             'loss': round(ep_loss / max(nb, 1), 4),
                             'acc':  round(ep_acc / max(nb, 1), 4)})
        return history

    # ── export / import ────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, list]:
        return {k: v.tolist() for k, v in self._params().items()}

    def load_state(self, state: Dict[str, list]):
        for k in self.PARAM_NAMES:
            if k in state:
                setattr(self, k, np.array(state[k], dtype=np.float64))


# ══════════════════════════════════════════════════════════════════════════
# Bootstrap training samples straight out of the KB (decision-tree data)
# ══════════════════════════════════════════════════════════════════════════
def _entry_samples(entry: Dict, seed_rng: random.Random) -> List[str]:
    samples = []
    kws = [k for k in entry.get('keywords', []) if k]
    answer = entry.get('answer', '')

    if kws:
        samples.append(' '.join(kws))
        for k in kws:
            samples.append(k)
        if len(kws) > 1:
            shuffled = kws[:]
            seed_rng.shuffle(shuffled)
            samples.append(' '.join(shuffled))

    if answer:
        toks = _preprocess(answer)
        if toks:
            samples.append(' '.join(toks[:18]))
            if len(toks) > 8:
                samples.append(' '.join(toks[:8]))

    # de-dupe while preserving order
    seen = set()
    out = []
    for s in samples:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_training_set(kb: List[Dict], seed: int = 42) -> Tuple[List[str], List[int], List[str]]:
    """Returns (texts, class_indices, label_ids) where label_ids[i] is the
    KB entry id for class index i (i == position of the entry in kb)."""
    rng = random.Random(seed)
    texts: List[str] = []
    class_idx: List[int] = []
    label_ids: List[str] = []
    for i, e in enumerate(kb):
        label_ids.append(e['id'])
        for text in _entry_samples(e, rng):
            texts.append(text)
            class_idx.append(i)
    return texts, class_idx, label_ids


# ══════════════════════════════════════════════════════════════════════════
# Public API — train / package / load / predict
# ══════════════════════════════════════════════════════════════════════════
def train_rnn_model(kb: List[Dict],
                     embed_dim: int = 32, hidden_dim: int = 48,
                     epochs: int = 40, batch_size: int = 32, lr: float = 0.01,
                     max_len: int = 20, seed: int = 42,
                     slm_id: str = '', slm_name: str = '') -> Dict[str, Any]:
    """Train a fresh GRU classifier on `kb` (the active SLM's knowledge
    base) and return a fully self-contained, JSON-serializable model
    package — the same dict that gets written to disk for export/import."""
    n_classes = len(kb)
    if n_classes < 2:
        raise ValueError('Minimal 2 entry KB diperlukan untuk melatih model RNN '
                          '(butuh setidaknya 2 kelas untuk diklasifikasikan)')

    embed_dim  = max(4, min(int(embed_dim), MAX_EMBED))
    hidden_dim = max(4, min(int(hidden_dim), MAX_HIDDEN))
    epochs     = max(1, min(int(epochs), MAX_EPOCHS))
    batch_size = max(1, min(int(batch_size), MAX_BATCH))
    max_len    = max(4, min(int(max_len), MAX_SEQ_LEN))
    lr         = float(lr) if lr and lr > 0 else 0.01

    texts, class_idx, label_ids = build_training_set(kb, seed=seed)
    if not texts:
        raise ValueError('Tidak ada teks yang bisa diekstrak dari KB (keywords/answer kosong)')

    vocab = Vocab.build(texts, max_vocab=MAX_VOCAB)

    X = np.zeros((len(texts), max_len), dtype=np.int64)
    lengths = np.zeros(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        ids, length, _ = encode_text(vocab, t, max_len)
        X[i] = ids
        lengths[i] = length
    y = np.array(class_idx, dtype=np.int64)

    clf = GRUClassifier(vocab_size=len(vocab), n_classes=n_classes,
                         embed_dim=embed_dim, hidden_dim=hidden_dim, seed=seed)
    history = clf.fit(X, lengths, y, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed)

    return {
        'format':      FORMAT_TAG,
        'version':     FORMAT_VERSION,
        'trained_at':  _now(),
        'source_slm':  {'id': slm_id, 'name': slm_name},
        'hyperparams': {
            'embed_dim':  embed_dim,
            'hidden_dim': hidden_dim,
            'max_len':    max_len,
            'vocab_size': len(vocab),
            'n_classes':  n_classes,
            'epochs':     epochs,
            'batch_size': batch_size,
            'lr':         lr,
            'seed':       seed,
        },
        'vocab':         vocab.to_list(),
        'labels':        label_ids,
        'train_samples': len(texts),
        'history':       history,
        'final_loss':    history[-1]['loss'] if history else None,
        'final_acc':     history[-1]['acc']  if history else None,
        'weights':       clf.to_dict(),
    }


def package_summary(package: Dict[str, Any]) -> Dict[str, Any]:
    """Metadata-only view of a model package (no weights/vocab payload) —
    what's safe to send back to the browser for a status display."""
    return {
        'trained_at':    package.get('trained_at'),
        'source_slm':    package.get('source_slm'),
        'hyperparams':   package.get('hyperparams'),
        'train_samples': package.get('train_samples'),
        'final_loss':    package.get('final_loss'),
        'final_acc':     package.get('final_acc'),
        'history_tail':  (package.get('history') or [])[-10:],
        'n_labels':      len(package.get('labels') or []),
    }


class LoadedRNNModel:
    """A trained model package reconstructed into live NumPy arrays, ready
    to score chat messages. Cache one of these per SLM (see app.py) —
    reconstructing from JSON on every request is unnecessary work."""

    def __init__(self, package: Dict[str, Any]):
        if package.get('format') != FORMAT_TAG:
            raise ValueError(f'Format model tidak dikenali (harus "{FORMAT_TAG}")')
        hp = package['hyperparams']
        self.vocab   = Vocab.from_list(package['vocab'])
        self.labels  = package['labels']
        self.max_len = hp['max_len']
        self.clf = GRUClassifier(vocab_size=hp['vocab_size'], n_classes=hp['n_classes'],
                                  embed_dim=hp['embed_dim'], hidden_dim=hp['hidden_dim'],
                                  seed=hp.get('seed', 42))
        self.clf.load_state(package['weights'])
        self.summary = package_summary(package)

    def predict(self, text: str, kb: List[Dict], top_k: int = 3,
                threshold: float = 0.5, max_unk_ratio: float = 0.7) -> Dict[str, Any]:
        ids, length, unk_ratio = encode_text(self.vocab, text, self.max_len)
        X = ids.reshape(1, -1)
        lengths = np.array([length])
        probs = self.clf.predict_proba(X, lengths)[0]
        order = np.argsort(-probs)[:max(int(top_k), 1)]

        kb_by_id = {e['id']: e for e in kb}
        topk = []
        for idx in order:
            eid = self.labels[idx] if idx < len(self.labels) else None
            entry = kb_by_id.get(eid)
            topk.append({
                'entry_id': eid,
                'score':    round(float(probs[idx]), 4),
                'keywords': entry.get('keywords', []) if entry else [],
                'stale':    entry is None,
            })

        best = topk[0] if topk else {'entry_id': None, 'score': 0.0, 'stale': True}
        best_entry = kb_by_id.get(best['entry_id'])

        # Guard against over-confident predictions on mostly-unrecognized
        # (out-of-vocabulary / off-topic) input — softmax always sums to 1,
        # so a gibberish query can still look "confident" otherwise.
        in_scope = unk_ratio <= max_unk_ratio
        matched = best_entry is not None and best['score'] >= threshold and in_scope

        return {
            'entry':     best_entry if matched else None,
            'score':     best['score'],
            'stale':     best['stale'],
            'unk_ratio': round(unk_ratio, 3),
            'matched':   matched,
            'topk':      topk,
        }
