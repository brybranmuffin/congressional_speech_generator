"""Calculate Evidence-Motivation Index (EMI) scores for a speech.

EMI = (closeness to the evidence pole) - (closeness to the intuition pole),
measured as a cosine-similarity difference (following sparse_sae_emi_pipeline.ipynb).

Three methods:
  * w2v  — average the speech's Word2Vec token vectors, compare to evidence /
           intuition seed centroids.
  * bert — mean-pooled BERT activation, compare to the centered evidence /
           intuition prototype vectors.
  * gpt2 — same as BERT with the GPT-2 model and prototypes.

All artifacts live in app/models (bert/, gpt2/, w2v/).
"""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths (resolve next to this file; app/ -> /app in the container) ──────────
MODELS_DIR     = Path(__file__).resolve().parent / "models"
BERT_MODEL_DIR = str(MODELS_DIR / "bert")
GPT2_MODEL_DIR = str(MODELS_DIR / "gpt2")
W2V_PATH       = str(MODELS_DIR / "w2v" / "congressional_embeddings.txt")

BERT_EV_PROTO = str(MODELS_DIR / "bert" / "bert_evidence_prototype_centered.npy")
BERT_IN_PROTO = str(MODELS_DIR / "bert" / "bert_intuition_prototype_centered.npy")
GPT2_EV_PROTO = str(MODELS_DIR / "gpt2" / "gpt2_evidence_prototype_centered.npy")
GPT2_IN_PROTO = str(MODELS_DIR / "gpt2" / "gpt2_intuition_prototype_centered.npy")

# Match the activation-collection scripts so text is tokenized as the poles were.
BERT_MAX_LENGTH = 256
GPT2_MAX_LENGTH = 512

# W2V seed words (from sparse_sae_emi_pipeline.ipynb, Cell 6).
EVIDENCE_SEEDS = [
    "accurate", "exact", "intelligence", "precise", "search",
    "analyse", "examination", "investigate", "procedure", "show",
    "analysis", "examine", "investigation", "process", "statistics",
    "correct", "expert", "knowledge", "proof", "study",
    "correction", "explore", "lab", "question", "trial",
    "data", "fact", "learn", "read", "real",
    "dossier", "find", "logic", "reason", "true",
    "education", "findings", "logical", "research", "truth",
    "evidence", "information", "method", "science", "truthful",
    "evident", "inquiry", "pinpoint", "scientific",
]
INTUITION_SEEDS = [
    "advice", "doubt", "mislead", "suggestion", "belief",
    "fake", "mistaken", "suspicion", "believe",
    "mistrust", "view", "bogus", "feeling", "opinion",
    "viewpoint", "genuine", "perspective", "wrong",
    "deceive", "guess", "phony", "deception", "gut",
    "dishonest", "instinct", "propaganda", "dishonesty",
    "intuition", "sense", "distrust", "lie", "suggest",
]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity; 0.0 if either vector is zero-length."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── BERT / GPT-2 activation ───────────────────────────────────────────────────
@lru_cache(maxsize=2)
def _load_encoder(model_dir: str):
    """Load (and cache) a tokenizer + encoder so weights load from disk once."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token          # GPT-2 has no pad token
    model = AutoModel.from_pretrained(model_dir).eval()
    logger.info("Loaded encoder from %s", model_dir)
    return tok, model


def get_activation(text, model_dir, max_length=512):
    tok, model = _load_encoder(model_dir)

    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=max_length, padding=True)
    with torch.no_grad():
        hidden = model(**enc, output_hidden_states=True).hidden_states[-1]  # (1, seq, dim)

    mask = enc["attention_mask"].unsqueeze(-1).float()
    return ((hidden * mask).sum(1) / mask.sum(1)).squeeze(0).numpy()        # (dim,)


@lru_cache(maxsize=4)
def _load_prototype(path: str) -> np.ndarray:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Prototype not found: {path}")
    return np.load(path)


def _contextual_emi(text, model_dir, ev_path, in_path, max_length):
    """EMI = cos(activation, evidence) - cos(activation, intuition)."""
    act       = get_activation(text, model_dir, max_length=max_length)
    ev_proto  = _load_prototype(ev_path)
    in_proto  = _load_prototype(in_path)
    if act.shape != ev_proto.shape:
        raise ValueError(
            f"Activation dim {act.shape} != prototype dim {ev_proto.shape}. "
            f"With no SAE, prototypes must be in activation space "
            f"({act.shape[0]}-dim), not SAE-feature space."
        )
    return _cos(act, ev_proto) - _cos(act, in_proto)


def calculate_bert_emi(text: str) -> float:
    return _contextual_emi(text, BERT_MODEL_DIR, BERT_EV_PROTO, BERT_IN_PROTO, BERT_MAX_LENGTH)


def calculate_gpt2_emi(text: str) -> float:
    return _contextual_emi(text, GPT2_MODEL_DIR, GPT2_EV_PROTO, GPT2_IN_PROTO, GPT2_MAX_LENGTH)


# ── Word2Vec EMI ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_w2v():
    """Load W2V vectors and build evidence / intuition seed centroids (cached)."""
    from gensim.models import KeyedVectors  # heavy import, only when w2v is used

    logger.info("Loading W2V embeddings from %s (1-2 min)...", W2V_PATH)
    wv = KeyedVectors.load_word2vec_format(W2V_PATH, binary=False, no_header=True)

    ev_centroid = np.array([wv[w] for w in EVIDENCE_SEEDS if w in wv], dtype=np.float32).mean(axis=0)
    in_centroid = np.array([wv[w] for w in INTUITION_SEEDS if w in wv], dtype=np.float32).mean(axis=0)
    logger.info("W2V loaded: vocab=%d  dim=%d", len(wv), wv.vector_size)
    return wv, ev_centroid, in_centroid


def calculate_w2v_emi(text: str):
    """EMI = cos(speech_vec, evidence_centroid) - cos(speech_vec, intuition_centroid).

    Returns None if no token in the text is in the W2V vocabulary.
    """
    wv, ev_centroid, in_centroid = _load_w2v()

    vecs = [wv[w] for w in text.lower().split() if w in wv]
    if not vecs:
        return None
    speech_vec = np.mean(vecs, axis=0)
    return _cos(speech_vec, ev_centroid) - _cos(speech_vec, in_centroid)


# ── API ───────────────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


# Sync `def` so FastAPI runs this blocking, torch/gensim-heavy handler in a
# threadpool instead of stalling the event loop.
@app.post("/calculate_emi")
def calculate_emi(
    w2v: bool = Query(...),
    bert: bool = Query(...),
    gpt2: bool = Query(...),
    text: str = Query(...),
):
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="text must be a non-empty string")

    result: dict[str, float | None] = {}
    try:
        if w2v:
            result["w2v_emi"] = calculate_w2v_emi(text)
        if bert:
            result["bert_emi"] = calculate_bert_emi(text)
        if gpt2:
            result["gpt2_emi"] = calculate_gpt2_emi(text)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not result:
        raise HTTPException(status_code=422, detail="Enable at least one of w2v, bert, gpt2.")

    return result
