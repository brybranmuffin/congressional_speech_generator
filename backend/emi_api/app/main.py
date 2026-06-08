"""Combined backend: GPT-2 speech generation + Evidence Minus Intuition (EMI) scoring.

Two endpoints served from one app/container:

  * POST /generate_speech — sample a congressional-style speech from the fine-tuned
    GPT-2 LM-head model (app/models/gpt2).
  * POST /calculate_emi   — score text on the EMI, the cosine-similarity difference
    between closeness to the evidence pole and the intuition pole
    (following sparse_sae_emi_pipeline.ipynb). Methods:
      - w2v  — average the speech's Word2Vec token vectors vs. seed centroids.
      - bert — mean-pooled BERT activation vs. centered prototype vectors.
      - gpt2 — same with the GPT-2 model and prototypes.

All artifacts live in app/models (bert/, gpt2/, w2v/).
"""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel

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
    """Return (tokenizer, encoder) for activations, loading weights once.

    For GPT-2 we reuse the generation LM's base transformer rather than loading a
    second copy of the weights; for BERT we load the base model via AutoModel.
    """
    if model_dir == GPT2_MODEL_DIR:
        tok, lm, _ = load_gpt2_lm()
        return tok, lm.transformer  # GPT2Model — the encoder shared with generation

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token          # GPT-2 has no pad token
    model = AutoModel.from_pretrained(model_dir).eval()
    logger.info("Loaded encoder from %s", model_dir)
    return tok, model


def get_activation(text, model_dir, max_length=512):
    tok, model = _load_encoder(model_dir)
    dev = next(model.parameters()).device  # encoder may sit on CPU (BERT) or the LM's device (GPT-2)

    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=max_length, padding=True).to(dev)
    with torch.no_grad():
        hidden = model(**enc, output_hidden_states=True).hidden_states[-1]  # (1, seq, dim)

    mask = enc["attention_mask"].unsqueeze(-1).float()
    return ((hidden * mask).sum(1) / mask.sum(1)).squeeze(0).cpu().numpy()  # (dim,)


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


# ── GPT-2 speech generation ───────────────────────────────────────────────────
def pick_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_gpt2_lm():
    """Load (and cache) the fine-tuned GPT-2 LM-head model for generation.

    No parameters, so every caller shares one cache entry (and one set of weights):
    generation uses the LM head here, EMI reuses ``model.transformer`` as the
    encoder. Returns (tokenizer, model, device).
    """
    model_dir = GPT2_MODEL_DIR
    if not Path(model_dir).is_dir():
        raise FileNotFoundError(
            f"GPT-2 model dir not found: {model_dir}. "
            "Expected the fine-tuned checkpoint in app/models/gpt2."
        )

    dev = pick_device()
    logger.info("Loading GPT-2 LM from %s onto %s", model_dir, dev)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    model.to(dev)

    logger.info("Loaded GPT-2 LM — %d parameters", sum(p.numel() for p in model.parameters()))
    return tokenizer, model, dev


def run_generation(
    prompt: str,
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    seed: int | None = None,
) -> str:
    """Sample one speech conditioned on ``prompt`` via nucleus sampling."""
    tokenizer, model, dev = load_gpt2_lm()

    if seed is not None:
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(dev)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ── API ───────────────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_model() -> None:
    """Best-effort warm the GPT-2 LM so the first generation isn't slow."""
    try:
        load_gpt2_lm()
    except FileNotFoundError as exc:
        logger.warning("GPT-2 LM not warmed: %s", exc)


# Sync `def` so FastAPI runs this blocking, torch-heavy handler in a threadpool
# instead of stalling the event loop.
@app.post("/generate_speech")
def generate_speech(
    party: str = Query(...),
    topic: str = Query(...),
    stance: str = Query("support"),
    max_new_tokens: int = Query(200, ge=1, le=1024),
    temperature: float = Query(0.9, gt=0.0, le=2.0),
    seed: int | None = Query(None),
):
    PROMPT = f"Mr. Speaker, I rise today as a member of the {party} party in {stance} of {topic}"

    try:
        speech = run_generation(
            PROMPT,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "party": party,
        "topic": topic,
        "prompt": PROMPT,
        "speech": speech,
    }


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
