import logging
import os
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, GPT2LMHeadModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = str(Path(__file__).resolve().parent / "model_weights")


def pick_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_model(model_dir: str = MODEL_DIR):
    """Load (and cache) the fine-tuned GPT-2 model and tokenizer.

    Cached so the weights load from disk once and every request reuses the warm
    objects. Returns (tokenizer, model, device).
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model dir not found: {model_dir}. "
            "Place the fine-tuned checkpoint in app/model_weights."
        )

    dev = pick_device()
    logger.info("Loading GPT-2 from %s onto %s", model_dir, dev)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    model.to(dev)

    logger.info("Loaded — %d parameters", sum(p.numel() for p in model.parameters()))
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
    tokenizer, model, dev = load_model()

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


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_model() -> None:
    """Best-effort warm the model at startup so the first request isn't slow."""
    try:
        load_model()
    except FileNotFoundError as exc:
        logger.warning("Model not warmed: %s", exc)


@app.post("/generate_speech")
def generate_speech(
    party: str = Query(...),
    topic: str = Query(...),
    stance: str = Query("support"),
    max_new_tokens: int = Query(200, ge=1, le=1024),
    temperature: float = Query(0.9, gt=0.0, le=2.0),
    seed: int | None = Query(None),
    # x_api_key: str = Header(...)
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
