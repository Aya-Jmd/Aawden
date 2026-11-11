"""Beam endpoint for the Aawden demo: Aya Expanse 8B with the two-stage adapter, live.

Every verdict produced by this endpoint is the same code path as the measurement
(`scripts/wsp_cross_model.render_for` + the `{"verdict":"` prefill), so what a visitor sees is the
model the table describes, not a re-implementation.

Security posture:
  * The endpoint is public (`authorized=False`) so the organiser can call it without a Beam account.
  * Every request must carry `X-Passcode`, compared against AAWDEN_DEMO_PASSCODE, which is set as a
    Beam runtime secret at deploy time. The Beam workspace token is never placed in this file, in an
    image, or in a request; deployment reads it from the repo `.env` exactly as `beam_launch.py` does.
  * Rate limit: a fixed window per client IP, so a shared passcode cannot burn the GPU unbounded.
  * The container keeps warm 5 minutes after the last request and then scales to zero.

Weights come from the Hub by pinned revision (CohereLabs/aya-expanse-8b @ 5062468b) using the
workspace secret AAWDEN_AYA_HF_TOKEN as HF_TOKEN. The adapter ships next to this file under
`demo/adapter-aya8b-curr-s17/` (180 MB, git-ignored; copy it from the cluster before deploying).

Deploy (after the user approves spending on Beam):

    python scripts/beam_launch.py deploy-demo --passcode <value> --max-cost-usd 0.50
"""
# No `from __future__ import annotations` here: the route functions are defined inside a factory,
# and string annotations would hide `Request` from FastAPI, turning it into a query parameter.
import hmac
import os
import time
from pathlib import Path

try:
    from beam import Image, asgi
except ImportError:  # Imported locally for tests without the SDK.
    Image = asgi = None

try:  # Model at module level: FastAPI cannot bind a body model defined inside a factory.
    from pydantic import BaseModel

    class ReviewIn(BaseModel):
        case_id: str
        condition: str = "incorrect_suggestion"  # neutral | correct_suggestion | incorrect_suggestion
except ImportError:
    ReviewIn = None

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "CohereLabs/aya-expanse-8b"
MODEL_REVISION = "5062468bf9bc0c6035fd64e06274333ec127d980"
ADAPTER_DIR = Path("demo/adapter-aya8b-curr-s17")  # relative to the synced app directory
PREFILL = '{"verdict":"'
MAX_NEW_TOKENS = 6
KEEP_WARM = 300
RATE_LIMIT = (30, 60.0)  # requests per client per window (seconds)


def image():
    if Image is None:
        return None
    return Image(python_version="python3.11").add_python_packages(
        ["torch==2.5.1", "transformers==4.46.3", "peft==0.13.2", "accelerate==1.1.1",
         "fastapi==0.115.5", "pydantic==2.10.3", "sentencepiece==0.2.0", "uvicorn==0.32.1"])


def load():
    """Load the pinned base model and merge the adapter. Runs once per container."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set; the gated Aya weights need the workspace secret")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, token=token, torch_dtype=torch.bfloat16, device_map="cuda")
    adapter_path = ROOT / ADAPTER_DIR
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter not shipped with the app: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def _verdict(model, tokenizer, case, condition: str):
    import sys

    import torch

    sys.path.insert(0, str(ROOT))
    from security_wsp.prompts import render
    from security_wsp.models import Case  # noqa: F401  (schema check where available)

    request = render(case, condition, wording="reviewer")
    text = tokenizer.apply_chat_template(
        [dict(m) for m in request.messages], tokenize=False, add_generation_prompt=True) + PREFILL
    ids = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    tail = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    token = tail.split('"')[0].strip()
    if token.startswith("not_vulnerable"):
        return "not vulnerable"
    if token.startswith("vulnerable"):
        return "vulnerable"
    return "no verdict"


def make_app(load_fn=load):
    from fastapi import FastAPI, Header, HTTPException, Request

    app = FastAPI(title="Aawden demo")
    state: dict = {"loaded": None, "hits": {}}
    passcode = os.environ.get("AAWDEN_DEMO_PASSCODE", "")

    def check(pw: str | None):
        if not passcode:
            raise HTTPException(503, "endpoint has no passcode configured")
        if not pw or not hmac.compare_digest(pw.encode(), passcode.encode()):
            raise HTTPException(401, "bad passcode")

    @app.get("/health")
    def health():
        return {"ok": True, "model": MODEL_ID, "adapter": str(ADAPTER_DIR)}

    @app.post("/review")
    def review(body: ReviewIn, request: Request, x_passcode: str | None = Header(None)):
        check(x_passcode)
        ip = request.client.host if request.client else "?"
        now = time.time()
        window, limit = RATE_LIMIT
        hits = [t for t in state["hits"].get(ip, []) if now - t < window] + [now]
        state["hits"][ip] = hits
        if len(hits) > limit:
            raise HTTPException(429, "rate limit")
        if state["loaded"] is None:
            state["loaded"] = load_fn()
        model, tokenizer = state["loaded"]
        case = _load_case(body.case_id)
        if body.condition not in ("neutral", "correct_suggestion", "incorrect_suggestion"):
            raise HTTPException(422, "unknown condition")
        return {"case_id": body.case_id, "condition": body.condition,
                "verdict": _verdict(model, tokenizer, case, body.condition)}

    return app


def _load_case(case_id: str):
    import json
    from types import SimpleNamespace

    data = json.loads((ROOT / "datasets/plain-pooled.json").read_text())
    for c in data["cases"]:
        if c["case_id"] == case_id:
            return SimpleNamespace(**c)
    raise KeyError(case_id)


if asgi is not None:
    @asgi(name="aawden-demo", gpu="RTX4090", cpu=2, memory="16Gi", timeout=300,
          keep_warm_seconds=KEEP_WARM, authorized=False, image=image(),
          secrets=["AAWDEN_AYA_HF_TOKEN", "AAWDEN_DEMO_PASSCODE"],
          env={"HF_TOKEN": os.environ.get("AAWDEN_AYA_HF_TOKEN", "")},
          )
    def web_server():
        return make_app()
