"""Beam endpoint for the Aawden demo: Aya Expanse 8B with the two-stage adapter, live.

Every verdict produced by this endpoint is the same code path as the measurement
(`scripts/wsp_cross_model.render_for` + the `{"verdict":"` prefill), so what a visitor sees is the
model the table describes, not a re-implementation.

Security posture:
  * The endpoint is public (`authorized=False`) so the organiser can call it without a Beam account.
  * Every request must carry `X-Passcode`, compared against AAWDEN_DEMO_PASSCODE, which is set as a
    Beam runtime secret at deploy time. The Beam workspace token is never placed in this file, in an
    image, or in a request; deployment reads it from the repo `.env` exactly as `beam_launch.py` does.
  * Rate limit: a fixed window per caller, plus one per container. Beam's proxy terminates the
    connection, so `request.client.host` is the proxy, not the visitor: the caller is read from the
    forwarded headers instead. Those are caller-controlled, so the rate limit is a speed bump and
    not a boundary.
  * Spend ceiling: every served request charges the GPU seconds it plausibly billed to a counter
    kept per UTC day on the `aawden-v3` volume, and requests are refused once the day's budget is
    gone. This, not the rate limit, is what bounds the bill when someone loops the endpoint.
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
import json
import os
import threading
import time
from pathlib import Path

try:
    from beam import Image, Volume, asgi
except ImportError:  # Imported locally for tests without the SDK.
    Image = Volume = asgi = None

try:  # Model at module level: FastAPI cannot bind a body model defined inside a factory.
    from typing import Optional

    from pydantic import BaseModel

    class ReviewIn(BaseModel):
        case_id: str
        condition: str = "incorrect_suggestion"  # neutral | correct_suggestion | incorrect_suggestion

    class CodeIn(BaseModel):
        """A free-form file. The reviewer's claim is stated outright: there is no oracle to flip.

        `case_id` requests replay a saved fixture whose label is executable. A submitted file has no
        label, so `reviewer_verdict` is what the reviewer says, not what the code does — the endpoint
        compares the model's two answers and makes no claim about which one is right.

        `Optional[...]` rather than `str | None`: this class body is evaluated at import, and the
        deploy CLI evaluates the module with its own interpreter (3.9), where that syntax raises.
        """
        code: str
        reviewer_verdict: Optional[str] = None  # vulnerable | not_vulnerable_for_target, or absent
        target_category: str = "sql_injection"
        language: str = "python"
        threat_model: Optional[str] = None
except ImportError:
    ReviewIn = None
    CodeIn = None

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "CohereLabs/aya-expanse-8b"
MODEL_REVISION = "5062468bf9bc0c6035fd64e06274333ec127d980"
ADAPTER_DIR = Path("demo/adapter-aya8b-curr-s17")  # relative to the synced app directory
PREFILL = '{"verdict":"'
MAX_NEW_TOKENS = 6
KEEP_WARM = 300
# RTX4090 has 22 GiB usable and an 8B model in bf16 is about 16 GiB of weights before activations, which
# OOMs on the first request. RTX5090 is a serverless option with 32 GiB, so it holds the weights with
# room to run. deploy_demo checks that this matches its --gpu so the two cannot drift.
GPU = os.environ.get("AAWDEN_DEMO_GPU", "RTX5090")
_CASE_COUNT = len(json.loads((ROOT / "datasets/plain-pooled.json").read_text())["cases"])
RATE_LIMIT = (30, 60.0)  # requests per caller per window (seconds)
GLOBAL_RATE_LIMIT = (60, 60.0)  # requests per container per window, whatever the caller claims to be
# The spend ceiling. A request charges the GPU seconds it plausibly billed: the wall clock since the
# previous request, capped at the keep-warm window (the container stayed up that long because of it),
# plus a full cold-start load when the model is not in memory yet. Requests are refused once the
# UTC day's total is spent. Under-counted by at most one request, because the last request's own
# handling time is only billed by the next request's gap.
SPEND_HOURLY_USD = 1.09  # configs/beam.json hourly_usd.RTX5090
SPEND_DAILY_USD = 2.00  # about 1.8 GPU-hours of warm container per UTC day
COLD_START_SECONDS = 180  # observed ~150 s; round up so the ceiling errs towards refusing early
VOLUME_NAME = "aawden-v3"
VOLUME_MOUNT = "/volumes/aawden-v3"
SPEND_PATH = Path(VOLUME_MOUNT) / "demo-spend.json"
MAX_CODE_CHARS = 20_000  # about 5k tokens: one screen of code plus room for the answer
# The two categories the measured set covers. A third category would need a threat model the model
# was never asked about, so it is refused rather than guessed at.
CATEGORIES = ("sql_injection", "access_control")
# Used only when a submitted file arrives without a threat model. These are the most common texts
# in the measured plain set for each category, so a free-form prompt is shaped like a measured one.
DEFAULT_THREAT_MODEL = {
    "sql_injection": ("Python sqlite3; username is untrusted. A lookup must treat the supplied name as "
                      "a literal value; it must not return other users. The database schema and "
                      "connection are trusted."),
    "access_control": ("records and the authenticated actor session are trusted. record_id is "
                       "attacker-controlled. Only the owner may read a record. Login and tenant "
                       "membership alone do not permit reading another owner's record. Missing "
                       "records return None."),
}


def image():
    """The runtime image.

    torch must come from the cu128 index, not PyPI: the RTX5090 is Blackwell (sm_120), and wheels
    published on PyPI before 2.7 ship no sm_120 kernels. With torch 2.5.1 the model loads its
    weights and then fails the first CUDA call with "no kernel image is available for execution on
    the device". torch 2.7.1+cu128 is the first build that covers this card.

    peft must match the release that wrote adapter_config.json (peft_version 0.21.2), whose config
    carries fields the 0.13 line does not parse.
    """
    if Image is None:
        return None
    return (
        Image(python_version="python3.11")
        .add_commands(["pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128"])
        .add_python_packages(["transformers==4.46.3", "peft==0.21.2", "accelerate==1.1.1",
                              "fastapi==0.115.5", "pydantic==2.10.3", "sentencepiece==0.2.0",
                              "uvicorn==0.32.1"])
    )


def load():
    """Load the pinned base model and merge the adapter. Runs once per container."""
    import torch

    # peft 0.21 reads torch.distributed.tensor.DTensor, but on torch 2.5 the submodule is only an
    # attribute after an explicit import; without this it raises AttributeError during inject_adapter.
    try:
        import torch.distributed.tensor  # noqa: F401
    except Exception:  # pragma: no cover - older torch without DTensor is fine
        pass
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("AAWDEN_AYA_HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("no Hub token: set the AAWDEN_AYA_HF_TOKEN secret on this endpoint")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, token=token, torch_dtype=torch.bfloat16, device_map="cuda")
    adapter_path = ROOT / ADAPTER_DIR
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter not shipped with the app: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def _verdict(model, tokenizer, case, condition: str, wording: str = "reviewer"):
    import sys

    import torch

    # The package lives under src/, not the repo root. Locally an editable-install .pth hides that;
    # inside the container nothing does, so both roots go on the path.
    for root in (str(ROOT / "src"), str(ROOT)):
        if root not in sys.path:
            sys.path.insert(0, root)
    from security_wsp.prompts import render

    request = render(case, condition, wording=wording)
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


def _client_ip(request) -> str:
    """The caller's address, as far as it can honestly be told behind Beam's proxy.

    `request.client.host` is the proxy's peer address, which is the same string for every visitor,
    so keying the rate limit on it puts the whole world in one bucket. The forwarded headers are the
    only better signal available here, and they are caller-controlled: the leftmost X-Forwarded-For
    entry is whatever the caller typed. So this returns a usable bucket, not an identity -- the
    passcode is the boundary, and the daily spend ceiling is what actually bounds the bill.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    real = request.headers.get("x-real-ip", "").strip()
    if real:
        return real
    return request.client.host if request.client else "?"


def _spend_read() -> dict:
    """The counter left by an earlier container. {} when the volume is absent or unwritable."""
    try:
        record = json.loads(SPEND_PATH.read_text())
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _spend_write(utc_day: str, seconds: float) -> None:
    """Persist the day's total. Write-then-rename so a killed container cannot leave half a file."""
    try:
        SPEND_PATH.parent.mkdir(parents=True, exist_ok=True)
        scratch = SPEND_PATH.with_name(SPEND_PATH.name + ".tmp")
        scratch.write_text(json.dumps({"utc_day": utc_day, "seconds": round(seconds, 1)}))
        scratch.replace(SPEND_PATH)
    except OSError:
        pass  # read-only volume: the in-container counter still holds for this container's lifetime


def make_app(load_fn=load):
    from fastapi import FastAPI, Header, HTTPException, Request

    app = FastAPI(title="Aawden demo")
    today = time.strftime("%Y-%m-%d", time.gmtime())
    saved = _spend_read()
    # Seed once per container: a cold start must not hand out a fresh budget, or the ceiling is only
    # as strong as the scale-to-zero timer.
    state: dict = {
        "loaded": None,
        "hits": {},
        "last_at": None,
        "spend_day": today,
        "spend_seconds": float(saved.get("seconds", 0.0)) if saved.get("utc_day") == today else 0.0,
    }
    spend_lock = threading.Lock()
    passcode = os.environ.get("AAWDEN_DEMO_PASSCODE", "")

    def check(pw: str | None):
        if not passcode:
            raise HTTPException(503, "endpoint has no passcode configured")
        if not pw or not hmac.compare_digest(pw.encode(), passcode.encode()):
            raise HTTPException(401, "bad passcode")

    def charge(seconds: float) -> float:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        with spend_lock:
            if today != state["spend_day"]:
                state["spend_day"], state["spend_seconds"] = today, 0.0
            state["spend_seconds"] += max(0.0, seconds)
            total = state["spend_seconds"]
            _spend_write(today, total)
        return total

    def gate(request, pw: str | None) -> None:
        """Passcode, then the rate limits, then the day's spend ceiling.

        Runs before the model is touched, so a refused request costs no GPU. The charge covers the
        time this container has already been billing since the previous request, which is why it is
        taken here rather than after the work.
        """
        check(pw)
        ip = _client_ip(request)
        now = time.time()
        window, limit = RATE_LIMIT
        hits = [t for t in state["hits"].get(ip, []) if now - t < window] + [now]
        state["hits"][ip] = hits
        gwindow, glimit = GLOBAL_RATE_LIMIT
        global_hits = [t for t in state["hits"].get("\0all", []) if now - t < gwindow] + [now]
        state["hits"]["\0all"] = global_hits
        if len(hits) > limit or len(global_hits) > glimit:
            raise HTTPException(429, "rate limit")
        # A caller that rewrites the forwarded header gets a fresh per-IP bucket every time, so the
        # table would otherwise grow without bound. Drop expired keys instead of leaking them.
        if len(state["hits"]) > 5000:
            for key in [k for k, v in state["hits"].items() if not v or now - v[-1] >= window]:
                del state["hits"][key]
        previous = state["last_at"]
        state["last_at"] = now
        billed = min(max(0.0, now - previous), float(KEEP_WARM)) if previous else 0.0
        if state["loaded"] is None:
            billed += COLD_START_SECONDS
        spent = charge(billed)
        if spent * SPEND_HOURLY_USD / 3600.0 >= SPEND_DAILY_USD:
            raise HTTPException(
                503, f"this demo has used its {SPEND_DAILY_USD:.2f} USD GPU budget for the UTC day")

    @app.get("/health")
    def health(request: Request):
        """Ungated on purpose: it is what tells the organiser the container is awake.

        It also reports what the caller looks like from in here and what the day has cost, because
        those two are the only way to tell from outside whether the forwarded headers carry anything
        real and whether the ceiling is holding.
        """
        today = time.strftime("%Y-%m-%d", time.gmtime())
        spent = state["spend_seconds"] if state["spend_day"] == today else 0.0
        return {
            "ok": True,
            "model": MODEL_ID,
            "adapter": str(ADAPTER_DIR),
            "client_ip": _client_ip(request),
            "peer": request.client.host if request.client else None,
            "forwarded_for": request.headers.get("x-forwarded-for"),
            "real_ip": request.headers.get("x-real-ip"),
            "spend": {
                "utc_day": today,
                "gpu_seconds": round(spent, 1),
                "usd": round(spent * SPEND_HOURLY_USD / 3600.0, 4),
                "daily_usd": SPEND_DAILY_USD,
                "persisted": bool(_spend_read()),
            },
        }

    @app.get("/diagnose")
    def diagnose(request: Request, x_passcode: str | None = Header(None)):
        """Walk load() and one verdict, returning the traceback of the first failure.

        Gated by the same passcode as /review. It exists because the container is not reachable by
        the log-streaming CLI from this machine, so a failure there has to report itself.
        """
        import traceback

        # Same gate as the routes: this one loads the model and runs a verdict, so it is the most
        # expensive request on the endpoint.
        gate(request, x_passcode)
        steps: list[dict] = []

        def step(name, fn):
            try:
                value = fn()
            except Exception as exc:  # noqa: BLE001 - surfacing any failure is the point
                steps.append({"step": name, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                              "traceback": traceback.format_exc()[-4000:]})
                return None
            steps.append({"step": name, "ok": True})
            return value

        def need(path, label):
            if not path.exists():
                raise FileNotFoundError(f"{label} missing: {path}")
            return True

        step("adapter_present", lambda: need(ROOT / ADAPTER_DIR / "adapter_config.json", "adapter config"))
        step("adapter_weights", lambda: need(ROOT / ADAPTER_DIR / "adapter_model.safetensors", "adapter weights"))
        step("dataset_present", lambda: need(ROOT / "datasets/plain-pooled.json", "dataset"))
        step("token_present", lambda: (os.environ.get("AAWDEN_AYA_HF_TOKEN") or os.environ.get("HF_TOKEN"))
             or (_ for _ in ()).throw(RuntimeError("no Hub token in the environment")))

        def versions():
            import torch
            from peft import __version__ as peft_version
            from transformers import __version__ as transformers_version

            return {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "capability": ".".join(str(x) for x in torch.cuda.get_device_capability()),
                    "device": torch.cuda.get_device_name(0),
                    "transformers": transformers_version, "peft": peft_version}

        info = step("versions", versions)
        if isinstance(info, dict):
            steps[-1]["value"] = info
        def import_package():
            import sys as _sys

            for root in (str(ROOT / "src"), str(ROOT)):
                if root not in _sys.path:
                    _sys.path.insert(0, root)
            from security_wsp.prompts import render

            return render

        step("import_package", import_package)
        loaded = step("load", load_fn)
        if loaded:
            model, tokenizer = loaded
            step("chat_template", lambda: tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True))
            case = step("load_case", lambda: _load_case("c_cab5d4f0e9d4"))
            if case is not None:
                step("verdict", lambda: _verdict(model, tokenizer, case, "incorrect_suggestion"))
        return {"steps": steps}

    @app.post("/review")
    def review(body: ReviewIn, request: Request, x_passcode: str | None = Header(None)):
        gate(request, x_passcode)
        if state["loaded"] is None:
            state["loaded"] = load_fn()
        model, tokenizer = state["loaded"]
        if body.condition not in ("neutral", "correct_suggestion", "incorrect_suggestion"):
            raise HTTPException(422, "unknown condition")
        try:
            case = _load_case(body.case_id)
        except KeyError:
            # An unknown case is the caller's mistake, not a server fault: answer 404 with the count
            # rather than letting KeyError become a 500.
            raise HTTPException(404, f"unknown case_id; this endpoint serves {_CASE_COUNT} fixed cases")
        return {"case_id": body.case_id, "condition": body.condition,
                "verdict": _verdict(model, tokenizer, case, body.condition)}

    @app.post("/review-code")
    def review_code(body: CodeIn, request: Request, x_passcode: str | None = Header(None)):
        """Review a file the caller supplies. Two answers, so the deference question is visible.

        There is no oracle for submitted code, so the endpoint does not score correctness. It returns
        the model's own verdict with no reviewer, then its verdict with the caller's claimed verdict
        attached — the same prompt shape as the measured runs. Whether the claim is right is the
        caller's to know; what the endpoint shows is whether the model moved.
        """
        gate(request, x_passcode)
        if len(body.code) > MAX_CODE_CHARS:
            raise HTTPException(413, f"code is {len(body.code)} characters; the limit is {MAX_CODE_CHARS}")
        if not body.code.strip():
            raise HTTPException(422, "code is empty")
        if body.target_category not in CATEGORIES:
            raise HTTPException(422, f"target_category must be one of {list(CATEGORIES)}")
        claim = body.reviewer_verdict
        if claim is not None and claim not in ("vulnerable", "not_vulnerable_for_target"):
            raise HTTPException(422, "reviewer_verdict must be vulnerable, not_vulnerable_for_target, or omitted")
        if state["loaded"] is None:
            state["loaded"] = load_fn()
        model, tokenizer = state["loaded"]
        case = _case_from_code(body)
        # Generate would fail on a prompt longer than the model's window; refuse it here with a
        # number the caller can act on, rather than a 500 after a 30-second load.
        _check_context(tokenizer, model, case)
        alone = _verdict(model, tokenizer, case, "neutral")
        if claim is None:
            return {"case_id": case.case_id, "language": case.language,
                    "target_category": case.target_category, "reviewer_verdict": None,
                    "verdict_alone": alone, "verdict_with_reviewer": None, "moved": None}
        # `correct_suggestion` states the oracle verdict, which here is the caller's claim, so the
        # reviewer line reads exactly as it does in a measured run.
        with_reviewer = _verdict(model, tokenizer, case, "correct_suggestion")
        return {"case_id": case.case_id, "language": case.language,
                "target_category": case.target_category, "reviewer_verdict": claim,
                "verdict_alone": alone, "verdict_with_reviewer": with_reviewer,
                "moved": None if "no verdict" in (alone, with_reviewer) else alone != with_reviewer}

    return app


def _case_from_code(body) -> "Case":
    """Wrap submitted source as a `Case` so `render` builds the identical prompt.

    Only `code`, `language`, `target_category` and `threat_model` reach the model (`Case.visible`),
    and `oracle_verdict` is read only to word the reviewer's opinion. The remaining fields are
    required by the dataclass and carry no meaning here: there is no family, split or executable
    label for a file that arrives over HTTP, and the case_id is a hash of the source so an identical
    submission answers identically.
    """
    import hashlib
    import sys

    for root in (str(ROOT / "src"), str(ROOT)):
        if root not in sys.path:
            sys.path.insert(0, root)
    from security_wsp.data import Case

    code = body.code.replace("\r\n", "\n")
    if not code.endswith("\n"):
        code += "\n"
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    return Case(
        case_id="code_" + digest[:12], family_id="submitted", source_id="submitted", split="submitted",
        language=body.language, target_category=body.target_category, code=code,
        threat_model=body.threat_model or DEFAULT_THREAT_MODEL[body.target_category],
        oracle_verdict=body.reviewer_verdict or "vulnerable",
        source_sha256=digest, oracle_test_ids=(), label_status="caller_submitted_no_oracle",
        license="caller-submitted")


def _check_context(tokenizer, model, case):
    """Refuse a prompt that will not fit the model's window, before paying for a generation."""
    import sys

    from fastapi import HTTPException

    for root in (str(ROOT / "src"), str(ROOT)):
        if root not in sys.path:
            sys.path.insert(0, root)
    from security_wsp.prompts import render

    window = getattr(getattr(model, "config", None), "max_position_embeddings", None) or 8192
    # The reviewer line is the longer of the two prompts, so checking it bounds both.
    request = render(case, "correct_suggestion", wording="reviewer")
    text = tokenizer.apply_chat_template([dict(m) for m in request.messages], tokenize=False,
                                         add_generation_prompt=True) + PREFILL
    length = len(tokenizer(text)["input_ids"])
    if length + MAX_NEW_TOKENS > window:
        raise HTTPException(
            413, f"prompt is {length} tokens and the model window is {window}; submit a shorter file")


def _load_case(case_id: str):
    """Load one fixture as a real `Case`, because `render` calls `case.visible()` on it.

    A SimpleNamespace is not a substitute: `render` goes through `Case.visible`, so anything that is
    not a `Case` raises AttributeError on the first request. Reading the dataset with the project's
    own loader also means the endpoint renders byte-identical prompts to the measured runs.
    """
    import json
    import sys

    for root in (str(ROOT / "src"), str(ROOT)):
        if root not in sys.path:
            sys.path.insert(0, root)
    from security_wsp.data import Case

    data = json.loads((ROOT / "datasets/plain-pooled.json").read_text())
    for item in data["cases"]:
        if item["case_id"] == case_id:
            return Case.from_dict(item)
    raise KeyError(case_id)


if asgi is not None:
    # The volume is what makes the spend ceiling survive scale-to-zero. concurrent_requests stays at
    # its default of 1, so the counter has a single writer.
    @asgi(name="aawden-demo", gpu=GPU, cpu=2, memory="32Gi", timeout=300,
          keep_warm_seconds=KEEP_WARM, authorized=False, image=image(),
          secrets=["AAWDEN_AYA_HF_TOKEN", "AAWDEN_DEMO_PASSCODE"],
          volumes=[Volume(name=VOLUME_NAME, mount_path=VOLUME_MOUNT)] if Volume is not None else [],
          )
    def web_server():
        return make_app()
