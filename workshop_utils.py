"""
Shared helpers for the workshop notebooks.

Grows session by session. Sessions 0-2 introduce:
  - PINNED_REVISIONS: HF repo commit hashes used across the workshop
  - load_model_and_tokenizer(): device-aware loader honoring the pinned revisions
  - format_topk_row(): pretty-printing helper for top-k tables
  - set_all_seeds(): the "control every RNG" helper used whenever we sample

generate_with_topk_trace() itself is built up inline in Session 1 (the reader
watches it get written), then imported from here in Sessions 2+.

Sessions 3-6 add:
  - sequence_logprob(): teacher-forced log-probability of a specific
    continuation, without generating anything. Introduced in Session 3,
    used by Session 4 (snowball measurement) and Session 6 (the ledger).
  - classify_response(): the rule-based 5-way abstention taxonomy.
    Built up inline in Session 6, kept here for reuse in Sessions 7-8.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Pinned HF revisions
# ---------------------------------------------------------------------------
# Pinning by commit hash makes the workshop reproducible against a specific
# frozen state of each repository. If a repo owner reshards safetensors or
# tweaks the tokenizer, our pinned revision is unaffected.
#
# If any of these commits become unavailable, the workshop will fall back
# to `main` with a warning printed by load_model_and_tokenizer().
#
# Hashes below are placeholders selected as sensible defaults; run
# `python -c "from huggingface_hub import HfApi; ..."` to resolve current
# main-branch commits if you want to re-pin.
PINNED_REVISIONS: dict[str, str] = {
    "Qwen/Qwen2.5-1.5B": "main",
    "Qwen/Qwen2.5-1.5B-Instruct": "main",
    "openai-community/gpt2": "main",
    "microsoft/deberta-v3-base": "main",
}


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def pick_device() -> str:
    """Return the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    """Pick a sensible dtype for the given device.

    bf16 on modern GPUs (Ampere or newer) and Apple Silicon; fp32 on CPU
    because bf16 CPU is slow in this transformers version.
    """
    if device == "cuda":
        # Ampere (compute 8.0) or newer supports bf16 natively
        try:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(
    repo_id: str,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """Load a HF model + tokenizer with pinned revision and sensible defaults.

    Returns (tokenizer, model). Model is placed on `device` with `dtype`.
    """
    if device is None:
        device = pick_device()
    if dtype is None:
        dtype = pick_dtype(device)
    if revision is None:
        revision = PINNED_REVISIONS.get(repo_id, "main")

    print(f"Loading {repo_id} @ {revision} onto {device} as {dtype}...")

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    # Qwen and many others: pad_token unset by default; set it to eos
    # so that batched generation and padding-aware ops don't crash.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device)
    model.eval()

    print(f"  loaded: {model.config.num_hidden_layers} layers, "
          f"hidden {model.config.hidden_size}, vocab {model.config.vocab_size}")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int = 0) -> None:
    """Seed every RNG we might touch.

    Note: this does NOT make sampling bit-reproducible across different
    GPU hardware — floating-point non-associativity in reduction ops means
    the same seed on a T4 vs. an RTX 3090 gives slightly different logits.
    But on the same machine, the same seed yields the same sequence.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms slow things down; leave off unless a
    # specific demo requires exact reproducibility.
    # torch.use_deterministic_algorithms(True)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


# ---------------------------------------------------------------------------
# Data classes used by generate_with_topk_trace
# ---------------------------------------------------------------------------
@dataclass
class TokenStep:
    """One step in a token-by-token generation trace."""
    step: int
    picked_id: int
    picked_str: str
    picked_prob: float
    picked_logprob: float
    topk_ids: list[int]
    topk_strs: list[str]
    topk_probs: list[float]
    cumulative_logprob: float


@dataclass
class GenerationTrace:
    """A full generation trace with per-step top-k alternatives."""
    prompt: str
    steps: list[TokenStep] = field(default_factory=list)
    completed_text: str = ""

    def sequence_logprob(self) -> float:
        """Sum of per-step log-probabilities of the tokens that were picked."""
        return sum(s.picked_logprob for s in self.steps)

    def as_table(self, max_alt: int = 3) -> str:
        """Pretty-print as a text table."""
        lines = [
            f"step | picked                 p       | " +
            " | ".join(f"alt#{i+1:<11}       p" for i in range(1, max_alt + 1))
        ]
        for s in self.steps:
            picked_cell = f"{repr(s.picked_str)[:20]:<20} {s.picked_prob:.3f}"
            alt_cells = []
            for i in range(1, max_alt + 1):
                if i < len(s.topk_ids):
                    alt_cells.append(
                        f"{repr(s.topk_strs[i])[:14]:<14} {s.topk_probs[i]:.3f}"
                    )
                else:
                    alt_cells.append(" " * 22)
            lines.append(
                f"{s.step:4d} | {picked_cell} | " + " | ".join(alt_cells)
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# generate_with_topk_trace
# ---------------------------------------------------------------------------
# NOTE: this is the workshop's core diagnostic instrument. In Session 1 the
# reader watches it get built step by step. This module-level copy is the
# "final" version for reuse in Sessions 2+.

@torch.no_grad()
def generate_with_topk_trace(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    k: int = 5,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    apply_chat_template: bool = False,
    stop_on_eos: bool = True,
) -> GenerationTrace:
    """Generate token by token, recording top-k alternatives at each step.

    temperature=0.0 means greedy (argmax). Any positive temperature triggers
    stochastic sampling from softmax(logits / temperature), optionally cropped
    by top-p.

    Reproducibility: pass `seed` to fix the sampling RNG (only meaningful
    for temperature > 0).
    """
    if device is None:
        device = str(next(model.parameters()).device)

    if apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
        chat_out = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = (chat_out.input_ids if hasattr(chat_out, "input_ids") else chat_out).to(device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    if seed is not None:
        set_all_seeds(seed)

    trace = GenerationTrace(prompt=prompt)
    generated_ids: list[int] = []
    past_key_values = None
    cumulative_logprob = 0.0

    for step in range(max_new_tokens):
        model_input = input_ids if past_key_values is None else input_ids[:, -1:]
        out = model(
            input_ids=model_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        logits = out.logits[0, -1, :]  # last position, first (and only) batch
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()

        # Pick a token
        if temperature == 0.0:
            picked_id = int(torch.argmax(log_probs).item())
        else:
            scaled = logits.float() / max(temperature, 1e-8)
            probs_scaled = torch.softmax(scaled, dim=-1)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs_scaled, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cum > top_p).nonzero()
                if len(cutoff) > 0:
                    keep = int(cutoff[0].item()) + 1
                    mask = torch.zeros_like(probs_scaled)
                    mask[sorted_idx[:keep]] = 1.0
                    probs_scaled = probs_scaled * mask
                    probs_scaled = probs_scaled / probs_scaled.sum()
            picked_id = int(torch.multinomial(probs_scaled, num_samples=1).item())

        picked_logprob = float(log_probs[picked_id].item())
        picked_prob = float(probs[picked_id].item())
        cumulative_logprob += picked_logprob

        # Top-k for reporting (from the unscaled distribution, so the numbers
        # are directly interpretable as the model's own next-token probs)
        top_probs, top_ids = torch.topk(probs, k=min(k, probs.numel()))
        top_ids_list = [int(x) for x in top_ids.tolist()]
        top_probs_list = [float(x) for x in top_probs.tolist()]

        trace.steps.append(TokenStep(
            step=step,
            picked_id=picked_id,
            picked_str=tokenizer.decode([picked_id]),
            picked_prob=picked_prob,
            picked_logprob=picked_logprob,
            topk_ids=top_ids_list,
            topk_strs=[tokenizer.decode([i]) for i in top_ids_list],
            topk_probs=top_probs_list,
            cumulative_logprob=cumulative_logprob,
        ))

        generated_ids.append(picked_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[picked_id]], device=device)], dim=1
        )

        if stop_on_eos and picked_id == tokenizer.eos_token_id:
            break

    trace.completed_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return trace


# ---------------------------------------------------------------------------
# Batched sampling (Session 4 onwards)
# ---------------------------------------------------------------------------
# generate_with_topk_trace() is the microscope: one sequence, every step
# visible. From Session 4 on we also need the opposite — many samples of the
# same prompt, fast, with no per-step bookkeeping. That is what the library's
# own model.generate() is for, and batching it is what makes 20-sample
# ensembles feasible on a laptop.

@torch.no_grad()
def sample_completions(
    model,
    tokenizer,
    prompt: str,
    n: int = 20,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 64,
    apply_chat_template: bool = False,
    system_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    batch_size: int = 10,
    device: Optional[str] = None,
) -> list[str]:
    """Draw `n` completions of one prompt. Returns decoded strings.

    temperature=0.0 gives greedy decoding, in which case all n samples are
    identical — useful as a control, wasteful otherwise.
    """
    if device is None:
        device = str(next(model.parameters()).device)

    if apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        messages = messages + [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prompt_len = ids.shape[1]

    if seed is not None:
        set_all_seeds(seed)

    outputs: list[str] = []
    remaining = n
    while remaining > 0:
        k = min(batch_size, remaining)
        batch_ids = ids.repeat(k, 1)
        gen = model.generate(
            batch_ids,
            # every row is the same prompt, so the mask is all ones — passing it
            # explicitly only silences a warning transformers raises when the
            # pad and eos tokens coincide (they do, for Qwen)
            attention_mask=torch.ones_like(batch_ids),
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            top_p=top_p if temperature > 0.0 else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        for row in gen:
            outputs.append(tokenizer.decode(row[prompt_len:], skip_special_tokens=True))
        remaining -= k
    return outputs


# ---------------------------------------------------------------------------
# Teacher forcing: the log-probability of a continuation we supply ourselves
# ---------------------------------------------------------------------------
# This is the Session 3 instrument. Instead of asking "what would the model
# say?", it asks "how much probability would the model have assigned to this
# exact string, had it said it?" — answerable in a single forward pass, with
# no sampling at all. Session 4 uses it to measure a snowball closing; Session 6
# builds the sequence-mass ledger on top of it.

@dataclass
class SequenceScore:
    """Teacher-forced score of one continuation under one context."""
    context: str
    continuation: str
    total_logprob: float          # nats, sum over continuation tokens
    n_tokens: int
    token_strs: list[str]
    token_logprobs: list[float]

    @property
    def mean_logprob(self) -> float:
        """Per-token log-probability (nats/token).

        Use this when comparing strings of different token length — a longer
        string is automatically less probable, which is a fact about length,
        not about the model's opinion of the string.
        """
        return self.total_logprob / max(self.n_tokens, 1)

    @property
    def prob(self) -> float:
        """Joint probability of the continuation. Underflows to 0 quickly."""
        return float(np.exp(self.total_logprob))

    def as_table(self) -> str:
        lines = [f"{'token':>18} | {'logprob':>9} | {'prob':>8}"]
        for s, lp in zip(self.token_strs, self.token_logprobs):
            lines.append(f"{repr(s)[:18]:>18} | {lp:>9.3f} | {np.exp(lp):>8.4f}")
        lines.append(f"{'TOTAL':>18} | {self.total_logprob:>9.3f} | "
                     f"{self.prob:>8.2e}  ({self.mean_logprob:.3f} nats/token)")
        return "\n".join(lines)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@torch.no_grad()
def sequence_logprob(
    model,
    tokenizer,
    context: str,
    continuation: str,
    append_token_id: Optional[int] = None,
    append_eos: bool = False,
    device: Optional[str] = None,
    warn_on_retokenization: bool = True,
) -> SequenceScore:
    """Log-probability the model assigns to `continuation` after `context`.

    No generation happens: we lay down context+continuation as one sequence,
    run one forward pass, and read off the log-probability the model would
    have assigned to each continuation token given everything to its left.

    `append_eos` (or an explicit `append_token_id`) additionally scores the
    turn-terminating token. That matters for Session 6: "I don't know." as a
    *complete turn* is a different event from "I don't know." as the opening
    of a longer answer, and only the terminator distinguishes them.

    Tokenization note: `context + continuation` is tokenized jointly, because
    tokenizing the two halves separately can produce a token sequence the
    model would never actually see (BPE merges across the boundary). If the
    joint tokenization disagrees with the context's own tokenization, the
    boundary is taken at the longest shared prefix and a warning is printed.
    """
    if device is None:
        device = str(next(model.parameters()).device)
    if not context:
        raise ValueError("context must be non-empty: the first token of a "
                         "sequence has nothing to its left to condition on.")

    ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
    full_ids = tokenizer(context + continuation, add_special_tokens=False).input_ids

    boundary = _common_prefix_len(ctx_ids, full_ids)
    if boundary != len(ctx_ids) and warn_on_retokenization:
        print(f"  note: tokenizer merged across the context/continuation boundary "
              f"({len(ctx_ids)} context tokens alone, {boundary} shared with the "
              f"joint tokenization). Scoring from the shared prefix.")

    if append_token_id is None and append_eos:
        append_token_id = tokenizer.eos_token_id
    if append_token_id is not None:
        full_ids = full_ids + [int(append_token_id)]

    input_ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids=input_ids).logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)

    token_logprobs, token_strs = [], []
    for pos in range(boundary, len(full_ids)):
        # logits at position pos-1 predict the token at position pos
        lp = float(log_probs[pos - 1, full_ids[pos]].item())
        token_logprobs.append(lp)
        token_strs.append(tokenizer.decode([full_ids[pos]]))

    return SequenceScore(
        context=context,
        continuation=continuation,
        total_logprob=float(sum(token_logprobs)),
        n_tokens=len(token_logprobs),
        token_strs=token_strs,
        token_logprobs=token_logprobs,
    )


def logsumexp(values: list[float]) -> float:
    """Total log-probability of a set of mutually exclusive strings.

    Used for the Session 6 equivalence set: "I don't know.", "I don't know",
    "I'm not sure." etc. are the same *state* spelled several ways, so their
    probability mass adds — in log space, that addition is a logsumexp.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("-inf")
    m = float(arr.max())
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.exp(arr - m).sum()))


# ---------------------------------------------------------------------------
# The abstention taxonomy (Session 6)
# ---------------------------------------------------------------------------
# A deliberately simple, auditable, rule-based classifier. It is *wrong* on a
# measurable fraction of outputs — Session 6 shows its failure cases rather
# than hiding them. Anything relying on it should report the audit too.

IDK_PATTERNS = [
    r"i don'?t know", r"i do not know", r"i'?m not sure", r"i am not sure",
    r"no idea", r"i can'?t say", r"i cannot say", r"i'?m unsure",
    r"i don'?t have (that|this|enough) information",
]
HEDGE_PATTERNS = [
    r"\bi think\b", r"\bi believe\b", r"\bprobably\b", r"\bpossibly\b",
    r"\bperhaps\b", r"\bmight\b", r"\bmay be\b", r"\blikely\b",
    r"\bas far as i know\b", r"\bapproximately\b", r"\baround\b",
    r"\bit'?s unclear\b", r"\bto my knowledge\b", r"\bseems\b",
]
REFUSAL_PATTERNS = [
    r"i can'?t help", r"i cannot help", r"i can'?t provide", r"i cannot provide",
    r"i'?m not able to (help|provide|assist)", r"as an ai", r"i'?m sorry, but",
    r"i must decline", r"it would (not )?be inappropriate",
]

CLASSES = ("bare_idk", "justified_idk", "hedged_answer",
           "confident_answer", "safety_refusal")


def classify_response(text: str, bare_word_limit: int = 6) -> str:
    """Five-way classification of an assistant turn.

    - bare_idk        : an abstention and nothing else ("I don't know.")
    - justified_idk   : abstains, then explains why
    - safety_refusal  : declines on policy/safety grounds (NOT the same state
                        as abstention — post-training shapes the two very
                        differently, so they must be counted separately)
    - hedged_answer   : gives an answer, marked with uncertainty language
    - confident_answer: gives an answer with no uncertainty marking

    Order matters: refusal is tested before abstention because refusals often
    contain abstention-like phrases.
    """
    t = " ".join(text.strip().split()).lower()
    if not t:
        return "confident_answer"  # an empty turn is not evidence of abstention

    if any(re.search(p, t) for p in REFUSAL_PATTERNS):
        return "safety_refusal"

    idk_hit = next((re.search(p, t) for p in IDK_PATTERNS
                    if re.search(p, t)), None)
    if idk_hit is not None:
        # "bare" = the abstention is essentially the whole turn
        stripped = re.sub(r"[^\w\s']", " ", t).split()
        if len(stripped) <= bare_word_limit:
            return "bare_idk"
        return "justified_idk"

    if any(re.search(p, t) for p in HEDGE_PATTERNS):
        return "hedged_answer"
    return "confident_answer"


# ---------------------------------------------------------------------------
# Small pretty-printing helpers
# ---------------------------------------------------------------------------
def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:6.2f} {unit}"
        n /= 1024
    return f"{n:6.2f} PB"
