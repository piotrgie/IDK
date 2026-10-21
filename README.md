# Workshop notebooks — *Chasing "I Don't Know" in Modern Language Models*

---

## Project scope (context for readers and for future generations of this material)

This workshop series is a self-paced, demo-driven introduction to how modern language models actually work under the hood, with a specific focus on **why they hallucinate and how the "I don't know" behavior is shaped (and often removed) by post-training**. It is aimed at scientifically-literate readers — working researchers, senior technical staff, graduate students, engineers new to LLMs — who are competent in Python but new to transformer internals, and who (correctly) mistrust confident LLM output but lack the vocabulary and diagnostic tools to interrogate it. The series builds up from token-by-token generation on a small model (Qwen2.5-1.5B and its instruction-tuned sibling, plus GPT-2 small for one demo), through a confident but math-free tour of the transformer architecture, through the theory of why generation models must hallucinate, through the semantic-attractor framing of model behavior, and lands — by Sessions 6–8 — in a scaled-down reproduction of original research on the abstention state: how "I don't know" carries real probability mass in pretrained models, how post-training either drains that mass or starves inflow to it, and how the standard spread-based hallucination detectors (semantic entropy, self-consistency) have a structural blind spot exactly where post-training bites hardest. The workshop's differentiator is that most transformer tutorials stop at "here is how attention works" and most hallucination tutorials start at "here are ten detection methods" — this series bridges the two, running end-to-end on a laptop, and ends with the reader having personally reproduced a research finding rather than merely read about it. The parent project targets researchers with a confirmatory study at scale; this workshop teaches the *method* at ~1/30th the scale so a curious reader can go on to read the paper with grounded intuition. All sessions are designed to run on a laptop (Sessions 0–5 CPU-fine); 
---

## What's in this directory

- `session_0_setup.py` — Environment setup and a 10-minute sanity check. Framing paragraph on what open-weight models let us measure vs. what commercial APIs (Claude, GPT, Gemini) don't.
- `session_1_watching.py` — Watching a model generate one token at a time. Builds `generate_with_topk_trace()`, the diagnostic instrument used across the whole series.
- `session_2_under_the_hood.py` — Confident tour of the transformer architecture: embeddings, attention, MLP, residual stream, KV cache, logit lens. No matrix math; every concept demonstrated with a small snippet.
- `session_3_reading_the_mind.py` — Confidence ≠ correctness, measured: a factual battery where right and wrong answers carry the same probability; temperature as a monotone reshaping (never changes the ranking); top-k/top-p as adaptive cropping; the teacher-forcing scorer built by hand (`sequence_logprob`) and used to measure knowledge as a log-odds gap; the Kadavath P(True) self-verification probe with its small-model calibration limits shown honestly.
- `session_4_why_hallucinate.py` — Intrinsic vs. extrinsic hallucination; the three inevitability arguments (computability, calibration/monofacts, grading incentives) with the arithmetic behind the last two; snowballing measured, plus the first causal intervention (prefilling one token into the assistant turn).
- `session_5_attractors.py` — The attractor gallery: random-number basin (sampled *and* read directly), iterated-paraphrase fixed point, GPT-2 repetition limit cycle with self-reinforcement, sycophancy under challenge, persona drift. Vocabulary: basin, width, depth, escape.
- `session_6_abstention.py` — The abstention experiment on the matched Qwen2.5-1.5B base/instruct pair: generation sweep (reachability), sequence-mass ledger with equivalence set and register control (mass), termination probe (persistence). One model in memory at a time.
- `workshop_utils.py` — Shared helpers, added to as sessions introduce them. Session 0 loads models; Session 1 adds `generate_with_topk_trace`; Session 3 adds `sequence_logprob`; Session 4 adds `sample_completions`; Session 6 adds `classify_response`.
- `requirements.txt` — Pinned versions. Reproducibility depends on these.
- `figures/` — Generated figures; each notebook creates its own on first run.
- `widgets/basin_landscape.html` — Self-contained interactive double-well landscape (no CDN, no backend) embedded in Session 5. Two mechanism sliders — mass drain vs. inflow gate — showing how the same extinct behavior can come from either cause.

## Format

Notebooks are stored as `.py` files in **jupytext percent format**:

- Cells are separated by `# %%` (code) and `# %% [markdown]` (markdown).
- This is git-friendly (real diffs on real lines) and lets you edit in any Python editor.
- Convert to `.ipynb` when you want a Jupyter notebook to run:

```bash
jupytext --to notebook session_0_setup.py
# produces session_0_setup.ipynb
```

Or open the `.py` directly in Jupyter Lab / VS Code — both understand jupytext natively when the `jupytext` package is installed. The `.py` file becomes the source of truth; regenerate the `.ipynb` any time.

## Install

Python 3.11 is required (some sessions use features that regress on 3.12 in the pinned `transformers` version).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional but recommended: a Hugging Face account and CLI login for cached model downloads.

```bash
huggingface-cli login
```

Qwen2.5 and GPT-2 are not gated and require no token. Llama models (used only in an optional Session 7 extension) are gated and require accepting a license on the Hugging Face model page first.

## Compute

- **Sessions 0–5:** CPU is sufficient. A GPU (any modern discrete GPU with ≥6 GB VRAM) speeds things up but is not required.
- **Sessions 6–8:** a GPU is recommended. Any of: RTX 3060 / 4060 / equivalent laptop GPU, Apple Silicon (MPS backend), or a free Colab T4. CPU still works but is slow.

Disk: ~5 GB for Qwen2.5-1.5B base + instruct + GPT-2 small + NLI model (Session 8 only). Models cache to `~/.cache/huggingface/` by default.

## Reproducibility

The workshop pins model repositories by **commit hash**, not just repo name. Repository contents change under stable names (tokenizer updates, config tweaks, safetensors reshards). Pinning by commit hash makes runs reproducible against a specific frozen state.

Pinned revisions are declared in `workshop_utils.py` under `PINNED_REVISIONS`.

Reproducibility caveats stated inline in Session 0:

- Cross-hardware: same seed on different GPU types (T4 vs. 3090 vs. Apple M-series) will not produce bit-identical logits. Conclusions are robust; last few decimal places aren't.
- CPU vs. GPU: not bit-identical either. Where a demo depends on exact numbers, the reference device is stated.
- Sampling (`do_sample=True`): reproducible only with all RNG state controlled. Helper code does this explicitly.

## Running the notebooks

```bash
# One-time: verify Session 0 sanity check
jupytext --to notebook session_0_setup.py
jupyter lab session_0_setup.ipynb   # or: jupyter notebook, or open in VS Code

# Sessions 1 and 2 depend on Session 0 having downloaded models successfully.
```

## Troubleshooting

Session 0 includes an inline troubleshooting flowchart. The four common failure modes:

1. **HF token missing** for gated repos → `huggingface-cli login`
2. **Insufficient disk** for model cache → clear `~/.cache/huggingface/` or symlink to a larger drive
3. **torch/CUDA version mismatch** → reinstall torch matching your CUDA version, per PyTorch's install matrix
4. **Tokenizer version mismatch** → `pip install -r requirements.txt` with the pinned versions in this directory

## Status and known gaps

Current state (2026-10): Sessions 0–6 are drafted as jupytext `.py` sources; Sessions 7–8 are not yet written. The core instruments (`sequence_logprob`, `sample_completions`, `classify_response`, the Session 6 probe functions) have been smoke-tested against the cached Qwen2.5-1.5B checkpoints; the notebooks themselves have not all been executed end-to-end.

Known gaps to close before publication:

- ~~**`PINNED_REVISIONS` are still `"main"` placeholders.**~~ **Resolved 2026-10-21:** all four repos are pinned to the commit hashes of the smoke-tested snapshots (verified to exist upstream). 
- **The `torch==2.4.1` pin predates Blackwell GPUs.** On an RTX 50-series card (sm_120), the pinned stack runs CPU-only; such readers need a `torch>=2.7` cu128 build, which breaks the exact-pin promise. Older GPUs and Colab T4 are unaffected. Needs a note in Session 0 or a dual pin.
- Session-level numbers quoted in the markdown (e.g. the ~10.6-nat Paris/Berlin gap in Session 3, the Faulkner/Beckett hook) were verified on Qwen2.5-1.5B-Instruct in float32 on CPU; exact decimals will drift across devices per the Session 0 reproducibility caveats.

## Relation to the parent project

- `../idk_workshops_plan.md` — the plan document these notebooks implement
- `../idk_paper_plan_draft.md` — the research paper at scale; this workshop is its pedagogical companion
- `../idk_attractor_research_summary.md` — the frozen results of the earlier study these notebooks scale down and reproduce
