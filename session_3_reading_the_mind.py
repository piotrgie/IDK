# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Session 3 — Reading the Model's Mind: Logits, Probabilities, and What the Model *Almost* Said
#
# **Series:** *Chasing "I Don't Know" in Modern Language Models*
#
# **Session length:** ~60 min.
# **Compute:** CPU sufficient; GPU optional (the candidate-ranking demo is ~100 forward passes and is noticeably faster on GPU).
# **Prerequisite:** Sessions 0–2. In particular `generate_with_topk_trace` from Session 1.
#
# ## Framing
#
# Session 1 showed you the model's opinions, step by step. Session 2 showed you the machinery that produces them. This session is about what those opinions are *worth* — and it is built around the single most important fact about them:
#
# > **The model's confidence and the model's correctness are largely unrelated.**
#
# You will see the model assign essentially the same high probability to a right answer and a wrong one, with nothing in the numbers to tell you which is which. Then you will build the instrument that makes the rest of this workshop possible: **teacher forcing** — computing the probability the model assigns to any text *you* choose, without generating anything. That one function turns the model from a text producer into a measurable object, and it is the tool Sessions 4 and 6 are built on.
#
# ---

# %%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from workshop_utils import (
    load_model_and_tokenizer,
    pick_device,
    pick_dtype,
    set_all_seeds,
    sample_completions,
)

os.makedirs("figures", exist_ok=True)

device = pick_device()
dtype = pick_dtype(device)

set_all_seeds(0)
tokenizer, model = load_model_and_tokenizer(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device=device,
    dtype=dtype,
)


# %% [markdown]
# ## Part 1 — The hook: same confidence, opposite truth
#
# A battery of short factual questions with unique, checkable answers. For each one we will record three things: the model's own answer (greedy), the probability it assigned to that answer, and the probability it assigned to the *correct* answer in the same context. The last two need a tool we haven't built yet, so for the moment, borrow it — Part 4 opens it up and has you write it yourself.

# %%
from workshop_utils import sequence_logprob   # built by hand in Part 4

BATTERY = [
    # (question, correct answer)
    ("What is the capital of Australia? Answer with the city name only.", "Canberra"),
    ("What is the capital of Kiribati? Answer with the city name only.", "Tarawa"),
    ("In what year was the Norwegian painter Johan Christian Dahl born? "
     "Answer with the year only.", "1788"),
    ("Who won the 1969 Nobel Prize in Literature? Answer with the name only.", "Samuel Beckett"),
    ("What is the atomic number of tungsten? Answer with the number only.", "74"),
    ("In what year was the University of Coimbra founded? Answer with the year only.", "1290"),
    ("What is the capital of the Canadian province of Manitoba? "
     "Answer with the city name only.", "Winnipeg"),
    ("Who composed the opera 'Rusalka'? Answer with the composer's name only.", "Dvorak"),
    ("In what year did the Chernobyl disaster occur? Answer with the year only.", "1986"),
    ("What is the highest mountain in Poland? Answer with the name only.", "Rysy"),
]


def chat_context(question):
    """The exact string the model sees before its own turn begins."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )


battery_rows = []
for question, correct in BATTERY:
    ctx = chat_context(question)
    own = sample_completions(model, tokenizer, ctx, n=1, temperature=0.0,
                             max_new_tokens=12, apply_chat_template=False)[0].strip()
    # probability of each answer as a COMPLETE turn (answer + end-of-turn):
    lp_own = sequence_logprob(model, tokenizer, ctx, own,
                              append_eos=True, warn_on_retokenization=False).total_logprob
    lp_correct = sequence_logprob(model, tokenizer, ctx, correct,
                                  append_eos=True, warn_on_retokenization=False).total_logprob
    is_right = correct.lower().rstrip(".") in own.lower()
    battery_rows.append({"q": question, "correct": correct, "own": own,
                         "right": is_right, "lp_own": lp_own, "lp_correct": lp_correct})

print(f"{'':4}{'question':<44}{'model said':<22}{'P(said)':>9}{'P(correct)':>11}")
for r in battery_rows:
    print(f"{'OK ' if r['right'] else 'ERR'} {r['q'][:42]:<44}{r['own'][:20]:<22}"
          f"{np.exp(r['lp_own']):>9.3f}{np.exp(r['lp_correct']):>11.1e}")


# %% [markdown]
# Read the table twice.
#
# **First pass — the right answers.** Winnipeg, 1986, 74: probability near 1.0. The model knows these, and the numbers say so.
#
# **Second pass — the wrong answers.** On a typical run of this checkpoint, the 1969 Nobel question gets *"William Faulkner"* at around P ≈ 0.2 — a decisive, fluent, wrong answer (Faulkner won in 1949; 1969 was Samuel Beckett, sitting down at P ≈ 0.007). The Rusalka question gets Smetana at even higher probability; Dvořák, the actual composer, is thirty times less probable. Nothing *in the probability itself* separates these from the correct rows above — confidence is a property of the distribution, and the distribution does not know when it is wrong.
#
# **And one more thing, which is the seed of everything later:** look at the P(correct) column on the wrong rows. The correct answers are not at zero. They carry real, measurable probability — orders of magnitude above noise — in the very context where the model chose something else. The model, in a precise sense, *almost said them*. This session's title is not a metaphor.
#
# ### The deep dive: where does the correct answer rank?
#
# Take the Nobel question and score a whole slate of plausible candidates — laureates from nearby years, and famous authors who never won — each as a complete turn in the same context. This is ~15 forward passes, no sampling, and it reconstructs the model's entire preference ordering over answers.

# %%
NOBEL_Q = "Who won the 1969 Nobel Prize in Literature? Answer with the name only."
NOBEL_CANDIDATES = [
    "Samuel Beckett",          # correct (1969)
    "William Faulkner",        # 1949
    "Ernest Hemingway",        # 1954
    "Albert Camus",            # 1957
    "John Steinbeck",          # 1962
    "Jean-Paul Sartre",        # 1964 (declined)
    "Yasunari Kawabata",       # 1968
    "Alexander Solzhenitsyn",  # 1970
    "Pablo Neruda",            # 1971
    "Heinrich Boll",           # 1972
    "Jorge Luis Borges",       # never won
    "Graham Greene",           # never won
    "Vladimir Nabokov",        # never won
    "W. H. Auden",             # never won
    "Virginia Woolf",          # never won
]

ctx = chat_context(NOBEL_Q)
cand_lp = np.array([
    sequence_logprob(model, tokenizer, ctx, name, append_eos=True,
                     warn_on_retokenization=False).total_logprob
    for name in NOBEL_CANDIDATES
])
# renormalize over the candidate slate so the numbers read as a ranking
cand_p = np.exp(cand_lp - cand_lp.max())
cand_p /= cand_p.sum()
order = np.argsort(-cand_p)

fig, ax = plt.subplots(figsize=(9, 5))
names = [NOBEL_CANDIDATES[i] for i in order]
probs = [cand_p[i] for i in order]
colors = ["#2e7d32" if n == "Samuel Beckett" else "#c62828" for n in names]
ax.barh(names[::-1], probs[::-1], color=colors[::-1])
ax.set_xlabel("probability (renormalized over the candidate slate)")
ax.set_title("Who won the 1969 Nobel Prize in Literature?\n"
             "The model's preference ordering — correct answer in green")
plt.tight_layout()
plt.savefig("figures/session3_nobel_ranking.png", dpi=120)
plt.show()

rank_correct = int(np.where(np.array(names) == "Samuel Beckett")[0][0]) + 1
print(f"Correct answer rank: {rank_correct}/{len(NOBEL_CANDIDATES)}, "
      f"p = {cand_p[0]:.3f} (renormalized)")


# %% [markdown]
# This figure is the session's thesis in one picture: a confident, well-shaped preference ordering, with the truth somewhere in the middle of it. The model has *an* opinion about every candidate; it just has no access to which opinion is right. (A caveat you have seen before, in Session 1's granularity demo: the slate is ours, so the renormalized numbers are relative preferences among candidates we chose, not absolute probabilities over all possible answers.)
#
# ---
#
# ## Part 2 — Logits, softmax, and what temperature actually does
#
# Time to be precise about the numbers behind that figure. Session 1 established the pipeline: the model's final layer emits one raw score per vocabulary token — the **logits** — and softmax turns them into probabilities. Temperature is a single number that reshapes this step:
#
# $$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
#
# Three facts follow from the formula, two obvious and one that most people get wrong:
#
# 1. **T → 0** concentrates all mass on the top logit: sampling becomes greedy decoding.
# 2. **T → ∞** flattens everything toward uniform: sampling becomes noise.
# 3. **Temperature never changes the ranking.** Dividing by T is a monotone transform of the logits. The top-1 at T = 0.1 is the top-1 at T = 2.0. If the model's most-preferred answer is wrong, no temperature makes it right — temperature only changes how often the sampler wanders to the alternatives.
#
# Watch all three on real distributions — one sharp, one contested:

# %%
def next_token_logits(prompt):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        return model(ids).logits[0, -1, :].float().cpu()


sharp_logits = next_token_logits("The chemical symbol for gold is")
flat_logits = next_token_logits("On a rainy day like last Monday, the")

TEMPS = [0.25, 0.7, 1.0, 2.0]
fig, axes = plt.subplots(2, len(TEMPS), figsize=(15, 6), sharey="row")
for row, (logits, label) in enumerate([(sharp_logits, "sharp: 'The chemical symbol for gold is'"),
                                       (flat_logits, "contested: 'On a rainy day…, the'")]):
    top_vals, top_ids = torch.topk(logits, k=10)
    toks = [tokenizer.decode([int(i)]) for i in top_ids]
    for col, T in enumerate(TEMPS):
        p = torch.softmax(logits / T, dim=-1)[top_ids]
        ax = axes[row, col]
        ax.bar(range(10), p.numpy(), color="#455a64")
        ax.set_xticks(range(10))
        ax.set_xticklabels([repr(t)[:8] for t in toks], rotation=60, ha="right", fontsize=7)
        if row == 0:
            ax.set_title(f"T = {T}")
        if col == 0:
            ax.set_ylabel(f"{label}\nprobability", fontsize=8)
plt.suptitle("Same logits, four temperatures: the shape changes, the ranking never does", fontsize=11)
plt.tight_layout()
plt.savefig("figures/session3_temperature.png", dpi=120)
plt.show()


# %%
# The same fact as a single curve: entropy of the distribution vs. temperature.
T_grid = np.linspace(0.05, 3.0, 60)


def entropy_at(logits, T):
    p = torch.softmax(logits / T, dim=-1)
    p = p[p > 0]
    return float(-(p * p.log()).sum())


fig, ax = plt.subplots(figsize=(8, 4.2))
for logits, label, color in [(sharp_logits, "sharp prompt", "#1565c0"),
                             (flat_logits, "contested prompt", "#c62828")]:
    ax.plot(T_grid, [entropy_at(logits, T) for T in T_grid], color=color, label=label)
ax.set_xlabel("temperature")
ax.set_ylabel("entropy of next-token distribution (nats)")
ax.set_title("Temperature is an entropy dial — and the two prompts stay distinguishable at every T")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("figures/session3_entropy_vs_T.png", dpi=120)
plt.show()


# %% [markdown]
# ## Part 3 — Top-k and top-p: cropping the tail
#
# The other two knobs in every API call are not reshapers but **croppers**: they cut the distribution down to a candidate set *before* sampling.
#
# - **top-k** keeps the k highest-probability tokens, always exactly k of them.
# - **top-p** (nucleus sampling) keeps the smallest set whose cumulative probability exceeds p — an *adaptive* set: a handful of tokens on a sharp distribution, hundreds on a flat one.
#
# Why crop at all? Because the tail is long and collectively heavy: thousands of individually-negligible tokens that sum to several percent. Sample long enough and one of them fires, and a single absurd token can derail everything after it (Session 1's left-to-right commitment; Session 4 makes that mechanism explicit). Cropping trades a little diversity for a floor under quality.

# %%
fig, ax = plt.subplots(figsize=(8.5, 4.5))
for logits, label, color in [(sharp_logits, "sharp prompt", "#1565c0"),
                             (flat_logits, "contested prompt", "#c62828")]:
    p_sorted = torch.softmax(logits, dim=-1).sort(descending=True).values
    cum = torch.cumsum(p_sorted, dim=0)
    nucleus_size = int((cum < 0.9).sum()) + 1
    ax.plot(range(1, 201), cum[:200].numpy(), color=color,
            label=f"{label}: top-p=0.9 keeps {nucleus_size} tokens")
ax.axhline(0.9, color="k", ls=":", lw=1)
ax.set_xlabel("tokens, sorted by probability")
ax.set_ylabel("cumulative probability")
ax.set_xscale("log")
ax.set_title("The nucleus is adaptive: same p, very different candidate sets")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/session3_nucleus.png", dpi=120)
plt.show()


# %% [markdown]
# Note what cropping *removes*, because it matters for this workshop: rare-but-legitimate continuations. A continuation that lives at p = 0.001 is simply deleted by top-p = 0.9 on a sharp distribution. Keep that in mind for Session 6, where the question will be whether a specific rare continuation — an abstention — is in the distribution at all. Sampling through a cropped decoder is the wrong instrument for that question, which is exactly why the next part exists.
#
# ---
#
# ## Part 4 — Teacher forcing: scoring text the model never said
#
# Everything so far inspected distributions one step at a time. The move that unlocks the rest of the workshop is scoring a whole **sequence** — and the key realization is that you can do it for *any* text, not just text the model generated.
#
# The chain rule of probability says the probability of a token sequence is the product of each token's probability given everything before it:
#
# $$P(t_1, t_2, \ldots, t_n \mid \text{context}) = \prod_{i=1}^{n} P(t_i \mid \text{context}, t_1, \ldots, t_{i-1})$$
#
# In log space the product becomes a sum — the cumulative log-probability you already met in Session 1. And here is the trick: the model computes $P(t_i \mid \text{everything before})$ at *every* position in *one* forward pass. So to score a candidate continuation, you don't generate anything. You lay down context + continuation as if it had been said — this is called **teacher forcing** — run one pass, and read off the probability of each continuation token. Write it:

# %%
@torch.no_grad()
def continuation_logprob_simple(model, tokenizer, context, continuation):
    """Log-probability of `continuation` following `context`. One forward pass.

    Educational version: assumes the tokenizer does not merge tokens across
    the context/continuation boundary (the workshop_utils version checks).
    """
    ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
    full_ids = tokenizer(context + continuation, add_special_tokens=False).input_ids

    input_ids = torch.tensor([full_ids], device=device)
    logits = model(input_ids=input_ids).logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)

    # logits at position i-1 are the model's prediction FOR position i,
    # so continuation token i is scored by the distribution at i-1
    total = 0.0
    for pos in range(len(ctx_ids), len(full_ids)):
        total += float(log_probs[pos - 1, full_ids[pos]].item())
    return total


# Sanity check against the production version used in Part 1:
ctx = chat_context("What is the capital of Japan?")
simple = continuation_logprob_simple(model, tokenizer, ctx, "Tokyo")
full = sequence_logprob(model, tokenizer, ctx, "Tokyo").total_logprob
print(f"simple version: {simple:.4f} nats")
print(f"utils version:  {full:.4f} nats   (same number — the utils one adds "
      f"boundary-merge checks and end-of-turn scoring)")


# %% [markdown]
# Twelve lines. This is the instrument. Three things it gives you that sampling cannot:
#
# 1. **No noise.** The number is exact (up to floating point), identical on every run. No seeds, no ensembles.
# 2. **Access to the unspoken.** You can score text the model would essentially never emit — and find out whether it was *improbable* or *impossible*. Session 6 rests entirely on this distinction.
# 3. **Fair comparison.** Two candidate answers scored in the same context are measured on exactly the same scale.
#
# ### Comparing whole sentences
#
# One wrinkle: a sentence's first token needs something to be conditioned on, so standalone sentences are compared by scoring them against a **shared neutral context**. As long as the context is identical, the *difference* in log-probabilities is a meaningful quantity — it is the model's log-odds between the two sentences.

# %%
STUB = "Fact:"
for sentence in [
    " Paris is the capital of France.",
    " Berlin is the capital of France.",
    " Paris is the capital of Germany.",
]:
    lp = continuation_logprob_simple(model, tokenizer, STUB, sentence)
    n_tok = len(tokenizer(sentence, add_special_tokens=False).input_ids)
    print(f"{sentence.strip():<38} {lp:8.2f} nats total   {lp / n_tok:6.2f} nats/token")


# %% [markdown]
# On the reference run the true sentence scores about **10.6 nats** above either false one — a factor of e^10.6 ≈ 40,000 in probability. That gap *is* the model's knowledge of this fact, expressed in its native unit. Both error directions (wrong city, wrong country) are punished by a similar amount, which is reassuring: the model is scoring the *claim*, not a surface pattern.
#
# Per-token normalization (`nats/token`, last column) matters whenever the compared strings have different lengths — a longer sentence accumulates more factors < 1, which is a fact about length, not about belief. You met this in Session 1; it will carry the weight in Session 6's ledger, where it decides whether abstentions and answers can be compared at all.
#
# ### What the model knows but does not say
#
# Now close the loop on Part 1. The battery table showed correct answers carrying real probability in contexts where the model said something else. How real? Compare against a floor — a fluent-sounding absurdity scored in the same context:

# %%
print(f"{'question':<40}{'P(correct)':>12}{'P(absurd)':>12}{'ratio':>10}")
for r in battery_rows:
    if r["right"]:
        continue
    ctx = chat_context(r["q"])
    lp_absurd = sequence_logprob(model, tokenizer, ctx, "Purple monkey dishwasher",
                                 append_eos=True, warn_on_retokenization=False).total_logprob
    ratio = np.exp(r["lp_correct"] - lp_absurd)
    print(f"{r['q'][:38]:<40}{np.exp(r['lp_correct']):>12.1e}{np.exp(lp_absurd):>12.1e}"
          f"{ratio:>9.0f}x")


# %% [markdown]
# The correct answers sit orders of magnitude above the floor. **The model "knows" more than it "says"** — not as a slogan but as a measured inequality: the knowledge is present in the distribution, and the sampler, which takes the argmax neighborhood and crops the rest, discards it. That gap between distribution and emission is where this workshop lives from here on.
#
# ---
#
# ## Part 5 — Does the model know when it's wrong? The Kadavath probe
#
# If the knowledge is in the distribution, can the model be made to *report* it? Kadavath et al. (2022), *"Language Models (Mostly) Know What They Know"*, asked exactly this: show a model a question together with a proposed answer, and read its probability for "True". Their headline findings: large models are surprisingly well-calibrated at this self-evaluation task — and that calibration *degrades* under RLHF post-training (a cost you will meet again in Session 7).
#
# The probe is ten lines with the instrument from Part 4: pose the verification question, score `"Yes"` vs. `"No"` as complete answers, normalize.

# %%
def p_true(question, proposed_answer):
    """The model's probability that `proposed_answer` is correct."""
    ctx = chat_context(
        f"Question: {question}\nProposed answer: {proposed_answer}\n"
        "Is the proposed answer correct? Reply Yes or No only."
    )
    lp_yes = sequence_logprob(model, tokenizer, ctx, "Yes",
                              warn_on_retokenization=False).total_logprob
    lp_no = sequence_logprob(model, tokenizer, ctx, "No",
                             warn_on_retokenization=False).total_logprob
    return float(np.exp(lp_yes) / (np.exp(lp_yes) + np.exp(lp_no)))


# For every battery question, probe three proposals: the model's own greedy
# answer, the correct answer, and a plausible-but-wrong distractor.
DISTRACTORS = ["Sydney", "Honiara", "1888", "William Golding", "68",
               "1390", "Regina", "Wagner", "1989", "Giewont"]

probe_rows = []
for r, distractor in zip(battery_rows, DISTRACTORS):
    bare_q = r["q"].split(" Answer with")[0]
    for proposal, truth in [(r["correct"], True), (distractor, False)]:
        probe_rows.append({
            "q": bare_q, "proposal": proposal, "truth": truth,
            "p_true": p_true(bare_q, proposal),
        })

fig, ax = plt.subplots(figsize=(8.5, 4.5))
for truth, color, label in [(True, "#2e7d32", "proposal is actually correct"),
                            (False, "#c62828", "proposal is actually wrong")]:
    vals = [p["p_true"] for p in probe_rows if p["truth"] == truth]
    ax.scatter(np.random.default_rng(0 if truth else 1).uniform(-0.12, 0.12, len(vals))
               + (1 if truth else 0), vals, color=color, s=70, alpha=0.8, label=label)
ax.set_xticks([0, 1])
ax.set_xticklabels(["wrong proposals", "correct proposals"])
ax.set_ylabel("model's P(True) for the proposal")
ax.set_ylim(-0.03, 1.03)
ax.set_title("Self-verification on the battery: direction mostly right, calibration poor")
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=9, loc="center right")
plt.tight_layout()
plt.savefig("figures/session3_ptrue.png", dpi=120)
plt.show()

mean_true = np.mean([p["p_true"] for p in probe_rows if p["truth"]])
mean_false = np.mean([p["p_true"] for p in probe_rows if not p["truth"]])
print(f"mean P(True) on correct proposals: {mean_true:.3f}")
print(f"mean P(True) on wrong proposals:   {mean_false:.3f}")
worst = min((p for p in probe_rows if p["truth"]), key=lambda p: p["p_true"])
print(f"worst miss on a correct proposal:  P(True)={worst['p_true']:.3f} "
      f"for {worst['proposal']!r} ({worst['q'][:40]})")


# %% [markdown]
# **Read the two clouds, then the overlap.** On a typical run the green cloud sits clearly above the red one — within a question, the model usually assigns more "True" to the truth than to the distractor, confirming in miniature that the knowledge is accessible by asking. But the clouds overlap, the absolute values are far from 0 and 1, and there are individual failures worth looking at: on the reference run, the model *generates* "Tarawa" correctly for the Kiribati question, then assigns it P(True) ≈ 0.05 when asked to verify it. Producing an answer and endorsing it are different computations, and at 1.5B they visibly disagree.
#
# Two honest framings to carry away:
#
# - **Kadavath's result is a scale story.** Calibrated self-knowledge *emerges* with model size; a 1.5B model shows the direction, not the calibration. What you just measured is the small-model end of their curve, and it is appropriately messy.
# - **The information was in the distribution all along.** Note what we did *not* do: we never asked the model to say "I'm not sure." We read its uncertainty directly from probabilities it assigns anyway. The gap between that internal signal and the model's *verbal* behavior — fluent, unhedged, confident prose on top of a P ≈ 0.2 answer — is a gap post-training actively widens. Which is precisely where this workshop is headed.
#
# ---

# %% [markdown]
# ## So what, for a working scientist
#
# - **Temperature 0 is not "truth mode."** The ranking is fixed before temperature touches anything; greedy decoding just serves the top of the ranking deterministically. If the model's preferred answer is wrong (Part 1's Faulkner), T = 0 gives you the same wrong answer every time — *more* consistently than T = 0.7 would. **Practical implication:** turning the temperature down makes outputs reproducible, not reliable, and a consistent answer across reruns at T = 0 carries zero additional evidence of correctness (it's the same draw repeated).
#
# - **What you can and cannot get through an API.** Everything in Parts 4–5 requires scoring *arbitrary* text under the model, and no major commercial API offers that: OpenAI returns top-20 log-probs for tokens the model *generates* (useful — you can read the margin on its own answer), Anthropic and Google return effectively none. **Practical implication:** the P(True) probe survives the translation — ask any chat model to verify a claim in a *fresh* conversation and you get the sampled verdict (and with OpenAI's `logprobs`, the actual probability of "Yes"). Verification-in-fresh-context is the API-compatible shadow of what you measured here, and it is cheap enough to make a habit.
#
# - **Fluency is not confidence, and confidence is not knowledge.** You now have the three-layer picture: the prose sounds certain (that's register, trained in), the distribution may or may not be concentrated (that's confidence, readable with logprobs), and the concentrated answer may or may not be true (that's the part no internal signal fully reaches). **Practical implication:** when a model's answer matters, the question to ask is never "does it sound sure" — it is "what would this model have said to the alternatives," and on open models you can literally compute that.
#
# ---

# %% [markdown]
# ## Exercises
#
# **1. Rank candidates against ground truth.** Build a 10-question battery in your own field with a closed slate of 5 candidate answers each. Score every candidate as a complete turn (Part 1's method) and check: how often is the top-log-prob candidate the correct one? Compare against the model's sampled accuracy on the same questions at T = 0.7 (20 samples). Which protocol is more accurate — scoring or sampling? Why might they differ at all?
#
# **2. Answer diversity vs. temperature.** For one contested question, sweep T over {0.1, 0.4, 0.7, 1.0, 1.5, 2.0} with 20 samples each; count distinct answers at each T. Plot the count against the Part 2 entropy curve for the same prompt's first token. Where do they diverge, and what does that say about how much of answer diversity is decided at the first token?
#
# **3. The ledger preview.** Take the battery's confident-wrong rows. For each, compute the complete-turn log-probability of a correct answer, a differently-worded correct answer ("Beckett" vs. "Samuel Beckett"), and your own absurd control. How sensitive is P(correct) to wording? This sensitivity is exactly why Session 6 scores an *equivalence set* of spellings rather than a single string — you are discovering the design requirement yourself.
#
# **4. (Harder) A reliability diagram for P(True).** Extend Part 5 to 40 proposals (20 true, 20 false), bin them by P(True), and plot observed accuracy per bin against the diagonal. Then — if you have the disk space from Session 0 — repeat on the base model `Qwen/Qwen2.5-1.5B` with the same prompts in plain `Q:/A:` format. Kadavath et al. report that RLHF *hurts* this calibration. Does the base model's diagram look better, worse, or just different? (Session 6 gives you the tools to make this comparison properly.)

# %% [markdown]
# ## What you just did
#
# - Measured, on a ten-question battery, that the model's probability for its own answer does not separate right answers from wrong ones — and that correct answers carry real probability even when the model says something else.
# - Reconstructed a model's full preference ordering over an answer slate with ~15 forward passes and no sampling.
# - Unpacked temperature as a monotone reshaping that never changes the ranking, and top-k/top-p as adaptive cropping of the tail.
# - Wrote the teacher-forcing scorer yourself — twelve lines that turn "what did the model say" into "what probability does the model assign to anything" — and verified it against the workshop's production version.
# - Measured the model's knowledge of a fact as a log-odds gap (~10.6 nats for Paris vs. Berlin) and its unspoken knowledge as a distance above an absurd floor.
# - Ran the Kadavath self-verification probe and saw both the signal (truth scores above distractors) and the small-model honesty (poor calibration, generation/verification disagreement).
#
# ## Forward hook
#
# The model knows more than it says — you have now measured that gap directly. **Session 4** asks the uncomfortable follow-up: why does the gap exist at all? The answer is not an engineering shortfall. There are three independent reasons — one computational, one statistical, one economic — why a model of this kind *must* sometimes assert things it cannot know, and you will compute the arithmetic behind two of them yourself.
