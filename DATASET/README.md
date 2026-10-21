# Supplementary data — *Chasing the "I Don't Know" Attractor*

Curated data release accompanying the talk
**"Chasing the 'I Don't Know' Attractor: Probability-Space Accounting of Abstention in
Instruction-Tuned Language Models"** (Piotr Gieszcz, DevAI / Data Science Summit 2026).

**Scope.** This is a *curated subset* of the project's experimental record: the files
sufficient to verify every quantitative claim made in the talk. It is not the full
project's dataset — probe-level dumps and instruments belonging to work in progress
(follow-up paper material) are withheld and may be shared on request.

Laboratory models: `Meta-Llama-3.1-8B` / `-Instruct`, `Qwen2.5-7B` / `-Instruct`
(bf16, open weights). Frontier arm (behavioural, preliminary): Opus 5, Sonnet,
Haiku 4.5, GPT-5.1, GPT-4.1-mini. All response text in this release is
model-generated; questions are PopQA-derived or constructed (see `corpus/`).
Log-probabilities are natural logs (nats). Response lengths are counted in tokens.

---

## Claim → data map

| talk claim | folder |
|---|---|
| 0 / 94,560 bare terminal "I don't know." across 4 open models | `01_zero_bare_idk/` |
| the form is rare, not forbidden: 4 / 240,000 at T=1.0, matching predicted E=4.40 | `02_rare_not_forbidden/` |
| pretraining funds the state: IDK ≈1,000× above the nonsense floor, 100% of questions | `03_pretrained_prices/` |
| qwen cannot stop: EOS wall ≈1e-9, 0/6,240 forced-IDK turns end; dwell 4 sentences vs llama's 1 | `04_qwen_cannot_stop/` |
| llama cannot say it: toll decomposition −16.15 nats (opener 20% / words 42% / period 31% / stop 7%) | `05_llama_toll_road/` |
| the calibrated gate: opener mass 0.91→0.06 (ρ=−0.65); 22,012× suppression on "42"; 3,606 short abstentions, zero bare | `06_calibrated_gate/` + `05`, `01` |
| fabrication on invented entities 2.6% vs 44.8% (17×); forced-opener mitigation 0.448→0.056, 93% answer retention | `07_fabrication_and_mitigation/` |
| open problems: hedge+invented-specific 26% (llama), accounting 52/26/14/8 vs 69/9/1/21 | `08_open_problems/` |
| detector table: SE 0.796/0.654 → first-sentence 0.817/0.841 → one-pass 0.831/0.877, ceiling 0.945/0.926 | `09_detectors/` |
| frontier 2026: zero spontaneous bare IDK; instruction recovers 85–87%, 0 false abstentions / 390; the Cambrian cliff; 84/84 bracketed substitution | `10_frontier_preliminary/` |

---

## Folders and files

### `corpus/`
Question sets. Ground truth is constructed, never model-sourced: admissible provenance
tags are `by-construction`, `database:PopQA/Wikidata`, `primary-source`, `corpus-attested`.
- `e2_corpus.json` / `.csv` — Block A: 300 PopQA entities (author / director / screenwriter),
  stratified over log-popularity (`s_pop`), with gold aliases and provenance per question.
- `e2_corpus.blockC.json` — 31 by-construction unanswerable questions (false premise,
  ill-posed, private info, future unknowable, invented entities).
- `e2_corpus.blockCROSS.json` + `blockCROSS_provenance.csv` — 80 template-matched pairs
  in a 2×2 of looks-answerable × answer-exists (settled / open / stable / ill-posed).
- `e2_corpus.roundE.json` — the 10 questions of the 240,000-draw run.
- One Block A question (`obscure_real`, Nantwich 1911) is excluded from all truth metrics —
  its gold value is unverifiable (the talk's "known but unverifiable" category).

### `01_zero_bare_idk/`
`raw_results.n312-*.csv.gz` (74,880 generations, 4 models × 312 questions × samples ×
decoding configs) and `raw_results.topp1-llama.csv.gz` (19,680, top-p=1.0 control) —
together the 94,560 generations behind the zero. Schema:
`model,family,variant,strategy,question_id,tier,sample_idx,response_text,classification`.
The `classification` column is the *permissive filter* used for exhaustive reading
(zero-claims are by reading survivors, not by trusting the classifier).

### `02_rare_not_forbidden/`
`raw_results.roundE.csv.gz` — 240,000 generations at T=1.0 on the 10 friendliest
questions (llama-instruct). Contains the 4 exact `"I don't know."` draws
(`popqa_3762914` ×2, `popqa_5553632`, `popqa_2687380`) matching the price-predicted
expectation of 4.40. Same schema as `01`.

### `03_pretrained_prices/`
`e1a_register_control.*.csv` — the "price shelf": forced-through log-probs of fixed
register sentences (IDK, hedges, "Yes.", nonsense floor "Purple staplers…") for every
question. Per-token price = `response_logprob / n_tokens` (the `logprob_per_token`
column includes EOS — a different quantity; do not mix the two).

### `04_qwen_cannot_stop/`
- `e1d_persistence.*.csv.gz` — forced-"I don't know." continuations, 6,240 per model
  across the 4 models (`label` = empty / continued; for qwen-instruct the turn ended
  0 times in its 6,240).
- `e13_sentence_budget[.base].csv` — P(turn ends | k-th sentence boundary) per
  model × response set; qwen-instruct abstention: 0.0040 / 0.0896 / 0.3906 / 0.7337.
- `e13_bracket[.base].csv` — dwell-time bracket under strict vs loose segmentation
  (abstention dwell: llama 1, qwen 4 — both segmenters agree).

### `05_llama_toll_road/`
- `e1b_inflow_probe.*.csv` — opener mass (probability the turn opens with an
  abstention opener) per question; the gate gradient 0.91→0.06.
- `e1b_opener_logprobs.*.csv` — priced openers (the −2.10 "I" toll etc.).
- `e30_idk_token_decomp.csv` — per-token decomposition of the bare-IDK price
  (llama-base, the worked −10.6-nat example).

### `06_calibrated_gate/`
- `raw_results.blockC.csv.gz` — 7,440 generations on Block C (the 22,012×
  "42" content-control lives here, with `e1b_*.proxyC.csv` carrying the probe side).
- The reachability column of the talk's 2×2 table (P(enters abstention) on invented
  entities, 0.96 / ≈0.50) is supported by the zero-shot labels in
  `07_fabrication_and_mitigation/e6_battery.csv.gz` (abstention-class share =
  grounded + deflected); the persistence column by `04_qwen_cannot_stop/e13_*`.

### `07_fabrication_and_mitigation/`
- `e6_battery.csv.gz` — 8,640 generations on invented entities, labelled
  grounded / deflected / hedged / fabricated (`arm` = zero-shot vs forced-IDK prefix;
  the 0.448 → 0.056 mitigation).
- `e9_post_idk_recovery.csv` — post-IDK retrieval: fraction of correct answers
  retained after a forced abstention opener (qwen 93%).

### `08_open_problems/`
- `raw_results.blockCROSS-*.csv.gz` — 4,800 generations on the 2×2 corpus.
- `e11_judged.csv` — speech-act labels (act / justified / fabricated) per answer;
- `e11_fab_judged.csv` — fabrication labels (`fab_sonnet`) for the hedge+cargo
  accounting (26%, 52/26/14/8 vs 69/9/1/21);
- `e11_judge_rubric.txt`, `e11_fab_rubric.txt` — the rubrics are part of the
  instrument; labels are reproducible only against these exact texts.

### `09_detectors/`
- `e23_detector_stack.csv` — the detector table (AUCs with paired-bootstrap CIs and
  a `resolution` column; identical judged ASSERTED population for all four rows).
- `e3_semantic_entropy[.firstsent].csv` — semantic entropy per question, full-turn
  vs first-sentence clustering (NLI: `microsoft/deberta-large-mnli`, bidirectional
  entailment).
- `e7_belief.csv` — one-pass quantities: slot entropy `H` and forced-slot `p_gold`.
- `e2_judged.v3.csv` + `e2_judge_rubric.v3.txt` — judged correctness labels for the
  detector population (rubric v3).
- `e2_samples.*.csv.gz` — the underlying 20 samples/question at T=1.0, all 4 models.

### `10_frontier_preliminary/`
Behavioural, smaller-n, marked preliminary in the talk.
- `e26_ladder*.csv` + `.meta.json` + summaries — instruction-compliance ladder on
  Opus 5 / Haiku 4.5 (CLI transport) and GPT-5.1 / GPT-4.1-mini (API): the frontier
  zeros, the 85–87% recovery, 0 false abstentions on answerable twins.
- `e28_substitution.csv` + summary + rubric — bracketed-substitution judging
  (84/84 long answers share the hedge-then-essay shape).
- `e29_deepdraw.*.csv` + `.meta.json` — 20-samples-per-cell deep draws behind the
  Cambrian cliff figure (language 1.00/0.70/1.00 · life 1.00/0.60/1.00 ·
  Cambrian 0.35/0.00/0.00).
- `e26_report.md`, `e29_report.md` — method notes for this arm.

---

## What is *not* here

Teacher-forcing position dumps (the full hazard instrument), all hidden-state
work (perturbation trajectories, flip statistics, layer-gain and depth/patching
runs), all committor / decision-budget data, the judge-relabelling audit
pipeline, and all withdrawn or superseded instruments. These belong to analyses
not presented in the talk or not yet published; the files above are sufficient
to check every number quoted in the talk. Contact the author for anything beyond.

## Reading notes

- Prices are in nats; ≈7 nats ≡ ×1,000, −16 nats ≈ 10,000,000:1 against.
  Per-token prices divide by token count so lengths compare fairly.
- Forced-through scoring ("price check, not a purchase") never generates; files in
  `03`, `05` and the probe halves of `06` are measurements of the model's
  distribution, not samples from it.
- Zero-claims (the 0/94,560, the 0/6,240) were verified by permissive filtering plus
  exhaustive reading of survivors — not by trusting any classifier column.
- bf16 reproducibility across library versions is ~3% on mid-range probabilities;
  treat smaller cross-run gaps as noise.

## Use and citation

Model outputs from Llama-3.1 and Qwen2.5 are subject to their respective licenses
(Llama 3.1 Community License; Apache-2.0). PopQA-derived questions follow PopQA's
licensing (Mallen et al. 2023). Constructed questions and all labels/rubrics:
© 2026 Piotr Gieszcz, released for research use with attribution.

> Gieszcz, P. (2026). *Chasing the "I Don't Know" Attractor: Probability-Space
> Accounting of Abstention in Instruction-Tuned Language Models.* Talk, DevAI /
> Data Science Summit 2026. Supplementary data.

Contact: piotr@atlab.waw.pl · materials: https://github.com/piotrgie/IDK
