# Paper Plan — Chapter Plan + INSIGHT Collection

**Working title**: The Retrieval Noise Taxonomy: Disentangling Stochastic, Systematic, and
Selection Effects on Self-Improvement Drift in Test-Time Continual Agents

**Target**: NeurIPS 2026 "Towards Test-Time Continual Learning Agents" (TTCL) workshop,
General Research Track, 4-9 pages. Submission deadline 2026-08-29.

**Status**: Plan mode (Socratic) in progress. Chapters locked so far: Thesis, Introduction,
Literature Review structure, Taxonomy operationalization, Ablation design, Sample size/analysis.
Remaining: base model choice, testbed integration, Method chapter completion, Results,
Discussion, Conclusion.

---

## [INSIGHT: thesis_statement]

Retrieval infrastructure is not a neutral storage layer but two distinct, potentially
opposite-signed noise channels (stochastic/approximate-search vs. systematic/staleness)
shaping whether test-time self-feedback drifts or stabilizes — demonstrated causally, not
correlationally, via matched-replay ablation that holds the starting task instance fixed by
construction across infra conditions, isolating infra as the sole source of divergence in
what follows.

## [INSIGHT: counterargument_and_rebuttal]

Objection: infra settings (small budgets, stale indices, aggressive eviction) correlate with
hard/long-horizon tasks, so the causal story could collapse into "hard tasks → drift" with
infra as a bystander. Rebuttal: matched-replay design (same starting task instance, varied
infra) removes difficulty as a confound by construction rather than statistical control;
secondary rebuttal is cross-testbed replication (AgentOdyssey + Evo-Memory) against the
objection that SkillLearnBench's drift finding is idiosyncratic to one model/task family.

## [INSIGHT: reader_takeaway]

Practitioners currently treat "retrieval quality" as one dial (more accurate = safer). The
paper's revision: approximate-search error and index staleness are mechanistically distinct
risks that can point in opposite directions across self-feedback rounds, so "use exact
search" and "refresh every N steps" are two separate tuning decisions, not one.

## [INSIGHT: intro_hook]

Illustrative (not yet-observed) scene: an AgentOdyssey agent learns "the merchant in Ward 3
only trades ore for silver" and writes it to memory. Staged side by side on the same fact:
(a) systematic failure — world state changes, index isn't rebuilt, agent keeps acting on and
generating new memories consistent with the stale rule, compounding the error; (b) stochastic
failure — same fact retrieved via approximate search, occasionally wrong but not directionally
biased, doesn't compound. Must be framed explicitly as illustrative/hypothetical in the draft,
not as a reported finding (no experiments have run yet).

## [INSIGHT: intro_gap_sentence]

"Nobody has shown that retrieval noise is not one failure mode but at least two
mechanistically distinct ones — stochastic and systematic — that can pull self-feedback drift
in opposite directions."

## [INSIGHT: intro_structure_ordering]

Introduction promises taxonomy-first: Section 2 defines the noise taxonomy (Related Work +
formal definitions), Section 3 is the matched-replay ablation method, Section 4 is results.
Chosen over ablation-first because the taxonomy is contribution #1 and needs to read as a
structural claim under test, not a post-hoc gloss on a surprising number.

## [INSIGHT: intro_contributions]

Three numbered bullets: (1) named taxonomy with formal definitions (stochastic/systematic/
selection), (2) causal matched-replay finding across two testbeds (AgentOdyssey, Evo-Memory),
(3) released harness (driftbench) as its own headline deliverable, not folded into the
empirics bullet.

## [INSIGHT: litreview_structure]

Three gap-subsections + one short instrumentation paragraph + one grounding lede sentence
(not four parallel subsections — that would read as a survey and dilute the argument):

1. **Drift phenomenon** — SkillLearnBench (COLM 2026, arXiv 2604.20087) establishes drift is
   real but offers no infra-level mechanism.
2. **Memory-as-correctness** — 2606.15903 and Agent Memory Characterization (arXiv 2606.06448)
   treat memory operations as a correctness problem, never as a noise source shaping learning
   dynamics; June 2026 "Externalization in LLM Agents" review as framing anchor neither paper
   adopts.
3. **Terminological near-neighbors** — WWW 2026 "Retrieval Collapses When AI Pollutes the Web"
   (arXiv 2602.16136, corpus-pollution at web scale) and Shumailov et al. model-collapse
   (Nature 2024, cross-generation synthetic-data retraining) share vocabulary but operate at
   the wrong scale/mechanism — this subsection's job is explicit differentiation, not a gap
   claim.

Borrowed instrumentation (one adoption sentence each, full detail deferred to Methods):
SeqMem-Eval (arXiv 2605.15384) recall-vs-proxy protocol, RDumb++ (arXiv 2601.15544) drift
metric, 2606.06448 staleness definition.

Grounding lede sentence (field lineage, not a subsection): EWC, Tent/CoTTA.

## [INSIGHT: litreview_conclusion_sentence]

"Across these threads, retrieval-infrastructure noise is either collapsed into a single
correctness problem, assumed away as a background condition, or examined at a scale/mechanism
that doesn't transfer to a single agent's private feedback loop — none decompose it into
mechanistically distinct channels or test their causal, potentially opposite-signed, effect on
self-feedback drift."

## [INSIGHT: litreview_pushback]

Shumailov critique (arXiv 2410.12954) addressed inline in the model-collapse paragraph of the
Lit Review, not deferred to Limitations — it qualifies the source theory being borrowed as
analogy, not our own result. One sentence: "Follow-up work has qualified the original collapse
dynamics under certain replay/mixing conditions (2410.12954), though the core
compounding-error mechanism motivating our hypothesis is not contested."

## [INSIGHT: taxonomy_operationalization]

- **Stochastic** = recall@k vs. exact search (standard ANN metric).
- **Systematic** = pending-write count at query time (2606.06448's staleness definition,
  applied directly).
- **Selection** = necessary-fact retention rate vs. an unbounded-memory oracle run of the same
  task instance. Novel metric, no direct precedent found — flagged as such, kept
  secondary/appendix-tier (not a headline channel on equal footing with the other two).

## [INSIGHT: ablation_design]

Matched-replay ablation is a **2-way factorial** crossing stochastic × systematic (3 levels
each, 9 cells, both testbeds) — needed to preempt the interaction objection (does staleness's
effect depend on search approximation level, or vice versa) rather than just one-at-a-time,
which can't detect that. Selection stays one-at-a-time against the shared (exact-search,
fresh-index) baseline, consistent with its secondary/appendix billing and its from-scratch
metric. The (baseline, baseline) grid corner doubles as the control condition, so no compute
is spent beyond the 9-cell grid plus a handful of Selection-only runs.

## [INSIGHT: sample_size_and_analysis]

Matched-replay's pairing (same instance across cells) analyzed via repeated-measures/
mixed-effects model, task instance as random effect; primary reported quantity is the
stochastic×systematic **interaction term** with CI, not a bare significance threshold.
N=15-20 instances/cell (no pilot data available; effect-size/CI framing chosen over
power-justified N given the deadline). ≈324 grid rollouts + ~70-100 Selection rollouts ≈ 400
total.

## [INSIGHT: base_model_and_stack]

**Base agent model**: Llama-3.1-8B-Instruct (primary), Qwen3-4B (optional cheap secondary/
robustness check) — both already validated by AgentOdyssey's own paper (Qwen3-4B smallest
open model tested, Llama-3.1-8B used for their MemoryLLM/MPlus baselines), so neither choice
is arbitrary. One consistent model across both testbeds (methodologically cleaner: one fewer
confound between testbeds). Evo-Memory's own paper tested only closed API models
(Gemini-2.5, Claude 3.5/3.7) — no open-weight precedent there, so our numbers on that testbed
won't be directly comparable to their published baselines; note as an acknowledged limitation,
not a blocker, since Evo-Memory's harness is used as a second matched-replay stage for our own
ablation, not a leaderboard to beat. Fixed-size-memory design (already locked) is required
regardless, per AgentOdyssey's own finding that Long-Context agents scale quadratically in
token cost over long horizons while fixed-size-memory/RAG agents stay linear.

**Inference stack**: vLLM (standard, well-supported for Llama-3.1-8B and Qwen3-4B; AWQ-INT4
quantization available if GPU budget on Lambda credits gets tight).

**Embedding model**: BAAI/bge-base-en-v1.5 (open-weight, 768-dim, fast — standard pairing
with Llama-family generators for RAG/agent-memory setups).

---

## [INSIGHT: headline_figure]

Two-panel figure. **Panel A (primary)**: marginal-effects plot — two lines with CI ribbons
showing drift vs. each channel independently at the other's baseline (stochastic level with
systematic held at baseline; systematic level with stochastic held at baseline). This is the
direct visual of "opposite-signed" — one line rising, one flat/falling — and the figure meant
to be screenshotted into a talk. **Panel B (supporting)**: the 3×3 heatmap with the
stochastic×systematic interaction term's estimate/CI annotated directly on it, doing the
robustness/confound job. Both testbeds shown as small multiples (side-by-side columns), not
pooled or overlaid in the figure itself — the figure shouldn't pre-commit to pooling before
the mixed-effects analysis decides whether testbed is pooled or modeled separately.

## [INSIGHT: falsification_criteria]

- **Null interaction term (CI includes zero) is NOT a falsification** — it means the two
  channels act additively/independently, which is the best-case robustness result: it answers
  the Introduction's confound-robustness question (effects don't depend on each other's
  setting) without touching the opposite-signed claim at all.
- **Same-signed main effects falsifies the headline claim specifically.** Each channel could
  still be real and mechanistically distinct, but "opposite-signed" is the citable hook and
  the taxonomy-naming rationale — losing it makes this a different, less citable paper, not a
  narrower-scope version of this one.
- **No detectable main effect from either channel falsifies the foundational premise**, not
  just contribution #1 — this contradicts the drift phenomenon borrowed from SkillLearnBench
  itself. The one result that triggers a full stop-and-rethink rather than a reframe.

## [INSIGHT: fallback_framing]

If only one channel (most likely systematic) shows a real effect: reframe contribution #1 from
"opposite-signed" to **"the taxonomy correctly predicts which noise type is consequential and
which isn't"** — a sharper, more actionable practitioner claim (tells them where to spend
engineering effort: index freshness over search accuracy, say) than drama for its own sake.
The working title never committed to "opposite-signed," so it survives this fallback
unchanged. The model-collapse analogy also survives/strengthens under this fallback, since
compounding error is specifically a systematic-noise signature — if systematic is the one
channel that's real, that's the cleanest possible confirmation the analogy pointed at the
right mechanism.

## [INSIGHT: conclusion_and_future_work]

One-paragraph conclusion: Retrieval infrastructure has been treated as a neutral storage
layer; this paper shows it's at least two mechanistically distinct noise channels with
different — possibly opposite — consequences for whether a self-improving agent's behavior
drifts. The practical payoff, regardless of which fallback scenario obtains, is
decision-relevant: practitioners get told *which* infra knob to spend engineering effort on,
not just "retrieval quality matters." The taxonomy, the matched-replay causal method, and the
released harness are offered as reusable infrastructure for a question the field has so far
only gestured at as future work.

Future directions: (1) extending the taxonomy to noise channels not covered here (e.g.
distributed/networked consolidation delay, not captured by either testbed); (2) adaptive infra
policies that respond to detected drift in real time; (3) formalizing the model-collapse
analogy into a derived theory — deliberately not attempted here (per the Lit Review's refusal
to force the bias/variance equivalence), but motivated properly by the results, especially
under the one-channel-real fallback.

## [INSIGHT: limitations]

Selection channel's retention-rate metric is novel, not adopted (flagged, kept secondary).
Evo-Memory testbed has no open-weight precedent, so results there aren't directly comparable
to its published closed-model baselines. N=15-20/cell uses effect-size/CI framing, not
power-justified (no pilot data). Model-collapse analogy is motivating narrative, not formal
derivation (Shumailov critique arXiv 2410.12954 addressed inline in Lit Review). Single base
model (Llama-3.1-8B-Instruct) — generalization across model scale untested.
**Compute-budget confound**: approximate search (aggressive HNSW) is faster than exact flat
search; under any wall-clock or step budget, "more approximate" conditions could complete more
steps than "more exact" conditions, confounding retrieval noise with effective compute/
exploration budget. Mitigation (locked as a design requirement, not just a caveat): fix a
per-episode STEP count, not wall-clock time, across every grid condition.

---

**Plan mode status: COMPLETE.** All six chapters (Introduction, Literature Review, Method,
Results, Discussion, Conclusion) have locked INSIGHT entries above. Proceeding to `full` mode
for drafting.

## Verified citations (see conversation log for full verification detail)

| Key | ArXiv / Venue | Role |
|---|---|---|
| SkillLearnBench | arXiv 2604.20087, COLM 2026 | Foundational drift finding |
| Agent Memory Characterization | arXiv 2606.06448 | Staleness definition |
| SeqMem-Eval | arXiv 2605.15384 | Eval protocol (recall-vs-proxy) |
| Evo-Memory | arXiv 2511.20857 | Second testbed |
| AgentOdyssey | arXiv 2606.24893 | Primary testbed (5 abilities: exploration, world-knowledge, episodic memory, skill learning, planning) |
| Audited Skill-Graph Self-Improvement | arXiv 2512.23760 | Names infra as future work |
| RDumb++ | arXiv 2601.15544 | Drift-detection metric |
| Shumailov et al. | Nature 2024 | Model-collapse theory (analogy, not derivation) |
| Dohmatob et al. | ICLR 2025 | Strong Model Collapse, bias-variance decomposition |
| Shumailov critique | arXiv 2410.12954 | Qualifies Nature 2024 findings — addressed inline |
| Retrieval Collapses (web) | WWW 2026 / arXiv 2602.16136 | Differentiated near-neighbor |
| Control-Plane Placement (memory forgetting) | arXiv 2606.15903 | Closest prior art — read in full, differentiated (passive/externally-edited memory vs. our active self-feedback setting) |

Classic baselines to cite in related work: EWC, Tent/CoTTA.

---

## Open items (not yet locked)

- Base agent model choice (open-weight, Lambda-hosted; gates rollout budget)
- Testbed integration specifics (AgentOdyssey/Evo-Memory adapter details)
- Method chapter completion (compute budget, episode/round definition)
- Results, Discussion, Conclusion chapters
