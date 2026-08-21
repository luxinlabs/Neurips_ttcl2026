# Paper Review: The Retrieval Noise Taxonomy: Disentangling Stochastic, Systematic, and Selection Effects on Self-Improvement Drift in Test-Time Continual Agents

## Paper Metadata

- **Authors**: Luxin Zhang (sole author)
- **Venue**: NeurIPS 2026 Workshop, Towards Test-Time Continual Learning Agents (TTCL), General Research Track
- **Year**: 2026
- **Domain**: Machine learning — test-time continual learning, agent memory systems
- **Paper Type**: Position paper (self-declared), with an unusually detailed experimental-methods section attached

## Executive Summary

This paper argues that retrieval infrastructure serving an agent's memory is not a neutral component but at least two mechanistically distinct noise sources — approximate-search ("stochastic") and index-staleness ("systematic") — that may drive a self-improving agent's test-time drift in opposite directions. It proposes a matched-replay causal ablation to test this, specified in full (grid, sample size, analysis plan, falsification criteria) before any run, and releases a tested diagnostic harness (`driftbench`) implementing the taxonomy over FAISS.

The conceptual contribution is genuine: the taxonomy fills a real, citably-demonstrated gap (a companion paper in the same space explicitly names "infrastructure-level factors" as unaddressed future work), and the causal method is more carefully reasoned than most workshop submissions — the confound analysis, the factorial-vs-one-at-a-time design justification, and the pre-registered falsification criteria all show real methodological maturity. The harness is real, public, and unit-tested, which is unusual and commendable for a paper with zero empirical results.

That said, my overall assessment is **borderline, leaning toward reject**, and the reasoning matters more than the label. The paper's problem is not that it lacks data — the CFP explicitly welcomes position papers, and I take that framing at face value rather than penalizing it reflexively. The problem is that roughly two-thirds of the paper (Section 3 in its entirety) is not written as a position-paper argument; it is written as the Methods section of an empirical paper that has not yet produced results. A position paper should be evaluated on the strength of its argument and evidence synthesis; this paper spends most of its length on experimental-design detail that cannot itself be evaluated (sample size adequacy, whether the statistical model is even correctly specified — see W2 below) until the experiment actually runs. That is a genre mismatch, not just a completeness gap, and it is the central tension a reviewer has to resolve.

## Summary of Contributions

1. A named taxonomy of retrieval-infrastructure noise (stochastic, systematic, and a secondary selection channel) with formal, checkable definitions.
2. A matched-replay causal ablation method that isolates infrastructure from task-difficulty confounding by construction, with a pre-registered analysis plan and falsification criteria.
3. A released, unit-tested diagnostic harness (`driftbench`) implementing all three channels.

## Strengths

### S1: The taxonomy closes a citably real gap, not an invented one
Section 2 doesn't just assert novelty — it quotes a companion paper (Huang & Huang, the audited skill-graph framework) explicitly naming "infrastructure-level factors (caching, retrieval freshness, memory management)" as unaddressed future work, and shows that the paper anchoring the field's drift phenomenon (SkillLearnBench) holds retrieval infrastructure fixed and unexamined throughout its own design. This is evidence-based gap identification, not the generic "prior work hasn't considered X" move that pads most related-work sections.

### S2: The matched-replay confound argument is genuinely rigorous
Section 3.1's reasoning — that harder tasks plausibly force smaller budgets, staler indices, and more aggressive eviction simultaneously, so a naive infra-vs-drift correlation could just be a difficulty confound — is exactly the objection a sharp reviewer would raise, and the paper pre-empts it by construction (same starting instance across every condition) rather than by promising a post-hoc statistical adjustment. Section 3.2's justification for a 2-way factorial specifically over the two headline channels (to estimate the interaction term needed to defend "opposite-signed" against that same confound) is likewise the right design choice for the right stated reason, not a default.

### S3: The harness is real and independently verifiable, which most workshop submissions — position paper or not — do not offer
Forty passing unit tests covering the three noise-channel wrappers, the grid builder, the matched-replay orchestrator, and the recall-vs-proxy checkpointing (Section 3.6) is a substantive artifact contribution independent of whether the eventual experiment confirms the hypothesis. I was able to independently verify (via the accompanying repository) that this claim is accurate as stated, not aspirational.

## Weaknesses

### W1: The paper is the wrong genre for a position paper (Critical)
A position paper is fundamentally an argument-and-synthesis document. This paper's actual argumentative content — the taxonomy, the two motivating analogies (model-collapse for the systematic channel, retrieval-robustness noise-type findings for the stochastic channel), and the practical reframe in Section 5 — comprises roughly a third of the paper's length. The remaining two-thirds (all of Section 3, half of Section 4) is written in the register and level of procedural detail of an empirical paper's Methods section: exact grid levels, exact sample size, exact base model and inference stack, exact step-count-vs-wall-clock design decision. None of that procedural detail can be *evaluated* by a reader in the way position-paper content can — a reader can agree or disagree that the taxonomy is a useful distinction, but cannot meaningfully assess whether N=15–20 is the right sample size, or whether the mixed-effects model is correctly specified (see W2), without data to check it against. The paper reads like an empirical submission that was reframed as a position paper after the fact rather than conceived as one — which, per the paper's own Round 1 peer-review history, is exactly what happened. That is a legitimate strategy, but it does not automatically produce a strong position paper; it produces a position paper with an oversized, unevaluable appendix bolted to the front of it.

### W2: The mixed-effects design has an unaddressed statistical confusion, not just an acknowledged limitation (Major)
Section 3.4 states that "sampling variance is nested inside that same instance-level term regardless of which grid cell produced it, so it inflates the residual rather than biasing the interaction estimate." This is imprecise in a way that matters. With exactly one rollout per (task instance × condition) cell — which is what the design in Section 3.2 specifies — there is no way to separate within-cell sampling variance from between-instance heterogeneity at all, because there is only one observation per cell to estimate both quantities from. It is not that sampling variance "inflates the residual" in a benign, already-modeled way; it is that the residual and the instance-level random effect are *not separately identifiable* under N=1 replicate per cell without additional assumptions the text doesn't state. The paper's own hedge ("a reviewer could reasonably ask for multiple LLM samples per condition... a natural extension") correctly identifies the fix but understates the problem it fixes: this isn't an optional robustness check, it's arguably required for the mixed-effects model as described to be estimable in the way Section 3.3 claims. I'd want to see either (a) an explicit statement of how the model handles this (e.g., treating sampling noise as fully absorbed into an assumed-known measurement-error term, with justification), or (b) the multi-sample design promoted from "natural extension" to part of the core method.

### W3: Selection noise's billing is inconsistent between title and body (Major)
The paper's title gives "Stochastic, Systematic, and Selection" grammatically equal weight. The body immediately and repeatedly demotes Selection to secondary status — "kept secondary throughout this paper," "appendix-tier ablation," "a metric with no precedent in prior work." A reader who stops at the title reasonably expects three co-equal contributions; a reader who finishes Section 2 learns the paper is really about two. This isn't fatal, but it's a real expectation-setting problem a title should not create, and it is entirely fixable without touching the taxonomy itself.

### W4: The related-work search does not appear to have been systematic, and the paper's own phrasing admits it
Section 2 describes Yang's control-plane-placement paper as "the closest work to the present paper found in any literature search conducted for it" — phrasing that, perhaps unintentionally, reveals the search process was iterative and ad hoc rather than a defined systematic search (fixed query set, defined databases, inclusion/exclusion criteria). All 18 references are ML-preprint-flavored (arXiv, ICLR, CVPR, COLM, one Nature paper); there is no evidence of a search against systems/database venues (VLDB, SIGMOD, OSDI, SOSP) on stale-index serving or approximate-index correctness tradeoffs, despite the paper's own framing as bridging ML and systems concerns. Given the taxonomy borrows two of its three metrics from systems literature already, that venue gap is a real hole in "closest prior art" claims, not just a stylistic nitpick.

## Methodology Assessment

| Criterion | Rating (1-5) | Assessment |
|-----------|:---:|------------|
| Soundness | 3 | The argument structure is internally consistent and the confound-handling logic (W1's procedural-detail aside) is sound. Docked for W2 — a real, not cosmetic, gap in the statistical design as specified. |
| Novelty | 3 | The individual metrics (recall@k, staleness, FIFO/LRU/importance eviction) are all pre-existing; the novel contribution is the taxonomy's *framing* — decomposing "retrieval quality" into causally distinguishable channels for self-feedback settings specifically — and the matched-replay isolation method applied to this problem. Real but incremental, not a new primitive. |
| Reproducibility | 3 | The harness code is public, tested, and (independently verified) accurate to its own description — genuinely good practice. But the actual experiment the paper is "about" cannot be run end-to-end by anyone right now, including the author: the testbed adapters are explicit `NotImplementedError` stubs. Infrastructure reproducibility and claim reproducibility are not the same thing, and only the former currently exists. |
| Experimental Design | 4 | The strongest dimension. Matched-replay, factorial-for-interaction/OFAT-for-secondary-channel, and pre-registered falsification criteria (including the non-obvious and correct call that a null interaction term is *not* falsifying) reflect real methodological sophistication. Docked one point for W2. |
| Statistical Rigor | 2 | Honest about the absence of a power calculation, which I credit — but W2 is a design-specification problem, not merely an unvalidated-until-data-exists problem, and that pulls this rating down further than the paper's own self-assessment acknowledges. |
| Scalability | 3 | The ~400-rollout budget is concretely bounded and justified against a real constraint (AgentOdyssey's own finding that long-context agents scale quadratically). No discussion of whether the taxonomy or method generalizes past the two testbeds or past 4–8B-parameter models. |

## Questions for the Authors

1. Under the design in Section 3.2 (one rollout per instance × condition cell), how does the mixed-effects model in Section 3.3 separately identify sampling variance from instance-level heterogeneity? If it doesn't, what assumption is being made instead, and where should that assumption be stated?
2. If Selection noise is secondary and appendix-tier throughout the body, would the paper's core claim survive dropping it from the title entirely, or is there a reason it needs top billing despite the demotion?
3. What, specifically, makes this paper's contribution time-sensitive enough to publish now, in this genre, rather than waiting for the (already-scoped, already-implemented-as-a-harness) experiment to actually run and submitting as a standard empirical paper? I ask this not rhetorically — the CFP explicitly welcomes position papers, so there may be a good answer (e.g., claiming priority on the taxonomy before a concurrent group publishes something similar) — but the paper doesn't state one, and it would strengthen the position-paper framing considerably if it did.

## Minor Issues

- No figures or diagrams anywhere in the paper. Even without data, a schematic of the three-channel taxonomy or the 3×3 experimental grid would substantially improve legibility and is achievable without any experimental results.
- "found in any literature search conducted for it" (Section 2) — informal, slightly self-undermining phrasing; standard "to our knowledge" phrasing would read more carefully to a reviewer without changing the underlying (accurate, appropriately hedged) claim.
- The AI Usage Disclosure is thorough and appropriately specific, which I credit, but its level of detail (drafting, harness implementation, an "internal simulated peer-review pass") may draw more scrutiny at a venue with less settled norms around AI-assisted authorship than NeurIPS currently has. Not a defect in the paper as submitted, but worth the author's awareness going in.
- Citation numbering/keying is currently by short mnemonic tag in the Markdown source (e.g., `[SkillLearnBench]`); confirm this resolves correctly to the venue's expected in-text citation style in the final compiled version.

## Literature Positioning

The paper positions itself carefully and mostly fairly against the three literatures it does engage with — SkillLearnBench (drift phenomenon), the memory-systems-as-correctness thread (Omri et al., Yang), and the terminologically-adjacent collapse literatures (web-scale retrieval collapse, model collapse) — with specific, checkable differentiation claims rather than hand-waving. The synthesis paragraph closing Section 2 (assumed-away / collapsed-into-correctness / wrong-scale) is a clean, accurate three-way parallel that actually matches what the preceding paragraphs established, which is not always true of related-work syntheses. The gap, per W4, is coverage breadth: the search appears confined to ML-preprint venues and does not show evidence of having checked systems/database literature on the same underlying mechanisms (staleness, approximate indexing) the taxonomy borrows metrics from.

## Recommendations

**Overall Assessment**: Borderline (leaning Weak Reject in current form; would move to Weak Accept with the W1/W2 fixes below)

**Confidence**: Medium-High — I have verified the harness's factual claims against the actual repository and independently re-checked every citation and the venue's formatting requirements, so my assessment of what's *stated accurately* is high-confidence. My assessment of the statistical design gap (W2) is based on standard mixed-effects-model identifiability reasoning and I'm confident in it, but I have not run the analysis myself since no data exists yet.

**Contribution Level**: Moderate — a real, evidence-grounded conceptual contribution (the taxonomy and the causal method) undercut by a genre mismatch (W1) and one genuine statistical design gap (W2) rather than by lack of data per se.

### Actionable Suggestions for Improvement

1. **Resolve the genre mismatch (W1)** — either compress Section 3's procedural detail substantially and foreground the argumentative content (making this a leaner, more genuinely position-paper-shaped submission), or be explicit in the Introduction about why the full experimental specification belongs in a position paper for this venue (see Question 3).
2. **Fix or explicitly own W2** — either state the identifiability assumption the current single-replicate design relies on, or promote the multi-sample-per-cell design from "natural extension" to the paper's actual specified method, since the harness can now support it directly (see accompanying code update).
3. **Resolve the title/body inconsistency on Selection noise (W3)** — either soften the title's equal billing or give Selection a genuine, non-secondary role; the current split-the-difference framing invites exactly the kind of "your title oversells" critique a stricter venue would flag harder than a workshop will.
4. **Broaden the related-work search to systems venues (W4)** before claiming "closest work... found in any literature search conducted for it" — either confirm no closer systems-literature match exists, or cite what does.
5. **Add at least one schematic figure** — the taxonomy and the experimental grid are both natural candidates and require no experimental results to produce.
