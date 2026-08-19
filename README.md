# The Retrieval Noise Taxonomy

Position paper + diagnostic harness for the NeurIPS 2026 workshop [Towards Test-Time Continual Learning Agents (TTCL)](https://ttcl-agents.github.io/), General Research Track.

**Thesis**: retrieval infrastructure serving an agent's memory is not a neutral storage layer. It introduces at least two mechanistically distinct kinds of noise — *stochastic* (approximate nearest-neighbor search) and *systematic* (index staleness) — that can pull a self-improving agent's test-time drift in opposite directions. A third, secondary channel, *selection* noise (memory eviction under a fixed budget), is also instrumented but treated as appendix-tier evidence. See `paper/draft.md` for the full argument.

## Status

This is a **position paper**: the taxonomy, the causal matched-replay method (with its analysis plan and falsification criteria specified in advance), and the released harness are the contribution. No experimental results exist yet — real integration against the AgentOdyssey and Evo-Memory testbeds is in progress. See the paper's Limitations section for the full, honest list of what's not yet done.

## Repository layout

```
paper/              Paper source
  draft.md            Working draft (Markdown), including the bilingual abstract
  paper.tex           NeurIPS-submission-format LaTeX (see paper/README.md to compile)
  references.bib      BibTeX bibliography, every entry independently verified

driftbench/          The diagnostic harness (the released artifact)
  index/                Three noise-channel wrappers over FAISS:
                           approx_index.py    - stochastic channel (recall@k vs. exact search)
                           shadow_index.py    - systematic channel (staleness / pending-write count)
                           eviction.py        - selection channel (budgeted memory, FIFO/LRU/importance)
                           noisy_memory.py    - composes all three into one memory store
  agent/                 Generic self-feedback episode loop (observe/retrieve/act/reflect/write),
                          injectable LLM/embedding/testbed interfaces, deterministic fakes for testing
  eval/                  SeqMem-Eval-style recall-vs-proxy checkpointing
  metrics/               RDumb++-style drift metric (KL divergence over episodes)
  replay/                Matched-replay orchestrator (same task instance across every grid condition)
  config.py              Experimental grid builder (3x3 factorial + selection one-at-a-time)

testbeds/            AgentOdyssey / Evo-Memory adapters — documented stubs, real integration pending
configs/             Default experimental grid (configs/grid_default.yaml)
tests/               40+ unit tests, all passing (see below)
PAPER_PLAN.md        Full Chapter Plan + INSIGHT collection from the Socratic planning process
```

## Running the tests

```
conda env create -f environment.yml
conda activate ttcl
python -m pytest tests/ -v
```

Note: `pip install faiss-cpu` fails on recent Python versions without `swig` installed
system-wide; the conda-based `environment.yml` avoids that entirely and is the verified
working path.

## Compiling the paper

See `paper/README.md`.

## License

Not yet specified.
