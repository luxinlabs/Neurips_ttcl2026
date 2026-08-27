"""Run the matched-replay factorial on GPU (MLX Metal + embeddings on MPS).

AgentOdyssey / Evo-Memory adapters are still stubs, so this uses the synthetic
revising-fact world in testbeds/synthetic_world.py — same matched-replay
orchestrator, same 3x3 stochastic x systematic grid, same SeqMem checkpoints.

Examples:
  .venv/bin/python -m experiments.run_experiment --smoke
  .venv/bin/python -m experiments.run_experiment --backend mlx
  .venv/bin/python -m experiments.run_experiment --backend heuristic --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from driftbench.agent.episode_loop import run_self_feedback_episode
from driftbench.config import ConditionSpec, ExperimentGrid, build_grid
from driftbench.eval.seqmem_protocol import SeqMemTrace
from driftbench.index.noisy_memory import NoisyMemoryStore
from driftbench.replay.matched_replay import ReplayResult, run_matched_replay
from testbeds.synthetic_world import MemoryGatedPolicy, SyntheticWorldAdapter, lore_notes

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRID = ROOT / "configs" / "grid_default.yaml"
DEFAULT_OUT = ROOT / "results" / "matched_replay.jsonl"

# 4-bit Llama-3.1-8B on Metal — the paper's INT4 fallback, since this machine
# is Apple GPU (no CUDA / vLLM). HuggingFace id may require a license grant.
DEFAULT_MLX_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
FALLBACK_MLX_MODELS = (
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
)


@dataclass
class TaskInstance:
    instance_id: str


def _load_llm(backend: str, model: str, max_tokens: int, temperature: float):
    if backend == "heuristic":
        print("LLM backend: heuristic (CPU, deterministic note-follower)", flush=True)
        return MemoryGatedPolicy(), {"backend": "heuristic"}
    if backend == "mlx":
        from driftbench.agent.mlx_client import MLXClient

        tried = [model, *[m for m in FALLBACK_MLX_MODELS if m != model]]
        last_err: Exception | None = None
        for candidate in tried:
            try:
                print(f"Loading MLX model {candidate} on Apple GPU...", flush=True)
                client = MLXClient(
                    model=candidate, max_tokens=max_tokens, temperature=temperature
                )
                print(f"MLX device: {client.device}", flush=True)
                return client, client.info()
            except Exception as exc:  # noqa: BLE001 — try the next public weight
                last_err = exc
                print(f"  failed to load {candidate}: {exc}", flush=True)
        raise RuntimeError(f"could not load any MLX model; last error: {last_err}")
    raise ValueError(f"unknown backend: {backend}")


def _load_embedder(backend: str, embed_model: str):
    if backend == "heuristic":
        from driftbench.agent.fakes import FakeEmbedder

        embedder = FakeEmbedder(dim=32)
        print("Embedder: FakeEmbedder dim=32 (hash, CPU)", flush=True)
        return embedder, {"backend": "fake", "dim": embedder.dim}
    from driftbench.agent.embedding import BGEEmbedder

    print(f"Loading embedder {embed_model}...", flush=True)
    embedder = BGEEmbedder(model_name=embed_model)
    print(f"Embedder device: {embedder.device} dim={embedder.dim}", flush=True)
    return embedder, {
        "backend": "bge",
        "model": embed_model,
        "device": embedder.device,
        "dim": embedder.dim,
    }


def _summarize(results: list[ReplayResult]) -> list[dict]:
    grouped: dict[str, list[ReplayResult]] = defaultdict(list)
    for r in results:
        grouped[r.condition_name].append(r)

    rows = []
    for name, items in grouped.items():
        terminals = [t.trace.checkpoints[-1] for t in items if t.trace.checkpoints]
        n = len(terminals)
        if n == 0:
            continue

        def mean(xs: list[float]) -> float:
            return sum(xs) / len(xs)

        def se(xs: list[float]) -> float:
            if len(xs) < 2:
                return 0.0
            m = mean(xs)
            var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
            return var**0.5 / (len(xs) ** 0.5)

        proxy = [c.proxy_success for c in terminals]
        recall = [c.recall_score for c in terminals]
        drift = [c.drift_kl for c in terminals]
        stale = [c.staleness for c in terminals]
        r_at_k = [c.recall_at_k for c in terminals]
        rows.append(
            {
                "condition": name,
                "n": n,
                "proxy_success_mean": mean(proxy),
                "proxy_success_se": se(proxy),
                "recall_mean": mean(recall),
                "recall_se": se(recall),
                "drift_kl_mean": mean(drift),
                "drift_kl_se": se(drift),
                "staleness_mean": mean(stale),
                "recall_at_k_mean": mean(r_at_k),
            }
        )
    rows.sort(key=lambda r: r["condition"])
    return rows


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("No results.")
        return
    headers = (
        "condition",
        "n",
        "proxy",
        "recall",
        "drift_kl",
        "staleness",
        "recall@k",
    )
    print()
    print(f"{headers[0]:<42} {headers[1]:>4} {headers[2]:>8} {headers[3]:>8} {headers[4]:>8} {headers[5]:>9} {headers[6]:>8}")
    print("-" * 96)
    for row in rows:
        print(
            f"{row['condition']:<42} {row['n']:>4} "
            f"{row['proxy_success_mean']:>8.3f} {row['recall_mean']:>8.3f} "
            f"{row['drift_kl_mean']:>8.3f} {row['staleness_mean']:>9.2f} "
            f"{row['recall_at_k_mean']:>8.3f}"
        )


def _write_jsonl(path: Path, results: list[ReplayResult], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"type": "meta", **meta}) + "\n")
        for r in results:
            rec = {
                "type": "result",
                "condition_name": r.condition_name,
                "instance_id": r.instance_id,
                "replicate": r.replicate,
                "checkpoints": [asdict(c) for c in r.trace.checkpoints],
            }
            f.write(json.dumps(rec) + "\n")
        f.write(json.dumps({"type": "summary", "rows": _summarize(results)}) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("mlx", "heuristic"), default="mlx")
    p.add_argument("--model", default=DEFAULT_MLX_MODEL)
    p.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--smoke", action="store_true", help="tiny grid: 3 instances, 12 steps, 1 replicate, no selection")
    p.add_argument("--n-instances", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--include-selection", action="store_true", default=None)
    p.add_argument("--no-selection", action="store_true")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--k-retrieve", type=int, default=4)
    p.add_argument(
        "--seed-notes",
        type=int,
        default=0,
        help="pre-fill memory with this many instance-matched lore notes so ANN can miss",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw = __import__("yaml").safe_load(args.grid.read_text())
    n_instances = args.n_instances
    n_samples = args.n_samples
    n_steps = args.n_steps
    include_selection = True
    if args.no_selection:
        include_selection = False
    elif args.include_selection:
        include_selection = True

    if args.smoke:
        n_instances = n_instances or 3
        n_samples = n_samples or 1
        n_steps = n_steps or 12
        include_selection = False if args.include_selection is None and not args.no_selection else include_selection
    else:
        n_instances = n_instances or int(raw.get("n_instances_per_cell", 18))
        n_samples = n_samples or int(raw.get("n_samples_per_cell", 2))
        n_steps = n_steps or 24

    grid: ExperimentGrid = build_grid(args.grid)
    n_conditions = len(grid.factorial_cells) + (len(grid.selection_conditions) if include_selection else 0)
    n_rollouts = n_instances * n_conditions * n_samples
    n_oracle = n_instances if include_selection else 0
    n_total = n_rollouts + n_oracle
    print(
        f"Grid: {len(grid.factorial_cells)} factorial cells"
        f"{' + ' + str(len(grid.selection_conditions)) + ' selection' if include_selection else ''}"
        f" | {n_instances} instances x {n_samples} replicates = {n_rollouts} rollouts"
        f"{f' + {n_oracle} oracle' if n_oracle else ''}"
        f" x {n_steps} steps"
        f"{f' | seed_notes={args.seed_notes}' if args.seed_notes else ''}",
        flush=True,
    )

    llm, llm_info = _load_llm(args.backend, args.model, args.max_tokens, args.temperature)
    embedder, emb_info = _load_embedder(args.backend, args.embed_model)
    dim = embedder.dim
    instances = [TaskInstance(f"synth-{i:03d}") for i in range(n_instances)]
    started = time.time()
    done_count = 0

    def episode_runner(
        instance: TaskInstance,
        condition: ConditionSpec,
        store: NoisyMemoryStore,
        replicate: int,
    ) -> SeqMemTrace:
        nonlocal done_count
        seed = (_rng_seed(instance.instance_id) ^ hash(condition.name) ^ (replicate * 1_000_003)) & 0xFFFFFFFF
        if hasattr(llm, "set_seed"):
            llm.set_seed(seed)
        adapter = SyntheticWorldAdapter(n_steps=n_steps)
        if args.seed_notes:
            for note in lore_notes(instance.instance_id, args.seed_notes):
                store.write(embedder.embed_one(note), payload=note)
        t0 = time.time()
        trace = run_self_feedback_episode(
            instance_id=instance.instance_id,
            adapter=adapter,
            memory=store,
            llm=llm,
            embedder=embedder,
            n_steps=n_steps,
            k_retrieve=args.k_retrieve,
        )
        trace.condition_name = condition.name
        done_count += 1
        last = trace.checkpoints[-1] if trace.checkpoints else None
        extra = ""
        if last is not None:
            extra = (
                f" proxy={last.proxy_success:.2f} recall={last.recall_score:.2f}"
                f" stale={last.staleness} r@k={last.recall_at_k:.2f}"
            )
        print(
            f"[{done_count}/{n_total}] {instance.instance_id} {condition.name} "
            f"rep={replicate} {time.time() - t0:.1f}s{extra}",
            flush=True,
        )
        return trace

    results = run_matched_replay(
        instances=instances,
        grid=grid,
        episode_runner=episode_runner,
        dim=dim,
        include_selection=include_selection,
        n_samples_per_cell=n_samples,
        measure_retention=include_selection,
    )
    elapsed = time.time() - started
    rows = _summarize(results)
    _print_table(rows)

    meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": elapsed,
        "backend": args.backend,
        "llm": llm_info,
        "embedder": emb_info,
        "n_instances": n_instances,
        "n_steps": n_steps,
        "n_samples": n_samples,
        "include_selection": include_selection,
        "grid": str(args.grid),
        "n_rollouts": n_rollouts,
        "smoke": args.smoke,
        "seed_notes": args.seed_notes,
    }
    _write_jsonl(args.out, results, meta)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print(f"\nWrote {args.out} and {summary_path} in {elapsed:.1f}s", flush=True)
    return 0


def _rng_seed(instance_id: str) -> int:
    import hashlib

    return int(hashlib.sha256(instance_id.encode()).hexdigest()[:8], 16)


if __name__ == "__main__":
    sys.exit(main())
