from driftbench.agent.episode_loop import run_self_feedback_episode
from driftbench.agent.fakes import FakeEmbedder, FakeLLM, FakeTestbedAdapter
from driftbench.index.approx_index import ApproxIndexConfig
from driftbench.index.noisy_memory import NoisyMemoryStore
from driftbench.index.shadow_index import ShadowIndexConfig


def _fresh_baseline_memory(dim: int) -> NoisyMemoryStore:
    return NoisyMemoryStore(
        dim=dim,
        stochastic=ApproxIndexConfig(kind="flat"),
        systematic=ShadowIndexConfig(rebuild_cadence=1),
    )


def test_episode_runs_and_checkpoints_every_step():
    dim = 16
    memory = _fresh_baseline_memory(dim)
    llm = FakeLLM(default_response="trade ore")
    embedder = FakeEmbedder(dim=dim)
    adapter = FakeTestbedAdapter(n_steps=3)

    trace = run_self_feedback_episode(
        instance_id="task-0",
        adapter=adapter,
        memory=memory,
        llm=llm,
        embedder=embedder,
        n_steps=5,  # loop should still stop at adapter's n_steps=3 via done
        checkpoint_every=1,
    )

    assert trace.task_instance_id == "task-0"
    assert len(trace.checkpoints) == 3
    assert [cp.episode for cp in trace.checkpoints] == [0, 1, 2]


def test_recall_probe_answered_correctly_scores_one():
    dim = 8
    memory = _fresh_baseline_memory(dim)
    llm = FakeLLM(
        default_response="act",
        keyed_responses={"What trades ore for silver?": "the merchant runs the trade"},
    )
    embedder = FakeEmbedder(dim=dim)
    adapter = FakeTestbedAdapter(n_steps=1, probe_question="What trades ore for silver?", probe_answer="the merchant")

    trace = run_self_feedback_episode(
        instance_id="task-0", adapter=adapter, memory=memory, llm=llm, embedder=embedder, n_steps=1
    )

    assert trace.checkpoints[0].recall_score == 1.0


def test_recall_probe_answered_incorrectly_scores_zero():
    dim = 8
    memory = _fresh_baseline_memory(dim)
    llm = FakeLLM(default_response="act", keyed_responses={"What trades ore for silver?": "no idea"})
    embedder = FakeEmbedder(dim=dim)
    adapter = FakeTestbedAdapter(n_steps=1, probe_question="What trades ore for silver?", probe_answer="the merchant")

    trace = run_self_feedback_episode(
        instance_id="task-0", adapter=adapter, memory=memory, llm=llm, embedder=embedder, n_steps=1
    )

    assert trace.checkpoints[0].recall_score == 0.0


def test_self_generated_feedback_is_written_to_memory_and_later_retrievable():
    dim = 16
    memory = _fresh_baseline_memory(dim)
    llm = FakeLLM(default_response="act")
    embedder = FakeEmbedder(dim=dim)
    adapter = FakeTestbedAdapter(n_steps=2)

    run_self_feedback_episode(
        instance_id="task-0", adapter=adapter, memory=memory, llm=llm, embedder=embedder, n_steps=2
    )

    # Every step writes one self-generated feedback note; after 2 steps the
    # memory should hold 2 payloads, proving the reflect->write leg fired.
    report = memory.retrieve(embedder.embed_one("start of task-0"), k=5)
    assert len(report.retrieved_ids) >= 1
    assert all(payload != "" for payload in report.retrieved_payloads)
