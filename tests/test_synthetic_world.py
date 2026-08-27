from driftbench.agent.episode_loop import run_self_feedback_episode
from driftbench.agent.fakes import FakeEmbedder, FakeLLM
from driftbench.index.approx_index import ApproxIndexConfig
from driftbench.index.noisy_memory import NoisyMemoryStore
from driftbench.index.shadow_index import ShadowIndexConfig
from testbeds.synthetic_world import MemoryGatedPolicy, SyntheticWorldAdapter, lore_notes


def _fresh_memory(dim: int) -> NoisyMemoryStore:
    return NoisyMemoryStore(
        dim=dim,
        stochastic=ApproxIndexConfig(kind="flat"),
        systematic=ShadowIndexConfig(rebuild_cadence=1),
    )


def test_lore_notes_are_deterministic_and_instance_specific():
    a = lore_notes("synth-000", 5)
    b = lore_notes("synth-000", 5)
    c = lore_notes("synth-001", 5)
    assert a == b
    assert a != c
    assert len(a) == 5


def test_reset_is_deterministic_per_instance():
    a = SyntheticWorldAdapter(n_steps=8)
    b = SyntheticWorldAdapter(n_steps=8)
    obs_a = a.reset("synth-001")
    obs_b = b.reset("synth-001")
    assert obs_a == obs_b
    assert a.old_good == b.old_good
    assert a.new_good == b.new_good
    assert a.old_good != a.new_good


def test_revision_changes_current_good_and_recall_probe():
    adapter = SyntheticWorldAdapter(n_steps=6, revision_at=2)
    adapter.reset("synth-000")
    assert adapter.recall_probe()[1] == adapter.old_good
    adapter.step("offer nothing")
    second = adapter.step("offer nothing")  # delivers the UPDATE observation
    assert "UPDATE" in second.observation
    assert adapter.current_good == adapter.new_good
    assert adapter.recall_probe()[1] == adapter.new_good


def test_action_scoring_requires_the_current_good():
    adapter = SyntheticWorldAdapter(n_steps=4, revision_at=1)
    adapter.reset("synth-002")
    before = adapter.step(f"offer {adapter.old_good} in trade")
    assert before.success is True
    after = adapter.step(f"offer {adapter.old_good} in trade")  # now stale
    assert after.success is False
    correct = adapter.step(f"offer {adapter.new_good} in trade")
    assert correct.success is True


def test_heuristic_policy_tracks_a_stated_revision():
    policy = MemoryGatedPolicy()
    prompt = (
        "Observation: You are at the Ward 3 merchant's stall and need ore.\n"
        "Relevant memory:\n"
        "- The Ward 3 merchant trades ore for silver.\n"
        "- UPDATE: the merchant in Ward 3 now trades ore for gold, not silver.\n"
        "Next action:"
    )
    action = policy.generate(prompt)
    assert "gold" in action
    assert "silver" not in action


def test_fresh_index_lets_heuristic_adapt_after_revision():
    dim = 16
    memory = _fresh_memory(dim)
    llm = MemoryGatedPolicy()
    embedder = FakeEmbedder(dim=dim)
    adapter = SyntheticWorldAdapter(n_steps=8, revision_at=3)

    trace = run_self_feedback_episode(
        instance_id="synth-010",
        adapter=adapter,
        memory=memory,
        llm=llm,
        embedder=embedder,
        n_steps=8,
    )
    assert trace.checkpoints[-1].proxy_success > 0.0
    # After revision, a fresh index should make the new good retrievable,
    # so closed-book recall of the current good should succeed at least once.
    assert any(cp.recall_score == 1.0 for cp in trace.checkpoints)


def test_episode_recall_prompt_includes_transcript_not_just_probe():
    dim = 8
    memory = _fresh_memory(dim)
    llm = FakeLLM(default_response="act")
    embedder = FakeEmbedder(dim=dim)
    adapter = SyntheticWorldAdapter(n_steps=1, revision_at=99)

    calls: list[str] = []
    orig = llm.generate

    def wrapped(prompt: str) -> str:
        calls.append(prompt)
        return orig(prompt)

    llm.generate = wrapped
    run_self_feedback_episode(
        instance_id="synth-000", adapter=adapter, memory=memory, llm=llm, embedder=embedder, n_steps=1
    )
    recall_prompts = [c for c in calls if "Without retrieving from memory" in c]
    assert recall_prompts
    assert "Episode so far:" in recall_prompts[0]
