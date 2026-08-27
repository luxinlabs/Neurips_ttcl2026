"""Synthetic self-feedback world for the matched-replay experiment.

AgentOdyssey / Evo-Memory adapters are still stubs. This environment
implements the paper's illustrative scene as a runnable testbed: a merchant
trade rule is observed, later revised, and subsequent merchant queries do not
restate the answer — so retrieved (possibly stale or approximate) memory is
load-bearing.

Each `instance_id` deterministically picks the old/new goods and distractor
facts, which is what matched-replay needs: the same starting instance across
every infrastructure condition.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from driftbench.agent.interfaces import StepResult

GOODS = ("silver", "gold", "copper", "jade", "silk", "iron", "pearl", "amber")
DISTRACTORS = (
    ("the baker in Ward 1", "sells rye bread"),
    ("the guard in Ward 2", "accepts the blue token"),
    ("the scribe in Ward 4", "records debts in ink"),
    ("the fisher in Ward 5", "trades trout for salt"),
    ("the smith in Ward 6", "repairs iron tools"),
    ("the herbalist in Ward 7", "brews willow tea"),
    ("the cartwright in Ward 8", "rents oak wagons"),
    ("the miner in Ward 9", "hauls coal at dawn"),
    ("the weaver in Ward 1", "dyes cloth crimson"),
    ("the innkeep in Ward 2", "boards travellers overnight"),
    ("the priest in Ward 4", "blesses river crossings"),
    ("the scout in Ward 5", "marks trails with chalk"),
)


def _rng_seed(instance_id: str) -> int:
    return int(hashlib.sha256(instance_id.encode()).hexdigest()[:8], 16)


def lore_notes(instance_id: str, n: int) -> list[str]:
    """Deterministic distractor notes used to pre-fill memory.

    The v1 factorial left recall@k at 1.0 because HNSW over ~16 episode
    writes is effectively exact. Seeding a larger lore set is what lets the
    stochastic channel actually miss neighbours, without changing the matched
    starting instance.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    seed = _rng_seed(instance_id)
    notes = []
    for i in range(n):
        who, what = DISTRACTORS[(seed + i) % len(DISTRACTORS)]
        notes.append(f"City lore {i}: {who} {what} (record {i}).")
    return notes


@dataclass
class SyntheticWorldAdapter:
    """N-step world with one revising fact plus stable distractors.

    Merchant-facing steps are scored on whether the action names the *current*
    trade good. Distractor steps always succeed so they fill memory without
    washing out the merchant-rule signal the taxonomy is about.
    """

    n_steps: int = 24
    revision_at: int | None = None
    n_distractors: int = 8

    _step: int = field(default=0, init=False)
    _instance_id: str = field(default="", init=False)
    _phase: str = field(default="merchant", init=False)
    old_good: str = field(default="silver", init=False)
    new_good: str = field(default="gold", init=False)
    distractors: list[tuple[str, str]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.revision_at is None:
            self.revision_at = max(2, self.n_steps // 3)

    def reset(self, instance_id: str) -> str:
        self._instance_id = instance_id
        self._step = 0
        self._phase = "merchant"
        seed = _rng_seed(instance_id)
        old_idx = seed % len(GOODS)
        new_idx = (seed // len(GOODS) + 1) % len(GOODS)
        if new_idx == old_idx:
            new_idx = (old_idx + 1) % len(GOODS)
        self.old_good = GOODS[old_idx]
        self.new_good = GOODS[new_idx]
        start = (seed // 17) % len(DISTRACTORS)
        self.distractors = [
            DISTRACTORS[(start + i) % len(DISTRACTORS)] for i in range(self.n_distractors)
        ]
        return (
            "Reply with one short line: offer <good> in trade for ore. "
            f"The merchant in Ward 3 trades ore for {self.old_good}."
        )

    @property
    def current_good(self) -> str:
        assert self.revision_at is not None
        return self.new_good if self._step >= self.revision_at else self.old_good

    def step(self, action: str) -> StepResult:
        success = True if self._phase == "distractor" else self._score_merchant(action)
        self._step += 1
        done = self._step >= self.n_steps
        observation = self._observation()
        return StepResult(observation=observation, success=success, done=done)

    def recall_probe(self) -> tuple[str, str]:
        return (
            "What does the Ward 3 merchant currently trade ore for?",
            self.current_good,
        )

    def _score_merchant(self, action: str) -> bool:
        text = action.lower()
        current = self.current_good
        other = self.new_good if current == self.old_good else self.old_good
        mentions_current = current in text
        mentions_other = other in text
        if mentions_current and not mentions_other:
            return True
        if mentions_current and mentions_other:
            return text.rfind(current) >= text.rfind(other)
        return False

    def _observation(self) -> str:
        assert self.revision_at is not None
        if self._step == self.revision_at:
            self._phase = "merchant"
            return (
                f"UPDATE: the merchant in Ward 3 now trades ore for {self.new_good}, "
                f"not {self.old_good}. Reply: offer <good> in trade for ore."
            )
        if self._step % 2 == 1 and self.distractors:
            self._phase = "distractor"
            who, what = self.distractors[(self._step // 2) % len(self.distractors)]
            return f"You learn: {who} {what}. Acknowledge in one short sentence."
        self._phase = "merchant"
        return (
            "You are at the Ward 3 merchant's stall and need ore. "
            "Reply: offer <good> in trade for ore."
        )


class MemoryGatedPolicy:
    """Deterministic policy that follows retrieved notes, then the observation.

    Used by `--backend heuristic` to exercise the full matched-replay loop
    without a GPU model. A real MLX LLM is the experiment proper.
    """

    def generate(self, prompt: str) -> str:
        if "Without retrieving from memory" in prompt or "Without looking anything up" in prompt:
            return self._answer_probe(prompt)
        if "write a memory note" in prompt.lower():
            return self._reflect(prompt)
        return self._act(prompt)

    def _act(self, prompt: str) -> str:
        good = self._latest_merchant_good(prompt)
        if good:
            return f"offer {good} in trade for ore"
        who, what = self._latest_distractor(prompt)
        if who:
            return f"remember that {who} {what}"
        return "look around"

    def _reflect(self, prompt: str) -> str:
        m = re.search(r"Result:\s*(.*)", prompt)
        result = m.group(1).strip() if m else prompt[-200:]
        good = self._latest_merchant_good(prompt)
        if good:
            return f"The Ward 3 merchant trades ore for {good}."
        return f"Noted: {result}"

    def _answer_probe(self, prompt: str) -> str:
        good = self._latest_merchant_good(prompt)
        return good if good else "unknown"

    def _latest_merchant_good(self, prompt: str) -> str | None:
        matches = re.findall(
            r"(?:now )?trades ore for ([a-z]+)",
            prompt.lower(),
        )
        return matches[-1] if matches else None

    def _latest_distractor(self, prompt: str) -> tuple[str, str]:
        for who, what in DISTRACTORS:
            if who.lower() in prompt.lower():
                return who, what
        return "", ""
