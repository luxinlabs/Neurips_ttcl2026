"""These stubs aren't functional yet (see module docstrings for the pending
TODOs) — these tests only pin down two things: (1) they fail loudly rather
than silently pretending to work, and (2) they expose the method surface
episode_loop.py and matched_replay.py actually call, so a future real
implementation can't accidentally drop a required method."""
import pytest

from testbeds.agentodyssey_adapter import AgentOdysseyAdapter
from testbeds.evomemory_adapter import EvoMemoryAdapter


@pytest.mark.parametrize("adapter_cls", [AgentOdysseyAdapter, EvoMemoryAdapter])
def test_stub_adapters_raise_not_implemented_on_construction(adapter_cls):
    with pytest.raises(NotImplementedError):
        adapter_cls()


@pytest.mark.parametrize("adapter_cls", [AgentOdysseyAdapter, EvoMemoryAdapter])
def test_stub_adapters_expose_the_required_interface_methods(adapter_cls):
    required_methods = {"reset", "step", "recall_probe"}
    assert required_methods.issubset(set(dir(adapter_cls)))
