import importlib.util
from pathlib import Path


def load_autopilot_module():
    path = Path(__file__).parents[1] / "scripts" / "autopilot.py"
    spec = importlib.util.spec_from_file_location("tianchi_autopilot", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_champion_requires_strict_improvement():
    module = load_autopilot_module()
    assert module.is_strict_improvement(0.721, 0.7209)
    assert not module.is_strict_improvement(0.7209, 0.7209)
    assert not module.is_strict_improvement(0.720900000000001, 0.7209)
