import pytest

from opinion_mining.anchor_config import AnchorV1Config


def test_anchor_v1_has_explicit_deterministic_enabled_fixes():
    config = AnchorV1Config()

    assert config.enabled_fixes == (
        "official_offsets",
        "implicit_opinion_sentinel",
        "multi_relation",
    )
    assert config.use_official_offsets is True
    assert config.use_implicit_opinion_sentinel is True
    assert config.preserve_multi_relation is True
    assert config.to_neural_config().relation_top_k == 2
    assert config.config_hash() == AnchorV1Config().config_hash()


def test_anchor_v1_rejects_unapproved_fix_names():
    with pytest.raises(ValueError, match="unsupported anchor fix"):
        AnchorV1Config(enabled_fixes=("threshold_search",))
