import pytest
from tests.test_binary import create_sample_model
from intellibranch.runtime import MAX_INPUT_BYTES, compute_entropy, truncate_to_rune_boundary


def test_predict_and_forward():
    model = create_sample_model()
    token_ids = [1, 2]
    probs = model.forward(token_ids)
    assert len(probs) == model.header.num_classes
    assert abs(sum(probs) - 1.0) < 1e-4

    slots = model.predict_slots(token_ids)
    assert slots.total >= 1
    assert slots.primary.index in (0, 1)


def test_truncate_to_rune_boundary():
    text = "Hello world! This is a long query."
    truncated = truncate_to_rune_boundary(text, 10)
    assert len(truncated.encode("utf-8")) <= 10

    # Test multibyte korean
    korean_text = "안녕하세요 반갑습니다."
    truncated_ko = truncate_to_rune_boundary(korean_text, 8)
    assert len(truncated_ko.encode("utf-8")) <= 8
    # Should decode cleanly
    assert isinstance(truncated_ko, str)


def test_compute_entropy():
    import numpy as np
    probs = np.array([0.5, 0.5], dtype=np.float32)
    ent = compute_entropy(probs)
    assert abs(ent - 1.0) < 1e-4
