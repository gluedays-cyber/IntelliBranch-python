import math
import numpy as np
import pytest

from intellibranch.ops import (
    dot_product,
    gelu,
    gelu_array,
    l2_normalize,
    matmul_vec_add,
    mean_pooling,
    safe_clamp,
    softmax,
)


def test_gelu():
    test_cases = [
        (0.0, 0.0, 1e-5),
        (1.0, 0.841192, 1e-4),
        (-1.0, -0.158808, 1e-4),
        (2.0, 1.9545, 1e-3),
        (-2.0, -0.0454, 1e-3),
    ]
    for inp, expected, eps in test_cases:
        res = gelu(inp)
        assert abs(res - expected) <= eps, f"GELU({inp}) = {res}; expected {expected}"


def test_gelu_array():
    inputs = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    expected = np.array([0.0, 0.841192, -0.158808], dtype=np.float32)
    out = gelu_array(inputs)
    np.testing.assert_allclose(out, expected, atol=1e-4)


def test_softmax():
    logits = np.array([2.0, 1.0, 0.1], dtype=np.float32)
    probs = softmax(logits, 1.0)

    assert len(probs) == 3
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert abs(np.sum(probs) - 1.0) < 1e-5
    assert probs[0] > probs[1] > probs[2]


def test_mean_pooling():
    emb_dim = 4
    embedding_table = np.array([
        1.0, 2.0, 3.0, 4.0,  # token 0
        5.0, 6.0, 7.0, 8.0,  # token 1
    ], dtype=np.float32)

    token_ids = [0, 1]
    out = mean_pooling(token_ids, embedding_table, emb_dim)
    expected = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_matmul_vec_add():
    in_dim = 2
    out_dim = 3
    vec = np.array([1.0, 2.0], dtype=np.float32)
    weights = np.array([
        1.0, 0.5, 0.0,  # row 0
        0.0, 1.0, 2.0,  # row 1
    ], dtype=np.float32)
    bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    out = matmul_vec_add(vec, weights, bias, in_dim, out_dim)
    expected = np.array([1.1, 2.7, 4.3], dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_ops_numerical_hardening():
    # 1. safe_clamp
    assert safe_clamp(float("nan"), 100.0) == 0.0
    assert safe_clamp(float("inf"), 100.0) == 0.0
    assert safe_clamp(float("-inf"), 100.0) == 0.0
    assert safe_clamp(250.0, 100.0) == 100.0
    assert safe_clamp(-250.0, 100.0) == -100.0

    # 2. Extreme GELU
    ext = gelu(1e20)
    assert not math.isnan(ext) and not math.isinf(ext) and ext <= 100.0
    assert gelu(float("nan")) == 0.0

    # 3. Matmul with NaN/Inf
    corrupted_vec = np.array([float("nan"), float("inf")], dtype=np.float32)
    corrupted_w = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    corrupted_b = np.array([float("nan"), 10.0], dtype=np.float32)
    out = matmul_vec_add(corrupted_vec, corrupted_w, corrupted_b, 2, 2)
    assert not np.any(np.isnan(out)) and not np.any(np.isinf(out))

    # 4. Softmax with degenerated logits
    degen_logits = np.array([float("nan"), float("inf"), float("-inf")], dtype=np.float32)
    probs = softmax(degen_logits, 1.0)
    assert not np.any(np.isnan(probs)) and not np.any(np.isinf(probs))
