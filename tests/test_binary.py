import glob
import io
import os
import numpy as np
import pytest

from intellibranch.binary import (
    CURRENT_FORMAT_VERSION,
    ChecksumFailedError,
    Header,
    InvalidMagicError,
    MAGIC_BYTES,
    MAX_SEQUENCE_TOKENS,
    Weights,
    deserialize_model,
    load_binary_model,
    save_binary_model,
    serialize_model,
)
from intellibranch.runtime import InferenceModel
from intellibranch.tokenizer import MergeRule


def create_sample_model(version: int = 2) -> InferenceModel:
    header = Header(
        magic=MAGIC_BYTES,
        version=version,
        vocab_size=4,
        embedding_dim=4,
        hidden_dim=6,
        num_classes=2,
    )
    labels = ["Refund", "Delivery"]
    vocab = ["[PAD]", "refund", "cancel", "delivery"]
    rules = [MergeRule(token1=1, token2=2, target=3)]

    pos_len = MAX_SEQUENCE_TOKENS * header.embedding_dim
    weights = Weights(
        embedding=np.array([i * 0.05 for i in range(16)], dtype=np.float32),
        positional=np.array([i * 0.01 for i in range(pos_len)], dtype=np.float32),
        w1=np.array([i * 0.02 for i in range(24)], dtype=np.float32),
        b1=np.full(6, 0.1, dtype=np.float32),
        w2=np.array([i * 0.03 for i in range(12)], dtype=np.float32),
        b2=np.full(2, 0.2, dtype=np.float32),
    )
    return InferenceModel(header, labels, vocab, rules, weights)


def test_serialization_round_trip():
    orig_model = create_sample_model(version=2)
    stream = io.BytesIO()
    serialize_model(stream, orig_model)

    raw_bytes = stream.getvalue()
    assert len(raw_bytes) > 0

    parsed_model = deserialize_model(io.BytesIO(raw_bytes))

    assert parsed_model.header == orig_model.header
    assert parsed_model.labels == orig_model.labels
    assert parsed_model.vocab == orig_model.vocab
    assert len(parsed_model.merge_rules) == len(orig_model.merge_rules)
    assert parsed_model.merge_rules[0] == orig_model.merge_rules[0]

    np.testing.assert_allclose(parsed_model.weights.embedding, orig_model.weights.embedding)
    np.testing.assert_allclose(parsed_model.weights.positional, orig_model.weights.positional)
    np.testing.assert_allclose(parsed_model.weights.w1, orig_model.weights.w1)
    np.testing.assert_allclose(parsed_model.weights.b1, orig_model.weights.b1)
    np.testing.assert_allclose(parsed_model.weights.w2, orig_model.weights.w2)
    np.testing.assert_allclose(parsed_model.weights.b2, orig_model.weights.b2)


def test_corrupted_magic_rejection():
    orig_model = create_sample_model(version=2)
    stream = io.BytesIO()
    serialize_model(stream, orig_model)
    raw = bytearray(stream.getvalue())

    # Corrupt magic byte
    raw[0] = ord("X")

    with pytest.raises(ChecksumFailedError):
        deserialize_model(io.BytesIO(raw))


def test_checksum_verification_failure():
    orig_model = create_sample_model(version=2)
    stream = io.BytesIO()
    serialize_model(stream, orig_model)
    raw = bytearray(stream.getvalue())

    # Corrupt byte in tensor section
    raw[-40] ^= 0xFF

    with pytest.raises(ChecksumFailedError):
        deserialize_model(io.BytesIO(raw))


def test_load_existing_model_weights():
    existing_bins = glob.glob("weights/*.bin")
    assert len(existing_bins) > 0, "no binary model weights found in weights/"
    for bin_path in existing_bins:
        model = load_binary_model(bin_path)
        assert model.header.magic == MAGIC_BYTES
        assert model.header.version in (1, 2)
        assert len(model.labels) == model.header.num_classes
        assert len(model.vocab) == model.header.vocab_size
        assert len(model.weights.embedding) == model.header.vocab_size * model.header.embedding_dim
