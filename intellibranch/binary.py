import hashlib
import io
import os
import struct
from dataclasses import dataclass
from typing import BinaryIO, List
import numpy as np

from intellibranch.tokenizer import BPETokenizer, MergeRule

MAGIC_BYTES = b"IBRN"
CURRENT_FORMAT_VERSION = 2
MAX_SEQUENCE_TOKENS = 128


class SerializationError(Exception):
    """Base exception for model serialization and deserialization errors."""
    pass


class InvalidMagicError(SerializationError):
    pass


class UnsupportedVersionError(SerializationError):
    pass


class ChecksumFailedError(SerializationError):
    pass


class InvalidTensorDimError(SerializationError):
    pass


@dataclass
class Header:
    magic: bytes
    version: int
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    num_classes: int


@dataclass
class Weights:
    embedding: np.ndarray   # [vocab_size * embedding_dim] float32
    positional: np.ndarray  # [max_sequence_tokens * embedding_dim] float32
    w1: np.ndarray          # [embedding_dim * hidden_dim] float32
    b1: np.ndarray          # [hidden_dim] float32
    w2: np.ndarray          # [hidden_dim * num_classes] float32
    b2: np.ndarray          # [num_classes] float32


def serialize_model(stream: BinaryIO, model: "InferenceModel") -> None:
    """Serializes the model structure into Little-Endian bytes and appends a SHA-256 checksum."""
    hasher = hashlib.sha256()

    def write_bytes(data: bytes) -> None:
        stream.write(data)
        hasher.update(data)

    # 1. Header Block (magic 4B, version 4B, vocab_size 4B, emb_dim 4B, hidden_dim 4B, num_classes 4B)
    header_bytes = struct.pack(
        "<4sIIIII",
        model.header.magic,
        model.header.version,
        model.header.vocab_size,
        model.header.embedding_dim,
        model.header.hidden_dim,
        model.header.num_classes,
    )
    write_bytes(header_bytes)

    # 2. Labels Block
    write_bytes(struct.pack("<I", len(model.labels)))
    for label in model.labels:
        label_bytes = label.encode("utf-8")
        write_bytes(struct.pack("<I", len(label_bytes)))
        write_bytes(label_bytes)

    # 3. Vocabulary Block
    write_bytes(struct.pack("<I", len(model.vocab)))
    for token in model.vocab:
        token_bytes = token.encode("utf-8")
        write_bytes(struct.pack("<I", len(token_bytes)))
        write_bytes(token_bytes)

    # 4. Merge Rules Block
    write_bytes(struct.pack("<I", len(model.merge_rules)))
    for rule in model.merge_rules:
        write_bytes(struct.pack("<III", rule.token1, rule.token2, rule.target))

    # 5. Tensor Blocks
    emb_floats = np.ascontiguousarray(model.weights.embedding, dtype=np.float32)
    write_bytes(emb_floats.tobytes())

    if model.header.version >= 2:
        pos_len = MAX_SEQUENCE_TOKENS * model.header.embedding_dim
        pos_floats = np.ascontiguousarray(model.weights.positional, dtype=np.float32)
        if len(pos_floats) < pos_len:
            padded = np.zeros(pos_len, dtype=np.float32)
            padded[: len(pos_floats)] = pos_floats
            pos_floats = padded
        else:
            pos_floats = pos_floats[:pos_len]
        write_bytes(pos_floats.tobytes())

    w1_floats = np.ascontiguousarray(model.weights.w1, dtype=np.float32)
    write_bytes(w1_floats.tobytes())

    b1_floats = np.ascontiguousarray(model.weights.b1, dtype=np.float32)
    write_bytes(b1_floats.tobytes())

    w2_floats = np.ascontiguousarray(model.weights.w2, dtype=np.float32)
    write_bytes(w2_floats.tobytes())

    b2_floats = np.ascontiguousarray(model.weights.b2, dtype=np.float32)
    write_bytes(b2_floats.tobytes())

    # 6. SHA-256 Checksum (32 bytes)
    checksum = hasher.digest()
    stream.write(checksum)


def deserialize_model(stream: BinaryIO) -> "InferenceModel":
    """Parses Little-Endian bytes, validates header & SHA-256 checksum, and instantiates an InferenceModel."""
    # Use buffered reading to calculate hash simultaneously
    raw_content = stream.read()
    if len(raw_content) < 24 + 32:
        raise SerializationError("file is too short to be a valid model")

    body_bytes = raw_content[:-32]
    expected_checksum = raw_content[-32:]

    computed_checksum = hashlib.sha256(body_bytes).digest()
    if computed_checksum != expected_checksum:
        raise ChecksumFailedError("checksum verification failed: model file corrupted")

    reader = io.BytesIO(body_bytes)

    # 1. Header Block
    header_data = reader.read(24)
    if len(header_data) < 24:
        raise SerializationError("failed to read complete header")

    magic, version, vocab_size, embedding_dim, hidden_dim, num_classes = struct.unpack("<4sIIIII", header_data)
    if magic != MAGIC_BYTES:
        raise InvalidMagicError(f"invalid binary format: expected {MAGIC_BYTES!r}, got {magic!r}")

    if version not in (1, 2):
        raise UnsupportedVersionError(f"unsupported format version {version}, expected 1 or 2")

    header = Header(
        magic=magic,
        version=version,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )

    # 2. Labels Block
    (num_labels,) = struct.unpack("<I", reader.read(4))
    if num_labels != num_classes:
        raise SerializationError(f"num labels ({num_labels}) does not match num_classes ({num_classes})")

    labels: List[str] = []
    for _ in range(num_labels):
        (str_len,) = struct.unpack("<I", reader.read(4))
        labels.append(reader.read(str_len).decode("utf-8"))

    # 3. Vocabulary Block
    (vocab_count,) = struct.unpack("<I", reader.read(4))
    if vocab_count != vocab_size:
        raise SerializationError(f"vocab count ({vocab_count}) does not match vocab_size ({vocab_size})")

    vocab: List[str] = []
    for _ in range(vocab_count):
        (str_len,) = struct.unpack("<I", reader.read(4))
        vocab.append(reader.read(str_len).decode("utf-8"))

    # 4. Merge Rules Block
    (num_rules,) = struct.unpack("<I", reader.read(4))
    merge_rules: List[MergeRule] = []
    for _ in range(num_rules):
        t1, t2, target = struct.unpack("<III", reader.read(12))
        merge_rules.append(MergeRule(token1=t1, token2=t2, target=target))

    # 5. Tensor Blocks
    emb_count = vocab_size * embedding_dim
    embedding = np.frombuffer(reader.read(emb_count * 4), dtype="<f4").astype(np.float32)
    if len(embedding) != emb_count:
        raise SerializationError("failed to read complete embedding weights")

    pos_len = MAX_SEQUENCE_TOKENS * embedding_dim
    if version >= 2:
        positional = np.frombuffer(reader.read(pos_len * 4), dtype="<f4").astype(np.float32)
        if len(positional) != pos_len:
            raise SerializationError("failed to read complete positional weights")
    else:
        positional = np.zeros(pos_len, dtype=np.float32)

    w1_count = embedding_dim * hidden_dim
    w1 = np.frombuffer(reader.read(w1_count * 4), dtype="<f4").astype(np.float32)
    if len(w1) != w1_count:
        raise SerializationError("failed to read complete W1 weights")

    b1 = np.frombuffer(reader.read(hidden_dim * 4), dtype="<f4").astype(np.float32)
    if len(b1) != hidden_dim:
        raise SerializationError("failed to read complete B1 weights")

    w2_count = hidden_dim * num_classes
    w2 = np.frombuffer(reader.read(w2_count * 4), dtype="<f4").astype(np.float32)
    if len(w2) != w2_count:
        raise SerializationError("failed to read complete W2 weights")

    b2 = np.frombuffer(reader.read(num_classes * 4), dtype="<f4").astype(np.float32)
    if len(b2) != num_classes:
        raise SerializationError("failed to read complete B2 weights")

    weights = Weights(
        embedding=embedding,
        positional=positional,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
    )

    from intellibranch.runtime import InferenceModel
    return InferenceModel(
        header=header,
        labels=labels,
        vocab=vocab,
        merge_rules=merge_rules,
        weights=weights,
    )


def save_binary_model(file_path: str, model: "InferenceModel") -> None:
    """Serializes an InferenceModel into a Little-Endian binary file with SHA-256 validation."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "wb") as f:
        serialize_model(f, model)


def load_binary_model(file_path: str) -> "InferenceModel":
    """Reads and validates an IntelliBranch binary model from the filesystem."""
    with open(file_path, "rb") as f:
        return deserialize_model(f)
