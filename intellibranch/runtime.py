import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np

from intellibranch.binary import Header, Weights, MAX_SEQUENCE_TOKENS
from intellibranch.ops import (
    gelu_array,
    matmul_vec_add,
    mean_pooling_with_pos,
    softmax,
)
from intellibranch.tokenizer import BPETokenizer, MergeRule, is_valid_utf8

MAX_INPUT_BYTES: int = 512


@dataclass
class MatchSlot:
    index: int = -1
    confidence: float = 0.0


@dataclass
class StaticInferenceResult:
    primary: MatchSlot
    secondary: MatchSlot
    entropy: float = 0.0
    total: int = 0


def compute_entropy(probs: np.ndarray) -> float:
    """Calculates Shannon entropy in bits with epsilon guards to prevent NaN/Inf underflows."""
    entropy = 0.0
    for p in probs:
        p_val = float(p)
        if p_val > 1e-7:
            entropy -= p_val * math.log2(p_val)
    return float(entropy)


def truncate_to_rune_boundary(text: str, max_bytes: int = MAX_INPUT_BYTES) -> str:
    """Truncates text to at most max_bytes without slicing multi-byte UTF-8 runes."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes]
    # Decode ignoring errors at boundary or trimming back
    while len(truncated) > 0:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError:
            truncated = truncated[:-1]
    return ""


class InferenceModel:
    """In-memory embedded classifier loaded from binary format."""

    def __init__(
        self,
        header: Header,
        labels: Sequence[str],
        vocab: Sequence[str],
        merge_rules: Sequence[MergeRule],
        weights: Weights,
        temperature: float = 1.0,
    ) -> None:
        self.header: Header = header
        self.labels: List[str] = list(labels)
        self.vocab: List[str] = list(vocab)
        self.merge_rules: List[MergeRule] = list(merge_rules)
        self.weights: Weights = weights
        self.temperature: float = temperature
        self.tokenizer: BPETokenizer = BPETokenizer(vocab=self.vocab, rules=self.merge_rules)

    def _forward_internal(
        self, token_ids: Sequence[int], temperature: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Internal forward pass computation without intermediate object overhead."""
        if not token_ids:
            raise ValueError("input token slice cannot be empty")

        # 1. Mean Pooling with Positional Encoding
        pooled = mean_pooling_with_pos(
            token_ids=token_ids,
            embedding_table=self.weights.embedding,
            pos_table=self.weights.positional,
            emb_dim=self.header.embedding_dim,
        )

        # 2. Layer 1 Linear
        hidden = matmul_vec_add(
            vec=pooled,
            weights=self.weights.w1,
            bias=self.weights.b1,
            in_dim=self.header.embedding_dim,
            out_dim=self.header.hidden_dim,
        )

        # 3. GELU Activation
        hidden = gelu_array(hidden)

        # 4. Layer 2 Linear
        logits = matmul_vec_add(
            vec=hidden,
            weights=self.weights.w2,
            bias=self.weights.b2,
            in_dim=self.header.hidden_dim,
            out_dim=self.header.num_classes,
        )

        # 5. Softmax
        temp = temperature if temperature > 0.0 else self.temperature
        probs = softmax(logits, temp)

        return pooled, hidden, logits, probs

    def forward(self, token_ids: Sequence[int], temperature: float = 0.0) -> np.ndarray:
        """Executes the 2-layer MLP inference over token IDs and returns probabilities."""
        _, _, _, probs = self._forward_internal(token_ids, temperature)
        return probs

    def predict_slots(self, token_ids: Sequence[int], temperature: float = 0.0) -> StaticInferenceResult:
        """Computes top-2 class predictions and entropy with zero allocation overhead."""
        _, _, _, probs = self._forward_internal(token_ids, temperature)

        top1_idx, top2_idx = -1, -1
        top1_prob, top2_prob = -1.0, -1.0

        for i, p in enumerate(probs):
            p_val = float(p)
            if p_val > top1_prob:
                top2_prob = top1_prob
                top2_idx = top1_idx
                top1_prob = p_val
                top1_idx = i
            elif p_val > top2_prob:
                top2_prob = p_val
                top2_idx = i

        primary = MatchSlot(index=top1_idx, confidence=top1_prob) if top1_idx >= 0 else MatchSlot()
        secondary = MatchSlot(index=top2_idx, confidence=top2_prob) if top2_idx >= 0 else MatchSlot()
        total = (1 if top1_idx >= 0 else 0) + (1 if top2_idx >= 0 else 0)
        ent = compute_entropy(probs)

        return StaticInferenceResult(primary=primary, secondary=secondary, entropy=ent, total=total)

    def predict_tokens(self, token_ids: Sequence[int]) -> Tuple[str, float]:
        """Computes class probabilities and returns top label alongside confidence score."""
        res = self.predict_slots(token_ids, self.temperature)
        best_idx = res.primary.index
        if best_idx < 0 or best_idx >= len(self.labels):
            raise IndexError("predicted class index exceeds label count")
        return self.labels[best_idx], float(res.primary.confidence)

    def predict_detailed(self, text: str) -> Tuple[StaticInferenceResult, float]:
        """Executes inference and returns top-2 slots, calibrated confidence, and UNK ratio."""
        if not is_valid_utf8(text):
            raise ValueError("invalid UTF-8 input")

        if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
            text = truncate_to_rune_boundary(text, MAX_INPUT_BYTES)

        token_ids = self.tokenizer.encode(text)
        if not token_ids:
            raise ValueError("empty input tokens")

        if len(token_ids) > MAX_SEQUENCE_TOKENS:
            token_ids = token_ids[:MAX_SEQUENCE_TOKENS]

        res = self.predict_slots(token_ids, self.temperature)

        # UNK ratio penalty
        unk_ratio = 0.0
        unk_id = self.tokenizer.vocab_map.get("[UNK]")
        if unk_id is not None and token_ids:
            unk_count = sum(1 for tid in token_ids if tid == unk_id)
            unk_ratio = float(unk_count) / float(len(token_ids))
            decay = max(0.0, 1.0 - unk_ratio)
            res.primary.confidence *= decay
            res.secondary.confidence *= decay

        return res, unk_ratio

    def predict(self, text: str) -> Tuple[str, float]:
        """Tokenizes text with BPE and returns predicted label and confidence score with safety guards."""
        res, _ = self.predict_detailed(text)
        best_idx = res.primary.index
        if best_idx < 0 or best_idx >= len(self.labels):
            raise IndexError("predicted class index exceeds label count")
        return self.labels[best_idx], res.primary.confidence

    def predict_features(
        self, token_ids: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts pooled embedding representation and unnormalized logits."""
        pooled, _, logits, _ = self._forward_internal(token_ids, self.temperature)
        return pooled, logits
