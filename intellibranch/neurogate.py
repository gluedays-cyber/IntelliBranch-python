import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple
import numpy as np

from intellibranch.binary import load_binary_model, MAX_SEQUENCE_TOKENS
from intellibranch.ops import dot_product, l2_normalize, softmax
from intellibranch.policy import DispatchPolicy, default_dispatch_policy
from intellibranch.runtime import (
    InferenceModel,
    MAX_INPUT_BYTES,
    compute_entropy,
    truncate_to_rune_boundary,
)

MAX_GATE_CLASSES: int = 16
MAX_GATE_EMB_DIM: int = 64

RouteAction = Callable[[Any, Any], Any]
AmbiguousAction = Callable[[Any, str, str, Any], Any]
PipelineAction = Callable[[Any, str, str, Any], Any]


def pipeline_key(primary: str, secondary: str) -> str:
    return f"{primary}->{secondary}"


@dataclass
class AnchorRule:
    class_index: int
    mask: int
    weight: float
    keywords: List[str]


@dataclass
class GateTrace:
    input_text: str
    token_ids: List[int] = field(default_factory=list)
    subwords: List[str] = field(default_factory=list)
    unknown_token_ratio: float = 0.0
    cosine_similarity: float = 1.0
    is_ood: bool = False
    anchor_bitmask: int = 0
    triggered_anchors: List[str] = field(default_factory=list)
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    predicted_label: str = ""
    secondary_label: str = ""
    confidence: float = 0.0
    margin: float = 0.0
    entropy: float = 0.0
    log_sum_exp: float = 0.0
    free_energy: float = 0.0
    threshold: float = 0.75
    is_ambiguous: bool = False
    is_pipeline: bool = False
    is_fallback: bool = False
    fallback_reason: str = ""
    latency_micros: int = 0


@dataclass
class _GateEvaluation:
    primary_idx: int = 0
    secondary_idx: int = -1
    is_fallback: bool = False
    is_pipeline: bool = False
    is_ambiguous: bool = False


class GateRouteBuilder:
    """Provides fluent API chaining for binding routes and anchor soft biases."""

    def __init__(self, gate: "NeuroGate", class_index: int, label: str) -> None:
        self.gate: NeuroGate = gate
        self.class_index: int = class_index
        self.label: str = label

    def with_anchor(self, weight: float, *keywords: str) -> "GateRouteBuilder":
        """Registers anchor keywords that inject a soft additive bias into this class's logit."""
        with self.gate.lock:
            mask = 0
            clean_keywords: List[str] = []
            for kw in keywords:
                cleaned = kw.strip().lower()
                if not cleaned:
                    continue
                clean_keywords.append(cleaned)
                if cleaned not in self.gate.anchor_dict:
                    if len(self.gate.anchor_dict) < 64:
                        m = 1 << len(self.gate.anchor_dict)
                        self.gate.anchor_dict[cleaned] = m
                mask |= self.gate.anchor_dict.get(cleaned, 0)

            model = self.gate.model
            if model is not None and model.tokenizer is not None:
                for kw in clean_keywords:
                    toks = model.tokenizer.encode(kw)
                    for tid in toks:
                        self.gate.anchor_token_map[tid] = self.gate.anchor_token_map.get(tid, 0) | mask

            self.gate.anchor_rules.append(
                AnchorRule(
                    class_index=self.class_index,
                    mask=mask,
                    weight=weight,
                    keywords=clean_keywords,
                )
            )
        return self

    def bind(self, label: str, handler: RouteAction) -> "GateRouteBuilder":
        """Allows continuing chaining for additional routes."""
        return self.gate.bind(label, handler)


class NeuroGate:
    """Coordinates a single shared neural backbone with a geometric 3-head gate."""

    def __init__(self, model: InferenceModel | None = None) -> None:
        self.lock = threading.RLock()
        self.model: InferenceModel | None = model
        self.labels: List[str] = []
        self.routes: List[RouteAction | None] = []
        self.label_to_index: Dict[str, int] = {}
        self.class_count: int = 0

        self.anchor_dict: Dict[str, int] = {}
        self.anchor_token_map: Dict[int, int] = {}
        self.anchor_rules: List[AnchorRule] = []

        self.domain_centroid: np.ndarray = np.zeros(MAX_GATE_EMB_DIM, dtype=np.float32)
        self.has_centroid: bool = False
        self.min_cosine_sim: float = 0.25

        self.policy: DispatchPolicy = default_dispatch_policy()
        self.pipelines: Dict[str, PipelineAction] = {}
        self.ambiguous_handler: AmbiguousAction | None = None
        self.fallback_handler: RouteAction | None = lambda ctx, payload: None

        if model is not None:
            self._init_model(model)

    @classmethod
    def from_file(cls, model_path: str) -> "NeuroGate":
        """Loads an InferenceModel from disk and constructs a NeuroGate."""
        model = load_binary_model(model_path)
        return cls(model)

    def _init_model(self, model: InferenceModel) -> None:
        with self.lock:
            self.model = model
            self.labels = []
            self.routes = []
            self.label_to_index = {}
            self.class_count = 0

            for i, lbl in enumerate(model.labels):
                if i >= MAX_GATE_CLASSES:
                    break
                self.labels.append(lbl)
                self.routes.append(None)
                self.label_to_index[lbl] = i
                self.class_count += 1

            self._compute_baseline_centroid(model)

    def _compute_baseline_centroid(self, model: InferenceModel) -> None:
        emb_dim = model.header.embedding_dim
        if emb_dim > MAX_GATE_EMB_DIM or emb_dim == 0 or len(model.weights.embedding) == 0:
            return

        total_sum = np.zeros(emb_dim, dtype=np.float64)
        valid_tokens = 0

        for tok_id, word in enumerate(model.vocab):
            if word in ("[PAD]", "[UNK]"):
                continue
            offset = tok_id * emb_dim
            if offset + emb_dim <= len(model.weights.embedding):
                total_sum += model.weights.embedding[offset : offset + emb_dim].astype(np.float64)
                valid_tokens += 1

        if valid_tokens > 0:
            avg_emb = (total_sum / float(valid_tokens)).astype(np.float32)
            normalized, _ = l2_normalize(avg_emb)
            self.domain_centroid[:emb_dim] = normalized
            self.has_centroid = True

    def calibrate_domain_centroid(self, samples: Sequence[Any]) -> "NeuroGate":
        """Calculates the true manifold center from sample dataset sentences."""
        with self.lock:
            if self.model is None or not samples:
                return self

            emb_dim = min(self.model.header.embedding_dim, MAX_GATE_EMB_DIM)
            total_sum = np.zeros(emb_dim, dtype=np.float64)
            valid_count = 0

            for s in samples:
                text = s.text if hasattr(s, "text") else s[0]
                tokens = self.model.tokenizer.encode(text)
                if not tokens:
                    continue
                pooled, _ = self.model.predict_features(tokens)
                total_sum += pooled[:emb_dim].astype(np.float64)
                valid_count += 1

            if valid_count > 0:
                avg = (total_sum / float(valid_count)).astype(np.float32)
                normalized, _ = l2_normalize(avg)
                self.domain_centroid[:emb_dim] = normalized
                self.has_centroid = True

        return self

    def set_domain_boundary(self, centroid: Sequence[float], min_cosine: float) -> "NeuroGate":
        with self.lock:
            dim = min(len(centroid), MAX_GATE_EMB_DIM)
            c = np.array(centroid[:dim], dtype=np.float32)
            normalized, _ = l2_normalize(c)
            self.domain_centroid[:dim] = normalized
            self.has_centroid = True
            self.min_cosine_sim = min_cosine
        return self

    def set_min_cosine_sim(self, threshold: float) -> "NeuroGate":
        with self.lock:
            self.min_cosine_sim = threshold
        return self

    def set_policy(self, policy: DispatchPolicy) -> "NeuroGate":
        with self.lock:
            self.policy = policy
        return self

    def bind(self, label: str, handler: RouteAction) -> GateRouteBuilder:
        """Registers an action handler for a target class label."""
        with self.lock:
            if label not in self.label_to_index:
                if self.class_count >= MAX_GATE_CLASSES:
                    raise ValueError(f"registered classes exceed {MAX_GATE_CLASSES}")
                idx = self.class_count
                self.labels.append(label)
                self.routes.append(handler)
                self.label_to_index[label] = idx
                self.class_count += 1
            else:
                idx = self.label_to_index[label]
                self.routes[idx] = handler

            return GateRouteBuilder(gate=self, class_index=idx, label=label)

    def bind_pipeline(self, primary: str, secondary: str, handler: PipelineAction) -> "NeuroGate":
        with self.lock:
            self.pipelines[pipeline_key(primary, secondary)] = handler
        return self

    def ambiguous(self, handler: AmbiguousAction) -> "NeuroGate":
        with self.lock:
            self.ambiguous_handler = handler
        return self

    def fallback(self, handler: RouteAction) -> "NeuroGate":
        with self.lock:
            self.fallback_handler = handler
        return self

    def swap_model(self, new_model: InferenceModel) -> None:
        self._init_model(new_model)

    def get_model(self) -> InferenceModel | None:
        return self.model

    def inspect(self, text: str) -> GateTrace:
        """Evaluates input text across all 3 heads and returns detailed diagnostics."""
        start = time.perf_counter()
        model = self.model
        if model is None:
            return GateTrace(
                input_text=text,
                is_fallback=True,
                fallback_reason="model not loaded",
                latency_micros=int((time.perf_counter() - start) * 1_000_000),
            )

        if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
            text = truncate_to_rune_boundary(text, MAX_INPUT_BYTES)

        tokens = model.tokenizer.encode(text)
        if not tokens:
            return GateTrace(
                input_text=text,
                is_fallback=True,
                fallback_reason="empty input tokens",
                latency_micros=int((time.perf_counter() - start) * 1_000_000),
            )
        if len(tokens) > MAX_SEQUENCE_TOKENS:
            tokens = tokens[:MAX_SEQUENCE_TOKENS]

        subwords: List[str] = []
        unk_count = 0
        unk_id = model.tokenizer.vocab_map.get("[UNK]")
        for tid in tokens:
            if unk_id is not None and tid == unk_id:
                unk_count += 1
            if tid < len(model.vocab):
                subwords.append(model.vocab[tid])
        unk_ratio = float(unk_count) / float(len(tokens))

        with self.lock:
            emb_dim = min(model.header.embedding_dim, MAX_GATE_EMB_DIM)
            num_classes = self.class_count

            pooled, raw_logits = model.predict_features(tokens)
            pooled = pooled[:emb_dim]
            raw_logits = raw_logits[:num_classes]

            # Head 1: L2 Cosine Out-of-Domain Guard
            norm_pooled, _ = l2_normalize(pooled)
            cosine_sim = 1.0
            is_ood = False
            if self.has_centroid:
                cosine_sim = dot_product(norm_pooled, self.domain_centroid[:emb_dim])
                if cosine_sim < self.min_cosine_sim:
                    is_ood = True

            # Head 2: Anchor Bitmask & Symbolic Bias
            text_bitmask = 0
            triggered_anchors: List[str] = []
            lower_text = text.lower()
            for kw, mask in self.anchor_dict.items():
                if kw in lower_text:
                    text_bitmask |= mask
                    triggered_anchors.append(kw)

            adjusted_logits = raw_logits.copy()
            for rule in self.anchor_rules:
                matched_bits = text_bitmask & rule.mask
                if matched_bits != 0:
                    count = bin(matched_bits).count("1")
                    adjusted_logits[rule.class_index] += rule.weight * count

            # Head 3: Softmax, Margin, Entropy
            probs = softmax(adjusted_logits, model.temperature)
            prob_map: Dict[str, float] = {self.labels[i]: float(probs[i]) for i in range(num_classes)}

            best_idx, second_idx = 0, 1 if num_classes > 1 else -1
            best_score, second_score = -1.0, -1.0
            for i in range(num_classes):
                p = float(probs[i])
                if p > best_score:
                    second_score = best_score
                    second_idx = best_idx
                    best_score = p
                    best_idx = i
                elif p > second_score:
                    second_score = p
                    second_idx = i

            calibrated_conf = best_score * (1.0 - unk_ratio)
            calibrated_second = second_score * (1.0 - unk_ratio) if second_idx >= 0 else 0.0
            margin = calibrated_conf - calibrated_second
            entropy = compute_entropy(probs)

            best_label = self.labels[best_idx]
            second_label = self.labels[second_idx] if second_idx >= 0 else ""

            # LogSumExp and Free Energy
            max_logit = float(np.max(adjusted_logits))
            sum_exp = float(np.sum(np.exp(adjusted_logits - max_logit)))
            log_sum_exp = max_logit + math.log(sum_exp) if sum_exp > 0 else max_logit
            free_energy = -log_sum_exp

            # OOD & Energy Guard
            ood_reason = ""
            if self.has_centroid and cosine_sim < self.min_cosine_sim:
                is_ood = True
                ood_reason = f"cosine similarity {cosine_sim:.4f} below domain threshold {self.min_cosine_sim:.4f} (OOD)"
            elif self.policy.min_log_sum_exp > 0 and log_sum_exp < self.policy.min_log_sum_exp:
                is_ood = True
                ood_reason = f"free energy {free_energy:.4f} (logSumExp {log_sum_exp:.4f}) below in-distribution threshold {self.policy.min_log_sum_exp:.4f} (OOD)"
            elif entropy > self.policy.max_entropy:
                is_ood = True
                ood_reason = f"prediction entropy {entropy:.4f} exceeds limit {self.policy.max_entropy:.4f} (OOD)"
            elif unk_ratio >= 0.5:
                is_ood = True
                ood_reason = f"excessive unknown tokens ({unk_ratio:.2f} >= 0.50)"

            latency_micros = int((time.perf_counter() - start) * 1_000_000)
            trace = GateTrace(
                input_text=text,
                token_ids=tokens,
                subwords=subwords,
                unknown_token_ratio=unk_ratio,
                cosine_similarity=cosine_sim,
                is_ood=is_ood,
                anchor_bitmask=text_bitmask,
                triggered_anchors=triggered_anchors,
                class_probabilities=prob_map,
                predicted_label=best_label,
                secondary_label=second_label,
                confidence=calibrated_conf,
                margin=margin,
                entropy=entropy,
                log_sum_exp=log_sum_exp,
                free_energy=free_energy,
                threshold=self.policy.high_threshold,
                latency_micros=latency_micros,
            )

            if is_ood:
                trace.is_fallback = True
                trace.fallback_reason = ood_reason
            elif trace.confidence < self.policy.low_threshold:
                trace.is_fallback = True
                trace.fallback_reason = f"confidence {trace.confidence:.4f} below low threshold {self.policy.low_threshold:.4f}"
            else:
                if second_label:
                    pipe_key = pipeline_key(best_label, second_label)
                    has_pipeline = pipe_key in self.pipelines
                    primary_anchors = any(
                        (text_bitmask & r.mask) != 0 for r in self.anchor_rules if r.class_index == best_idx
                    )
                    secondary_anchors = any(
                        (text_bitmask & r.mask) != 0 for r in self.anchor_rules if r.class_index == second_idx
                    )
                    has_both_anchors = primary_anchors and secondary_anchors
                    if calibrated_second >= self.policy.pipeline_threshold or (has_both_anchors and has_pipeline):
                        trace.is_pipeline = True

                if trace.confidence < self.policy.high_threshold or margin < self.policy.margin_cutoff:
                    trace.is_ambiguous = True

                if best_idx >= len(self.routes) or self.routes[best_idx] is None:
                    trace.is_fallback = True
                    trace.fallback_reason = f"label '{best_label}' has no bound route handler"

            return trace

    def _evaluate_fast(self, text: str) -> _GateEvaluation:
        model = self.model
        if model is None:
            return _GateEvaluation(is_fallback=True)

        if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
            text = truncate_to_rune_boundary(text, MAX_INPUT_BYTES)

        tokens = model.tokenizer.encode(text)
        if not tokens:
            return _GateEvaluation(is_fallback=True)
        if len(tokens) > MAX_SEQUENCE_TOKENS:
            tokens = tokens[:MAX_SEQUENCE_TOKENS]

        unk_count = 0
        unk_id = model.tokenizer.vocab_map.get("[UNK]")
        for tid in tokens:
            if unk_id is not None and tid == unk_id:
                unk_count += 1
        unk_ratio = float(unk_count) / float(len(tokens))

        with self.lock:
            emb_dim = min(model.header.embedding_dim, MAX_GATE_EMB_DIM)
            num_classes = self.class_count

            pooled, raw_logits = model.predict_features(tokens)
            pooled = pooled[:emb_dim]
            raw_logits = raw_logits[:num_classes]

            # [Head 1]
            norm_pooled, _ = l2_normalize(pooled)
            if self.has_centroid:
                cosine_sim = dot_product(norm_pooled, self.domain_centroid[:emb_dim])
                if cosine_sim < self.min_cosine_sim:
                    return _GateEvaluation(is_fallback=True)

            # [Head 2]
            text_bitmask = 0
            lower_text = text.lower()
            for kw, mask in self.anchor_dict.items():
                if kw in lower_text:
                    text_bitmask |= mask

            adjusted_logits = raw_logits.copy()
            for rule in self.anchor_rules:
                matched_bits = text_bitmask & rule.mask
                if matched_bits != 0:
                    count = bin(matched_bits).count("1")
                    adjusted_logits[rule.class_index] += rule.weight * count

            # [Head 3]
            probs = softmax(adjusted_logits, model.temperature)
            best_idx, second_idx = 0, 1 if num_classes > 1 else -1
            best_score, second_score = -1.0, -1.0
            for i in range(num_classes):
                p = float(probs[i])
                if p > best_score:
                    second_score = best_score
                    second_idx = best_idx
                    best_score = p
                    best_idx = i
                elif p > second_score:
                    second_score = p
                    second_idx = i

            calibrated_conf = best_score * (1.0 - unk_ratio)
            calibrated_second = second_score * (1.0 - unk_ratio) if second_idx >= 0 else 0.0
            margin = calibrated_conf - calibrated_second
            entropy = compute_entropy(probs)

            max_logit = float(np.max(adjusted_logits))
            sum_exp = float(np.sum(np.exp(adjusted_logits - max_logit)))
            log_sum_exp = max_logit + math.log(sum_exp) if sum_exp > 0 else max_logit

            if (
                (self.policy.min_log_sum_exp > 0 and log_sum_exp < self.policy.min_log_sum_exp)
                or unk_ratio >= 0.5
                or calibrated_conf < self.policy.low_threshold
                or entropy > self.policy.max_entropy
            ):
                return _GateEvaluation(is_fallback=True, primary_idx=best_idx, secondary_idx=second_idx)

            eval_res = _GateEvaluation(primary_idx=best_idx, secondary_idx=second_idx)

            if second_idx >= 0:
                pipe_key = pipeline_key(self.labels[best_idx], self.labels[second_idx])
                has_pipeline = pipe_key in self.pipelines
                primary_anchors = any(
                    (text_bitmask & r.mask) != 0 for r in self.anchor_rules if r.class_index == best_idx
                )
                secondary_anchors = any(
                    (text_bitmask & r.mask) != 0 for r in self.anchor_rules if r.class_index == second_idx
                )
                has_both_anchors = primary_anchors and secondary_anchors
                if calibrated_second >= self.policy.pipeline_threshold or (has_both_anchors and has_pipeline):
                    eval_res.is_pipeline = True

            if calibrated_conf < self.policy.high_threshold or margin < self.policy.margin_cutoff:
                eval_res.is_ambiguous = True

            if best_idx >= len(self.routes) or self.routes[best_idx] is None:
                eval_res.is_fallback = True

            return eval_res

    def _evaluate_fast_tokens(self, tokens: Sequence[int]) -> _GateEvaluation:
        model = self.model
        if model is None or not tokens:
            return _GateEvaluation(is_fallback=True)

        if len(tokens) > MAX_SEQUENCE_TOKENS:
            tokens = tokens[:MAX_SEQUENCE_TOKENS]

        unk_count = 0
        unk_id = model.tokenizer.vocab_map.get("[UNK]")
        for tid in tokens:
            if unk_id is not None and tid == unk_id:
                unk_count += 1
        unk_ratio = float(unk_count) / float(len(tokens))

        with self.lock:
            emb_dim = min(model.header.embedding_dim, MAX_GATE_EMB_DIM)
            num_classes = self.class_count

            pooled, raw_logits = model.predict_features(tokens)
            pooled = pooled[:emb_dim]
            raw_logits = raw_logits[:num_classes]

            # [Head 1]
            norm_pooled, _ = l2_normalize(pooled)
            if self.has_centroid:
                cosine_sim = dot_product(norm_pooled, self.domain_centroid[:emb_dim])
                if cosine_sim < self.min_cosine_sim:
                    return _GateEvaluation(is_fallback=True)

            # [Head 2]
            text_bitmask = 0
            for tid in tokens:
                if tid in self.anchor_token_map:
                    text_bitmask |= self.anchor_token_map[tid]

            adjusted_logits = raw_logits.copy()
            for rule in self.anchor_rules:
                matched_bits = text_bitmask & rule.mask
                if matched_bits != 0:
                    count = bin(matched_bits).count("1")
                    adjusted_logits[rule.class_index] += rule.weight * count

            # [Head 3]
            probs = softmax(adjusted_logits, model.temperature)
            best_idx, second_idx = 0, 1 if num_classes > 1 else -1
            best_score, second_score = -1.0, -1.0
            for i in range(num_classes):
                p = float(probs[i])
                if p > best_score:
                    second_score = best_score
                    second_idx = best_idx
                    best_score = p
                    best_idx = i
                elif p > second_score:
                    second_score = p
                    second_idx = i

            calibrated_conf = best_score * (1.0 - unk_ratio)
            calibrated_second = second_score * (1.0 - unk_ratio) if second_idx >= 0 else 0.0
            margin = calibrated_conf - calibrated_second
            entropy = compute_entropy(probs)

            max_logit = float(np.max(adjusted_logits))
            sum_exp = float(np.sum(np.exp(adjusted_logits - max_logit)))
            log_sum_exp = max_logit + math.log(sum_exp) if sum_exp > 0 else max_logit

            if (
                (self.policy.min_log_sum_exp > 0 and log_sum_exp < self.policy.min_log_sum_exp)
                or unk_ratio >= 0.5
                or calibrated_conf < self.policy.low_threshold
                or entropy > self.policy.max_entropy
            ):
                return _GateEvaluation(is_fallback=True, primary_idx=best_idx, secondary_idx=second_idx)

            eval_res = _GateEvaluation(primary_idx=best_idx, secondary_idx=second_idx)

            if second_idx >= 0:
                pipe_key = pipeline_key(self.labels[best_idx], self.labels[second_idx])
                has_pipeline = pipe_key in self.pipelines
                primary_anchors = any(
                    (text_bitmask & r.mask) != 0 for r in self.anchor_rules if r.class_index == best_idx
                )
                secondary_anchors = any(
                    (text_bitmask & r.mask) != 0 for r in self.anchor_rules if r.class_index == second_idx
                )
                has_both_anchors = primary_anchors and secondary_anchors
                if calibrated_second >= self.policy.pipeline_threshold or (has_both_anchors and has_pipeline):
                    eval_res.is_pipeline = True

            if calibrated_conf < self.policy.high_threshold or margin < self.policy.margin_cutoff:
                eval_res.is_ambiguous = True

            if best_idx >= len(self.routes) or self.routes[best_idx] is None:
                eval_res.is_fallback = True

            return eval_res

    def filter(self, ctx: Any, text: str, payload: Any = None) -> Any:
        """Evaluates query and executes the appropriate handler."""
        eval_res = self._evaluate_fast(text)

        with self.lock:
            # 1. Fallback Tier
            if eval_res.is_fallback:
                if self.fallback_handler is not None:
                    return self.fallback_handler(ctx, payload)
                return None

            # 2. Ambiguous Tier
            if eval_res.is_ambiguous:
                if self.ambiguous_handler is not None:
                    p_label = self.labels[eval_res.primary_idx]
                    s_label = self.labels[eval_res.secondary_idx] if eval_res.secondary_idx >= 0 else ""
                    return self.ambiguous_handler(ctx, p_label, s_label, payload)
                if self.fallback_handler is not None:
                    return self.fallback_handler(ctx, payload)
                return None

            # 3. Definite Route Tier
            if eval_res.primary_idx < 0 or eval_res.primary_idx >= self.class_count or self.routes[eval_res.primary_idx] is None:
                if self.fallback_handler is not None:
                    return self.fallback_handler(ctx, payload)
                return None

            return self.routes[eval_res.primary_idx](ctx, payload)

    def filter_pipeline(self, ctx: Any, text: str, payload: Any = None) -> Any:
        """Evaluates query supporting definite, multi-intent pipeline, and fallback executions."""
        eval_res = self._evaluate_fast(text)

        with self.lock:
            # 1. Fallback Tier
            if eval_res.is_fallback:
                if self.fallback_handler is not None:
                    return self.fallback_handler(ctx, payload)
                return None

            p_label = self.labels[eval_res.primary_idx]
            s_label = self.labels[eval_res.secondary_idx] if eval_res.secondary_idx >= 0 else ""

            # 2. Multi-Intent Pipeline Tier
            if eval_res.is_pipeline and s_label:
                pipe_key = pipeline_key(p_label, s_label)
                if pipe_key in self.pipelines and self.pipelines[pipe_key] is not None:
                    return self.pipelines[pipe_key](ctx, p_label, s_label, payload)

            # 3. Ambiguous Tier
            if eval_res.is_ambiguous:
                if self.ambiguous_handler is not None:
                    return self.ambiguous_handler(ctx, p_label, s_label, payload)
                if self.fallback_handler is not None:
                    return self.fallback_handler(ctx, payload)
                return None

            # 4. Definite Route Tier
            if eval_res.primary_idx < 0 or eval_res.primary_idx >= self.class_count or self.routes[eval_res.primary_idx] is None:
                if self.fallback_handler is not None:
                    return self.fallback_handler(ctx, payload)
                return None

            return self.routes[eval_res.primary_idx](ctx, payload)

    def filter_tokens(self, ctx: Any, tokens: Sequence[int], payload: Any = None) -> Any:
        """Evaluates pre-tokenized inputs with zero allocation."""
        eval_res = self._evaluate_fast_tokens(tokens)

        with self.lock:
            if eval_res.is_fallback:
                if self.fallback_handler is not None:
                    return self.fallback_handler(ctx, payload)
                return None

            if eval_res.is_ambiguous:
                if self.ambiguous_handler is not None:
                    p_label = self.labels[eval_res.primary_idx]
                    s_label = self.labels[eval_res.secondary_idx] if eval_res.secondary_idx >= 0 else ""
                    return self.ambiguous_handler(ctx, p_label, s_label, payload)
                if self.fallback_handler is not None:
                    return self.fallback_handler(ctx, payload)
                return None

            if eval_res.primary_idx < 0 or eval_res.primary_idx >= self.class_count or self.routes[eval_res.primary_idx] is None:
                if self.fallback_handler is not None:
                    return self.fallback_handler(ctx, payload)
                return None

            return self.routes[eval_res.primary_idx](ctx, payload)
