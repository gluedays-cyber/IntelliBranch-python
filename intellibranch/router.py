import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
import numpy as np

from intellibranch.binary import load_binary_model, MAX_SEQUENCE_TOKENS
from intellibranch.policy import DispatchPolicy, default_dispatch_policy
from intellibranch.runtime import (
    InferenceModel,
    MAX_INPUT_BYTES,
    compute_entropy,
    truncate_to_rune_boundary,
)
from intellibranch.telemetry import TelemetryEvent, TelemetryRingBuffer

RouteAction = Callable[[Any, Any], Any]
AmbiguousAction = Callable[[Any, str, str, Any], Any]
PipelineAction = Callable[[Any, str, str, Any], Any]


def pipeline_key(primary: str, secondary: str) -> str:
    return f"{primary}->{secondary}"


@dataclass
class RouteTrace:
    input_text: str
    token_ids: List[int] = field(default_factory=list)
    subwords: List[str] = field(default_factory=list)
    unknown_token_ratio: float = 0.0
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    predicted_label: str = ""
    secondary_label: str = ""
    confidence: float = 0.0
    margin: float = 0.0
    entropy: float = 0.0
    threshold: float = 0.75
    is_ambiguous: bool = False
    is_pipeline: bool = False
    is_fallback: bool = False
    fallback_reason: str = ""
    latency_micros: int = 0


class Router:
    """Coordinates in-memory inference routing with hot-swap, 3-tier safety, and telemetry feedback."""

    def __init__(self, model_path: str = "", default_threshold: float = 0.75, model: InferenceModel | None = None) -> None:
        self.lock = threading.RLock()
        self.threshold: float = default_threshold
        self.policy: DispatchPolicy = default_dispatch_policy()
        if default_threshold > 0.0:
            self.policy.high_threshold = default_threshold
            self.policy.low_threshold = default_threshold * 0.6

        self.routes: Dict[str, RouteAction] = {}
        self.pipelines: Dict[str, PipelineAction] = {}
        self.default_pipeline_action: PipelineAction | None = None
        self.ambiguous_action: AmbiguousAction | None = None
        self.fallback_action: RouteAction = lambda ctx, payload: None
        self.telemetry: TelemetryRingBuffer = TelemetryRingBuffer(1024)

        if model is not None:
            self.model: InferenceModel | None = model
        elif model_path:
            self.model = load_binary_model(model_path)
        else:
            self.model = None

    @classmethod
    def from_file(cls, model_path: str, default_threshold: float = 0.75) -> "Router":
        return cls(model_path=model_path, default_threshold=default_threshold)

    def reload(self, model_path: str) -> None:
        """Parses, validates, and atomically swaps model weights without interrupting active traffic."""
        new_model = load_binary_model(model_path)
        with self.lock:
            self.model = new_model

    def swap_model(self, new_model: InferenceModel) -> None:
        """Replaces the active inference model atomically."""
        with self.lock:
            self.model = new_model

    def get_model(self) -> InferenceModel | None:
        with self.lock:
            return self.model

    def enable_telemetry(self, capacity: int) -> "Router":
        """Configures or resizes the telemetry ring buffer for active learning feedback."""
        with self.lock:
            self.telemetry = TelemetryRingBuffer(capacity)
        return self

    def drain_telemetry(self) -> List[TelemetryEvent]:
        """Extracts all recorded routing events for drift monitoring and retraining."""
        with self.lock:
            return self.telemetry.drain()

    def _record_telemetry(
        self,
        text: str,
        primary: str,
        secondary: str,
        conf: float,
        entropy: float,
        is_ambiguous: bool,
        is_pipeline: bool,
        is_fallback: bool,
    ) -> None:
        if self.telemetry is not None and (is_ambiguous or is_pipeline or is_fallback):
            self.telemetry.push(
                TelemetryEvent(
                    input_text=text,
                    predicted_label=primary,
                    secondary_label=secondary,
                    confidence=conf,
                    entropy=entropy,
                    is_ambiguous=is_ambiguous,
                    is_pipeline=is_pipeline,
                    is_fallback=is_fallback,
                )
            )

    def set_policy(self, policy: DispatchPolicy) -> "Router":
        with self.lock:
            self.policy = policy
        return self

    def bind(self, label: str, action: RouteAction) -> "Router":
        with self.lock:
            self.routes[label] = action
        return self

    def bind_pipeline(self, primary: str, secondary: str, action: PipelineAction) -> "Router":
        with self.lock:
            self.pipelines[pipeline_key(primary, secondary)] = action
        return self

    def default_pipeline(self, action: PipelineAction) -> "Router":
        with self.lock:
            self.default_pipeline_action = action
        return self

    def ambiguous(self, action: AmbiguousAction) -> "Router":
        with self.lock:
            self.ambiguous_action = action
        return self

    def fallback(self, action: RouteAction) -> "Router":
        with self.lock:
            self.fallback_action = action
        return self

    def inspect(self, text: str) -> RouteTrace:
        """Evaluates input text and generates a full diagnostic RouteTrace."""
        start = time.perf_counter()
        with self.lock:
            model = self.model
            threshold = self.threshold

        if model is None:
            return RouteTrace(
                input_text=text,
                is_fallback=True,
                fallback_reason="model not loaded",
                threshold=threshold,
                latency_micros=int((time.perf_counter() - start) * 1_000_000),
            )

        if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
            text = truncate_to_rune_boundary(text, MAX_INPUT_BYTES)

        tokens = model.tokenizer.encode(text)
        if not tokens:
            return RouteTrace(
                input_text=text,
                is_fallback=True,
                fallback_reason="empty input tokens",
                threshold=threshold,
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

        unk_ratio = float(unk_count) / float(len(tokens)) if tokens else 0.0
        probs = model.forward(tokens, model.temperature)

        prob_map: Dict[str, float] = {}
        best_label = ""
        second_label = ""
        best_score = -1.0
        second_score = -1.0

        for i, p in enumerate(probs):
            if i < len(model.labels):
                lbl = model.labels[i]
                p_val = float(p)
                prob_map[lbl] = p_val
                if p_val > best_score:
                    second_score = best_score
                    second_label = best_label
                    best_score = p_val
                    best_label = lbl
                elif p_val > second_score:
                    second_score = p_val
                    second_label = lbl

        calibrated_conf = best_score * (1.0 - unk_ratio)
        calibrated_second = second_score * (1.0 - unk_ratio)
        margin = calibrated_conf - calibrated_second
        entropy = compute_entropy(probs)

        trace = RouteTrace(
            input_text=text,
            token_ids=tokens,
            subwords=subwords,
            unknown_token_ratio=unk_ratio,
            class_probabilities=prob_map,
            predicted_label=best_label,
            secondary_label=second_label,
            confidence=calibrated_conf,
            margin=margin,
            entropy=entropy,
            threshold=self.policy.high_threshold,
            latency_micros=int((time.perf_counter() - start) * 1_000_000),
        )

        with self.lock:
            if trace.confidence < self.policy.low_threshold:
                trace.is_fallback = True
                trace.fallback_reason = f"confidence {trace.confidence:.4f} below low threshold {self.policy.low_threshold:.4f}"
            elif unk_ratio >= 0.5:
                trace.is_fallback = True
                trace.fallback_reason = f"excessive unknown tokens ({unk_ratio:.2f} >= 0.50)"
            elif entropy > self.policy.max_entropy:
                trace.is_fallback = True
                trace.fallback_reason = f"prediction entropy {entropy:.4f} exceeds limit {self.policy.max_entropy:.4f} (OOD)"
            else:
                if second_label and calibrated_second >= self.policy.pipeline_threshold:
                    trace.is_pipeline = True
                if trace.confidence < self.policy.high_threshold or margin < self.policy.margin_cutoff:
                    trace.is_ambiguous = True
                if best_label not in self.routes:
                    trace.is_fallback = True
                    trace.fallback_reason = f"label '{best_label}' has no bound route handler"

        return trace

    def dispatch(self, ctx: Any, text: str, payload: Any = None) -> Any:
        """Executes inference and routes through a 3-tier decision pipeline (Definite / Ambiguous / Fallback)."""
        with self.lock:
            model = self.model

        if model is None:
            self._record_telemetry(text, "", "", 0.0, 0.0, False, False, True)
            return self.fallback_action(ctx, payload)

        try:
            res, unk_ratio = model.predict_detailed(text)
        except Exception:
            self._record_telemetry(text, "", "", 0.0, 0.0, False, False, True)
            return self.fallback_action(ctx, payload)

        if res.total == 0:
            self._record_telemetry(text, "", "", 0.0, 0.0, False, False, True)
            return self.fallback_action(ctx, payload)

        primary_idx = res.primary.index
        if primary_idx < 0 or primary_idx >= len(model.labels):
            self._record_telemetry(text, "", "", 0.0, 0.0, False, False, True)
            return self.fallback_action(ctx, payload)

        primary_label = model.labels[primary_idx]
        primary_conf = float(res.primary.confidence)

        secondary_label = ""
        secondary_conf = 0.0
        if res.total >= 2:
            sec_idx = res.secondary.index
            if 0 <= sec_idx < len(model.labels):
                secondary_label = model.labels[sec_idx]
                secondary_conf = float(res.secondary.confidence)

        margin = primary_conf - secondary_conf
        entropy = float(res.entropy)

        with self.lock:
            # 1. Fallback Isolation
            if primary_conf < self.policy.low_threshold or unk_ratio >= 0.5 or entropy > self.policy.max_entropy:
                self._record_telemetry(text, primary_label, secondary_label, primary_conf, entropy, False, False, True)
                return self.fallback_action(ctx, payload)

            # 2. Ambiguous Route
            is_ambiguous = primary_conf < self.policy.high_threshold or margin < self.policy.margin_cutoff
            if is_ambiguous:
                self._record_telemetry(text, primary_label, secondary_label, primary_conf, entropy, True, False, False)
                if self.ambiguous_action is not None:
                    return self.ambiguous_action(ctx, primary_label, secondary_label, payload)
                return self.fallback_action(ctx, payload)

            # 3. Definite Route
            action = self.routes.get(primary_label)
            if action is None:
                self._record_telemetry(text, primary_label, secondary_label, primary_conf, entropy, False, False, True)
                return self.fallback_action(ctx, payload)

            return action(ctx, payload)

    def dispatch_pipeline(self, ctx: Any, text: str, payload: Any = None) -> Any:
        """Routes requests with multi-intent support, executing pipeline handlers when both top-1 and top-2 qualify."""
        with self.lock:
            model = self.model

        if model is None:
            self._record_telemetry(text, "", "", 0.0, 0.0, False, False, True)
            return self.fallback_action(ctx, payload)

        try:
            res, unk_ratio = model.predict_detailed(text)
        except Exception:
            self._record_telemetry(text, "", "", 0.0, 0.0, False, False, True)
            return self.fallback_action(ctx, payload)

        if res.total == 0:
            self._record_telemetry(text, "", "", 0.0, 0.0, False, False, True)
            return self.fallback_action(ctx, payload)

        primary_idx = res.primary.index
        if primary_idx < 0 or primary_idx >= len(model.labels):
            self._record_telemetry(text, "", "", 0.0, 0.0, False, False, True)
            return self.fallback_action(ctx, payload)

        primary_label = model.labels[primary_idx]
        primary_conf = float(res.primary.confidence)

        secondary_label = ""
        secondary_conf = 0.0
        if res.total >= 2:
            sec_idx = res.secondary.index
            if 0 <= sec_idx < len(model.labels):
                secondary_label = model.labels[sec_idx]
                secondary_conf = float(res.secondary.confidence)

        entropy = float(res.entropy)

        with self.lock:
            # 1. Fallback Isolation
            if primary_conf < self.policy.low_threshold or unk_ratio >= 0.5 or entropy > self.policy.max_entropy:
                self._record_telemetry(text, primary_label, secondary_label, primary_conf, entropy, False, False, True)
                return self.fallback_action(ctx, payload)

            # 2. Multi-Intent Pipeline Dispatch
            if secondary_label and secondary_conf >= self.policy.pipeline_threshold:
                self._record_telemetry(text, primary_label, secondary_label, primary_conf, entropy, False, True, False)
                key = pipeline_key(primary_label, secondary_label)
                if key in self.pipelines:
                    return self.pipelines[key](ctx, primary_label, secondary_label, payload)
                if self.default_pipeline_action is not None:
                    return self.default_pipeline_action(ctx, primary_label, secondary_label, payload)

            # 3. Fallback to standard 3-tier routing
            margin = primary_conf - secondary_conf
            is_ambiguous = primary_conf < self.policy.high_threshold or margin < self.policy.margin_cutoff
            if is_ambiguous:
                self._record_telemetry(text, primary_label, secondary_label, primary_conf, entropy, True, False, False)
                if self.ambiguous_action is not None:
                    return self.ambiguous_action(ctx, primary_label, secondary_label, payload)
                return self.fallback_action(ctx, payload)

            action = self.routes.get(primary_label)
            if action is None:
                self._record_telemetry(text, primary_label, secondary_label, primary_conf, entropy, False, False, True)
                return self.fallback_action(ctx, payload)

            return action(ctx, payload)

    def dispatch_with_trace(self, ctx: Any, text: str, payload: Any = None) -> Tuple[RouteTrace, Any]:
        """Runs inference, collects full diagnostic telemetry, and dispatches to handler."""
        trace = self.inspect(text)

        with self.lock:
            self._record_telemetry(
                text=text,
                primary=trace.predicted_label,
                secondary=trace.secondary_label,
                conf=trace.confidence,
                entropy=trace.entropy,
                is_ambiguous=trace.is_ambiguous,
                is_pipeline=trace.is_pipeline,
                is_fallback=trace.is_fallback,
            )

            if trace.is_fallback:
                return trace, self.fallback_action(ctx, payload)

            if trace.is_pipeline:
                key = pipeline_key(trace.predicted_label, trace.secondary_label)
                if key in self.pipelines:
                    return trace, self.pipelines[key](ctx, trace.predicted_label, trace.secondary_label, payload)
                if self.default_pipeline_action is not None:
                    return trace, self.default_pipeline_action(ctx, trace.predicted_label, trace.secondary_label, payload)

            if trace.is_ambiguous and self.ambiguous_action is not None:
                return trace, self.ambiguous_action(ctx, trace.predicted_label, trace.secondary_label, payload)

            action = self.routes.get(trace.predicted_label)
            if action is None:
                return trace, self.fallback_action(ctx, payload)

            return trace, action(ctx, payload)
