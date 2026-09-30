"""IntelliBranch: Microsecond Neural Routing and Filtering Framework for Python."""

from intellibranch.binary import (
    Header,
    Weights,
    MergeRule,
    save_binary_model,
    load_binary_model,
    serialize_model,
    deserialize_model,
)
from intellibranch.neurogate import (
    NeuroGate,
    GateRouteBuilder,
    GateTrace,
    AnchorRule,
)
from intellibranch.ops import (
    gelu,
    gelu_array,
    safe_clamp,
    mean_pooling,
    mean_pooling_with_pos,
    matmul_vec_add,
    softmax,
    l2_normalize,
    dot_product,
)
from intellibranch.policy import (
    DispatchPolicy,
    default_dispatch_policy,
)
from intellibranch.router import (
    Router,
    RouteTrace,
)
from intellibranch.runtime import (
    InferenceModel,
    MatchSlot,
    StaticInferenceResult,
    compute_entropy,
    truncate_to_rune_boundary,
)
from intellibranch.telemetry import (
    TelemetryEvent,
    TelemetryRingBuffer,
)
from intellibranch.tokenizer import (
    BPETokenizer,
    is_valid_utf8,
)
from intellibranch.trainer import (
    TrainConfig,
    default_train_config,
    DataSample,
    load_csv_dataset,
    train_model,
)

__version__ = "2.0.0"

__all__ = [
    "AnchorRule",
    "BPETokenizer",
    "DataSample",
    "DispatchPolicy",
    "GateRouteBuilder",
    "GateTrace",
    "Header",
    "InferenceModel",
    "MatchSlot",
    "MergeRule",
    "NeuroGate",
    "RouteTrace",
    "Router",
    "StaticInferenceResult",
    "TelemetryEvent",
    "TelemetryRingBuffer",
    "TrainConfig",
    "Weights",
    "compute_entropy",
    "default_dispatch_policy",
    "default_train_config",
    "deserialize_model",
    "dot_product",
    "gelu",
    "gelu_array",
    "is_valid_utf8",
    "l2_normalize",
    "load_binary_model",
    "load_csv_dataset",
    "matmul_vec_add",
    "mean_pooling",
    "mean_pooling_with_pos",
    "safe_clamp",
    "save_binary_model",
    "serialize_model",
    "softmax",
    "train_model",
    "truncate_to_rune_boundary",
]
