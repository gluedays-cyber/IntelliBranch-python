import math
from typing import Sequence, Tuple
import numpy as np

# Sqrt2OverPi is sqrt(2 / pi) for the GELU tanh approximation.
SQRT_2_OVER_PI: float = 0.7978845608

# GeluCoeff is the polynomial coefficient for GELU tanh approximation.
GELU_COEFF: float = 0.044715

# NumericalClampLimit protects float32 values from overflowing into Inf/NaN.
NUMERICAL_CLAMP_LIMIT: float = 100.0


def safe_clamp(val: float, limit: float = NUMERICAL_CLAMP_LIMIT) -> float:
    """Restricts a float32 within [-limit, limit] and replaces NaN/Inf with 0.0."""
    if math.isnan(val) or math.isinf(val):
        return 0.0
    if val < -limit:
        return -limit
    if val > limit:
        return limit
    return val


def gelu(x: float) -> float:
    """Calculates the Gaussian Error Linear Unit activation using the standard tanh approximation.
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    """
    x_clamped = safe_clamp(x, NUMERICAL_CLAMP_LIMIT)
    cube = x_clamped * x_clamped * x_clamped
    inner = safe_clamp(SQRT_2_OVER_PI * (x_clamped + GELU_COEFF * cube), NUMERICAL_CLAMP_LIMIT)
    tanh_val = math.tanh(inner)
    return safe_clamp(0.5 * x_clamped * (1.0 + tanh_val), NUMERICAL_CLAMP_LIMIT)


def gelu_array(arr: np.ndarray) -> np.ndarray:
    """Vectorized GELU activation function with numerical clamp protection."""
    clamped = np.nan_to_num(arr, nan=0.0, posinf=NUMERICAL_CLAMP_LIMIT, neginf=-NUMERICAL_CLAMP_LIMIT)
    clamped = np.clip(clamped, -NUMERICAL_CLAMP_LIMIT, NUMERICAL_CLAMP_LIMIT)
    cube = clamped ** 3
    inner = np.clip(SQRT_2_OVER_PI * (clamped + GELU_COEFF * cube), -NUMERICAL_CLAMP_LIMIT, NUMERICAL_CLAMP_LIMIT)
    tanh_val = np.tanh(inner)
    out = 0.5 * clamped * (1.0 + tanh_val)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=NUMERICAL_CLAMP_LIMIT, neginf=-NUMERICAL_CLAMP_LIMIT), -NUMERICAL_CLAMP_LIMIT, NUMERICAL_CLAMP_LIMIT).astype(np.float32)


def gelu_derivative(x: float) -> float:
    """Calculates d(GELU(x))/dx for backpropagation."""
    cube = x * x * x
    u = SQRT_2_OVER_PI * (x + GELU_COEFF * cube)
    tanh_u = math.tanh(u)
    du = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEFF * x * x)
    sech2 = 1.0 - tanh_u * tanh_u
    return 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2 * du


def gelu_derivative_array(x: np.ndarray) -> np.ndarray:
    """Vectorized d(GELU(x))/dx for backpropagation."""
    cube = x ** 3
    u = SQRT_2_OVER_PI * (x + GELU_COEFF * cube)
    tanh_u = np.tanh(u)
    du = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEFF * (x ** 2))
    sech2 = 1.0 - tanh_u ** 2
    return (0.5 * (1.0 + tanh_u) + 0.5 * x * sech2 * du).astype(np.float32)


def mean_pooling_with_pos(
    token_ids: Sequence[int],
    embedding_table: np.ndarray,
    pos_table: np.ndarray | None,
    emb_dim: int,
) -> np.ndarray:
    """Computes the average embedding vector across token IDs with learned positional embeddings."""
    seq_len = len(token_ids)
    if seq_len == 0:
        raise ValueError("cannot pool over zero tokens")

    out = np.zeros(emb_dim, dtype=np.float32)
    max_seq = 0
    if pos_table is not None and len(pos_table) > 0 and emb_dim > 0:
        max_seq = len(pos_table) // emb_dim

    for pos, tok_id in enumerate(token_ids):
        tok_offset = tok_id * emb_dim
        if tok_offset + emb_dim > len(embedding_table):
            raise IndexError("token ID exceeds embedding table bounds")

        emb_vec = embedding_table[tok_offset : tok_offset + emb_dim].astype(np.float32)
        emb_vec = np.clip(np.nan_to_num(emb_vec, nan=0.0), -NUMERICAL_CLAMP_LIMIT, NUMERICAL_CLAMP_LIMIT)

        if max_seq > 0 and pos < max_seq:
            pos_offset = pos * emb_dim
            pos_vec = pos_table[pos_offset : pos_offset + emb_dim].astype(np.float32)
            emb_vec = gelu_array(emb_vec + pos_vec)

        out += emb_vec
        out = np.clip(np.nan_to_num(out, nan=0.0), -NUMERICAL_CLAMP_LIMIT, NUMERICAL_CLAMP_LIMIT)

    inv_len = 1.0 / float(seq_len)
    out *= inv_len
    return np.clip(np.nan_to_num(out, nan=0.0), -NUMERICAL_CLAMP_LIMIT, NUMERICAL_CLAMP_LIMIT).astype(np.float32)


def mean_pooling(token_ids: Sequence[int], embedding_table: np.ndarray, emb_dim: int) -> np.ndarray:
    """Computes the average embedding vector across the given token IDs without positional encoding."""
    return mean_pooling_with_pos(token_ids, embedding_table, None, emb_dim)


def matmul_vec_add(vec: np.ndarray, weights: np.ndarray, bias: np.ndarray, in_dim: int, out_dim: int) -> np.ndarray:
    """Computes out = vec * weights + bias where vec is [in_dim], weights is [in_dim, out_dim] row-major,
    bias is [out_dim], and out is [out_dim].
    """
    v = np.clip(np.nan_to_num(vec[:in_dim].astype(np.float32), nan=0.0), -NUMERICAL_CLAMP_LIMIT, NUMERICAL_CLAMP_LIMIT)
    b = np.clip(np.nan_to_num(bias[:out_dim].astype(np.float32), nan=0.0), -NUMERICAL_CLAMP_LIMIT, NUMERICAL_CLAMP_LIMIT)
    w = np.clip(np.nan_to_num(weights.reshape((in_dim, out_dim)).astype(np.float32), nan=0.0), -NUMERICAL_CLAMP_LIMIT, NUMERICAL_CLAMP_LIMIT)

    res = np.dot(v, w) + b
    return np.clip(np.nan_to_num(res, nan=0.0), -NUMERICAL_CLAMP_LIMIT, NUMERICAL_CLAMP_LIMIT).astype(np.float32)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Computes numerically stable softmax probabilities over logits with temperature scaling."""
    n = len(logits)
    if n == 0:
        raise ValueError("empty logits")

    if temperature <= 0.0 or math.isnan(temperature) or math.isinf(temperature):
        temperature = 1.0

    inv_temp = 1.0 / temperature
    arr = np.clip(np.nan_to_num(logits.astype(np.float32), nan=0.0), -NUMERICAL_CLAMP_LIMIT, NUMERICAL_CLAMP_LIMIT) * inv_temp
    max_logit = float(np.max(arr))

    shifted = arr - max_logit
    exp_vals = np.exp(shifted)
    exp_vals = np.nan_to_num(exp_vals, nan=0.0, posinf=0.0, neginf=0.0)

    sum_exp = float(np.sum(exp_vals))
    if sum_exp <= 0.0 or math.isnan(sum_exp) or math.isinf(sum_exp):
        return np.full(n, 1.0 / float(n), dtype=np.float32)

    return (exp_vals / sum_exp).astype(np.float32)


def l2_normalize(vec: np.ndarray) -> Tuple[np.ndarray, float]:
    """Computes normalized = vec / ||vec||2 with numerical safety.
    Returns (normalized_vector, euclidean_norm).
    """
    v = np.nan_to_num(vec.astype(np.float32), nan=0.0)
    norm = float(np.linalg.norm(v))
    if norm < 1e-7 or math.isnan(norm) or math.isinf(norm):
        return np.zeros_like(v, dtype=np.float32), 0.0
    return (v / norm).astype(np.float32), norm


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the dot product between two vectors."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(np.dot(a[:n].astype(np.float32), b[:n].astype(np.float32)))
