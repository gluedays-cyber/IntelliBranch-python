import csv
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np

from intellibranch.binary import CURRENT_FORMAT_VERSION, Header, MAGIC_BYTES, MAX_SEQUENCE_TOKENS, Weights
from intellibranch.ops import gelu, gelu_derivative, matmul_vec_add, mean_pooling_with_pos, softmax
from intellibranch.runtime import InferenceModel
from intellibranch.tokenizer import BPETokenizer


@dataclass
class TrainConfig:
    """Hyperparameter specification for offline model training."""
    embedding_dim: int = 64
    hidden_dim: int = 128
    target_vocab_size: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    epochs: int = 100
    batch_size: int = 32
    patience: int = 5
    seed: int = 42


def default_train_config() -> TrainConfig:
    return TrainConfig()


@dataclass
class DataSample:
    text: str
    label: str


def load_csv_dataset(file_path: str) -> List[DataSample]:
    """Reads training data from a two-column (text, label) CSV file."""
    samples: List[DataSample] = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("CSV dataset is empty")

        if len(header) < 2:
            raise ValueError("CSV must contain at least 2 columns (text, label)")

        for row in reader:
            if len(row) < 2:
                continue
            text = row[0].strip()
            label = row[1].strip()
            if text and label:
                samples.append(DataSample(text=text, label=label))

    if not samples:
        raise ValueError("dataset contains no valid samples")

    return samples


class AdamWState:
    """Maintains first and second momentum vectors for AdamW parameter updates."""

    def __init__(self, size: int) -> None:
        self.m: np.ndarray = np.zeros(size, dtype=np.float32)
        self.v: np.ndarray = np.zeros(size, dtype=np.float32)
        self.t: int = 0

    def step(
        self,
        param: np.ndarray,
        grad: np.ndarray,
        lr: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        eps: float,
    ) -> None:
        self.t += 1
        t_float = float(self.t)
        bc1 = 1.0 - (beta1 ** t_float)
        bc2 = 1.0 - (beta2 ** t_float)

        self.m = beta1 * self.m + (1.0 - beta1) * grad
        self.v = beta2 * self.v + (1.0 - beta2) * (grad * grad)

        m_hat = self.m / bc1
        v_hat = self.v / bc2

        step_update = lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * param)
        param -= step_update


@dataclass
class _EncodedSample:
    tokens: List[int]
    class_id: int


def train_model(samples: Sequence[DataSample], cfg: TrainConfig | None = None) -> InferenceModel:
    """Executes the complete training pipeline including BPE, AdamW optimization, and Early Stopping."""
    if cfg is None:
        cfg = default_train_config()

    random.seed(cfg.seed)
    np_rng = np.random.RandomState(cfg.seed)

    # 1. Unique labels
    label_map: Dict[str, int] = {}
    labels: List[str] = []
    for s in samples:
        if s.label not in label_map:
            label_map[s.label] = len(labels)
            labels.append(s.label)

    num_classes = len(labels)
    if num_classes < 2:
        raise ValueError(f"dataset must contain at least 2 distinct classes, found {num_classes}")

    # 2. Train BPE Tokenizer
    corpus = [s.text for s in samples]
    tokenizer = BPETokenizer.train(corpus, cfg.target_vocab_size)
    vocab_size = tokenizer.vocab_size()

    # 3. Encode dataset
    class_buckets: Dict[int, List[_EncodedSample]] = {c: [] for c in range(num_classes)}
    for s in samples:
        tokens = tokenizer.encode(s.text)
        if not tokens:
            tokens = [0]
        cid = label_map[s.label]
        class_buckets[cid].append(_EncodedSample(tokens=tokens, class_id=cid))

    train_set: List[_EncodedSample] = []
    val_set: List[_EncodedSample] = []

    for b in class_buckets.values():
        random.shuffle(b)
        if len(b) <= 2:
            train_set.extend(b)
            val_set.extend(b)
        else:
            val_cnt = max(1, int(len(b) * 0.15))
            train_set.extend(b[val_cnt:])
            val_set.extend(b[:val_cnt])

    # 4. Initialize Network Weights
    header = Header(
        magic=MAGIC_BYTES,
        version=CURRENT_FORMAT_VERSION,
        vocab_size=vocab_size,
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=num_classes,
    )

    pos_len = MAX_SEQUENCE_TOKENS * cfg.embedding_dim
    emb_scale = math.sqrt(1.0 / float(cfg.embedding_dim))
    w1_scale = math.sqrt(2.0 / float(cfg.embedding_dim))
    w2_scale = math.sqrt(2.0 / float(cfg.hidden_dim))

    emb_weights = (np_rng.uniform(-1.0, 1.0, vocab_size * cfg.embedding_dim) * emb_scale).astype(np.float32)
    pos_weights = (np_rng.uniform(-1.0, 1.0, pos_len) * emb_scale).astype(np.float32)
    w1_weights = (np_rng.uniform(-1.0, 1.0, cfg.embedding_dim * cfg.hidden_dim) * w1_scale).astype(np.float32)
    b1_weights = np.zeros(cfg.hidden_dim, dtype=np.float32)
    w2_weights = (np_rng.uniform(-1.0, 1.0, cfg.hidden_dim * num_classes) * w2_scale).astype(np.float32)
    b2_weights = np.zeros(num_classes, dtype=np.float32)

    # Initialize AdamW
    opt_emb = AdamWState(len(emb_weights))
    opt_pos = AdamWState(len(pos_weights))
    opt_w1 = AdamWState(len(w1_weights))
    opt_b1 = AdamWState(len(b1_weights))
    opt_w2 = AdamWState(len(w2_weights))
    opt_b2 = AdamWState(len(b2_weights))

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    patience_counter = 0

    best_weights: Weights | None = None

    batch_size = cfg.batch_size
    if len(train_set) <= 32:
        batch_size = 4
    elif batch_size > len(train_set):
        batch_size = len(train_set)
    batch_size = max(1, batch_size)

    # Training Loop
    for epoch in range(1, cfg.epochs + 1):
        random.shuffle(train_set)

        grad_emb = np.zeros_like(emb_weights)
        grad_pos = np.zeros_like(pos_weights)
        grad_w1 = np.zeros_like(w1_weights)
        grad_b1 = np.zeros_like(b1_weights)
        grad_w2 = np.zeros_like(w2_weights)
        grad_b2 = np.zeros_like(b2_weights)

        accum_count = 0

        for idx, sample in enumerate(train_set):
            # Forward pass
            pooled = mean_pooling_with_pos(sample.tokens, emb_weights, pos_weights, cfg.embedding_dim)
            z1 = matmul_vec_add(pooled, w1_weights, b1_weights, cfg.embedding_dim, cfg.hidden_dim)
            a1 = np.array([gelu(v) for v in z1], dtype=np.float32)
            z2 = matmul_vec_add(a1, w2_weights, b2_weights, cfg.hidden_dim, num_classes)
            probs = softmax(z2, 1.0)

            # Backward pass
            dz2 = probs.copy()
            dz2[sample.class_id] -= 1.0

            grad_w2 += np.outer(a1, dz2).flatten()
            grad_b2 += dz2

            da1 = np.dot(w2_weights.reshape((cfg.hidden_dim, num_classes)), dz2)
            dz1 = np.array([da1[h] * gelu_derivative(z1[h]) for h in range(cfg.hidden_dim)], dtype=np.float32)

            grad_w1 += np.outer(pooled, dz1).flatten()
            grad_b1 += dz1

            d_mean = np.dot(w1_weights.reshape((cfg.embedding_dim, cfg.hidden_dim)), dz1)

            inv_len = 1.0 / float(len(sample.tokens))
            for pos, tok in enumerate(sample.tokens):
                tok_offset = tok * cfg.embedding_dim
                pos_offset = pos * cfg.embedding_dim
                for e in range(cfg.embedding_dim):
                    sum_val = emb_weights[tok_offset + e] + (
                        pos_weights[pos_offset + e] if pos_offset + e < len(pos_weights) else 0.0
                    )
                    g = d_mean[e] * inv_len * gelu_derivative(sum_val)
                    grad_emb[tok_offset + e] += g
                    if pos_offset + e < len(grad_pos):
                        grad_pos[pos_offset + e] += g

            accum_count += 1

            if accum_count % batch_size == 0 or idx == len(train_set) - 1:
                scale = 1.0 / float(accum_count)
                opt_emb.step(emb_weights, grad_emb * scale, cfg.learning_rate, 0.0, cfg.beta1, cfg.beta2, cfg.epsilon)
                opt_pos.step(pos_weights, grad_pos * scale, cfg.learning_rate, 0.0, cfg.beta1, cfg.beta2, cfg.epsilon)
                opt_w1.step(w1_weights, grad_w1 * scale, cfg.learning_rate, cfg.weight_decay, cfg.beta1, cfg.beta2, cfg.epsilon)
                opt_b1.step(b1_weights, grad_b1 * scale, cfg.learning_rate, 0.0, cfg.beta1, cfg.beta2, cfg.epsilon)
                opt_w2.step(w2_weights, grad_w2 * scale, cfg.learning_rate, cfg.weight_decay, cfg.beta1, cfg.beta2, cfg.epsilon)
                opt_b2.step(b2_weights, grad_b2 * scale, cfg.learning_rate, 0.0, cfg.beta1, cfg.beta2, cfg.epsilon)

                grad_emb.fill(0.0)
                grad_pos.fill(0.0)
                grad_w1.fill(0.0)
                grad_b1.fill(0.0)
                grad_w2.fill(0.0)
                grad_b2.fill(0.0)
                accum_count = 0

        # Evaluate Train & Validation
        train_loss = 0.0
        train_correct = 0
        for sample in train_set:
            p_vec = mean_pooling_with_pos(sample.tokens, emb_weights, pos_weights, cfg.embedding_dim)
            h_vec = matmul_vec_add(p_vec, w1_weights, b1_weights, cfg.embedding_dim, cfg.hidden_dim)
            act_vec = np.array([gelu(v) for v in h_vec], dtype=np.float32)
            logits = matmul_vec_add(act_vec, w2_weights, b2_weights, cfg.hidden_dim, num_classes)
            pr = softmax(logits, 1.0)
            p_target = max(float(pr[sample.class_id]), 1e-7)
            train_loss -= math.log(p_target)
            if int(np.argmax(pr)) == sample.class_id:
                train_correct += 1

        train_loss /= float(len(train_set))
        train_acc = float(train_correct) / float(len(train_set))

        val_loss = 0.0
        val_correct = 0
        for sample in val_set:
            p_vec = mean_pooling_with_pos(sample.tokens, emb_weights, pos_weights, cfg.embedding_dim)
            h_vec = matmul_vec_add(p_vec, w1_weights, b1_weights, cfg.embedding_dim, cfg.hidden_dim)
            act_vec = np.array([gelu(v) for v in h_vec], dtype=np.float32)
            logits = matmul_vec_add(act_vec, w2_weights, b2_weights, cfg.hidden_dim, num_classes)
            pr = softmax(logits, 1.0)
            p_target = max(float(pr[sample.class_id]), 1e-7)
            val_loss -= math.log(p_target)
            if int(np.argmax(pr)) == sample.class_id:
                val_correct += 1

        val_loss /= float(len(val_set))
        val_acc = float(val_correct) / float(len(val_set))

        # Check Early Stopping
        is_better = False
        if len(samples) < 50:
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                is_better = True
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                is_better = True

        if is_better:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = Weights(
                embedding=emb_weights.copy(),
                positional=pos_weights.copy(),
                w1=w1_weights.copy(),
                b1=b1_weights.copy(),
                w2=w2_weights.copy(),
                b2=b2_weights.copy(),
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience and epoch >= 30:
                print(f"[Early Stopping] Triggered at epoch {epoch} (Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f})")
                break

        if epoch % 10 == 0 or epoch == cfg.epochs:
            print(f"Epoch {epoch:3d}/{cfg.epochs:3d} - Train Loss: {train_loss:.4f} (Acc: {train_acc*100.0:.1f}%) | Val Loss: {val_loss:.4f} (Acc: {val_acc*100.0:.1f}%)")

    if best_weights is None:
        best_weights = Weights(
            embedding=emb_weights,
            positional=pos_weights,
            w1=w1_weights,
            b1=b1_weights,
            w2=w2_weights,
            b2=b2_weights,
        )

    return InferenceModel(
        header=header,
        labels=labels,
        vocab=tokenizer.vocab,
        merge_rules=tokenizer.merge_rules,
        weights=best_weights,
    )
