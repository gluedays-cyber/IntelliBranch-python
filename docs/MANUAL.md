# IntelliBranch: Embedded Neural AI Manual & Tutorial for Python Developers

This guide provides Python developers and system architects with a deep-dive technical manual and hands-on tutorial for **IntelliBranch: An Engine That Directly Creates and Runs Its Own Domain Artificial Intelligence in Python**. Stop borrowing third-party cloud models or heavy deep learning daemons—learn how to structure domain knowledge, generate lightweight neural networks from scratch in seconds, and execute microsecond AI-driven control flow with pure Python and NumPy.

---

## Table of Contents

1. [Architectural Mental Model](#1-architectural-mental-model)
   - [Two Phases: Offline Training vs. In-Memory Routing](#11-two-phases-offline-training-vs-in-memory-routing)
2. [The 4-Step Operational Workflow](#2-the-4-step-operational-workflow)
3. [Keyword & API Reference Manual](#3-keyword--api-reference-manual)
   - [Constructor: `Router`](#31-router)
   - [3-Tier Criteria: `DispatchPolicy` & `set_policy`](#32-dispatchpolicy--set_policy)
   - [Branch Binding: `bind`](#33-bind)
   - [Borderline Safety: `ambiguous`](#34-ambiguous)
   - [Multi-Intent: `bind_pipeline` & `default_pipeline`](#35-bind_pipeline--default_pipeline)
   - [Safety Isolation: `fallback`](#36-fallback)
   - [Inference & Branching: `dispatch` & `dispatch_pipeline`](#37-dispatch--dispatch_pipeline)
   - [Whitebox Observability: `inspect` & `RouteTrace`](#38-inspect--routetrace)
   - [Atomic Hot-Swap: `reload` & `swap_model`](#39-reload--swap_model)
   - [Active Learning: `enable_telemetry` & `drain_telemetry`](#310-enable_telemetry--drain_telemetry)
   - [NeuroGate 3-Head Engine: `NeuroGate`](#311-neurogate-3-head-geometric-intelligent-filter-engine)
4. [End-to-End Production Tutorial](#4-end-to-end-production-tutorial)
   - [Step 1: Structuring Domain Knowledge (`dataset.csv`)](#step-1-structuring-domain-knowledge-datasetcsv)
   - [Step 2: Training & Model Generation (`ib-train`)](#step-2-training--model-generation-ib-train)
   - [Step 3: Microsecond Live Routing (`dispatch`)](#step-3-microsecond-live-routing-dispatch)
5. [Advanced Production Recipes](#5-advanced-production-recipes)
   - [Thread-Safe Atomic Hot-Reloading](#51-thread-safe-atomic-hot-reloading)
   - [Telemetry & Active Learning Feedback Loop](#52-telemetry--active-learning-feedback-loop)
   - [Semantic LLM Gateway & Cloud Bypass](#53-semantic-llm-gateway--cloud-bypass)
6. [IBRN Binary Wire Format Specification](#6-ibrn-binary-wire-format-specification)

---

## 1. Architectural Mental Model

In standard Python, control flow branching over unstructured strings relies on brittle string checking:

```python
# Standard Python: String matching
if "refund" in query or "cancel" in query:
    process_refund(query)
```

This collapses when callers use colloquial phrasing, typos, or context switches:
- *"Sent the return box a week ago, when do I get my money back?"* (Missed)
- *"Cancel shipment delay notifications"* (False positive trigger)

**IntelliBranch** replaces fragile pattern matching with **continuous latent vector coordinate proximity**:

```text
[ Input Text ] ("can u refund order #49281")
     │
     ▼
[ BPE Tokenizer ] ────── Splits into statistical subword tokens -> robust against typos & slang
     │
     ▼
[ 64-D Latent Coordinates + Positional Embeddings ] ── Domain intents mapped into compact vector space
     │
     ▼
[ Non-Linear GELU Mean Pooling ] ── Breaks permutation invariance, capturing word order
     │
     ▼
[ 128-D Hidden Layer ] ── Evaluates context combinations
     │
     ▼
[ Softmax Distribution + Shannon Entropy ] ── Converts logits to calibrated probabilities
     │
     ▼
[ 3-Tier Branch Dispatch ] ── Directly executes bound Python function in microseconds
```

### 1.1. Two Phases: Offline Training vs. In-Memory Routing

| Dimension | Phase 1: Model Training (Offline) | Phase 2: Router Dispatch (Live In-Memory Inference) |
| :--- | :--- | :--- |
| **What happens?** | Reads your `dataset.csv` and builds compact weights in **under 2 seconds** | Loads the `.bin` weights into RAM and routes requests in **microseconds** |
| **Output / Result** | A single portable binary file (`intent.bin`, < 150 KB) | Immediate execution of your Python handler (`router.bind(...)`) |
| **Runtime Resource** | Run once during deployment or server bootstrap | Consumes < 150 KB RAM and **0% background CPU** when idle |

---

## 2. The 4-Step Operational Workflow

```
1. Prepare CSV Dataset ──> 2. Compile Model (.bin) ──> 3. Bind Handlers in Python ──> 4. Dispatch Microsecond Traffic
   [data/intents.csv]        [python -m ...train]        [router.bind("Refund", fn)]       [router.dispatch(None, q, payload)]
```

---

## 3. Keyword & API Reference Manual

### 3.1. `Router`

Constructs an in-memory routing core from an existing `intent.bin` model file:

```python
from intellibranch import Router

router = Router(model_path="weights/intent.bin", default_threshold=0.60)
```

### 3.2. `DispatchPolicy` & `set_policy`

Defines production 3-tier routing boundaries:

```python
from intellibranch import DispatchPolicy

policy = DispatchPolicy(
    high_threshold=0.75,      # Above this: Definite execution
    low_threshold=0.40,       # Below this: Immediate Fallback isolation
    margin_cutoff=0.15,       # Minimum required gap between Top-1 and Top-2
    max_entropy=2.0,          # Shannon entropy limit for Out-of-Domain (OOD) isolation
    pipeline_threshold=0.30,  # Minimum secondary confidence for multi-intent
)
router.set_policy(policy)
```

### 3.3. `bind`

Registers a Python callable for a target class label:

```python
def handle_refund(ctx, payload):
    print(f"Refunding: {payload}")

router.bind("Refund", handle_refund)
```

### 3.4. `ambiguous`

Handles borderline requests where two intents compete closely:

```python
def handle_ambiguous(ctx, primary, secondary, payload):
    print(f"Ambiguous between {primary} and {secondary}. Prompt user for clarification.")

router.ambiguous(handle_ambiguous)
```

### 3.5. `bind_pipeline` & `default_pipeline`

Handles queries containing co-occurring multi-intent instructions:

```python
def handle_return_reship(ctx, primary, secondary, payload):
    print(f"Multi-intent pipeline: {primary} -> {secondary}")

router.bind_pipeline("Refund", "Delivery", handle_return_reship)
```

### 3.6. `fallback`

Default safety handler for low confidence, high entropy, or unhandled labels:

```python
router.fallback(lambda ctx, payload: print(f"Escalated to human support: {payload}"))
```

### 3.7. `dispatch` & `dispatch_pipeline`

Executes classification and dispatches to the registered handler:

```python
router.dispatch(None, "I want my money back", {"user_id": 42})
router.dispatch_pipeline(None, "refund delivery", {"user_id": 42})
```

### 3.8. `inspect` & `RouteTrace`

Returns complete whitebox diagnostic telemetry without executing business handlers:

```python
trace = router.inspect("can you refund my order")
print(f"Predicted: {trace.predicted_label}, Conf: {trace.confidence:.2f}, Latency: {trace.latency_micros} μs")
```

### 3.9. `reload` & `swap_model`

Atomically updates model weights without process restarts:

```python
router.reload("weights/new_model.bin")
```

### 3.10. `enable_telemetry` & `drain_telemetry`

Bounded ring buffer for collecting edge cases and active learning feedback:

```python
router.enable_telemetry(capacity=2048)
events = router.drain_telemetry()
```

### 3.11. `NeuroGate`: 3-Head Geometric Intelligent Filter Engine

Coordinates:
- **Head 1**: Geometric L2 Cosine Out-of-Domain Guard
- **Head 2**: Symbolic Anchor Bitmask & Soft-Bias
- **Head 3**: Stack Softmax, Margin, and Free Energy Gating

```python
from intellibranch import NeuroGate

gate = NeuroGate.from_file("weights/demo_cs.bin")
gate.bind("Refund", handle_refund).with_anchor(1.3, "refund", "money", "card")
gate.set_min_cosine_sim(0.35)

trace = gate.inspect("refund delivery")
gate.filter(None, "please refund the money to my card", None)
```

---

## 4. End-to-End Production Tutorial

### Step 1: Structuring Domain Knowledge (`dataset.csv`)

```csv
text,label
i want to cancel my payment and request a refund,Refund
where is my package delivery,Delivery
forgot my account login password,Account
```

### Step 2: Training & Model Generation (`ib-train`)

```bash
python -m intellibranch.cli.train --data data/sample_dataset.csv --out weights/intent.bin --epochs 60 --vocab 200
```

### Step 3: Microsecond Live Routing (`dispatch`)

```python
from intellibranch import Router

router = Router(model_path="weights/intent.bin", default_threshold=0.60)
router.bind("Refund", lambda ctx, q: print(f"Refund: {q}"))
router.bind("Delivery", lambda ctx, q: print(f"Delivery: {q}"))
router.fallback(lambda ctx, q: print(f"Fallback: {q}"))

router.dispatch(None, "cancel order and refund money", "query_payload")
```

---

## 5. Advanced Production Recipes

### 5.1. Thread-Safe Atomic Hot-Reloading

IntelliBranch `Router` and `NeuroGate` support thread-safe atomic model replacement under live concurrent traffic without locks blocking the read path.

### 5.2. Telemetry & Active Learning Feedback Loop

Drain uncertain or out-of-domain queries recorded in the ring buffer to retrain the next version of your weights.

### 5.3. Semantic LLM Gateway & Cloud Bypass

Filter routine user requests in microseconds locally ($0.00 cost) and escalate only genuine ambiguous or high-entropy requests to OpenAI/Claude.

---

## 6. IBRN Binary Wire Format Specification

- **Magic Signature**: `IBRN` (4 Bytes Little-Endian)
- **Version**: `2` (uint32)
- **Hyperparameters**: VocabSize, EmbeddingDim, HiddenDim, NumClasses (4x uint32)
- **Labels Block**: Length-prefixed UTF-8 strings
- **Vocabulary Block**: Length-prefixed UTF-8 subwords
- **BPE Merge Rules**: `(token1, token2, target)` tuples (3x uint32)
- **Tensors (IEEE 754 float32)**:
  - Embedding Table: `[VocabSize * EmbeddingDim]`
  - Positional Embeddings: `[128 * EmbeddingDim]`
  - Layer 1 Weights: `[EmbeddingDim * HiddenDim]`
  - Layer 1 Bias: `[HiddenDim]`
  - Layer 2 Weights: `[HiddenDim * NumClasses]`
  - Layer 2 Bias: `[NumClasses]`
- **Checksum**: SHA-256 (32 Bytes over all preceding bytes)
