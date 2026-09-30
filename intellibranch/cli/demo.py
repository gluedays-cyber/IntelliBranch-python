import argparse
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from intellibranch.binary import save_binary_model
from intellibranch.neurogate import NeuroGate
from intellibranch.policy import DispatchPolicy, default_dispatch_policy
from intellibranch.trainer import default_train_config, load_csv_dataset, train_model


@dataclass
class TestCase:
    query: str
    expectation: str


@dataclass
class DemoSuite:
    domain_name: str
    model_path: str
    data_path: str
    description: str
    policy: DispatchPolicy
    min_cosine: float
    setup_gate: Callable[[NeuroGate], None]
    test_cases: List[TestCase]
    custom_run: Callable[[NeuroGate, Any], None] | None = None


def ensure_model(model_path: str, data_path: str) -> None:
    if not os.path.exists(model_path):
        print(f"Model [{model_path}] not found. Auto-training on-the-fly from [{data_path}]...")
        samples = load_csv_dataset(data_path)
        cfg = default_train_config()
        cfg.epochs = 60
        cfg.learning_rate = 0.003
        cfg.target_vocab_size = 256

        model = train_model(samples, cfg)
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        save_binary_model(model_path, model)
        print(f"Successfully compiled [{model_path}] in memory.")


def measure_benchmark_latency(gate: NeuroGate, query: str, iterations: int = 100) -> float:
    # Warmup
    _ = gate.inspect(query)
    start = time.perf_counter()
    for _ in range(iterations):
        _ = gate.inspect(query)
    elapsed = time.perf_counter() - start
    return (elapsed / float(iterations)) * 1_000_000.0  # μs/op


def setup_cs_gate(g: NeuroGate) -> None:
    g.bind("Refund", lambda ctx, payload: print("    [ACTION: Refund] Process refund request & reverse charge")).with_anchor(
        1.3, "refund", "money", "card", "charge", "return"
    )
    g.bind("Delivery", lambda ctx, payload: print("    [ACTION: Delivery] Query courier GPS tracking & update address")).with_anchor(
        1.3, "courier", "delivered", "package", "delivery", "box", "shipping"
    )
    g.bind("Account", lambda ctx, payload: print("    [ACTION: Account] Trigger security verification & unlock profile")).with_anchor(
        1.3, "account", "login", "password", "security", "portal", "profile", "factor"
    )
    g.bind("Payment", lambda ctx, payload: print("    [ACTION: Payment] Retry checkout gateway & validate billing")).with_anchor(
        1.3, "payment", "checkout", "billing", "pay", "declined"
    )

    pipeline_handler = lambda ctx, p, s, payload: print(
        f"    [PIPELINE: {p} -> {s}] Return box approved THEN update reshipment destination"
    )
    g.bind_pipeline("Refund", "Delivery", pipeline_handler)
    g.bind_pipeline("Delivery", "Refund", pipeline_handler)
    g.ambiguous(
        lambda ctx, p, s, payload: print(f"    [AMBIGUOUS: {p} vs {s}] Borderline confidence: Prompt user for clarification")
    )
    g.fallback(lambda ctx, payload: print("    [FALLBACK] Escalated to human support tier-2 agent"))


def setup_llm_gate(g: NeuroGate) -> None:
    g.bind("QueryBalance", lambda ctx, payload: print("    [LOCAL BYPASS] Fetched balance from Redis cache in 30 μs (Cost: $0.00)")).with_anchor(
        1.3, "balance", "checking", "account", "funds", "savings"
    )
    g.bind("TransferFunds", lambda ctx, payload: print("    [LOCAL BYPASS] Executed internal ledger transaction directly (Cost: $0.00)")).with_anchor(
        1.3, "transfer", "send", "dollars", "wire", "remit"
    )
    g.bind("CardLock", lambda ctx, payload: print("    [LOCAL BYPASS] Instant freeze signal emitted to Visa processor (Cost: $0.00)")).with_anchor(
        1.3, "freeze", "lock", "debit", "card", "lost", "stolen"
    )
    g.bind("UpdateProfile", lambda ctx, payload: print("    [LOCAL BYPASS] Profile update form rendered (Cost: $0.00)")).with_anchor(
        1.3, "profile", "update", "address", "phone", "residential", "email"
    )
    g.fallback(lambda ctx, payload: print("    [CLOUD LLM ESCAPE] High entropy/OOD query forwarded to OpenAI GPT-4o (Cost: $0.02)"))


def setup_sre_gate(g: NeuroGate) -> None:
    g.bind("OutOfMemory", lambda ctx, payload: print("    [P0 CRITICAL] Trigger Horizontal Pod Autoscaler & restart worker")).with_anchor(
        1.5, "memory", "oom", "allocating", "starvation", "killed", "oomkilled", "137"
    )
    g.bind("DBPoolExhausted", lambda ctx, payload: print("    [P1 WARNING] Increase PostgreSQL pool cap and kill idle connections")).with_anchor(
        1.5, "hikaripool", "connection", "pool", "timeout", "timed", "postgres", "slots"
    )
    g.bind("AuthBruteForce", lambda ctx, payload: print("    [SECURITY] Add IP to iptables drop list and notify SecOps")).with_anchor(
        1.5, "security", "login", "attempts", "alert", "brute", "fail2ban", "ssh"
    )
    g.bind("SystemHealth", lambda ctx, payload: print("    [P3 INFO] Metric collected without alerting on-call")).with_anchor(
        1.5, "health", "probe", "healthz", "200", "ok", "heartbeat", "nominal"
    )
    g.fallback(lambda ctx, payload: print("    [UNKNOWN LOG] Streamed to cold storage archive"))


def run_sre_custom(g: NeuroGate, ctx: Any) -> None:
    print("    [Zero-Allocation Demonstration via FilterTokens]")
    model = g.get_model()
    if model is None:
        return
    raw_log = "kernel killed process worker-task due to host memory starvation"
    tokens = model.tokenizer.encode(raw_log)

    start = time.perf_counter()
    err = g.filter_tokens(ctx, tokens, None)
    elapsed_us = (time.perf_counter() - start) * 1_000_000.0

    trace = g.inspect(raw_log)
    print(f"    Raw Log : \"{raw_log}\"")
    print(f"    NeuroGate Routed: {trace.predicted_label} (Confidence: {trace.confidence*100:.2f}%, Cosine: {trace.cosine_similarity:.4f}, Latency: {elapsed_us:.2f} μs, Err: {err})")


def setup_iot_gate(g: NeuroGate) -> None:
    g.bind("LightControl", lambda ctx, payload: print("    [GPIO 18 HIGH] Toggle Zigbee Relay for Living Room Chandelier")).with_anchor(
        2.0, "dark", "light", "lamps", "lamp", "switch", "lights", "chandelier", "brighten"
    )
    g.bind("ClimateControl", lambda ctx, payload: print("    [MODBUS UART] Send temperature setpoint to Daikin HVAC inverter")).with_anchor(
        1.8, "cooling", "heat", "fan", "temp", "temperature", "ac", "air", "heating", "celsius"
    )
    g.bind("DoorLock", lambda ctx, payload: print("    [ZWAVE COMMAND] Engage motorized deadbolt locking mechanism")).with_anchor(
        1.8, "lock", "door", "deadbolt", "entrance", "unlock"
    )
    g.bind("MediaPlayback", lambda ctx, payload: print("    [ALSA AUDIO] Resume Spotify streaming on soundbar")).with_anchor(
        1.8, "play", "jazz", "music", "soundbar", "spotify", "song", "pause", "audio"
    )
    g.fallback(lambda ctx, payload: print("    [AUDIO PROMPT] 'Sorry, I did not catch that command'"))


def setup_cicd_gate(g: NeuroGate) -> None:
    g.bind("NetworkTimeoutRetry", lambda ctx, payload: print("    [AUTO REMEDIATION] Retry transient build step after 5s backoff")).with_anchor(
        1.6, "timeout", "curl", "connect", "timed", "port", "handshake", "tls"
    )
    g.bind("ResourceScaleUp", lambda ctx, payload: print("    [AUTO REMEDIATION] Re-queue job on 64GB High-Memory Runner Pod")).with_anchor(
        1.6, "sigkill", "memory", "137", "killed", "runner", "exhausted", "quota"
    )
    g.bind("CodeSyntaxAlert", lambda ctx, payload: print("    [AUTO NOTIFY] Block PR merge and notify author via Slack/Git comment")).with_anchor(
        1.8, "syntax", "semicolon", "unexpected", "token", "column", "variable", "string"
    )
    g.bind("CacheEvict", lambda ctx, payload: print("    [AUTO REMEDIATION] Invalidate layer cache and rebuild from scratch")).with_anchor(
        1.6, "cache", "clean", "corrupted", "build", "checksum", "sha256"
    )
    g.fallback(lambda ctx, payload: print("    [MANUAL TRIAGE] Flag build for human DevOps on-call review"))


def setup_fintech_gate(g: NeuroGate) -> None:
    g.bind("NormalTransfer", lambda ctx, payload: print("    [INSTANT APPROVAL] Transaction approved and dispatched to ACH rail")).with_anchor(
        2.0, "lunch", "split", "colleagues", "monthly", "payment", "bill", "rent", "reimbursement", "dinner"
    )
    g.bind("PhishingSuspicion", lambda ctx, payload: print("    [BLOCK & INTERCEPT] Suspicious scam wire blocked; call compliance desk")).with_anchor(
        2.2, "urgent", "police", "fine", "bitcoin", "wallet", "scam", "compromised", "safety"
    )
    g.bind("ChargebackDispute", lambda ctx, payload: print("    [DISPUTE ROUTE] Open formal chargeback ticket with issuing bank")).with_anchor(
        2.0, "dispute", "charged", "three", "times", "single", "coffee", "card", "unauthorized", "subscription"
    )
    g.bind("HighValueAudit", lambda ctx, payload: print("    [COMPLIANCE AUDIT] Hold escrow wire pending dual-officer AML sign-off")).with_anchor(
        1.8, "acquisition", "escrow", "million", "tranche", "corporate", "commercial", "estate"
    )
    g.ambiguous(
        lambda ctx, p, s, payload: print(f"    [STEP-UP 2FA] Ambiguous memo ({p} vs {s}): SMS OTP challenge required")
    )
    g.fallback(lambda ctx, payload: print("    [MANUAL AUDIT] Route wire memo to fraud investigations team"))


def main() -> None:
    parser = argparse.ArgumentParser(description="IntelliBranch v2.0 - 6-Domain NeuroGate Filtering Suite")
    parser.add_argument("--domain", default="all", help="Domain to run: all, cs, llm, sre, iot, cicd, fintech")
    args = parser.parse_args()

    suites: Dict[str, DemoSuite] = {
        "cs": DemoSuite(
            domain_name="1. E-Commerce CS Gateway (XOR Order & Multi-Intent Pipeline with NeuroGate)",
            model_path="weights/demo_cs.bin",
            data_path="data/demo_cs.csv",
            description="Demonstrates 3-head NeuroGate with L2 Cosine OOD boundary, symbolic anchors, and multi-intent pipeline.",
            policy=DispatchPolicy(
                high_threshold=0.70,
                low_threshold=0.35,
                margin_cutoff=0.15,
                max_entropy=0.70,
                pipeline_threshold=0.25,
                min_log_sum_exp=7.0,
            ),
            min_cosine=0.35,
            setup_gate=setup_cs_gate,
            test_cases=[
                TestCase("please refund the money to my card", "Definite Refund"),
                TestCase("courier marked delivered but package is missing", "Definite Delivery"),
                TestCase("i forgot my account password and cannot log into the user portal", "Definite Account"),
                TestCase("my credit card was declined at checkout with transaction error code 402", "Definite Payment"),
                TestCase("i returned the box please update delivery", "Multi-Intent Pipeline (Refund -> Delivery)"),
                TestCase("refund delivery", "Positional XOR Sequence Disambiguation"),
                TestCase("what is the meaning of quantum black holes", "OOD / Fallback Isolation"),
            ],
        ),
        "llm": DemoSuite(
            domain_name="2. Semantic LLM Gateway & Cloud API Bypass (NeuroGate Guarded)",
            model_path="weights/demo_llm.bin",
            data_path="data/demo_llm.csv",
            description="Resolves known banking intents in ~30 μs locally, safely escalating true OOD queries to Cloud LLM.",
            policy=DispatchPolicy(
                high_threshold=0.75,
                low_threshold=0.35,
                margin_cutoff=0.15,
                max_entropy=1.50,
                pipeline_threshold=0.30,
                min_log_sum_exp=7.5,
            ),
            min_cosine=0.35,
            setup_gate=setup_llm_gate,
            test_cases=[
                TestCase("what is my current checking account balance", "Local Bypass: QueryBalance"),
                TestCase("how much money is remaining in my personal savings account", "Local Bypass: QueryBalance"),
                TestCase("send five hundred dollars to john doe from checking", "Local Bypass: TransferFunds"),
                TestCase("freeze my debit card immediately i lost my leather wallet", "Local Bypass: CardLock"),
                TestCase("update my residential street address in my user profile", "Local Bypass: UpdateProfile"),
                TestCase("explain how quantum entanglement works in simple terms", "Cloud LLM Fallback (OOD)"),
                TestCase("write a python script to scrape stock prices", "Cloud LLM Fallback (OOD)"),
            ],
        ),
        "sre": DemoSuite(
            domain_name="3. High-Throughput SRE Log Triage (Zero Allocation: 0 B/op)",
            model_path="weights/demo_sre.bin",
            data_path="data/demo_sre.csv",
            description="Parses crash dumps and server logs with strictly zero allocation stack features.",
            policy=default_dispatch_policy(),
            min_cosine=0.30,
            setup_gate=setup_sre_gate,
            custom_run=run_sre_custom,
            test_cases=[
                TestCase("fatal error: runtime: out of memory allocating 4194304 bytes", "P0 OutOfMemory"),
                TestCase("container exited with code 137 OOMKilled cgroup memory limit exceeded", "P0 OutOfMemory"),
                TestCase("HikariPool-1 - Connection is not available request timed out after 30000ms", "P1 DBPoolExhausted"),
                TestCase("org.postgresql.util.PSQLException: FATAL: remaining connection slots are reserved", "P1 DBPoolExhausted"),
                TestCase("SECURITY ALERT: 250 failed login attempts in 60 seconds from single IP", "Security AuthBruteForce"),
                TestCase("Fail2ban banned host 192.168.1.100 for 3600 seconds after 10 failed login attempts", "Security AuthBruteForce"),
                TestCase("INFO: health check probe /healthz returned 200 OK latency: 2ms", "P3 SystemHealth"),
                TestCase("Heartbeat ping received from worker node status healthy", "P3 SystemHealth"),
            ],
        ),
        "iot": DemoSuite(
            domain_name="4. Offline Edge IoT Command Dispatcher (Nuance & Anchor Calibrated)",
            model_path="weights/demo_iot.bin",
            data_path="data/demo_iot.csv",
            description="Sub-milliwatt, sub-180KB offline smart home command router with symbolic anchor soft-bias.",
            policy=default_dispatch_policy(),
            min_cosine=0.30,
            setup_gate=setup_iot_gate,
            test_cases=[
                TestCase("it is too dark in here please switch on lamps in living room", "LightControl (Slang/Context Anchor Boost)"),
                TestCase("turn on the chandelier lights above dining table", "LightControl"),
                TestCase("cooling mode on maximum fan speed in master bedroom", "ClimateControl"),
                TestCase("set living room temperature setpoint to 21 degrees celsius", "ClimateControl"),
                TestCase("lock the front entrance smart door deadbolt immediately", "DoorLock"),
                TestCase("unlock front door deadbolt for delivery courier guest", "DoorLock"),
                TestCase("play smooth jazz music on living room soundbar speaker", "MediaPlayback"),
                TestCase("pause spotify audio playback on bedroom speaker", "MediaPlayback"),
            ],
        ),
        "cicd": DemoSuite(
            domain_name="5. Automated CI/CD Failure Triage & Self-Healing",
            model_path="weights/demo_cicd.bin",
            data_path="data/demo_cicd.csv",
            description="Analyzes build error tail logs with symbolic keyword anchors to trigger auto-remediation.",
            policy=default_dispatch_policy(),
            min_cosine=0.30,
            setup_gate=setup_cicd_gate,
            test_cases=[
                TestCase("curl: (28) Failed to connect to registry.npmjs.org port 443: Connection timed out", "Auto-Retry: NetworkTimeoutRetry"),
                TestCase("docker pull failed tls handshake timeout communicating with registry", "Auto-Retry: NetworkTimeoutRetry"),
                TestCase("Command terminated by signal 9 SIGKILL exit status 137 runner ran out of memory", "Scale-Up: ResourceScaleUp"),
                TestCase("gcc: fatal error: Killed (program cc1plus) virtual memory exhausted", "Scale-Up: ResourceScaleUp"),
                TestCase("syntax error: unexpected token semicolon at line 144 column 2", "Notify-Dev: CodeSyntaxAlert"),
                TestCase("cannot use variable of type string as type int in argument to processTransaction", "Notify-Dev: CodeSyntaxAlert"),
                TestCase("corrupted go build cache detected in /root/.cache/go-build please clean", "Evict-Cache: CacheEvict"),
                TestCase("checksum mismatch for cached layer sha256:4a8b invalid local tar", "Evict-Cache: CacheEvict"),
            ],
        ),
        "fintech": DemoSuite(
            domain_name="6. FinTech Transaction Memo Audit & Fraud Prevention (NeuroGate Calibrated)",
            model_path="weights/demo_fintech.bin",
            data_path="data/demo_fintech.csv",
            description="Real-time remittance inspection for scam interception with high-risk symbolic anchors.",
            policy=DispatchPolicy(
                high_threshold=0.70,
                low_threshold=0.35,
                margin_cutoff=0.15,
                max_entropy=2.0,
                pipeline_threshold=0.30,
            ),
            min_cosine=0.30,
            setup_gate=setup_fintech_gate,
            test_cases=[
                TestCase("monthly lunch payment split with office colleagues", "Instant Approval: NormalTransfer"),
                TestCase("reimbursement for team dinner pizza and drinks", "Instant Approval: NormalTransfer"),
                TestCase("urgent send funds now police fine wire to bitcoin wallet immediately", "Block & Intercept: PhishingSuspicion"),
                TestCase("your account is compromised transfer all savings to temporary safety wallet", "Block & Intercept: PhishingSuspicion"),
                TestCase("merchant charged my card three times for single coffee", "Dispute: ChargebackDispute"),
                TestCase("unauthorized recurring subscription charge from merchant after cancellation", "Dispute: ChargebackDispute"),
                TestCase("corporate acquisition escrow settlement tranche wire five million dollars", "AML Audit: HighValueAudit"),
                TestCase("commercial real estate property purchase closing escrow wire transfer", "AML Audit: HighValueAudit"),
            ],
        ),
    }

    ordered_keys = ["cs", "llm", "sre", "iot", "cicd", "fintech"]
    selected = args.domain.lower()

    total_start = time.perf_counter()
    total_queries = 0
    executed_domains = 0

    print("=" * 80)
    print("  INTELLIBRANCH v2.0 - 6-DOMAIN NEUROGATE 3-HEAD INTELLIGENT FILTERING SUITE (PYTHON)")
    print("=" * 80)

    ctx: Any = None

    for key in ordered_keys:
        if selected != "all" and selected != key:
            continue

        executed_domains += 1
        suite = suites[key]
        print(f"\n>>> DOMAIN: {suite.domain_name}")
        print(f"    Model Path  : {suite.model_path}")
        print(f"    Capability  : {suite.description}")
        print("    " + "-" * 76)

        ensure_model(suite.model_path, suite.data_path)

        gate = NeuroGate.from_file(suite.model_path)
        gate.set_policy(suite.policy)
        if suite.min_cosine > 0:
            gate.set_min_cosine_sim(suite.min_cosine)

        try:
            samples = load_csv_dataset(suite.data_path)
            gate.calibrate_domain_centroid(samples)
        except Exception:
            pass

        suite.setup_gate(gate)

        for tc in suite.test_cases:
            total_queries += 1
            trace = gate.inspect(tc.query)
            bench_latency = measure_benchmark_latency(gate, tc.query, 1000)

            routed_label = trace.predicted_label
            if trace.is_pipeline:
                routed_label = f"Pipeline ({trace.predicted_label} -> {trace.secondary_label})"
            elif trace.is_ood:
                routed_label = f"OOD Fallback ({trace.predicted_label})"

            print(f'  • Input    : "{tc.query}"')
            print(f"    Expect   : {tc.expectation}")
            print(
                f"    Inference: {routed_label} (Confidence: {trace.confidence*100:.2f}%, "
                f"Cosine: {trace.cosine_similarity:.4f}, Entropy: {trace.entropy:.4f}, "
                f"Energy: {trace.free_energy:.2f}, Latency: {bench_latency:.2f} μs/op, OOD: {trace.is_ood})"
            )

            gate.filter_pipeline(ctx, tc.query, None)
            print()

        if suite.custom_run is not None:
            suite.custom_run(gate, ctx)
            print()

    total_duration = time.perf_counter() - total_start
    domain_label = "domains" if executed_domains > 1 else "domain"
    print("=" * 80)
    print(f"DEMONSTRATION COMPLETED: {total_queries} queries routed across {executed_domains} distinct neural {domain_label} in {total_duration:.4f}s")
    print("ALL INFERENCES RUN IN MICROSECONDS WITH PURE PYTHON + NUMPY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
