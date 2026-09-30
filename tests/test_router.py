import os
import tempfile
import threading
import time
import pytest

from tests.test_binary import create_sample_model
from intellibranch.binary import save_binary_model
from intellibranch.policy import DispatchPolicy
from intellibranch.router import Router


def test_router_dispatch_and_fallback():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "router_test.bin")
        orig_model = create_sample_model()
        save_binary_model(model_path, orig_model)

        router = Router.from_file(model_path, default_threshold=0.1)

        state = {"refund": False, "delivery": False, "fallback": False}

        router.bind("Refund", lambda ctx, payload: state.update(refund=True))
        router.bind("Delivery", lambda ctx, payload: state.update(delivery=True))
        router.fallback(lambda ctx, payload: state.update(fallback=True))

        router.dispatch(None, "refund", None)
        assert state["refund"] or state["delivery"] or state["fallback"]

        # Strict threshold router causing fallback
        strict_router = Router.from_file(model_path, default_threshold=0.999999)
        strict_state = {"fallback": False}
        strict_router.bind("Refund", lambda ctx, payload: None)
        strict_router.fallback(lambda ctx, payload: strict_state.update(fallback=True))

        strict_router.dispatch(None, "refund", None)
        assert strict_state["fallback"] is True


def test_router_inspect_and_trace():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "router_trace_test.bin")
        orig_model = create_sample_model()
        save_binary_model(model_path, orig_model)

        router = Router.from_file(model_path, default_threshold=0.5)
        router.bind("Refund", lambda ctx, payload: None)

        trace = router.inspect("refund")
        assert len(trace.token_ids) > 0
        assert len(trace.class_probabilities) > 0
        assert trace.confidence > 0.0


def test_router_3tier_dispatch():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "router_3tier_test.bin")
        orig_model = create_sample_model()
        save_binary_model(model_path, orig_model)

        router = Router.from_file(model_path, default_threshold=0.5)
        policy = DispatchPolicy(
            high_threshold=0.80,
            low_threshold=0.30,
            margin_cutoff=0.20,
            max_entropy=2.0,
        )
        router.set_policy(policy)

        state = {"definite": False, "ambiguous": False, "fallback": False}
        router.bind("Refund", lambda ctx, payload: state.update(definite=True))
        router.ambiguous(lambda ctx, p, s, payload: state.update(ambiguous=True))
        router.fallback(lambda ctx, payload: state.update(fallback=True))

        trace = router.inspect("refund")
        router.dispatch(None, "refund", None)

        if trace.is_ambiguous:
            assert state["ambiguous"] is True
        elif trace.is_fallback:
            assert state["fallback"] is True
        else:
            assert state["definite"] is True


def test_router_ood_entropy_isolation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "router_ood_test.bin")
        orig_model = create_sample_model()
        save_binary_model(model_path, orig_model)

        router = Router.from_file(model_path, default_threshold=0.5)
        policy = DispatchPolicy(
            high_threshold=0.70,
            low_threshold=0.20,
            margin_cutoff=0.10,
            max_entropy=0.001,  # Strictly forces OOD isolation
        )
        router.set_policy(policy)

        state = {"fallback": False}
        router.bind("Refund", lambda ctx, payload: None)
        router.fallback(lambda ctx, payload: state.update(fallback=True))

        router.dispatch(None, "refund", None)
        assert state["fallback"] is True


def test_router_multi_intent_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "router_pipeline_test.bin")
        orig_model = create_sample_model()
        save_binary_model(model_path, orig_model)

        router = Router.from_file(model_path, default_threshold=0.5)
        policy = DispatchPolicy(
            high_threshold=0.80,
            low_threshold=0.20,
            margin_cutoff=0.15,
            max_entropy=3.0,
            pipeline_threshold=0.20,
        )
        router.set_policy(policy)

        state = {"specific": False, "default": False}

        def on_specific(ctx, p, s, payload):
            state["specific"] = True

        def on_default(ctx, p, s, payload):
            state["default"] = True

        router.bind_pipeline("Refund", "Delivery", on_specific)
        router.default_pipeline(on_default)

        router.dispatch_pipeline(None, "refund delivery", None)
        assert state["specific"] or state["default"]


def test_router_atomic_reload_concurrently():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path_a = os.path.join(tmp_dir, "model_a.bin")
        model_path_b = os.path.join(tmp_dir, "model_b.bin")

        model_a = create_sample_model()
        save_binary_model(model_path_a, model_a)

        model_b = create_sample_model()
        model_b.temperature = 0.8
        save_binary_model(model_path_b, model_b)

        router = Router.from_file(model_path_a, default_threshold=0.5)
        router.bind("Refund", lambda ctx, payload: None)
        router.fallback(lambda ctx, payload: None)

        stop_event = threading.Event()

        def reader():
            while not stop_event.is_set():
                router.dispatch(None, "refund my payment immediately", None)
                router.inspect("check status")

        def reloader():
            for i in range(20):
                if stop_event.is_set():
                    break
                target = model_path_b if i % 2 == 0 else model_path_a
                router.reload(target)
                time.sleep(0.002)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads.append(threading.Thread(target=reloader))

        for t in threads:
            t.start()

        time.sleep(0.1)
        stop_event.set()

        for t in threads:
            t.join()


def test_router_telemetry_drain():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "telemetry_test.bin")
        sample_model = create_sample_model()
        save_binary_model(model_path, sample_model)

        router = Router.from_file(model_path, default_threshold=0.5)
        router.enable_telemetry(4)

        router.dispatch(None, "unknown gibberish query 12345", None)
        router.dispatch(None, "another isolated query 67890", None)

        events = router.drain_telemetry()
        assert len(events) >= 2
        for ev in events:
            assert ev.is_fallback or ev.is_ambiguous or ev.is_pipeline

        drained_again = router.drain_telemetry()
        assert len(drained_again) == 0
