import os
import tempfile
import pytest

from intellibranch.binary import save_binary_model
from intellibranch.neurogate import NeuroGate
from intellibranch.trainer import DataSample, TrainConfig, train_model


def test_neurogate_basic_and_anchors():
    samples = [
        DataSample(text="turn on living room lamps", label="LightControl"),
        DataSample(text="it is too dark here please switch on light", label="LightControl"),
        DataSample(text="cooling mode maximum fan temperature bedroom", label="ClimateControl"),
        DataSample(text="set ac thermostat cool air heat", label="ClimateControl"),
    ]

    cfg = TrainConfig(
        epochs=50,
        embedding_dim=32,
        hidden_dim=32,
        target_vocab_size=64,
    )
    model = train_model(samples, cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "neurogate_test.bin")
        save_binary_model(model_path, model)

        gate = NeuroGate.from_file(model_path)

        state = {"light": False, "fallback": False}

        def on_light(ctx, payload):
            state["light"] = True

        def on_fallback(ctx, payload):
            state["fallback"] = True

        (
            gate.bind("LightControl", on_light)
            .with_anchor(1.5, "lamp", "lamps", "dark", "light")
            .bind("ClimateControl", lambda ctx, payload: None)
            .with_anchor(1.5, "cooling", "ac", "fan", "temp")
        )
        gate.fallback(on_fallback)

        # 1. Verify anchor boost reinforces LightControl
        state["light"] = False
        trace = gate.inspect("it is too dark in here please switch on lamps")
        gate.filter(None, "it is too dark in here please switch on lamps", None)
        assert state["light"] is True, f"Expected LightControl, trace: {trace}"

        # 2. Verify inspection trace
        assert trace.predicted_label == "LightControl"
        assert len(trace.triggered_anchors) > 0

        # 3. Verify strict OOD boundary rejection
        gate.set_min_cosine_sim(0.99)
        state["fallback"] = False
        gate.filter(None, "what is the meaning of quantum black holes", None)
        assert state["fallback"] is True
