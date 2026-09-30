import os
import tempfile
import pytest

from intellibranch.trainer import (
    DataSample,
    TrainConfig,
    load_csv_dataset,
    train_model,
)


def test_load_csv_dataset():
    samples = load_csv_dataset("data/sample_dataset.csv")
    assert len(samples) > 0
    assert samples[0].text
    assert samples[0].label


def test_train_model():
    samples = [
        DataSample(text="refund my purchase", label="Refund"),
        DataSample(text="cancel payment request refund", label="Refund"),
        DataSample(text="where is my delivery package", label="Delivery"),
        DataSample(text="track shipment delivery courier", label="Delivery"),
    ]
    cfg = TrainConfig(
        epochs=15,
        embedding_dim=16,
        hidden_dim=16,
        target_vocab_size=32,
        seed=123,
    )
    model = train_model(samples, cfg)

    assert model.header.vocab_size >= 10
    assert model.header.num_classes == 2
    assert "Refund" in model.labels
    assert "Delivery" in model.labels

    # Test inference on trained model
    pred_label, conf = model.predict("refund money")
    assert pred_label in model.labels
    assert conf > 0.0
