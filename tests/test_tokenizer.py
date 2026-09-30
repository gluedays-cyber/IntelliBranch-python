from intellibranch.tokenizer import BPETokenizer, is_valid_utf8


def test_bpe_train_and_encode():
    corpus = [
        "refund my order",
        "please cancel payment and refund",
        "where is my package delivery",
        "track delivery shipment",
    ]
    tok = BPETokenizer.train(corpus, target_vocab_size=50)
    assert tok.vocab_size() >= 10
    assert tok.vocab[0] == "[PAD]"
    assert tok.vocab[1] == "[UNK]"

    # Test encoding
    tokens = tok.encode("refund delivery")
    assert len(tokens) > 0
    decoded = tok.decode(tokens)
    assert "refund" in decoded
    assert "delivery" in decoded


def test_utf8_check():
    assert is_valid_utf8("hello world")
    assert is_valid_utf8("한국어 테스트")
