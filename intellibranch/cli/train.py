import argparse
import os
import sys

from intellibranch.binary import save_binary_model
from intellibranch.trainer import default_train_config, load_csv_dataset, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="IntelliBranch Offline Model Training CLI")
    parser.add_argument("--data", default="data/sample_dataset.csv", help="Path to CSV training dataset")
    parser.add_argument("--out", default="weights/model.bin", help="Output binary model path")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum training epochs")
    parser.add_argument("--vocab", type=int, default=150, help="Target BPE vocabulary size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")

    args = parser.parse_args()

    print(f"Loading dataset from: {args.data}")
    samples = load_csv_dataset(args.data)
    print(f"Loaded {len(samples)} training samples")

    cfg = default_train_config()
    cfg.epochs = args.epochs
    cfg.target_vocab_size = args.vocab
    cfg.learning_rate = args.lr
    cfg.batch_size = 16
    cfg.patience = 10

    print("Starting offline BPE + AdamW training pipeline...")
    model = train_model(samples, cfg)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"Serializing trained model to Little-Endian binary: {args.out}")
    save_binary_model(args.out, model)

    print("Training and binary export completed successfully.")
    print(f"Model saved at: {args.out} (Vocab: {model.header.vocab_size}, Classes: {model.header.num_classes})")


if __name__ == "__main__":
    main()
