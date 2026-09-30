import os
import sys

from intellibranch.binary import save_binary_model
from intellibranch.router import Router
from intellibranch.trainer import default_train_config, load_csv_dataset, train_model


# 1. Business Logic Handlers
def handle_refund(ctx, payload):
    print(f"[ACTION: Refund]   Processing refund for: '{payload}'")


def handle_delivery(ctx, payload):
    print(f"[ACTION: Delivery] Querying shipment tracking for: '{payload}'")


def handle_account(ctx, payload):
    print(f"[ACTION: Account]  Initiating account security for: '{payload}'")


def handle_fallback(ctx, payload):
    print(f"[FALLBACK: Safety] Isolated low-confidence request: '{payload}'")


def main():
    model_path = "weights/intent.bin"

    # Auto-compile model if missing (ensures instant zero-config clone & run)
    if not os.path.exists(model_path):
        print("Model weights not found. Compiling from data/sample_dataset.csv...")
        samples = load_csv_dataset("data/sample_dataset.csv")

        cfg = default_train_config()
        cfg.epochs = 50
        cfg.learning_rate = 0.005
        cfg.target_vocab_size = 250

        model = train_model(samples, cfg)
        os.makedirs("weights", exist_ok=True)
        save_binary_model(model_path, model)
        print("Model compilation completed.")

    # 2. Load compiled binary weights into memory (0.60 calibrated threshold)
    router = Router(model_path=model_path, default_threshold=0.60)

    # 3. Bind routes directly
    (
        router.bind("Refund", handle_refund)
        .bind("Delivery", handle_delivery)
        .bind("Account", handle_account)
        .fallback(handle_fallback)
    )

    # 4. Execute microsecond branch dispatch
    test_queries = [
        "I want to cancel my payment and request a refund",
        "When will my delivery package arrive",
        "Forgot my account password",
        "Please refund my purchase",
        "Track my shipment status",
        "Completely random gibberish noise 12345!@#$",
        "hey where is my stuff it was supposed to get here yesterday",
        "can u cancel order #49281? i bought it by mistake",
        "bruh the reset link is not sending to my email, fix this",
        "got charged twice on my card, refund the extra charge asap",
        "item arrived totally smashed, want my money back",
        "cant log into my acct keeps saying wrong password",
        "tracking says delivered but nothing is in my mailbox",
        "yo i typed the wrong apt number, can someone update the address before it ships",
        "sent the return box a week ago, when do i get my refund?",
        "locked out of my account after 3 tries... help pls",
        "ordered a large but you guys sent me a small",
        "any update on order #88412? hasnt moved in 4 days",
        "how do i just delete my account permanently? done with this site",
        "driver dumped the package in the rain, everything inside is ruined",
        "promo code didnt apply at checkout, can u refund the difference",
        "need a real person, this bot is completely useless",
        "can i change the delivery date? nobody will be home this friday",
        "my card was charged but never received any confirmation email or receipt",
        "lost access to my 2FA phone number, how do i get back in",
        "package has been stuck in transit for 10 days straight, is it lost or what",
    ]

    print("=== IntelliBranch Server Routing Started ===")
    ctx = None
    for query in test_queries:
        router.dispatch(ctx, query, query)
    print("=== All queries dispatched in microseconds ===")


if __name__ == "__main__":
    main()
