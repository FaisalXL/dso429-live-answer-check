"""Different metric entirely: NLI (Natural Language Inference) cross-encoder.
Instead of cosine similarity between independently-pooled embeddings, this
jointly encodes (reference, student_answer) and directly predicts whether
one entails, contradicts, or is neutral to the other. This is the class of
model actually trained to detect "sounds similar but says the opposite."
"""

from sentence_transformers import CrossEncoder
import torch

torch.set_num_threads(1)

REFERENCE = (
    "A smart contract on a blockchain could hold the customer's deposit in "
    "escrow when the pre-order is placed. The contract's code encodes the "
    "pickup deadline as a condition: if the store confirms fulfillment "
    "before the deadline, the deposit is released to the store; if the "
    "deadline passes without confirmation, the contract automatically "
    "refunds the customer, with no manual intervention needed from either "
    "party. A real risk is that the contract depends on an outside system, "
    "an oracle, to report whether the order was actually fulfilled; if that "
    "oracle is wrong or manipulated, the contract executes the wrong "
    "outcome even though the code ran correctly."
)

CASES = {
    "wrong_logic_1": (
        "The oracle holds the smart contract deposit and the store must "
        "confirm the escrow before the customer refunds the deadline "
        "automatically, which removes risk from the code."
    ),
    "wrong_logic_3_confident_but_backwards": (
        "A smart contract works by having the customer manually confirm the "
        "oracle before the store's escrow deadline releases the risk back to "
        "the deposit, which is what makes it trustworthy."
    ),
    "keyword_salad_1": (
        "Escrow deposit smart contract pickup deadline store confirm refund "
        "customer oracle risk manipulated outcome code correctly."
    ),
    "correct_novel_casual": (
        "basically you'd code the refund to happen on its own if the shop "
        "never checks in by the pickup time, so you don't have to hope they "
        "do the right thing. downside is the contract only knows what some "
        "outside source tells it really happened."
    ),
    "off_topic": (
        "Cryptocurrency and blockchain technology are changing how "
        "businesses handle payments and contracts in the modern economy."
    ),
}

model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
print("label mapping:", model.config.id2label)

for label, text in CASES.items():
    scores = model.predict([(REFERENCE, text)])
    probs = torch.softmax(torch.tensor(scores), dim=1)[0]
    labeled = {model.config.id2label[i]: round(p.item() * 100, 1) for i, p in enumerate(probs)}
    print(f"{label:38s} {labeled}")
