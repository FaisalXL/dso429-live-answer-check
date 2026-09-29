"""Prototype: score against two separate concept references (mechanism,
risk) instead of one blended reference, and see whether requiring BOTH
concepts to clear a bar fixes (a) the partial-credit blind spot and
(b) the false-positive problem, versus the current single-blended-max
approach.
"""

from sentence_transformers import SentenceTransformer, util
import torch

torch.set_num_threads(1)
model = SentenceTransformer("all-MiniLM-L6-v2")

CONCEPTS = {
    "mechanism": (
        "A smart contract could hold the deposit in escrow and automatically "
        "release it to the store or refund the customer depending on whether "
        "the store confirms fulfillment before the deadline, with no manual "
        "intervention needed from either party."
    ),
    "risk": (
        "A real risk is that the contract depends on an outside system, an "
        "oracle, to report whether the order was actually fulfilled in the "
        "physical world; if that oracle is wrong or manipulated, the "
        "contract executes the wrong outcome."
    ),
}
concept_labels = list(CONCEPTS.keys())
concept_embeddings = model.encode(
    [CONCEPTS[k] for k in concept_labels], convert_to_tensor=True, normalize_embeddings=True
)

TEST_CASES = {
    "correct_full_novel": (
        "You could set up an automated agreement where money is locked up "
        "until either side proves the deal went through by a set time; if "
        "nobody confirms in time, whoever paid gets their money back "
        "automatically. The catch is you need a trustworthy way to know "
        "whether the real-world delivery happened."
    ),
    "partial_escrow_only": (
        "A smart contract could hold the deposit in escrow and release it "
        "automatically based on whether the store confirms the order in "
        "time."
    ),
    "partial_oracle_risk_only": (
        "A real limitation of smart contracts is that they depend on an "
        "oracle to know what happened in the real world, and that oracle "
        "could be wrong or manipulated."
    ),
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
    "keyword_salad_2_reordered": (
        "Oracle risk code correctly manipulated outcome. Customer refund "
        "confirm store deadline. Pickup smart contract deposit escrow."
    ),
}

BOTH_CONCEPTS_THRESHOLD = 0.45  # each concept must individually clear this

print(f"{'case':38s} {'mechanism':>10s} {'risk':>10s} {'min(both)':>10s} {'passes_both':>12s}")
print("-" * 85)
for label, text in TEST_CASES.items():
    emb = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(emb, concept_embeddings)[0]
    mech, risk = sims[0].item(), sims[1].item()
    passes = mech >= BOTH_CONCEPTS_THRESHOLD and risk >= BOTH_CONCEPTS_THRESHOLD
    print(f"{label:38s} {mech*100:9.1f}% {risk*100:9.1f}% {min(mech,risk)*100:9.1f}% {str(passes):>12s}")
