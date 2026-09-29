from sentence_transformers import SentenceTransformer, util
import torch

torch.set_num_threads(1)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Three valid phrasings of the same correct answer, deliberately using
# different vocabulary/register/emphasis, to see whether "max similarity
# across references" closes the false-negative gap from the single-reference
# test without blowing open the false-positive gap.

REFERENCES = {
    "original_technical": (
        "A smart contract on a blockchain could hold the customer's deposit in "
        "escrow when the pre-order is placed. The contract's code encodes the "
        "pickup deadline as a condition: if the store confirms fulfillment "
        "before the deadline, the deposit is released to the store; if the "
        "deadline passes without confirmation, the contract automatically "
        "refunds the customer, with no manual intervention needed from either "
        "party. This removes the need to trust the store to issue refunds "
        "voluntarily, since the rule is enforced by code once deployed. A real "
        "risk is that the contract depends on an outside system, an oracle, to "
        "report whether the order was actually fulfilled in the physical "
        "world; if that oracle is wrong, unavailable, or manipulated, the "
        "contract will execute the wrong outcome even though the code itself "
        "ran correctly."
    ),
    "plain_language": (
        "You could set up an automated agreement where money is locked up "
        "until either side proves the deal went through by a set time; if "
        "nobody confirms in time, whoever paid gets their money back with "
        "nobody having to manually process it. The catch is you need some "
        "trustworthy way to know in the first place whether the real-world "
        "delivery happened, and that outside check is really the weak link."
    ),
    "alternate_framing": (
        "Instead of relying on the coffee shop's word that it will issue a "
        "refund, you could write the refund rule directly into code that runs "
        "on its own once the deadline is reached, so neither the customer nor "
        "the store has to take any action for the money to move correctly. "
        "The main limitation is that the contract can only see what it's told "
        "by an external data feed about whether the order was fulfilled, so "
        "it's only as trustworthy as that outside data source."
    ),
}

TEST_CASES = {
    "reference_itself": REFERENCES["original_technical"],
    "keyword_salad_no_reasoning": (
        "Escrow deposit smart contract pickup deadline store confirm refund "
        "customer oracle risk manipulated outcome code correctly."
    ),
    "wrong_logic_dense_keywords": (
        "The oracle holds the smart contract deposit and the store must "
        "confirm the escrow before the customer refunds the deadline "
        "automatically, which removes risk from the code."
    ),
    "correct_but_totally_different_wording": (
        "You could set up an automated agreement where money is locked up "
        "until either side proves the deal went through by a set time; if "
        "nobody confirms in time, whoever paid gets their money back with "
        "nobody having to manually process it. The catch is you need some "
        "trustworthy way to know in the first place whether the real-world "
        "delivery happened, and that outside check is really the weak link."
    ),
    "topic_adjacent_but_off_target": (
        "Cryptocurrency and blockchain technology are changing how businesses "
        "handle payments and contracts in the modern economy, offering more "
        "transparency and efficiency than traditional banking systems."
    ),
    "four_lines_keyword_matching": (
        "A smart contract uses escrow to hold the deposit. It checks the "
        "deadline and confirms with the store. If not confirmed it refunds "
        "the customer automatically. A risk is the oracle could be wrong."
    ),
    "correct_but_uses_third_framing_words": (
        "Rather than trusting the shop to refund you, the rule could be "
        "written into code that fires on its own when the deadline hits, so "
        "no one has to lift a finger for the money to move the right way. The "
        "weak point is that the code only knows what an outside data feed "
        "tells it about whether the order was actually delivered."
    ),
}

ref_embeddings = {
    name: model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    for name, text in REFERENCES.items()
}

print(f"{'case':38s} {'single(orig)':>13s} {'max(3 refs)':>12s}  best_matched_ref")
for label, text in TEST_CASES.items():
    emb = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    single = util.cos_sim(emb, ref_embeddings["original_technical"]).item()
    sims = {name: util.cos_sim(emb, e).item() for name, e in ref_embeddings.items()}
    best_ref = max(sims, key=sims.get)
    best_val = sims[best_ref]
    print(f"{label:38s} {single*100:12.1f}% {best_val*100:11.1f}%  {best_ref}")
