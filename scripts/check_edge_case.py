"""
scripts/check_edge_case.py
--------------------------
Runs 6 test cases through both candidate weight configurations to validate
that the fix doesn't overcorrect on good answers while catching bad ones.

  Option 1 — rebalanced weights : α=0.25 β=0.40 γ=0.15 δ=0.20
  Option 4 — faith × cov formula: α=0.30 β=0.30 γ=0.15 δ=0.25 (original weights,
             but Faithfulness term = faithfulness × claim_coverage)

Cases:
  0. Original edge case  — 1/6 grounded, 5 fabricated         (should be LOW)
  1. Fully grounded      — all sentences supported             (should be HIGH)
  2. Fully fabricated    — no sentences supported              (should be LOW)
  3. Partial coverage    — 3/5 sentences grounded              (should be MID)
  4. Contradictory       — sentences actively contradicted     (should be LOW)
  5. Topic drift         — correct facts, wrong topic          (should be MID-LOW)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import compute_h_score, compute_answer_relevance, THRESHOLD

# ── Test cases ────────────────────────────────────────────────────────────────
CASES = [
    {
        "label": "Case 0 — original edge case (1/6 grounded)",
        "context": (
            "The Eiffel Tower is located in Paris, France. "
            "It was constructed between 1887 and 1889 as the entrance arch for the 1889 World's Fair. "
            "The structure stands 330 metres tall and was designed by Gustave Eiffel."
        ),
        "answer": (
            "The Eiffel Tower is located in Paris, France. "
            "It was built in 1650 during the reign of Louis XIV. "
            "The tower is made entirely of marble imported from Italy. "
            "It has exactly 9,999 steps leading to the top. "
            "The tower was originally painted bright red. "
            "It serves as the world's tallest radio transmission mast."
        ),
        "query": "Where is the Eiffel Tower and when was it built?",
        "expect": "LOW",
    },
    {
        "label": "Case 1 — fully grounded (all sentences supported)",
        "context": (
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide "
            "to produce oxygen and energy in the form of glucose. "
            "It takes place primarily in the chloroplasts of plant cells. "
            "The green pigment chlorophyll absorbs light energy to drive the reaction."
        ),
        "answer": (
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide "
            "to produce oxygen and glucose. "
            "It occurs in the chloroplasts of plant cells. "
            "Chlorophyll, the green pigment, absorbs light to power the reaction."
        ),
        "query": "How does photosynthesis work in plants?",
        "expect": "HIGH",
    },
    {
        "label": "Case 2 — fully fabricated (no sentences supported)",
        "context": (
            "The Amazon River is the largest river by discharge volume of water in the world. "
            "It flows through Brazil and several other South American countries. "
            "The river is approximately 6,400 kilometres long."
        ),
        "answer": (
            "The Amazon River originates in the Swiss Alps near Geneva. "
            "It flows northward through central Africa before emptying into the Mediterranean Sea. "
            "The river is home to polar bears and arctic foxes. "
            "It was first mapped by Chinese explorers in 1200 AD."
        ),
        "query": "Where does the Amazon River flow and how long is it?",
        "expect": "LOW",
    },
    {
        "label": "Case 3 — partial coverage (3/5 sentences grounded)",
        "context": (
            "The human brain contains approximately 86 billion neurons. "
            "Neurons communicate through electrical and chemical signals called synapses. "
            "The cerebral cortex is responsible for higher-order thinking and consciousness."
        ),
        "answer": (
            "The human brain contains around 86 billion neurons. "
            "Neurons communicate via electrical and chemical signals at synapses. "
            "The cerebral cortex handles higher-order thinking. "
            "The brain runs entirely on a special type of crystal energy. "
            "Memories are stored in a dedicated memory organ called the hippocampus gland."
        ),
        "query": "How many neurons are in the human brain and how do they communicate?",
        "expect": "MID",
    },
    {
        "label": "Case 4 — contradictory (sentences contradicted by context)",
        "context": (
            "The Great Wall of China was built over many centuries, primarily during the Ming dynasty "
            "(1368–1644). It stretches approximately 21,196 kilometres in total. "
            "It was constructed using stone, brick, tamped earth, and wood."
        ),
        "answer": (
            "The Great Wall of China was built in a single year by Emperor Qin. "
            "It is only 500 kilometres long. "
            "It was constructed entirely from steel and concrete. "
            "The wall is invisible from space despite popular belief."
        ),
        "query": "When was the Great Wall of China built and how long is it?",
        "expect": "LOW",
    },
    {
        "label": "Case 5 — topic drift (correct facts, wrong topic)",
        "context": (
            "Mount Everest is the highest mountain on Earth at 8,849 metres above sea level. "
            "It is located in the Himalayas on the border between Nepal and Tibet. "
            "Edmund Hillary and Tenzing Norgay first summited it in 1953."
        ),
        "answer": (
            "The Pacific Ocean is the largest and deepest ocean on Earth. "
            "It covers more than 165 million square kilometres. "
            "The Mariana Trench, located in the Pacific, is the deepest known point at 11,034 metres."
        ),
        "query": "How tall is Mount Everest and where is it located?",
        "expect": "LOW (drift)",
    },
]

# ── Weight configs ────────────────────────────────────────────────────────────
OPT1 = {"alpha": 0.25, "beta": 0.40, "gamma": 0.15, "delta": 0.20, "formula": "standard"}
OPT4 = {"alpha": 0.30, "beta": 0.30, "gamma": 0.15, "delta": 0.25, "formula": "faith_x_cov"}


def composite(faith, cov, contra, rel, cfg):
    a, b, g, d = cfg["alpha"], cfg["beta"], cfg["gamma"], cfg["delta"]
    f = faith * cov if cfg["formula"] == "faith_x_cov" else faith
    return round(a * f + b * cov + g * (1 - contra) + d * rel, 4)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    header = f"{'Case':<48} {'Expect':<12} {'Orig':>6} {'Opt1':>6} {'Opt4':>6}"
    print("\n" + "="*len(header))
    print(header)
    print("="*len(header))

    orig_cfg = {"alpha": 0.30, "beta": 0.30, "gamma": 0.15, "delta": 0.25, "formula": "standard"}
    all_pass = True

    for case in CASES:
        nli  = compute_h_score(case["answer"], [case["context"]])
        rel  = compute_answer_relevance(case["query"], case["answer"])
        f, c, contra = nli["faithfulness"], nli["claim_coverage"], nli["contradiction"]

        orig  = composite(f, c, contra, rel, orig_cfg)
        opt1  = composite(f, c, contra, rel, OPT1)
        opt4  = composite(f, c, contra, rel, OPT4)

        expect = case["expect"]
        print(f"{case['label']:<48} {expect:<12} {orig:>6.4f} {opt1:>6.4f} {opt4:>6.4f}")

        # Detailed breakdown
        print(f"  NLI → faith={f:.3f}  cov={c:.3f}  contra={contra:.3f}  rel={rel:.3f}")

        # Flag if a "should be LOW" case passes threshold
        for name, score in [("Opt1", opt1), ("Opt4", opt4)]:
            if "LOW" in expect and score >= THRESHOLD:
                print(f"  ⚠  {name} FAIL: {score:.4f} >= {THRESHOLD} (expected low)")
                all_pass = False
            elif "HIGH" in expect and score < 0.55:
                print(f"  ⚠  {name} may overcorrect: {score:.4f} for a fully-grounded answer")
                all_pass = False

    print("="*len(header))
    print(f"\n{'Threshold':<48} {'':12} {'':>6} {THRESHOLD:>6} {THRESHOLD:>6}")
    print()
    if all_pass:
        print("✅ All cases behave as expected for both options.")
    else:
        print("⚠  One or more cases are flagged — review before applying to main pipeline.")

    sys.exit(0 if all_pass else 1)
