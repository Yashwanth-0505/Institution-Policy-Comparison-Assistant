from ml_pipeline.evaluation.dataset import load_eval_dataset
import logging

# Suppress the report printing
logging.getLogger().setLevel(logging.CRITICAL)

pairs = load_eval_dataset()

# Get the hybrid predictions
from ml_pipeline.evaluation.runner import _run_matcher_on_dataset
from ml_pipeline.config import load_config

config = load_config()
_, _, hybrid_pred_type, _ = _run_matcher_on_dataset(pairs, "hybrid", config)

# Find wording-only pairs and their predictions
print("Wording-only pairs and their classifications:")
for i, pair in enumerate(pairs):
    if pair.get("expected_change_type") == "wording_only":
        pred = hybrid_pred_type[i]
        status = "PASS" if pred == "wording_only" else "FAIL"
        print(f"\n{status}: {pair['id']}: predicted={pred}")
        print(f"   a: {pair['clause_a_text'][:70]}...")
        print(f"   b: {pair['clause_b_text'][:70]}...")
