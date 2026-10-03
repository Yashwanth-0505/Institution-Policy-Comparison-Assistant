"""Debug script to identify which wording-only pair is misclassified"""
from ml_pipeline.evaluation.runner import _run_matcher_on_dataset
from ml_pipeline.evaluation.dataset import load_eval_dataset
from ml_pipeline.config import load_config
from ml_pipeline.detector import _extract_signals

pairs = load_eval_dataset()
config = load_config()

# Get predictions
_, _, hybrid_pred_type, _ = _run_matcher_on_dataset(pairs, "hybrid", config)

# Find wording-only pairs
print("=" * 70)
print("WORDING-ONLY PAIRS ANALYSIS")
print("=" * 70)

for i, pair in enumerate(pairs):
    if pair.get("expected_change_type") == "wording_only":
        pred = hybrid_pred_type[i]
        status = "PASS" if pred == "wording_only" else "FAIL"
        
        # Get signals
        text_a = pair.get("clause_a_text", "")
        text_b = pair.get("clause_b_text", "")
        signals, ratio = _extract_signals(text_a, text_b)
        
        print(f"\n[{status}] {pair['id']}")
        print(f"   Expected: wording_only")
        print(f"   Predicted: {pred}")
        print(f"   Ratio: {ratio:.4f}")
        print(f"   Signals: {[s.signal_type for s in signals]}")
        print(f"   Text A: {text_a[:60]}...")
        print(f"   Text B: {text_b[:60]}...")
