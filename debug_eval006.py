"""Debug eval_006 modal change"""
import json
from ml_pipeline.detector import _extract_signals

with open('ml_pipeline/evaluation/data/labelled_pairs.json') as f:
    pairs = json.load(f)

p = [p for p in pairs if p['id'] == 'eval_006'][0]

print(f"\neval_006 Analysis:")
print(f"Category: {p['category']}")
print(f"\nText A:\n{p['clause_a_text']}\n")
print(f"Text B:\n{p['clause_b_text']}\n")

signals, ratio = _extract_signals(p['clause_a_text'], p['clause_b_text'])

print(f"Ratio: {ratio:.4f}")
print(f"Signals: {len(signals)}")
for s in signals:
    print(f"  - {s.signal_type}: {s.old_value} -> {s.new_value}")
