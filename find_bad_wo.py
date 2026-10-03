"""Find which wording-only pair is misclassified"""
import json

with open('ml_pipeline/evaluation/data/labelled_pairs.json') as f:
    pairs = json.load(f)

wo_pairs = [p for p in pairs if p.get('expected_change_type') == 'wording_only']

print(f"\nFound {len(wo_pairs)} wording-only pairs:\n")
for p in wo_pairs:
    print(f"ID: {p['id']}")
    print(f"  Category: {p['category']}")
    print(f"  A: {p['clause_a_text'][:70]}...")
    print(f"  B: {p['clause_b_text'][:70]}...")
    print()
