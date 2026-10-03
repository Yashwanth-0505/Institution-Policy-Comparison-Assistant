"""Test each wording-only pair individually"""
import json
from ml_pipeline.detector import _extract_signals, _classify
from ml_pipeline.aligner import HybridMatcher
from ml_pipeline.features import FeatureExtractor
from ml_pipeline.config import load_config
from ml_pipeline.models import ExtractedClause, SectionMetadata

with open('ml_pipeline/evaluation/data/labelled_pairs.json') as f:
    pairs = json.load(f)

wo_pairs = [p for p in pairs if p.get('expected_change_type') == 'wording_only']
config = load_config()
extractor = FeatureExtractor(config=config)

print("\nTesting each wording-only pair:\n")

for p in wo_pairs:
    text_a = p['clause_a_text']
    text_b = p['clause_b_text']
    
    signals, ratio = _extract_signals(text_a, text_b)
    
    # Create minimal clause objects for _classify
    clause_a = ExtractedClause(
        document_id="a", document_name="a.pdf", version="1", clause_id="a0",
        heading="", original_text=text_a, normalized_text=text_a, page_number=1,
        section_metadata=SectionMetadata(), flagged_for_review=False, review_reason=""
    )
    clause_b = ExtractedClause(
        document_id="b", document_name="b.pdf", version="1", clause_id="b0",
        heading="", original_text=text_b, normalized_text=text_b, page_number=1,
        section_metadata=SectionMetadata(), flagged_for_review=False, review_reason=""
    )
    
    from ml_pipeline.models import ClauseMatch, MatchType
    match = ClauseMatch(
        clause_a=clause_a, clause_b=clause_b, match_score=ratio,
        number_score=1.0, heading_score=1.0, embedding_score=0.9,
        tfidf_score=0.9, match_type=MatchType.HYBRID,
        needs_review=False, review_reason=""
    )
    
    change_type, needs_review, reason = _classify(signals, ratio, match)
    
    status = "PASS" if change_type.value == "wording_only" else f"FAIL -> {change_type.value}"
    print(f"{p['id']}: {status}")
    print(f"  Ratio: {ratio:.4f}")
    print(f"  Signals: {[s.signal_type for s in signals]}")
    print()
