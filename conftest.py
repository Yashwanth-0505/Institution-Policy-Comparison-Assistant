# Root conftest.py — makes d:/jig the pytest rootdir and adds ml_pipeline
# to sys.path so test imports resolve correctly.
import sys
from pathlib import Path

# Ensure d:\jig is on sys.path so both `ml_pipeline` and
# `ml_pipeline.tests` are importable.
root = Path(__file__).parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
