from __future__ import annotations

import importlib
import sys
from pathlib import Path

_eval_dir = str(Path(__file__).parent)
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

dataset = importlib.import_module("dataset")
EVAL_CASES = dataset.EVAL_CASES
EvalCase = dataset.EvalCase
