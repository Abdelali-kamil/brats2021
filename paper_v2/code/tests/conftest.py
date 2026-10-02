"""Make the tests import paper_v2/code/brats_gbm, not the first-phase package at the repo root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
