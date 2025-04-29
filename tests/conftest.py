# tests/conftest.py
import sys
from pathlib import Path

# Add repo root (parent of this file) to import search path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
