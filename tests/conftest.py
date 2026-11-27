import sys
from pathlib import Path

# Permite `from src import preprocess` ejecutando pytest desde la raiz.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
