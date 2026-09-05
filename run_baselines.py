"""Launch from any working directory; all writes stay in this isolated copy."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
os.environ["NUMBA_CACHE_DIR"] = str(ROOT / ".cache/numba")
os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from external_baselines.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
