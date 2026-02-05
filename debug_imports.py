
import sys
import time

def log(msg):
    print(f"[DEBUG] {time.strftime('%H:%M:%S')} - {msg}")
    sys.stdout.flush()

log("Starting debug script")

log("Importing numpy")
import numpy as np

log("Importing pandas")
import pandas as pd

log("Importing matplotlib")
import matplotlib.pyplot as plt

log("Importing sklearn")
from sklearn.metrics import f1_score

log("Importing torch")
import torch

log("Importing xgboost")
import xgboost

log("Importing local modules")
try:
    from src.core import InvertedIndex, PageContext
    log("Imported src.core")
except Exception as e:
    log(f"Failed to import src.core: {e}")

try:
    from src.models.dnn_model import DNNEntityLinker
    log("Imported DNNEntityLinker")
except Exception as e:
    log(f"Failed to import DNNEntityLinker: {e}")

log("Checking .env file")
from pathlib import Path
project_root = Path('.').resolve()
env_path = project_root / ".env"
if env_path.exists():
    log(f".env found at {env_path}")
    with open(env_path) as f:
        log(f".env content first line: {f.readline().strip()}")
else:
    log(".env not found")

log("Imports complete. Testing LMDB open (read-only)...")
try:
    import lmdb
    index_path = project_root / "Provided-Resources/PostingsLast"
    env = lmdb.open(str(index_path), readonly=True, lock=False)
    log("LMDB opened successfully")
    env.close()
except Exception as e:
    log(f"LMDB failed: {e}")

log("Debug complete")
