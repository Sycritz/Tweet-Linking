# TELS: Team Development Guide

This guide is for team members setting up the project on their local machines (especially Windows with VS Code). It explains the project's goal, current status, and the new modular architecture.

## 1. Project Overview
The goal is to build a **Tweet Entity Linking System (TELS)**. We take raw tweets and link specific mentions (n-grams) to the correct Wikipedia pages using a Machine Learning model (SVM, DNN, or XGBoost).

### The Pipeline:
1.  **Preprocessing**: Clean the tweet text.
2.  **Candidate Generation**: Look up n-grams in the Inverted Index (LMDB) to find potential Wikipedia matches.
3.  **Feature Extraction**: Calculate scores (commonness, similarity, rank, etc.) for each candidate.
4.  **Classification**: The model predicts which candidate is the "Correct" link (Binary Classification).

---

## 2. Today's "Big Refactor"
We have moved away from a single "MVP" script. The old structure used `sys.path.append` hacks that work differently on Windows and Linux, causing constant import errors.

**The code is now organized into a real Python package structure.**

### New Directory Structure:
```text
/src
  ├── core.py           <-- Core data structures (Candidate, InvertedIndex)
  ├── proto/            <-- Protocol Buffer generated files (_pb2.py)
  ├── preprocessing/    <-- Text cleaning and n-gram logic
  ├── candidate_generation/ <-- Candidate lookup logic
  ├── features/         <-- Feature calculation logic
  ├── models/           <-- SVM, DNN, XGBoost implementations
  └── utils/            <-- Dataset loaders and high-level wrappers
```

---

## 3. Mandatory Setup (Windows / VS Code)

### Fix Import Errors Immediately:
Standard Python tools (and VS Code/Pyright) look for imports starting from the project root.
1.  **DO NOT** use `sys.path.append`. It is now gone from the project.
2.  **Absolute Imports**: Always import from the root `src` folder.
    - ✅ `from src.core import Candidate`
    - ✅ `from src.proto import SerializedListNew_pb2`
3.  **VS Code Setting**: If VS Code is highlighting imports in red, ensure you have the workspace root opened as the primary folder.

### Protocol Buffers:
The `.proto` generated files are now in `src.proto`. Because they are binary-heavy, we have standardized the implementation in `src/core.py`:
```python
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
```
This ensures consistent behavior across OS boundaries (Windows/Linux).

---

## 4. Building the SVM Pipeline
If you are working on the SVM model, follow this modular flow:

### Step 1: Initialize Resources
```python
from src.core import InvertedIndex, PageContext

index = InvertedIndex("path/to/PostingsLast")
context = PageContext("path/to/PageIdToContext2")
```

### Step 2: Preprocess & Generate Candidates
```python
from src.candidate_generation.candidate_generator import generate_candidates

# This uses cleaning/ngrams internally
candidates = generate_candidates(tweet_text, index, context)
```

### Step 3: Feature Extraction
```python
from src.features.features_extractor import extract_features

for candidate in candidates:
    features = extract_features(candidate, total_score)
    # Features is a numpy array ready for the SVM
```

### Step 4: Model Prediction
Import the `SVMEntityLinker` from `src.models.svm_model` and use its `.predict()` method on the feature arrays.

---

## Tips for Teammates:
- **Naming**: Use `snake_case` for filenames. We converted all hyphenated files (e.g., `dnn-model.py` -> `dnn_model.py`) to avoid import syntax issues.
- **Paths**: Keep the `Provided-Resources` folder at the root. Do not move the LMDB databases unless you update your local paths.
