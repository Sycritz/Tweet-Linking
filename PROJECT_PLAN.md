# Tweet Entity Linking System (TELS) - Project Plan

## 📋 Project Overview

This is an **end-to-end Entity Linking (EL) system** that takes tweets and links mentioned entities to their corresponding Wikipedia pages. Think of it like auto-tagging tweets with Wikipedia links.

### What is Entity Linking?

1. **Recognition**: Find entity mentions in text (e.g., "Turing" in a tweet)
2. **Linking**: Connect those mentions to the correct Wikipedia page (e.g., "Alan Turing" Wikipedia page vs "Turing Award" vs "Turing test")

### The Challenge with Tweets

- Short, noisy text (typos, slang, abbreviations)
- Lack of context
- Ambiguity (e.g., "Paris" = city in France OR Paris Hilton OR Paris, Texas?)

---

## ✅ Project Requirements - What You MUST Deliver

### 1. Train Models (MANDATORY)

- ✅ **DNN (Deep Neural Network)** - REQUIRED
- ✅ **At least one other ML technique** (Random Forest, SVM, Gradient Boosting, etc.)
- ✅ Use **Meij revised dataset** for training
- ✅ Implement **at least 7 features** from the paper [1]

### 2. Modular Code Architecture

Your code must be clean and organized:

- ✅ **Text processing module** (cleaning tweets, extracting n-grams)
- ✅ **Feature extraction module** (building feature vectors)
- ✅ **Training/inference pipeline**

### 3. Evaluation

- ✅ Test on **ALL provided gold standards**
- ✅ Compare against **TagMe** and **AIDA** (baseline systems)
- ✅ Calculate metrics (Precision, Recall, F1-score)

### 4. Report

- ✅ Detailed, well-organized technical report
- ✅ Justify all design choices
- ✅ Present results with charts/tables
- ✅ Compare with baselines

---

## 🏗️ System Architecture - Recommended Design

```
┌─────────────────────────────────────────────────────────┐
│                    TELS Pipeline                          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  1. TEXT PREPROCESSING MODULE                            │
│  - Clean tweets (remove URLs, @mentions, #hashtags)      │
│  - Extract n-grams (1-grams to 6-grams)                  │
│  - Normalize text                                        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  2. CANDIDATE GENERATION MODULE                          │
│  - Look up n-grams in Inverted Index (PostingsLast)     │
│  - Generate (n-gram, PageID) candidate pairs            │
│  - Filter out low-confidence candidates                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  3. FEATURE EXTRACTION MODULE                            │
│  - Extract 7+ features for each candidate pair:         │
│    • Commonness (anchor frequency)                      │
│    • String similarity                                  │
│    • Context similarity (categories, abstract)          │
│    • Page rank, popularity                             │
│    • Entity type matching, etc.                        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  4. MACHINE LEARNING MODULE                              │
│  - Model 1: DNN (TensorFlow/PyTorch)                    │
│  - Model 2: Random Forest / XGBoost / SVM               │
│  - Training on Meij dataset                             │
│  - Predict: Is this candidate the correct link?         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  5. EVALUATION MODULE                                    │
│  - Test on gold standards                               │
│  - Compare with TagMe & AIDA                            │
│  - Calculate P, R, F1, Accuracy                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Features to Extract (from Paper)

You need **minimum 7 features**. Here are the key ones from the TELS paper:

1. **Commonness**: How often is this n-gram used as anchor to this page? (from inverted index)
2. **String Similarity**: Edit distance / Jaccard similarity between mention and page title
3. **Prior Probability**: P(entity | mention) 
4. **Context Similarity**: Cosine similarity between tweet words and page categories/abstract
5. **Page Popularity**: Number of page views (2019 data)
6. **Page Rank**: Authority score in Wikipedia graph
7. **Entity Type Match**: Does the candidate type match expected type?
8. **Redirect/Disambiguation flags**: Is it a redirect? Is it a disambiguation page?
9. **N-gram Length**: Length of the mention
10. **Position in Tweet**: Where does the mention appear?

### Feature Extraction Details

#### 1. Commonness
```
commonness(ngram, page) = count(ngram → page) / Σ count(ngram → all_pages)
```
This tells you how likely this n-gram refers to this specific page vs other pages.

#### 2. String Similarity
Use Levenshtein distance or Jaccard similarity:
```
similarity(ngram, page_title) = 1 - edit_distance(ngram, page_title) / max_length
```

#### 3. Prior Probability
```
P(page | ngram) = count(ngram → page) / total_occurrences(ngram)
```

#### 4. Context Similarity
Compare tweet words with Wikipedia page categories and abstract using:
- TF-IDF vectors
- Cosine similarity
- Word embeddings (Word2Vec, GloVe)

#### 5. Page Popularity
Use the page view count from 2019 (available in PageIdToContext2 dataset).

#### 6. Page Rank
Graph centrality measure (available in PageIdToContext2 dataset).

---

## 📅 Execution Plan for 100+ Mark

### Phase 1: Setup & Understanding (Days 1-2)

- [ ] Set up Python environment (Python 3.8+)
- [ ] Install dependencies: `lmdb`, `protobuf`, `numpy`, `pandas`, `sklearn`, `tensorflow`/`pytorch`
- [ ] Read the paper [1] thoroughly (take notes)
- [ ] Explore provided datasets (PostingsLast, PageIdToContext2)
- [ ] Run `InvertedIndexAccess.py` and `InterrogatePageIdToContextLmdb.py` to understand data
- [ ] Load and analyze Meij dataset structure
- [ ] Understand gold standard format

### Phase 2: Code Infrastructure (Days 3-5)

- [ ] Create project structure:
  ```
  Tweet-Linking/
  ├── src/
  │   ├── preprocessing/
  │   │   ├── __init__.py
  │   │   ├── text_cleaner.py
  │   │   └── ngram_extractor.py
  │   ├── candidate_generation/
  │   │   ├── __init__.py
  │   │   └── candidate_generator.py
  │   ├── features/
  │   │   ├── __init__.py
  │   │   ├── feature_extractor.py
  │   │   └── feature_utils.py
  │   ├── models/
  │   │   ├── __init__.py
  │   │   ├── dnn_model.py
  │   │   └── ml_model.py
  │   └── evaluation/
  │       ├── __init__.py
  │       └── evaluator.py
  ├── data/
  ├── models/
  ├── results/
  └── notebooks/
  ```
- [ ] Build text preprocessing module
  - Remove URLs, @mentions, hashtags (optional)
  - Lowercase normalization
  - Special character handling
- [ ] Build n-gram extraction functions (1-gram to 6-gram)
- [ ] Build inverted index query wrapper (wrap `InvertedIndexAccess.py`)
- [ ] Build PageID context query wrapper (wrap `InterrogatePageIdToContextLmdb.py`)

### Phase 3: Feature Engineering (Days 6-8)

- [ ] Implement feature extraction functions for each of 7-10 features
- [ ] Build feature vector generator for (mention, pageID) pairs
- [ ] Test feature extraction on sample data
- [ ] Create feature normalization/scaling pipeline
- [ ] Handle missing values and edge cases
- [ ] Save feature extraction pipeline

### Phase 4: Model Development (Days 9-12)

#### DNN Model (TensorFlow/Keras)
- [ ] Prepare Meij dataset (parse annotations)
- [ ] Create train/validation split (80/20)
- [ ] Design DNN architecture:
  - Input layer (7-10 features)
  - Hidden layers (2-3 dense layers with dropout)
  - Output layer (binary classification: link or not)
- [ ] Train with Adam optimizer, binary cross-entropy loss
- [ ] Hyperparameter tuning (learning rate, dropout, batch size)
- [ ] Save best model

#### Second Model (XGBoost/Random Forest)
- [ ] Train XGBoost classifier (recommended for tabular data)
- [ ] Or train Random Forest classifier
- [ ] Hyperparameter tuning (n_estimators, max_depth, etc.)
- [ ] Feature importance analysis
- [ ] Save best model

### Phase 5: Evaluation (Days 13-15)

- [ ] Load all gold standards:
  - Meij dataset (test set)
  - NEEL2014
  - NEEL2015
  - NEEL2016
  - Any others provided
- [ ] Run your TELS system on each gold standard
- [ ] Calculate metrics for each:
  - Precision
  - Recall
  - F1-score
  - Accuracy
- [ ] Compare with baselines:
  - Run TagMe on gold standards
  - Run AIDA on gold standards
  - Create comparison tables
- [ ] Error analysis:
  - What types of entities are hard to link?
  - Common failure patterns?
  - Examples of correct/incorrect predictions
- [ ] Generate visualizations (confusion matrices, PR curves, etc.)

### Phase 6: Report Writing (Days 16-18)

#### Report Structure

1. **Introduction** (1-2 pages)
   - Problem statement
   - Motivation
   - Contributions

2. **Related Work** (2-3 pages)
   - Overview of entity linking
   - Challenges with tweets
   - Existing systems (TagMe, AIDA, WAT, etc.)

3. **Methodology** (4-5 pages)
   - System architecture
   - Data preprocessing
   - Candidate generation
   - Feature engineering (detailed description of each feature)
   - Model architectures (DNN + second model)
   - Training procedure

4. **Experimental Setup** (2 pages)
   - Datasets description
   - Evaluation metrics
   - Baseline systems
   - Implementation details

5. **Results and Analysis** (3-4 pages)
   - Performance on each gold standard
   - Comparison with baselines (tables + charts)
   - Feature importance analysis
   - Error analysis
   - Discussion of findings

6. **Conclusion and Future Work** (1 page)
   - Summary of achievements
   - Limitations
   - Future improvements

7. **References**

8. **Appendix** (optional)
   - Code snippets
   - Additional results
   - Hyperparameter settings

---

## 👥 Task Delegation for Non-Coding Teammates

Since your teammates can't code, here's what they CAN do:

### Teammate 1 - Data Analysis & Documentation

- [ ] Analyze Meij dataset statistics:
  - How many tweets?
  - Average tweet length?
  - Number of annotations?
  - Entity distribution?
- [ ] Create visualizations (bar charts, histograms, pie charts)
- [ ] Document feature descriptions in detail
- [ ] Write **Related Work** section of report (read papers, summarize)
- [ ] Write **Introduction** section
- [ ] Proofread and format final report
- [ ] Create presentation slides

### Teammate 2 - Manual Evaluation & Testing

- [ ] Manually inspect sample predictions
  - Take 50 random tweets
  - Check if entity links are correct
  - Document errors
- [ ] Error analysis:
  - Categorize failure cases (wrong link, missed entity, etc.)
  - Create error taxonomy
- [ ] Run baseline tools (if they have web UIs or simple commands)
- [ ] Create result tables in Excel/Google Sheets
- [ ] Create comparison charts (bar charts, line plots)
- [ ] Help format report (figures, tables, references)
- [ ] Create presentation slides

---

## 🏆 Key Success Factors for 100+

1. ✨ **Strong Feature Engineering**: Go beyond basic features, add creative ones
   - Consider tweet-specific features (hashtags, mentions, emoji)
   - Use embeddings (Word2Vec, BERT) for semantic similarity
   
2. ✨ **Model Diversity**: DNN + ensemble methods (e.g., XGBoost)
   - Ensemble both models for even better results
   
3. ✨ **Thorough Evaluation**: Test on ALL gold standards, detailed error analysis
   - Don't just report numbers, analyze WHY your system works/fails
   
4. ✨ **Beat Baselines**: Your F1 should be competitive with TagMe/AIDA
   - Aim for F1 >= 0.70 (good), F1 >= 0.75 (excellent)
   
5. ✨ **Excellent Report**: Clear writing, good visualizations, deep analysis
   - Use LaTeX for professional formatting
   - High-quality figures and tables
   
6. ✨ **Code Quality**: Clean, modular, well-commented, reproducible
   - Include README with setup instructions
   - Requirements.txt with all dependencies
   - Example usage notebook

---

## 🔧 Technical Stack Recommendations

### Programming Language
- **Python 3.8+**

### Core Libraries
- `numpy` - numerical computing
- `pandas` - data manipulation
- `scikit-learn` - traditional ML models, metrics
- `lmdb` - database access (already provided)
- `protobuf` - data serialization (already provided)

### Deep Learning
**Option 1: TensorFlow/Keras (easier)**
```bash
pip install tensorflow
```

**Option 2: PyTorch (more flexible)**
```bash
pip install torch
```

### Machine Learning
```bash
pip install xgboost  # Recommended for second model
# OR
pip install lightgbm  # Alternative
```

### NLP & Text Processing
```bash
pip install nltk
pip install spacy
pip install gensim  # For Word2Vec embeddings
```

### Utilities
```bash
pip install tqdm  # Progress bars
pip install matplotlib seaborn  # Visualizations
pip install jupyter  # For notebooks
```

---

## 📊 Expected Deliverables

1. **Source Code**
   - Modular Python codebase
   - README.md with setup instructions
   - requirements.txt

2. **Trained Models**
   - Saved DNN model (.h5 or .pt)
   - Saved XGBoost/RF model (.pkl)

3. **Results**
   - Predictions on all gold standards
   - Evaluation metrics (CSV/JSON)
   - Visualizations (PNG/PDF)

4. **Report**
   - PDF document (15-20 pages)
   - Professional formatting
   - Clear figures and tables

5. **Presentation** (likely)
   - PowerPoint/PDF slides
   - Demo (if applicable)

---

## ❓ Questions to Finalize Architecture

Before we start coding, let's decide on:

1. **Programming Stack**: 
   - TensorFlow/Keras or PyTorch for DNN?
   - XGBoost, Random Forest, or SVM for second model?

2. **Feature Set**: 
   - Stick to basic 7 features or go for 10+ with embeddings?
   - Use pre-trained word embeddings (Word2Vec, GloVe)?

3. **Timeline**: 
   - How much time until deadline?
   - What's the priority order?

4. **Existing Code**: 
   - Any starter code or build from scratch?
   - Do the provided Python scripts work on your machine?

5. **Computing Resources**: 
   - GPU access for DNN training?
   - RAM constraints?

6. **Baseline Systems**:
   - Can we access TagMe and AIDA APIs?
   - Or do we need to run them locally?

---

## 🚀 Next Steps

1. **Review this plan** - Make sure you understand everything
2. **Answer the questions above** - So we can finalize tech stack
3. **Set up environment** - Install Python and dependencies
4. **Test provided scripts** - Make sure data access works
5. **Start Phase 1** - Begin exploration and setup

Let me know your thoughts and answers to the questions, and we can start building! 💪
