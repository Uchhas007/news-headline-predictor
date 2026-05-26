# 📰 News Headline Topic Classifier
### Multi-Class Text Classification — Comparative Analysis of ML & Deep Learning Models

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow?logo=scikit-learn)
![spaCy](https://img.shields.io/badge/spaCy-NLP-09a3d5?logo=spacy)
![Gensim](https://img.shields.io/badge/Gensim-Word2Vec-green)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline Overview](#-pipeline-overview)
- [Preprocessing Strategies](#-preprocessing-strategies)
- [Word Representations](#-word-representations)
- [Models & Architectures](#-models--architectures)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Evaluation Metrics](#-evaluation-metrics)
- [Key Findings & Results](#-key-findings--results)
- [Dependencies & Installation](#-dependencies--installation)
- [How to Run](#-how-to-run)
- [Experiment Summary Table](#-experiment-summary-table)

---

## 🧠 Project Overview

This project performs **4-class news headline topic classification** using a systematic comparative study across:

- **3 text preprocessing strategies** (No Preprocessing, Extreme, Optimum)
- **2 word representation techniques** (TF-IDF and Skip-gram Word2Vec embeddings)
- **8 model architectures** (Logistic Regression, DNN, SimpleRNN, GRU, LSTM, Bidirectional SimpleRNN, Bidirectional GRU, Bidirectional LSTM)

resulting in **24 distinct training experiments** evaluating every combination of preprocessing × representation × model.

The dataset consists of real-world HTML-formatted news headlines, each labelled under one of four topics: **Business**, **Science and Technology**, **World News**, and **Sports**.

---

## 📊 Dataset
## 📊 Dataset

This project uses a news headline dataset hosted on Kaggle.  
All data can be accessed and downloaded from the following link:

https://www.kaggle.com/datasets/uchhas007/news-headline-dataset
### Source & Format

The dataset is provided as two pre-split CSV files:

| File | Role | Rows | Columns |
|---|---|---|---|
| `Training_data_8.csv` | Model training & validation | **93,151** | `News Headline`, `News Topic` |
| `Test_data.csv` | Final held-out evaluation | **12,000** | `News Headline`, `News Topic` |

> Columns are renamed to `text` and `label` at load time for consistency.

---

### Class Distribution

#### Training Set — **Imbalanced**

| Class | Count | Percentage |
|---|---|---|
| Business | 36,316 | 38.99% |
| Science and Technology | 22,422 | 24.07% |
| World News | 19,118 | 20.52% |
| Sports | 15,295 | 16.42% |
| **Total** | **93,151** | **100%** |

> ⚠️ The training set is notably imbalanced. Business dominates at ~39% while Sports is the minority class at ~16%. This motivated using **Macro F1** as the primary evaluation metric rather than plain accuracy.

#### Test Set — **Perfectly Balanced**

| Class | Count | Percentage |
|---|---|---|
| Business | 3,000 | 25% |
| Science and Technology | 3,000 | 25% |
| World News | 3,000 | 25% |
| Sports | 3,000 | 25% |
| **Total** | **12,000** | **100%** |

The balanced test set ensures fair evaluation across all four categories regardless of training imbalance.

---

### Raw Text Format

Headlines are wrapped in HTML markup. A typical raw entry looks like:

```html
<html> <body> News Headlines:
 <br> <b> Nortel Networks Names Clent Richardson Chief Marketing Officer
 Nortel Networks Corp., North America's bigger maker of phone equipment,
 named Clent Richardson as its chief marketing officer... </b>
 ...
</body> </html>
```

This HTML noise (`<html>`, `<body>`, `<br>`, `<b>` tags and HTML entities like `&amp;`, `&quot;`, `&#39;`) is one of the key challenges the preprocessing pipeline must address.

---

### Text Length Statistics (Post HTML Stripping, Train Sample of 3,000)

| Metric | Value |
|---|---|
| Mean word count | ~39 words |
| Median word count | ~39 words |
| Std deviation (words) | ~9.7 |
| Min word count | 12 words |
| Max word count | 135 words |
| **95th percentile** | **54 words** |
| Mean character length | ~257 characters |

> **Implication:** `max_sequence_length = 60` tokens was chosen for RNN models to cover the 95th percentile of headline lengths without wasteful padding.

All four classes exhibit nearly identical text-length distributions, confirming that **length alone is not a discriminating feature** — the actual word content and vocabulary are what distinguish the categories.

---

### Label Encoding

Labels are integer-encoded using `sklearn.preprocessing.LabelEncoder`:

| Label String | Encoded Integer |
|---|---|
| Business | 0 |
| Science and Technology | 1 |
| Sports | 2 |
| World News | 3 |

---

## 📁 Project Structure

```
news-headline-classifier/
│
├── news_headline_predictor.ipynb   ← Main project notebook (all experiments)
├── Training_data_8.csv             ← Training data (93,151 rows)
├── Test_data.csv                   ← Test data (12,000 rows, balanced)
└── README.md
```

The notebook is self-contained and sequentially structured. All EDA, preprocessing, model training, evaluation, and result comparisons are embedded with output logs and visualisations.

---

## 🔄 Pipeline Overview

```
Raw HTML Text (CSV)
        │
        ▼
┌───────────────────┐
│  HTML Stripping   │  BeautifulSoup4 — decode entities, remove tags
└───────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│              3 Preprocessing Branches                       │
│                                                             │
│  [1] No Preprocessing   →  Raw HTML text as-is             │
│  [2] Extreme            →  SpaCy POS filter + Stemming      │
│  [3] Optimum            →  SpaCy Lemmatisation (no POS)     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│   Word Representation               │
│                                     │
│  TF-IDF  → Logistic Regression      │
│         → Deep Neural Network       │
│                                     │
│  Skip-gram Word2Vec → SimpleRNN     │
│                    → GRU            │
│                    → LSTM           │
│                    → Bi-SimpleRNN   │
│                    → Bi-GRU         │
│                    → Bi-LSTM        │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│   Evaluation                        │
│   • Accuracy                        │
│   • Macro F1 Score                  │
│   • Confusion Matrix                │
│   • Classification Report           │
└─────────────────────────────────────┘
```

---

## 🧹 Preprocessing Strategies

Three distinct preprocessing pipelines were implemented and applied independently.

### 1. No Preprocessing (Baseline)

The raw HTML text is passed directly to TF-IDF/tokeniser with zero cleaning. This serves as the baseline to measure the degrading effect of HTML noise on model performance.

```python
train_raw = train_df['text'].tolist()
```

### 2. Extreme Preprocessing

**Steps applied (in order):**

1. **HTML stripping** — `BeautifulSoup4` removes all tags and decodes HTML entities
2. **Lowercase** — all characters lowercased
3. **SpaCy tokenisation** — `en_core_web_sm` pipeline
4. **POS filtering** — only tokens tagged as `NOUN`, `VERB`, `ADJ`, or `ADV` are kept
5. **Stopword removal** — SpaCy's built-in stop word list applied
6. **Alpha filter** — non-alphabetic tokens and tokens shorter than 2 characters removed
7. **Stemming** — PorterStemmer reduces each token to its root form

```python
allowed_pos_tags = {'NOUN', 'VERB', 'ADJ', 'ADV'}
stemmer = PorterStemmer()

# For each SpaCy token:
# keep if: POS ∈ allowed_pos_tags AND not stop AND is_alpha AND len ≥ 2
# then: apply PorterStemmer
```

**Tradeoff:** Aggressive — reduces vocabulary substantially and removes noise, but PorterStemmer corrupts domain-specific discriminators (e.g., `"earnings"` → `"earn"`, `"league"` → `"leagu"`), which hurts both embedding quality and classification performance.

### 3. Optimum Preprocessing *(Recommended)*

**Steps applied (in order):**

1. **HTML stripping** — `BeautifulSoup4`
2. **Lowercase**
3. **Regex cleaning** — non-alphabetic characters and extra whitespace removed (`re.sub(r'[^a-z\s]', ' ', ...)`)
4. **SpaCy lemmatisation** — `tok2vec`, `tagger`, `morphologizer`, `lemmatizer` pipes active; `parser` and `ner` disabled for speed
5. **Stopword removal** — SpaCy stopwords
6. **Alpha + length filter** — non-alpha and length ≤ 1 removed

```python
nlp.select_pipes(enable=["tok2vec", "tagger", "morphologizer", "lemmatizer"])

def preprocess_text_optimum(parsed_doc):
    return ' '.join(
        token.lemma_
        for token in parsed_doc
        if not token.is_stop
        and token.is_alpha
        and len(token.lemma_) > 1
    )
```

**Why this is optimal:**
- **No POS filter** — in short news headlines, all word types (including determiners, prepositions in domain phrases) carry meaning
- **Lemmatisation over Stemming** — preserves real dictionary words (`running → run`, not `runn`), maintaining embedding coverage
- **Stopword removal** — reduces noise without losing discriminative vocabulary
- **HTML entity decoding** — `&amp;` → `&`, `&#39;` → `'` handled cleanly by BeautifulSoup

---

## 🔤 Word Representations

### TF-IDF

Applied to all three preprocessing variants. Fitted exclusively on training data; test data is only transformed (no data leakage).

```python
TfidfVectorizer(
    max_features = 50_000,   # large vocab preserves domain terms
    ngram_range  = (1, 2),   # unigrams + bigrams capture compound phrases
    sublinear_tf = True,     # log(1+tf) — dampens effect of very frequent terms
    min_df       = 2         # ignore tokens appearing in < 2 documents (noise)
)
```

| Preprocessing | Train Shape | Test Shape |
|---|---|---|
| No Preprocessing | (93151, 50000) | (12000, 50000) |
| Extreme | (93151, 50000) | (12000, 50000) |
| Optimum | (93151, 50000) | (12000, 50000) |

**Memory note:** TF-IDF matrices are stored as sparse matrices (`scipy.sparse`). Batched `.toarray()` conversion is used during DNN training to avoid OOM errors on GPU.

---

### Skip-gram Word2Vec

One Word2Vec model trained per preprocessing variant, exclusively on training data.

```python
Word2Vec(
    sentences   = tokenized_corpus,
    vector_size = 128,   # embedding dimension (compact but expressive for short headlines)
    window      = 5,     # context window — standard for short-to-medium texts
    sg          = 1,     # Skip-gram (sg=1); CBOW would be sg=0
    min_count   = 2,     # discard tokens with < 2 corpus occurrences
    workers     = 4,     # parallel threads
    epochs      = 10,    # sufficient convergence for 93k documents
    seed        = 42
)
```

**Embedding matrix construction:**
- Keras `Tokenizer` fitted on training texts only
- For each word in the tokenizer vocabulary: look up the Word2Vec vector; zero-initialise OOV words
- The resulting `embedding_matrix` (shape: `[vocab_size, 128]`) is loaded into a frozen `Embedding` layer

```python
# Sequence padding parameters
embedding_dim       = 128
max_sequence_length = 60   # covers 95th percentile of cleaned headline lengths
```

---

## 🏗️ Models & Architectures

### 1. Logistic Regression

- **Input:** TF-IDF sparse matrix
- **Solver:** `saga` (supports multinomial + L2, scales to large sparse data)
- **Multi-class strategy:** `multinomial`

```
TF-IDF vector (50000-d) → LogisticRegression(C=1.0, solver='saga', multinomial)
```

### 2. Deep Neural Network (DNN)

- **Input:** TF-IDF (dense batch conversion)
- **Architecture:** 4 fully-connected layers with BatchNorm and Dropout

```
Input (50000-d)
    │
    ├─ Dense(512, relu) → BatchNorm → Dropout(0.4)
    ├─ Dense(256, relu) → BatchNorm → Dropout(0.4)
    ├─ Dense(128, relu) → BatchNorm → Dropout(0.4)
    ├─ Dense(64,  relu) → BatchNorm → Dropout(0.4)
    │
    └─ Dense(4, softmax)
```

**Training details:**
- Optimizer: Adam (lr = 1e-3)
- Loss: Sparse Categorical Cross-Entropy
- Batch size: 512 (batched sparse → dense conversion)
- Validation split: 10% of training data
- EarlyStopping: patience = 3 (monitors `val_loss`)
- ReduceLROnPlateau: factor = 0.5, patience = 2, min_lr = 1e-5

### 3–8. RNN Models (Skip-gram Input)

Six architectures, all sharing the same embedding and output structure:

```
Embedding (128-d, Skip-gram weights, frozen, input_length=60)
    │
    └─ [RNN Layer]  →  Dense(64, relu) → Dropout(0.3) → Dense(4, softmax)
```

| Model | Layer Type | Bidirectional |
|---|---|---|
| SimpleRNN | `SimpleRNN(128)` | No |
| GRU | `GRU(128)` | No |
| LSTM | `LSTM(128)` | No |
| Bidirectional SimpleRNN | `Bidirectional(SimpleRNN(128))` | Yes |
| Bidirectional GRU | `Bidirectional(GRU(128))` | Yes |
| Bidirectional LSTM | `Bidirectional(LSTM(128))` | Yes |

**Common RNN hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `rnn_units` | 128 | Sufficient capacity for 4-class classification |
| `dropout` | 0.3 | Applied inside RNN layer and after |
| `recurrent_dropout` | 0.1 | Light regularisation on recurrent connections |
| `embedding_dim` | 128 | Matches Skip-gram vector size |
| `max_sequence_length` | 60 | Covers 95th percentile of headline lengths |
| `batch_size` | 256 | Optimal for T4 GPU throughput |
| `learning_rate` | 1e-3 | Standard Adam default |
| `epochs` | 15 | With EarlyStopping |

**Callbacks:**
```python
EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
```

**Bidirectional advantage:** Bidirectional wrappers process the token sequence in both forward and backward directions, effectively doubling the feature space and capturing left-to-right and right-to-left context simultaneously — critical for understanding news headline phrasing where context at both ends of the headline matters.

---

## 🔧 Hyperparameter Tuning

### Logistic Regression

Regularisation strength `C` was tuned via **3-fold cross-validation** on the optimum-preprocessed training data:

| C | CV Macro F1 (mean ± std) |
|---|---|
| 0.1 | (under-regularised — higher bias) |
| **1.0** | **Best — balanced bias-variance trade-off** |
| 10.0 | (over-fits training noise) |

**Decision:** `C = 1.0` selected and applied to all three preprocessing variants.

### DNN

Tuning was performed by manually monitoring `val_loss` across configurations. Final chosen values:

| Parameter | Tuned Value |
|---|---|
| Hidden units | [512, 256, 128, 64] |
| Dropout rate | 0.4 |
| Learning rate | 1e-3 |
| Batch size | 512 |
| Optimiser | Adam |

### RNN Models

Key observations during training:
- **128 RNN units** provided the best capacity without overfitting on this dataset
- **Dropout 0.3** (vs 0.5) avoided under-fitting while still regularising
- **recurrent_dropout 0.1** added light regularisation without slowing convergence
- Most RNN models converged between **epoch 5–10**; EarlyStopping prevented wasted compute

---

## 📏 Evaluation Metrics

All experiments are evaluated consistently using four metrics:

| Metric | Why it was chosen |
|---|---|
| **Accuracy** | Standard classification measure; reported alongside F1 |
| **Macro F1 Score** | Primary metric — averages F1 across all classes equally, making it robust to training-set class imbalance |
| **Confusion Matrix** | Identifies per-class errors and systematic misclassification patterns |
| **Classification Report** | Per-class precision, recall, and F1 for detailed diagnostics |

> **Why Macro F1?** The training set is imbalanced (Business 39%, Sports 16%). Accuracy alone would reward models that focus on the majority class. Macro F1 treats every class with equal weight, giving a fairer picture of true multi-class performance — especially important for the minority Sports class.

---

## 📈 Key Findings & Results

### Best vs Worst Configuration

| | **BEST** | **WORST** |
|---|---|---|
| **Model** | Bidirectional LSTM / Bidirectional GRU | SimpleRNN (Unidirectional) |
| **Preprocessing** | Optimum | No Preprocessing |
| **Representation** | Skip-gram | Skip-gram |
| **Why** | Bidirectional context + clean lemmatised embeddings | Vanishing gradients + HTML noise in raw embeddings |

---

### Preprocessing Effect

| Preprocessing | Behaviour | Typical Result |
|---|---|---|
| **Optimum** *(best)* | Clean, semantically rich tokens; high embedding coverage | Highest F1 across most models |
| **No Preprocessing** | HTML tags corrupt tokenisation; noisy vocabulary | Competitive for TF-IDF (IDF penalises `<html>`), poor for embeddings |
| **Extreme** *(worst)* | Aggressive POS filter and stemming strip discriminative domain terms | Loses information like `"earnings"`, `"league"`, `"technology"` |

**Key insight from EDA:** Class-specific vocabulary (e.g., *game, season, player* for Sports; *company, market, billion* for Business) is extremely discriminative. Any preprocessing that destroys or corrupts these domain-specific tokens (like aggressive stemming) directly hurts classification F1.

---

### Representation Effect

| Representation | Best With | Observation |
|---|---|---|
| **TF-IDF** | Logistic Regression | Strong performance on keyword-rich short headlines; bag-of-words captures domain vocabulary well |
| **Skip-gram** | Bidirectional LSTM / GRU | Semantic similarity captured; benefits sequence models that exploit word order |

**TF-IDF + LR insight:** Given that news headlines are short and vocabulary-rich (distinct per topic), TF-IDF's sparse explicit feature representation combined with Logistic Regression's linear boundary is surprisingly competitive against deeper models. It is also the fastest and most interpretable combination.

---

### Model Architecture Ranking

From best to worst (Macro F1, Optimum preprocessing):

1. 🥇 **Bidirectional LSTM** — captures bidirectional context with long-range memory gates
2. 🥈 **Bidirectional GRU** — similar to BiLSTM, fewer parameters, slightly faster
3. 🥉 **LSTM** — long-range dependencies without bidirectional context
4. **GRU** — efficient gated unit, slightly below LSTM
5. **DNN (TF-IDF)** — strong for bag-of-words, no sequential awareness
6. **Logistic Regression (TF-IDF)** — fast, interpretable, competitive baseline
7. **Bidirectional SimpleRNN** — bidirectional helps but no gating
8. **SimpleRNN** *(worst)* — vanishing gradient issue over 60-token sequences; struggles with long-range dependencies in headlines

---

### Class-Level Observations

- **Business** (majority class) — consistently easiest to classify across all models
- **Sports** (minority class) — benefits most from Macro F1; models trained with raw data tend to underperform here
- **Science and Technology vs World News** — the most commonly confused class pair; they share vocabulary around innovation, policy, and geopolitics

---

## 📦 Dependencies & Installation

### Requirements

```
python >= 3.10
tensorflow >= 2.12
scikit-learn >= 1.2
gensim >= 4.3
spacy >= 3.5
nltk >= 3.8
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.7
seaborn >= 0.12
beautifulsoup4 >= 4.12
```

### Installation

```bash
pip install nltk spacy gensim tensorflow scikit-learn beautifulsoup4 matplotlib seaborn pandas numpy
python -m spacy download en_core_web_sm
```

### NLTK Data

The notebook downloads the following NLTK corpora at runtime:

```python
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

---

## ▶️ How to Run

### Google Colab (Recommended)

1. Upload `Training_data_8.csv` and `Test_data.csv` to `/content/`
2. Open `news_headline_predictor.ipynb` in Colab
3. Set runtime to **GPU (T4)**
4. Run all cells top-to-bottom

> The notebook is tuned for T4 GPU. Key memory-safety features:
> - `tf.keras.backend.clear_session()` + `gc.collect()` after every Keras model
> - Sparse matrix batched conversion (`.toarray()` in chunks of 1024) for DNN
> - `ReduceLROnPlateau` and `EarlyStopping` prevent runaway training

### Local Execution

```bash
git clone <repo-url>
cd news-headline-classifier
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter notebook news_headline_predictor.ipynb
```

---

## 📋 Experiment Summary Table

All 24 model–preprocessing–representation combinations:

| # | Model | Preprocessing | Representation |
|---|---|---|---|
| 1 | Logistic Regression | No Preprocessing | TF-IDF |
| 2 | Logistic Regression | Extreme | TF-IDF |
| 3 | Logistic Regression | Optimum | TF-IDF |
| 4 | Deep Neural Network | No Preprocessing | TF-IDF |
| 5 | Deep Neural Network | Extreme | TF-IDF |
| 6 | Deep Neural Network | Optimum | TF-IDF |
| 7 | SimpleRNN | No Preprocessing | Skip-gram |
| 8 | SimpleRNN | Extreme | Skip-gram |
| 9 | SimpleRNN | Optimum | Skip-gram |
| 10 | GRU | No Preprocessing | Skip-gram |
| 11 | GRU | Extreme | Skip-gram |
| 12 | GRU | Optimum | Skip-gram |
| 13 | LSTM | No Preprocessing | Skip-gram |
| 14 | LSTM | Extreme | Skip-gram |
| 15 | LSTM | Optimum | Skip-gram |
| 16 | Bidirectional SimpleRNN | No Preprocessing | Skip-gram |
| 17 | Bidirectional SimpleRNN | Extreme | Skip-gram |
| 18 | Bidirectional SimpleRNN | Optimum | Skip-gram |
| 19 | Bidirectional GRU | No Preprocessing | Skip-gram |
| 20 | Bidirectional GRU | Extreme | Skip-gram |
| 21 | Bidirectional GRU | Optimum | Skip-gram |
| 22 | Bidirectional LSTM | No Preprocessing | Skip-gram |
| 23 | Bidirectional LSTM | Extreme | Skip-gram |
| 24 | Bidirectional LSTM | Optimum | Skip-gram |

---

## 📌 Design Decisions & Rationale

| Decision | Choice | Reason |
|---|---|---|
| Primary metric | Macro F1 | Training set is imbalanced; macro F1 weights all classes equally |
| TF-IDF ngram_range | (1, 2) | Bigrams (e.g., *"stock market"*, *"world cup"*) are highly informative for news topics |
| TF-IDF max_features | 50,000 | Captures full domain vocabulary across 4 news topics |
| Skip-gram vs CBOW | Skip-gram (`sg=1`) | Better for rare words and smaller datasets; preferred for domain-specific corpora |
| Embedding dim | 128 | Balances expressiveness and memory footprint; matches RNN unit count |
| max_sequence_length | 60 | Covers 95th percentile of cleaned headline word counts |
| Lemmatisation over Stemming | Lemmatisation | Preserves real dictionary words, maintains embedding coverage |
| POS filter excluded (Optimum) | Not applied | Short headlines need all word types; POS filtering over-reduces |
| Embedding frozen | `trainable=False` | Prevents overfitting with pre-trained Skip-gram weights on limited headlines |
| DNN batch size | 512 | Efficient for sparse→dense conversion on GPU |
| RNN batch size | 256 | Balances throughput and memory on T4 GPU |
| EarlyStopping patience | 3 | Prevents overfitting without stopping too early |

---

## 🧾 Acknowledgements

- Dataset: Custom news headline collection (4-class topic labels)
- SpaCy English model: `en_core_web_sm`
- Word embeddings: Gensim Word2Vec (skip-gram, trained on corpus)
- Deep learning framework: TensorFlow / Keras
- ML baseline: scikit-learn LogisticRegression

---

*This project was developed as part of a Natural Language Processing course lab assignment exploring the impact of preprocessing, word representation, and model architecture choices on multi-class text classification performance.*
