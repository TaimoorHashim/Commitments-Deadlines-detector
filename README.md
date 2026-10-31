# 📌 Commitment and Deadline Extraction System

An intelligent Natural Language Processing (NLP) system and modern web interface built with **Gradio**, **spaCy**, **NLTK**, and **Regular Expressions** to automatically detect commitments, assignees, actions, and deadlines in everyday written communication (emails, Slack messages, meeting notes, etc.).

---

## 🚀 Quick Start

### 1. Launch the Gradio Web Interface
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:7860
```

---

## 🌟 Key Features

### 🔍 1. Single Sentence Quick Inspector
- **Instant Extraction**: Test any sentence or select from preset examples.
- **Visual Status**: Clear status badges indicating whether a commitment was detected and the specific matched trigger phrase.
- **Structured Fields**:
  - 👤 **Responsible Person**: Extracted via spaCy Named Entity Recognition (`PERSON`) and grammatical subject analysis (`nsubj`/`PROPN`).
  - ⚡ **Action / Task**: Identified using action verb lemmatization and direct object subtree dependency parsing (`dobj`, `obj`, `attr`).
  - ⏰ **Deadline**: Detected using comprehensive regular expression patterns.
  - 🎯 **Trigger Matched**: Specific keyword/phrase that signaled the commitment.
- **Linguistic Breakdown**: Interactive tables displaying token POS tags, lemmas, dependency relations, and named entity recognition results.

### 📄 2. Document & Batch Analyzer
- **Multi-Sentence Text Area**: Paste complete emails, meeting minutes, or project chat transcripts.
- **KPI Summary Cards**: Real-time counts of total sentences, commitments detected, non-commitments, deadlines identified, and unique assignees.
- **Interactive Data Table**: View all sentences or filter dynamically by *Commitments Only* or *Non-Commitments Only*.
- **File Upload**: Upload `.txt` or `.csv` files for automated batch processing.
- **CSV Export**: One-click download of the structured results table.

### 📈 3. Model Evaluation & Benchmark
- **Benchmark Suite**: Automatically evaluates the model against the 20 benchmark sentences from `task.ipynb`.
- **Performance Metrics**:
  - **Accuracy: 90.0%**
  - **Precision: 83.33%**
  - **Recall: 100.0%**
  - **F1-Score: 90.91%**
- **Confusion Matrix**: Visual breakdown of True Positives (10), False Positives (2), True Negatives (8), and False Negatives (0).
- **Interactive Metric Chart**: Clean Matplotlib bar visualization.
- **Detailed Evaluation Table**: Full comparison of actual ground truth vs model prediction for every benchmark sentence.

### ⚙️ 4. Pattern Library & Rules
- Reference catalog of all monitored commitment trigger patterns (`\bi will\b`, `\bpromised to\b`, `\bmust\b`, etc.).
- Regular expressions for deadlines (relative days, specific dates, timeframes).
- Monitored action verbs list.
- Configurable settings:
  - Natural word ordering vs original token subtree order.
  - Extended deadline recognition (e.g. `on 12 November`, `by 5pm`).
  - Speaker pronoun resolution (`I` as Speaker, `We` as Team).

---

## 📁 Project Structure

```
d:\AI_Batch2\Commitment and deadlines\
├── app.py                     # Gradio web application (Blocks, Tabs, UI styling)
├── nlp_engine.py              # Core NLP extraction & evaluation pipeline
├── task.ipynb                 # Original Jupyter notebook with development & evaluation
└── README.md                  # Documentation and user guide
```

---

## 🛠️ Technology Stack
- **Python 3.14**
- **Gradio 6.31.0**
- **spaCy 3.8.18** (`en_core_web_sm`)
- **NLTK 3.10.4**
- **Pandas 3.0.6**
- **Matplotlib 3.11.2**

