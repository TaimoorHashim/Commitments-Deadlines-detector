"""
NLP Engine for Commitment and Deadline Extraction
Implements extraction logic based on task.ipynb with spaCy, regular expressions, and NLTK.
"""

import re
import spacy
import pandas as pd

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Base commitment patterns from notebook
BASE_COMMITMENT_PATTERNS = [
    r"\bi will\b",
    r"\bi'll\b",
    r"\bwill\b",
    r"\bpromised to\b",
    r"\bpromise to\b",
    r"\bneeds to\b",
    r"\bneed to\b",
    r"\bmust\b",
    r"\bhave to\b",
    r"\bhas to\b",
    r"\bshould\b",
]

# Action verbs from notebook
ACTION_VERBS = [
    "submit", "send", "finish", "complete",
    "prepare", "deliver", "review", "write",
    "create", "attend", "upload", "pay"
]

# Base deadline patterns from notebook
BASE_DEADLINE_PATTERNS = [
    r"\bby\s+(?:next\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    r"\bwithin\s+(?:\d+|one|two|three|four|five)\s+(?:day|days|week|weeks)\b",
    r"\bbefore\s+(?:next\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    r"\bon\s+(?:next\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    r"\b(?:today|tomorrow|tonight|yesterday)\b",
    r"\bnext\s+(?:week|month|year)\b",
    r"\bin\s+\d+\s+(?:day|days|week|weeks|month|months)\b",
    r"\bby\s+\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
    r"\bby\s+\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
]

# Extended deadline patterns for broader real-world coverage
EXTENDED_DEADLINE_PATTERNS = BASE_DEADLINE_PATTERNS + [
    r"\bon\s+\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
    r"\b(?:by|before|until)\s+end\s+of\s+(?:day|week|month)\b",
    r"\b(?:by|before|at)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)\b",
    r"\basap\b|\bas\s+soon\s+as\s+possible\b"
]

# Benchmark evaluation dataset from notebook
EVALUATION_DATA = [
    ("I will submit the assignment by Friday.", 1),
    ("Ahmed promised to send the report tomorrow.", 1),
    ("Sara needs to finish the presentation.", 1),
    ("I must complete my homework.", 1),
    ("We have to prepare the project.", 1),
    ("Bilal should review the document.", 1),
    ("I need to upload the files today.", 1),
    ("They will deliver the materials next week.", 1),
    ("I have to attend the meeting.", 1),
    ("She promised to call me tomorrow.", 1),
    ("The weather is pleasant today.", 0),
    ("I enjoy learning Python.", 0),
    ("The university is closed today.", 0),
    ("Ahmed likes playing cricket.", 0),
    ("The report is available online.", 0),
    ("We visited the university yesterday.", 0),
    ("The meeting was interesting.", 0),
    ("I am studying computer science.", 0),
    ("The weather will be pleasant tomorrow.", 0),
    ("We will see you at the university.", 0)
]


def detect_commitment(sentence: str, patterns=None) -> tuple[bool, str]:
    """Detects if sentence contains commitment pattern. Returns (bool, matched_pattern)."""
    if patterns is None:
        patterns = BASE_COMMITMENT_PATTERNS
    
    for pattern in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return True, match.group(0)
    return False, ""


def extract_person(sentence: str, resolve_pronouns: bool = False) -> str:
    """Extracts responsible person using spaCy NER and subject dependency detection."""
    doc = nlp(sentence)

    # 1. Named Entity Recognition
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text

    # 2. Grammatical subject detection (Proper Noun)
    for token in doc:
        if token.dep_ in ["nsubj", "nsubjpass"] and token.pos_ == "PROPN":
            return token.text

    # Optional: pronoun detection for speaker/team
    if resolve_pronouns:
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                if token.lower_ in ["i", "me", "myself"]:
                    return "I (Speaker)"
                elif token.lower_ in ["we", "us"]:
                    return "We (Team)"
                elif token.lower_ in ["you"]:
                    return "You (Assignee)"
                elif token.pos_ == "NOUN":
                    return token.text

    return "Unknown"


def extract_action(sentence: str, natural_order: bool = True) -> str | None:
    """
    Extracts action verb and its direct object phrase.
    natural_order=True retains the clean word order (e.g. 'submit the NLP assignment').
    natural_order=False mimics exact notebook token appending ('submit assignment the NLP').
    """
    doc = nlp(sentence)

    for token in doc:
        if token.lemma_.lower() in ACTION_VERBS:
            if natural_order:
                # Natural phrasing: collect verb and relevant subtree in sentence order
                obj_tokens = []
                for child in token.children:
                    if child.dep_ in ["dobj", "obj", "attr"]:
                        # Collect all words in the object phrase sorted by position
                        subtree_tokens = sorted(list(child.subtree), key=lambda t: t.i)
                        obj_tokens.extend([t.text for t in subtree_tokens])
                if obj_tokens:
                    return f"{token.text} {' '.join(obj_tokens)}"
                return token.text
            else:
                # Exact notebook behavior
                action = []
                for child in token.children:
                    if child.dep_ in ["dobj", "obj", "attr"]:
                        action.append(child.text)
                        for subchild in child.subtree:
                            if subchild != child:
                                action.append(subchild.text)
                if action:
                    return token.text + " " + " ".join(action)
                return token.text

    # Fallback: look for ROOT verb or auxiliary verb construct if no specific action verb matched
    return None


def extract_deadline(sentence: str, use_extended: bool = False) -> str | None:
    """Extracts deadline from sentence using regex patterns."""
    patterns = EXTENDED_DEADLINE_PATTERNS if use_extended else BASE_DEADLINE_PATTERNS
    for pattern in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return match.group(0)

    # Secondary check with spaCy DATE/TIME entities if extended mode
    if use_extended:
        doc = nlp(sentence)
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME"]:
                return ent.text

    return None


def extract_commitment_info(sentence: str, natural_order: bool = True, use_extended_deadlines: bool = False, resolve_pronouns: bool = False) -> dict:
    """Full extraction pipeline for a single sentence."""
    is_commitment, trigger = detect_commitment(sentence)
    
    result = {
        "Sentence": sentence.strip(),
        "Commitment Detected": is_commitment,
        "Trigger": trigger if is_commitment else "",
        "Responsible Person": "Unknown",
        "Action": None,
        "Deadline": None
    }

    if is_commitment:
        result["Responsible Person"] = extract_person(sentence, resolve_pronouns=resolve_pronouns)
        result["Action"] = extract_action(sentence, natural_order=natural_order)
        result["Deadline"] = extract_deadline(sentence, use_extended=use_extended_deadlines)

    return result


def analyze_text(text: str, natural_order: bool = True, use_extended_deadlines: bool = False, resolve_pronouns: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Processes paragraph/document by splitting into sentences and extracting commitments.
    Returns (DataFrame, summary_statistics).
    """
    if not text or not text.strip():
        empty_df = pd.DataFrame(columns=["Sentence", "Commitment Detected", "Trigger", "Responsible Person", "Action", "Deadline"])
        return empty_df, {"total": 0, "commitments": 0, "non_commitments": 0, "deadlines": 0, "persons": 0}

    doc = nlp(text.strip())
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    results = []
    for sent in sentences:
        res = extract_commitment_info(
            sent,
            natural_order=natural_order,
            use_extended_deadlines=use_extended_deadlines,
            resolve_pronouns=resolve_pronouns
        )
        results.append(res)

    df = pd.DataFrame(results)

    # Compute statistics
    total = len(df)
    commitments = int(df["Commitment Detected"].sum()) if total > 0 else 0
    non_commitments = total - commitments
    deadlines = int(df["Deadline"].notna().sum()) if total > 0 else 0
    unique_persons = int((df["Responsible Person"].replace("Unknown", None).dropna().nunique())) if total > 0 else 0

    stats = {
        "total": total,
        "commitments": commitments,
        "non_commitments": non_commitments,
        "deadlines": deadlines,
        "persons": unique_persons
    }

    return df, stats


def evaluate_model() -> tuple[dict, pd.DataFrame]:
    """Runs evaluation on standard benchmark dataset from task.ipynb."""
    actual_labels = []
    predicted_labels = []
    rows = []

    for sentence, actual in EVALUATION_DATA:
        pred_bool, trigger = detect_commitment(sentence)
        pred_int = int(pred_bool)
        actual_labels.append(actual)
        predicted_labels.append(pred_int)
        
        info = extract_commitment_info(sentence)
        rows.append({
            "Sentence": sentence,
            "Actual": "Commitment" if actual == 1 else "Non-Commitment",
            "Predicted": "Commitment" if pred_int == 1 else "Non-Commitment",
            "Correct": "✅ Yes" if actual == pred_int else "❌ No",
            "Person": info["Responsible Person"],
            "Action": info["Action"] or "-",
            "Deadline": info["Deadline"] or "-"
        })

    TP = sum(1 for a, p in zip(actual_labels, predicted_labels) if a == 1 and p == 1)
    FP = sum(1 for a, p in zip(actual_labels, predicted_labels) if a == 0 and p == 1)
    TN = sum(1 for a, p in zip(actual_labels, predicted_labels) if a == 0 and p == 0)
    FN = sum(1 for a, p in zip(actual_labels, predicted_labels) if a == 1 and p == 0)

    accuracy = (TP + TN) / len(actual_labels) if actual_labels else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1-Score": round(f1_score * 100, 2),
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Total": len(actual_labels)
    }

    results_df = pd.DataFrame(rows)
    return metrics, results_df

